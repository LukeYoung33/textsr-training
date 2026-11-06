"""Isolated full-frame objective extending the unchanged BasicSR SR lifecycle."""
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from basicsr.models.sr_model import SRModel
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.losses.fullframe_text_priority_loss import FullFrameTextPriorityLoss
from realesrgan.utils import YUV2RGB


@MODEL_REGISTRY.register()
class FullFrameTextPriorityModel(SRModel):
    """Use SRModel network/save/resume plumbing without the legacy line-loss init."""

    def __init__(self, opt):
        """@luna-comment
        Input: model options for scale-one V5 with offline aligned full-frame pairs.
        Output: initialized SRModel extension or an explicit unsupported-option error.
        Key Steps: reject silent preprocessing/degradation/queue behavior; validate backbone.
        Key Params: disabled legacy flags are allowed; only V5 scale-one is supported.
        State/Dependencies: delegates normal network/load/training initialization to SRModel.
        """
        if opt.get('scale', 1) != 1 or opt.get('network_g', {}).get('type') != 'BasicTextLineR4StudentY4UVV5':
            raise ValueError('FullFrameTextPriorityModel requires scale=1 and BasicTextLineR4StudentY4UVV5')
        if (opt.get('lq_preprocess') or {}).get('enabled', False):
            raise ValueError('full-frame model does not support lq_preprocess.enabled')
        for key in ('high_order_degradation', 'gt_usm', 'lq_usm', 'queue_size', 'queue'):
            if opt.get(key, False):
                raise ValueError('full-frame model does not support ' + key)
        from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import BasicTextLineR4StudentY4UVV5
        super().__init__(opt)

    def init_training_settings(self):
        """@luna-comment
        Input: SRModel options with train.fullframe_loss_opt and ordinary optimizer/scheduler.
        Output: isolated objective, optional EMA, optimizer and scheduler state.
        Key Steps: construct explicit full-frame objective; reuse BasicSR setup methods.
        Key Params: text_start_iter/ramp_iter delay local objectives; EMA defaults off.
        State/Dependencies: mutates new model state only; no legacy CP defaults are loaded.
        """
        self.net_g.train()
        train = self.opt['train']
        self.objective = FullFrameTextPriorityLoss(**train.get('fullframe_loss_opt', {})).to(self.device)
        self.text_start_iter = int(train.get('text_start_iter', 0))
        self.text_ramp_iter = int(train.get('text_ramp_iter', 0))
        if self.text_start_iter < 0 or self.text_ramp_iter < 0:
            raise ValueError('text schedule iterations must be nonnegative')
        self.ema_decay = float(train.get('ema_decay', 0))
        if not 0 <= self.ema_decay < 1:
            raise ValueError('ema_decay must be in [0,1)')
        if self.ema_decay:
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g')
            if load_path:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        """@luna-comment
        Input: F009 RGB batch, center/context masks, integer instance map and ROI lists.
        Output: device batch and packed native Y4UV LQ/GT available for one forward.
        Key Steps: reject missing contracts; transfer tensor fields; pack fixed color format.
        Key Params: RGB spatial dimensions must match and be multiples of eight.
        State/Dependencies: replaces per-step state; annotations never enter the generator.
        """
        required = ('lq_rgb', 'gt_rgb', 'valid_mask_rgb', 'context_valid_mask_rgb',
                    'text_mask_rgb', 'instance_map', 'text_rois')
        if any(key in data for key in ('kernel1', 'kernel2', 'sinc_kernel', 'kernel', 'gt_usm', 'lq_usm')):
            raise ValueError('full-frame batches do not support online degradation/USM fields')
        if any(key not in data for key in required):
            raise ValueError('missing F009 fields: ' + ','.join(k for k in required if k not in data))
        self.batch = {key: value.to(self.device) if torch.is_tensor(value) else value
                      for key, value in data.items()}
        self.lq_rgb, self.gt_rgb = self.batch['lq_rgb'], self.batch['gt_rgb']
        if self.lq_rgb.ndim != 4 or self.lq_rgb.shape[1] != 3 or self.lq_rgb.shape != self.gt_rgb.shape:
            raise ValueError('full-frame model requires paired B3HW RGB')
        if any(size % 8 for size in self.lq_rgb.shape[-2:]):
            raise ValueError('RGB canvas dimensions must be multiples of eight')
        if any(not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any() for t in (self.lq_rgb, self.gt_rgb)):
            raise ValueError('RGB inputs must be finite in [0,1]')
        self.lq, self.gt = rgb_to_y4uv(self.lq_rgb), rgb_to_y4uv(self.gt_rgb)

    def optimize_parameters(self, current_iter):
        """@luna-comment
        Input: current iteration and previously fed paired full-canvas batch.
        Output: one optimizer step, objective/ROI telemetry and optional EMA update.
        Key Steps: one generator forward; image and cropped local loss; backward/update.
        Key Params: linear text ramp leaves image reconstruction active throughout.
        State/Dependencies: BasicSR optimizer, scalar reduction and EMA state are reused.
        """
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq)
        elapsed = current_iter - self.text_start_iter
        scale = 0. if elapsed < 0 else min(1., elapsed / self.text_ramp_iter) if self.text_ramp_iter else 1.
        total, losses = self.objective(self.output, self.gt, self.batch, text_scale=scale)
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite full-frame objective')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.net_g.parameters()):
            raise FloatingPointError('nonfinite full-frame generator gradient')
        self.optimizer_g.step()
        losses['text_scale'] = total.new_tensor(scale)
        losses.update({'roi_' + key: total.new_tensor(value) for key, value in self.objective.last_stats.items()})
        self.log_dict = self.reduce_loss_dict(OrderedDict(losses))
        if self.ema_decay:
            self.model_ema(self.ema_decay)

    def get_current_visuals(self):
        """@luna-comment
        Input: current RGB source/GT and native output generated by train/test.
        Output: detached CPU RGB visual dictionary in the standard SRModel contract.
        Key Steps: decode fixed normalized UV without value-dependent range inference.
        Key Params: inference decoding clamps only visualization to [0,1].
        State/Dependencies: existing Y4UV and YUV2RGB conversion helpers; no network call.
        """
        return OrderedDict(lq=self.lq_rgb.detach().cpu(), gt=self.gt_rgb.detach().cpu(),
                           result=YUV2RGB(y4uv_to_yuv444(self.output)).detach().cpu())

    def diagnose_gradient_interaction(self):
        """@luna-comment
        Input: current fed F009 batch and enabled image/text objective settings.
        Output: generator gradient norms/cosine and measured-GT stationary-target audit.
        Key Steps: one diagnostic forward; separate autograd gradients; compare directions.
        Key Params: diagnostic is opt-in and never performs an optimizer step.
        State/Dependencies: objective telemetry is refreshed; parameter gradients unchanged.
        """
        parameters = [p for p in self.net_g.parameters() if p.requires_grad]
        with torch.enable_grad():
            prediction = self.net_g(self.lq)
            _, losses = self.objective(prediction, self.gt, self.batch)
            image = torch.autograd.grad(losses['l_image'], parameters, retain_graph=True, allow_unused=True)
            text = torch.autograd.grad(losses['l_text'], parameters, allow_unused=True)
        image_sq, text_sq, dot = 0., 0., 0.
        for left, right in zip(image, text):
            if left is not None:
                image_sq += float(left.square().sum())
            if right is not None:
                text_sq += float(right.square().sum())
            if left is not None and right is not None:
                dot += float((left * right).sum())
        image_norm, text_norm = image_sq ** .5, text_sq ** .5
        result = {'image_gradient_norm': image_norm, 'text_gradient_norm': text_norm,
                  'gradient_cosine': dot / max(image_norm * text_norm, 1e-12)}
        result.update({'gt_' + key: value for key, value in self.objective.audit_gt_gradient(self.gt, self.batch).items()})
        return result

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """@luna-comment
        Input: F009 validation loader, iteration, optional logger and image-save switch.
        Output: averaged configured image metrics and objective scalars, optional RGB files.
        Key Steps: skip empty trusted centers; same packed forward; aggregate each
            metric over supported tiles only and log skipped counts.
        Key Params: ordinary BasicSR image metrics are supported; no online OCR required.
        State/Dependencies: updates metric_results and BasicSR best-metric records.
        """
        val = self.opt.get('val', {})
        metrics = val.get('metrics') or {}
        result = {name: 0. for name in metrics}
        metric_counts = {name: 0 for name in metrics}
        seen, skipped = 0, 0
        objective_sum, count = 0., 0
        objective = getattr(self, 'objective', None)
        if objective is None:
            objective = FullFrameTextPriorityLoss(**self.opt.get('train', {}).get('fullframe_loss_opt', {})).to(self.device)
        for batch in dataloader:
            supported = batch['valid_mask_rgb'].flatten(1).sum(1) > 0
            seen += len(supported)
            supported_count = int(supported.sum())
            skipped += len(supported) - supported_count
            if not supported_count:
                continue
            self.feed_data(batch)
            self.test()
            with torch.no_grad():
                total, _ = objective(self.output, self.gt, self.batch)
            n = self.output.shape[0]
            objective_sum += float(total) * n
            visuals = self.get_current_visuals()
            for index in range(n):
                if not supported[index]:
                    continue
                sr = tensor2img(visuals['result'][index:index+1])
                gt = tensor2img(visuals['gt'][index:index+1])
                for name, options in metrics.items():
                    support = self.batch['valid_mask_rgb'][index, 0].detach().cpu().numpy().copy()
                    settings = deepcopy(options)
                    if settings.get('type') == 'calculate_psnr':
                        from basicsr.metrics.metric_util import to_y_channel
                        sr_metric, gt_metric = sr.astype(np.float64), gt.astype(np.float64)
                        if settings.get('test_y_channel', False):
                            sr_metric, gt_metric = to_y_channel(sr_metric), to_y_channel(gt_metric)
                        border = int(settings.get('crop_border', 0))
                        if border < 0:
                            raise ValueError('metric crop_border must be nonnegative')
                        if border:
                            support[:border] = support[-border:] = 0
                            support[:, :border] = support[:, -border:] = 0
                        if support.sum() <= 0:
                            continue
                        squared = (sr_metric - gt_metric) ** 2
                        if squared.ndim == 2:
                            squared = squared[..., None]
                        mse = (squared * support[..., None]).sum() / (support.sum() * squared.shape[2])
                        result[name] += float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)
                    else:
                        ys, xs = np.where(support > 0)
                        if not len(ys):
                            continue
                        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                        if not np.all(support[y0:y1, x0:x1] == 1):
                            raise ValueError('configured metric requires rectangular trusted support; use masked PSNR')
                        result[name] += calculate_metric({'img': sr[y0:y1, x0:x1], 'img2': gt[y0:y1, x0:x1]}, settings)
                    metric_counts[name] += 1
                if save_img:
                    paths = batch.get('lq_path', [])
                    name = Path(paths[index]).stem if len(paths) > index else f'tile_{count + index}'
                    imwrite(sr, str(Path(self.opt['path']['visualization']) / f'{name}_{count + index}_{current_iter}.png'))
            count += supported_count
        if count == 0:
            raise ValueError('full-frame validation loader is empty' if not seen else
                             'validation dataset has no trusted support')
        for name in metrics:
            if not metric_counts[name]:
                raise ValueError(f'validation metric {name} has no trusted support')
        self.metric_results = {name: value / metric_counts[name] for name, value in result.items()}
        self.validation_objective = objective_sum / count
        dataset_name = dataloader.dataset.opt.get('name', 'fullframe')
        if metrics:
            self._initialize_best_metric_results(dataset_name)
            for name, value in self.metric_results.items():
                self._update_best_metric_result(dataset_name, name, value, current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.validation_support = dict(seen=seen, supported=count, skipped=skipped, metric_counts=metric_counts)
        get_root_logger().info(f'Full-frame validation {dataset_name}: objective={self.validation_objective:.6g}, '
                              f'tiles={count}, seen={seen}, skipped={skipped}, metric_counts={metric_counts}')
        if tb_logger:
            tb_logger.add_scalar(f'validation/{dataset_name}/objective', self.validation_objective, current_iter)
