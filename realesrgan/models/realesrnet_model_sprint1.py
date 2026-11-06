"""
realesrnet_model_sprint1.py  –  Sprint-1 variant of RealESRNetModel

Key difference from realesrnet_model.py:

  optimize_parameters():
    The original code contains a hard-coded warm-up that forces
    cri_rec.loss_weight = 1.0 for the first 25 000 iterations regardless of
    what the yml says.  Given that CRNN feature activations are ~440x larger
    in magnitude than the pixel L1 loss (measured empirically), this warm-up
    makes l_rec completely dominate training for the first 25k steps, sending
    the model in the wrong direction.

    This version removes that override.  The loss_weight in the yml is
    respected from step 0.  If a warm-up is desired in the future, it should
    ramp *up* from 0 → target_weight rather than starting at an uncalibrated 1.0.

    Everything else (feed_data paths, validation, queue, colour conversion,
    etc.) is identical to the original.

Usage in yml:
    model_type: RealESRNetModelSprint1
"""

from realesrgan.models.realesrnet_model import RealESRNetModel
from realesrgan.models.ocr_schedule import get_ocr_loss_scale
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.losses import build_loss
from collections import OrderedDict


@MODEL_REGISTRY.register()
class RealESRNetModelSprint1(RealESRNetModel):
    """Sprint-1 model: removes the hard-coded cri_rec warm-up override."""

    def init_training_settings(self):
        """Initialize legacy losses and the independent optional CP loss.

        @shanlan-comment
        Input: model options containing the inherited loss blocks and optional ``train.cp_opt``.
        Output: initialized loss modules, including ``cri_cp`` when configured.
        Key Steps: delegate all existing setup first, then build CP without changing rec/ocr behavior.
        Key Params: ``cp_opt`` is independent; its schedule is read during optimization.
        State/Dependencies: mutates loss modules and optimizer state through the inherited setup.
        """
        super().init_training_settings()
        train_opt = self.opt.get('train', {})
        if train_opt.get('cp_opt'):
            self.cri_cp = build_loss(train_opt['cp_opt']).to(self.device)
        else:
            self.cri_cp = None

    def optimize_parameters(self, current_iter):
        """Compute configured reconstruction losses and update the generator.

        @shanlan-comment
        Input: current iteration number and model tensors prepared by ``feed_data``; YUV modes retain RGB GT in ``gt_rgb``.
        Output: one optimizer step, reduced scalar losses in ``log_dict``, and optional EMA update.
        Key Steps: run pixel/perceptual/recognition/OCR losses, build a YUV-compatible gradient target, then backpropagate.
        Key Params: configured loss blocks control attribution; ``grad_opt`` receives the same spatial mask as pixel loss.
        State/Dependencies: reads ``lq``, ``gt``, ``gt_rgb``, ``mask``, and loss modules; mutates generator parameters and logs.
        """
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = self.output.sum() * 0.0
        loss_dict = OrderedDict()

        # ── pixel loss ────────────────────────────────────────────────────────
        if self.cri_pix:
            from realesrgan.utils import RGB2YUV
            if self.color in ('yuv', 'nv12', 'nv21'):
                if isinstance(self.output, __import__('torch').Tensor):
                    l_pix = self.cri_pix(
                        self.output,
                        RGB2YUV(self.gt_rgb, out_format='yuv'),
                        mask=self.mask,
                    )
                else:
                    out_y, out_u, out_v = self.output[:]
                    gt_y, gt_u, gt_v = RGB2YUV(self.gt_rgb, out_format='NV21', split=True)
                    import torch.nn.functional as _F
                    l_pix = (
                        self.cri_pix(out_y, gt_y, mask=self.mask)
                        + self.cri_pix(out_u, gt_u, mask=_F.avg_pool2d(self.mask, 2, 2))
                        + self.cri_pix(out_v, gt_v, mask=_F.avg_pool2d(self.mask, 2, 2))
                    )
            else:
                l_pix = self.cri_pix(self.output, self.gt, mask=self.mask)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_pix2 and isinstance(self.output, __import__('torch').Tensor):
            from realesrgan.utils import RGB2YUV
            l_pix2 = self.cri_pix2(
                self.output,
                RGB2YUV(self.gt_rgb, out_format='yuv'),
                mask=self.mask,
            )
            l_total += l_pix2
            loss_dict['l_pix_yuv'] = l_pix2

        # ── VGG / perceptual loss ─────────────────────────────────────────────
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.cvtColor(self.output), self.gt_rgb)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # ── CRNN perceptual loss (rec) ─────────────────────────────────────────
        # NOTE: loss_weight is taken directly from the yml at every step.
        #       The original warm-up logic (forcing weight=1.0 for iter<25000)
        #       has been removed because it caused l_rec to dominate l_pix
        #       by ~1800x during the critical early training phase.
        if self.cri_rec:
            if self.color in ('yuv', 'nv12', 'nv21'):
                l_rec = self.cri_rec(self.cvtColor(self.output), self.gt_rgb, mask=self.mask)
            else:
                l_rec = self.cri_rec(self.output, self.gt, mask=self.mask)
            l_total += l_rec
            loss_dict['l_rec'] = l_rec

        # ── OCR CTC / text-region loss ────────────────────────────────────────
        if self.cri_ocr:
            train_opt = self.opt.get('train', {})
            ocr_scale = get_ocr_loss_scale(
                current_iter,
                train_opt.get('ocr_start_iter', 0),
                train_opt.get('ocr_ramp_iters', 0),
            )
            loss_dict['ocr_scale'] = self.output.new_tensor(ocr_scale)
            if ocr_scale > 0.0:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    l_ocr_raw = self.cri_ocr(
                        self.cvtColor(self.output), self.gt_rgb, self.gt_lines)
                else:
                    l_ocr_raw = self.cri_ocr(self.output, self.gt, self.gt_lines)
                l_ocr = l_ocr_raw * ocr_scale
                l_total += l_ocr
                loss_dict['l_ocr'] = l_ocr

        # ── text-line CRNN content perceptual loss (cp) ──────────────────────
        if self.cri_cp:
            train_opt = self.opt.get('train', {})
            cp_start = train_opt.get('cp_start_iter', 0)
            cp_ramp = train_opt.get('cp_ramp_iters', 0)
            cp_scale = get_ocr_loss_scale(current_iter, cp_start, cp_ramp)
            loss_dict['cp_scale'] = self.output.new_tensor(cp_scale)
            if not hasattr(self, 'gt_lines'):
                raise RuntimeError('CRNN content loss is enabled but gt_lines is missing')
            if cp_scale > 0.0:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    cp_output = self.cvtColor(self.output)
                    cp_target = self.gt_rgb
                else:
                    cp_output = self.output
                    cp_target = self.gt
                l_cp_raw = self.cri_cp(cp_output, cp_target, self.gt_lines)
                l_cp = l_cp_raw * cp_scale
                l_total += l_cp
                loss_dict['l_cp_raw'] = l_cp_raw
                loss_dict['l_cp'] = l_cp
                cp_stats = getattr(self.cri_cp, 'last_stats', {})
                for stat_name in ('accepted_lines', 'skipped_lines', 'clamp_ratio', 'width_p50', 'width_p95', 'width_max'):
                    if stat_name in cp_stats:
                        loss_dict[f'cp_{stat_name}'] = self.output.new_tensor(cp_stats[stat_name])
                for layer_name, layer_value in cp_stats.get('layer_losses', {}).items():
                    loss_dict[f'cp_layer_{layer_name}'] = self.output.new_tensor(layer_value)

        # ── gradient loss ─────────────────────────────────────────────────────
        if self.cri_grad:
            from realesrgan.utils import RGB2YUV
            if self.color in ('yuv', 'nv12', 'nv21'):
                grad_target = RGB2YUV(self.gt_rgb, out_format='yuv')
                grad_pred = self.output
            else:
                grad_target = self.gt
                grad_pred = self.output
            l_grad = self.cri_grad(grad_pred, grad_target, mask=self.mask)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
