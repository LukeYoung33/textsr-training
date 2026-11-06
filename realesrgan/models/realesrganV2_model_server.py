import copy
import numpy as np
import random
import torch
import cv2
from torch.nn import functional as F
from tqdm import tqdm
import os.path as osp
from collections import OrderedDict

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.losses import build_loss
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp, tensor2img, imwrite
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from realesrgan.utils import YUV2RGB, RGB2YUV, YUV2RGBConv
import editdistance


# --------------------------------------------------------------------------- #
# Blur kernel helpers (shared with RealESRNetModel)
# --------------------------------------------------------------------------- #

def _sample_odd_ksize(ksize_range, default=(3, 11)):
    if ksize_range is None:
        ksize_range = default
    k_low, k_high = int(ksize_range[0]), int(ksize_range[1])
    k_low = max(3, k_low)
    k_high = max(k_low, k_high)
    if k_low % 2 == 0:
        k_low += 1
    if k_high % 2 == 0:
        k_high -= 1
    if k_high < k_low:
        k_high = k_low
    if k_low == k_high:
        return k_low
    return random.randrange(k_low, k_high + 1, 2)


def _make_gaussian_kernel(k, sigma=0, device=None, dtype=torch.float32):
    if k % 2 == 0:
        k += 1
    g1 = cv2.getGaussianKernel(k, sigma)
    kernel = np.dot(g1, g1.T).astype(np.float32)
    kernel = torch.from_numpy(kernel).to(dtype=dtype)
    if device is not None:
        kernel = kernel.to(device)
    return kernel


def _make_motion_kernel(k, length=None, angle_deg=None, device=None, dtype=torch.float32):
    if k % 2 == 0:
        k += 1
    if length is None:
        length = k
    if angle_deg is None:
        angle_deg = np.random.rand() * 360.0
    kernel = np.zeros((k, k), dtype=np.float32)
    c = k // 2
    theta = np.deg2rad(angle_deg)
    dx = (length - 1) / 2.0 * np.cos(theta)
    dy = (length - 1) / 2.0 * np.sin(theta)
    x0 = int(round(c - dx)); y0 = int(round(c - dy))
    x1 = int(round(c + dx)); y1 = int(round(c + dy))
    cv2.line(kernel, (x0, y0), (x1, y1), 1.0, thickness=1)
    s = kernel.sum()
    if s <= 0:
        kernel[c, c] = 1.0
        s = 1.0
    kernel /= s
    kernel = torch.from_numpy(kernel).to(dtype=dtype)
    if device is not None:
        kernel = kernel.to(device)
    return kernel


def _build_blur_kernels(batch, k, mode, device, dtype):
    if mode == 'normal':
        return _make_gaussian_kernel(k, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
    if mode == 'sim':
        return torch.stack([_make_motion_kernel(k, device=device, dtype=dtype) for _ in range(batch)], dim=0)
    # 'rand': 50/50 gaussian vs motion per sample
    g_kernel = _make_gaussian_kernel(k, device=device, dtype=dtype)
    kernels = []
    for _ in range(batch):
        if random.random() < 0.5:
            kernels.append(g_kernel)
        else:
            kernels.append(_make_motion_kernel(k, device=device, dtype=dtype))
    return torch.stack(kernels, dim=0)


def _apply_degradation_blur(img, k, mode):
    b = img.size(0)
    kernel = _build_blur_kernels(b, k, mode, device=img.device, dtype=img.dtype)
    return filter2D(img, kernel)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@MODEL_REGISTRY.register()
class RealESRGANV2Model(SRGANModel):
    """RealESRGAN GAN model refactored to align feed_data / loss / optimization
    patterns with RealESRNetModel.

    Key changes vs the original RealESRGANModel:
    - Shared helper methods (_load_paired_data, _apply_color_conversion, etc.)
    - feed_data dispatches to v2 / v3 variants via config flags
    - _dequeue_and_enqueue supports gt_lines
    - optimize_parameters fixes the l_total/l_g_total bug and adds cri_ocr /
      cri_grad / cri_pix2 branches
    - nondist_validation uses the cleaner ned/recognition aggregation logic
    - Full 'yuv' / 'nv12' / 'nv21' color space support via YUV2RGBConv
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', 180)

        self.color = opt.get('color', 'rgb').lower()
        print(f'###################### COLOR is {self.color} ######################')
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.cvtColor = YUV2RGBConv().to(self.device)

        self.finetune_deg = opt.get('finetune_deg', None)

        if self.opt.get('use_feed_data_v2', False):
            print('Notice: Feed data V2 activated.')
        if self.opt.get('use_feed_data_v3', False):
            print('Notice: Feed data V3 activated.')

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        if train_opt.get('ppocr_percep_opt'):
            self.cri_ppocr_percep = build_loss(train_opt['ppocr_percep_opt']).to(self.device)
        else:
            self.cri_ppocr_percep = None
        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

    # ----------------------------------------------------------------------- #
    # Parameter annealing
    # ----------------------------------------------------------------------- #

    def _update_annealed_params(self, current_iter):
        """Update pixel-loss weight and D lr according to linear anneal schedules.

        Both schedules are driven purely by current_iter so resumed checkpoints
        automatically land on the correct value without any extra state.

        pixel_anneal_opt keys: start_iter, end_iter, start_weight, end_weight
        d_lr_anneal_opt  keys: start_iter, end_iter, start_lr,     end_lr

        D lr annealing is applied *after* the global LR scheduler fires, so it
        overrides the scheduler for optimizer_d while the schedule is active.
        """
        train_opt = self.opt['train']

        # ---- pixel loss weight ----
        pix_ann = train_opt.get('pixel_anneal_opt')
        if pix_ann and self.cri_pix is not None:
            start = pix_ann.get('start_iter', 0)
            end   = pix_ann.get('end_iter', 100000)
            w0    = pix_ann.get('start_weight', self.cri_pix.loss_weight)
            w1    = pix_ann.get('end_weight', w0 * 0.1)
            if current_iter >= start:
                t = min(float(current_iter - start) / max(end - start, 1), 1.0)
                self.cri_pix.loss_weight = w0 + (w1 - w0) * t

        # ---- discriminator learning rate ----
        d_lr_ann = train_opt.get('d_lr_anneal_opt')
        if d_lr_ann:
            start = d_lr_ann.get('start_iter', 0)
            end   = d_lr_ann.get('end_iter', 100000)
            lr0   = d_lr_ann.get('start_lr', train_opt['optim_d']['lr'])
            lr1   = d_lr_ann.get('end_lr', lr0 * 0.1)
            if current_iter >= start:
                t = min(float(current_iter - start) / max(end - start, 1), 1.0)
                new_lr = lr0 + (lr1 - lr0) * t
                for pg in self.optimizer_d.param_groups:
                    pg['lr'] = new_lr

    # ----------------------------------------------------------------------- #
    # Training pair pool
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, (
                f'queue size {self.queue_size} should be divisible by batch size {b}')
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            if hasattr(self, 'gt_lines'):
                self.queue_gt_lines = [None] * self.queue_size
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            if hasattr(self, 'queue_gt_lines'):
                idx_list = idx.tolist()
                self.queue_gt_lines = [self.queue_gt_lines[i] for i in idx_list]

            lq_dequeue = self.queue_lr[0:b].clone()
            gt_dequeue = self.queue_gt[0:b].clone()
            if hasattr(self, 'queue_gt_lines'):
                gt_lines_dequeue = copy.deepcopy(self.queue_gt_lines[0:b])

            self.queue_lr[0:b] = self.lq.clone()
            self.queue_gt[0:b] = self.gt.clone()
            if hasattr(self, 'gt_lines'):
                if not hasattr(self, 'queue_gt_lines'):
                    self.queue_gt_lines = [None] * self.queue_size
                self.queue_gt_lines[0:b] = copy.deepcopy(self.gt_lines)

            self.lq = lq_dequeue
            self.gt = gt_dequeue
            if hasattr(self, 'queue_gt_lines'):
                self.gt_lines = gt_lines_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr + b] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b] = self.gt.clone()
            if hasattr(self, 'gt_lines'):
                if not hasattr(self, 'queue_gt_lines'):
                    self.queue_gt_lines = [None] * self.queue_size
                self.queue_gt_lines[self.queue_ptr:self.queue_ptr + b] = copy.deepcopy(self.gt_lines)
            self.queue_ptr += b

    # ----------------------------------------------------------------------- #
    # Data helpers
    # ----------------------------------------------------------------------- #

    def _extract_data_scale(self, data=None, gt=None, lq=None):
        scale_value = None
        if data is not None and 'data_scale' in data:
            scale_value = data['data_scale']
            if torch.is_tensor(scale_value):
                scale_value = scale_value.flatten()[0].item()
            elif isinstance(scale_value, (list, tuple)):
                scale_value = scale_value[0]
        if scale_value is None and gt is not None and lq is not None:
            h_lq, w_lq = lq.shape[-2:]
            h_gt, w_gt = gt.shape[-2:]
            if h_lq > 0 and w_lq > 0:
                scale_h = h_gt // h_lq
                scale_w = w_gt // w_lq
                if scale_h == scale_w and scale_h > 0:
                    scale_value = scale_h
        if scale_value is None:
            scale_value = self.opt.get('scale', 1)
        return max(int(scale_value), 1)

    def _load_paired_data(self, data, apply_usm=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            if apply_usm and self.opt.get('gt_usm', False):
                self.gt = self.usm_sharpener(self.gt)
        if 'mask' in data and data['mask'] is not None:
            self.mask = data['mask'].to(self.device)
        else:
            self.mask = None
        if 'gt_lines' in data:
            self.gt_lines = copy.deepcopy(data['gt_lines'])
        elif hasattr(self, 'gt_lines'):
            del self.gt_lines
        self.current_data_scale = self._extract_data_scale(
            data=data, gt=getattr(self, 'gt', None), lq=self.lq)

    def _apply_color_conversion(self):
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.gt_rgb = self.gt
            self.lq_rgb = self.lq
            self.lq = RGB2YUV(self.lq, out_format=self.color)
            self.gt = RGB2YUV(self.gt, out_format=self.color)

    def _apply_noise(self, img, gaussian_noise_prob, noise_range, poisson_scale_range, gray_noise_prob):
        if np.random.uniform() < gaussian_noise_prob:
            return random_add_gaussian_noise_pt(
                img, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        return random_add_poisson_noise_pt(
            img, scale_range=poisson_scale_range, gray_prob=gray_noise_prob, clip=True, rounds=False)

    def _apply_jpeg(self, img, jpeg_range):
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*jpeg_range)
        img = torch.clamp(img, 0, 1)
        return self.jpeger(img, quality=jpeg_p)

    def _apply_paired_lq_degradation_v3(self, img):
        out = self._apply_noise(
            img,
            gaussian_noise_prob=self.opt['gaussian_noise_prob'],
            noise_range=self.opt['noise_range'],
            poisson_scale_range=self.opt['poisson_scale_range'],
            gray_noise_prob=self.opt['gray_noise_prob'])
        out = self._apply_jpeg(out, self.opt['jpeg_range'])
        out = self._apply_noise(
            out,
            gaussian_noise_prob=self.opt['gaussian_noise_prob2'],
            noise_range=self.opt['noise_range2'],
            poisson_scale_range=self.opt['poisson_scale_range2'],
            gray_noise_prob=self.opt['gray_noise_prob2'])
        out = self._apply_jpeg(out, self.opt['jpeg_range2'])
        return torch.clamp((out * 255.0).round(), 0, 255) / 255.

    # ----------------------------------------------------------------------- #
    # Paired-data shortcuts (train / val)
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def feed_data_train_pair(self, data, use_queue=False, apply_usm=False):
        self._load_paired_data(data, apply_usm=apply_usm)
        if use_queue:
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()
        self._apply_color_conversion()

    @torch.no_grad()
    def feed_data_val_pair(self, data):
        self._load_paired_data(data, apply_usm=False)
        if 'label_mask' in data:
            self.label_mask = data['label_mask'].to(self.device).float()
        if 'label_mask_gt' in data:
            self.label_mask = data['label_mask_gt'].to(self.device).float()

        lq_h, lq_w = self.lq.shape[2:]
        down_rate = self.opt['datasets']['val'].get('down_rate', 1)
        scale = self.current_data_scale
        pad_h = int(np.ceil(lq_h / down_rate) * down_rate - lq_h)
        pad_w = int(np.ceil(lq_w / down_rate) * down_rate - lq_w)
        self.lq = F.pad(self.lq, pad=(0, pad_w, 0, pad_h), mode='reflect')
        if hasattr(self, 'gt'):
            self.gt = F.pad(self.gt, pad=(0, pad_w * scale, 0, pad_h * scale), mode='reflect')
        if self.mask is not None:
            self.mask = F.pad(self.mask, pad=(0, pad_w * scale, 0, pad_h * scale), mode='reflect')
        self._apply_color_conversion()

    # ----------------------------------------------------------------------- #
    # Synthetic degradation helpers
    # ----------------------------------------------------------------------- #

    def _synthesize_lq_standard(self, gt):
        """Standard two-order degradation (original RealESRGAN pipeline)."""
        ori_h, ori_w = gt.size()[2:4]

        # --- first degradation ---
        out = filter2D(gt, self.kernel1)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        scale = (np.random.uniform(1, self.opt['resize_range'][1]) if updown_type == 'up'
                 else np.random.uniform(self.opt['resize_range'][0], 1) if updown_type == 'down'
                 else 1)
        out = F.interpolate(out, scale_factor=scale, mode=random.choice(['area', 'bilinear', 'bicubic']))
        out = self._apply_noise(out, self.opt['gaussian_noise_prob'], self.opt['noise_range'],
                                self.opt['poisson_scale_range'], self.opt['gray_noise_prob'])
        out = self._apply_jpeg(out, self.opt['jpeg_range'])

        # --- second degradation ---
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        scale = (np.random.uniform(1, self.opt['resize_range2'][1]) if updown_type == 'up'
                 else np.random.uniform(self.opt['resize_range2'][0], 1) if updown_type == 'down'
                 else 1)
        out = F.interpolate(
            out,
            size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)),
            mode=random.choice(['area', 'bilinear', 'bicubic']))
        out = self._apply_noise(out, self.opt['gaussian_noise_prob2'], self.opt['noise_range2'],
                                self.opt['poisson_scale_range2'], self.opt['gray_noise_prob2'])

        # --- sinc + JPEG (two random orders) ---
        target_size = (ori_h // self.opt['scale'], ori_w // self.opt['scale'])
        if np.random.uniform() < 0.5:
            out = F.interpolate(out, size=target_size, mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, self.sinc_kernel)
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
        else:
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
            out = F.interpolate(out, size=target_size, mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, self.sinc_kernel)

        return torch.clamp((out * 255.0).round(), 0, 255) / 255.

    def _synthesize_lq_v2(self, gt):
        """Two-order degradation with gaussian+motion blur (v2 variant)."""
        ori_h, ori_w = gt.size()[2:4]
        blur_mode = self.opt.get('blur_mode', 'rand')
        blur_ksize_range = self.opt.get('blur_ksize_range', (3, 11))
        blur_ksize_range2 = self.opt.get('blur_ksize_range2', blur_ksize_range)

        # --- first degradation ---
        if hasattr(self, 'kernel1') and self.kernel1 is not None:
            out = filter2D(gt, self.kernel1)
        else:
            out = _apply_degradation_blur(gt, _sample_odd_ksize(blur_ksize_range), blur_mode)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        scale = (np.random.uniform(1, self.opt['resize_range'][1]) if updown_type == 'up'
                 else np.random.uniform(self.opt['resize_range'][0], 1) if updown_type == 'down'
                 else 1)
        out = F.interpolate(out, scale_factor=scale, mode=random.choice(['area', 'bilinear', 'bicubic']))
        out = self._apply_noise(out, self.opt['gaussian_noise_prob'], self.opt['noise_range'],
                                self.opt['poisson_scale_range'], self.opt['gray_noise_prob'])
        out = self._apply_jpeg(out, self.opt['jpeg_range'])

        # --- second degradation ---
        if np.random.uniform() < self.opt['second_blur_prob']:
            if hasattr(self, 'kernel2') and self.kernel2 is not None:
                out = filter2D(out, self.kernel2)
            else:
                out = _apply_degradation_blur(out, _sample_odd_ksize(blur_ksize_range2), blur_mode)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        scale = (np.random.uniform(1, self.opt['resize_range2'][1]) if updown_type == 'up'
                 else np.random.uniform(self.opt['resize_range2'][0], 1) if updown_type == 'down'
                 else 1)
        out = F.interpolate(
            out,
            size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)),
            mode=random.choice(['area', 'bilinear', 'bicubic']))
        out = self._apply_noise(out, self.opt['gaussian_noise_prob2'], self.opt['noise_range2'],
                                self.opt['poisson_scale_range2'], self.opt['gray_noise_prob2'])

        target_size = (ori_h // self.opt['scale'], ori_w // self.opt['scale'])
        if np.random.uniform() < 0.5:
            out = F.interpolate(out, size=target_size, mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, self.sinc_kernel)
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
        else:
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
            out = F.interpolate(out, size=target_size, mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, self.sinc_kernel)

        return torch.clamp((out * 255.0).round(), 0, 255) / 255.

    def _synthesize_lq_v3(self, gt):
        """V3: no resize in second stage; sinc-only spatial op."""
        out = filter2D(gt, self.kernel1)
        ori_h, ori_w = gt.size()[2:4]
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        scale = (np.random.uniform(1, self.opt['resize_range'][1]) if updown_type == 'up'
                 else np.random.uniform(self.opt['resize_range'][0], 1) if updown_type == 'down'
                 else 1)
        mode=random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        out = self._apply_noise(out, self.opt['gaussian_noise_prob'], self.opt['noise_range'],
                                self.opt['poisson_scale_range'], self.opt['gray_noise_prob'])
        out = self._apply_jpeg(out, self.opt['jpeg_range'])

        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        out = self._apply_noise(out, self.opt['gaussian_noise_prob2'], self.opt['noise_range2'],
                                self.opt['poisson_scale_range2'], self.opt['gray_noise_prob2'])

        if np.random.uniform() < 0.5:
            out = filter2D(out, self.sinc_kernel)
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
        else:
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
            out = filter2D(out, self.sinc_kernel)

        out = F.interpolate(out, (ori_h, ori_w), mode=mode)

        return torch.clamp((out * 255.0).round(), 0, 255) / 255.

    # ----------------------------------------------------------------------- #
    # feed_data entry points
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def feed_data(self, data):
        if self.opt.get('use_feed_data_v3', False):
            return self._feed_data_synthetic(data, synth_fn=self._synthesize_lq_v3, use_paired_deg=True)
        if self.opt.get('use_feed_data_v2', False):
            return self._feed_data_synthetic(data, synth_fn=self._synthesize_lq_v2, use_paired_deg=False)
        return self._feed_data_synthetic(data, synth_fn=self._synthesize_lq_standard, use_paired_deg=False)

    @torch.no_grad()
    def _feed_data_synthetic(self, data, synth_fn, use_paired_deg=False):
        """Unified synthetic degradation path.

        Args:
            synth_fn: callable that takes a GT tensor and returns an LQ tensor.
            use_paired_deg: when falling back to paired data (finetune_deg), also
                apply light paired degradation (v3 behaviour).
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            self.gt = data['gt'].to(self.device)
            if 'gt_lines' in data:
                self.gt_lines = copy.deepcopy(data['gt_lines'])
            elif hasattr(self, 'gt_lines'):
                del self.gt_lines
            if self.opt.get('gt_usm', False):
                self.gt = self.usm_sharpener(self.gt)
            self.mask = data['mask'].to(self.device) if ('mask' in data and data['mask'] is not None) else None

            use_synthetic = self.finetune_deg is None or random.random() > self.finetune_deg
            if use_synthetic:
                self.kernel1 = data['kernel1'].to(self.device)
                self.kernel2 = data['kernel2'].to(self.device)
                self.sinc_kernel = data['sinc_kernel'].to(self.device)
                self.lq = synth_fn(self.gt)
            else:
                if use_paired_deg:
                    self._load_paired_data(data, apply_usm=True)
                    self.lq = self._apply_paired_lq_degradation_v3(self.lq)
                else:
                    self.lq = data['lq'].to(self.device)

            # GAN training requires USM-sharpened GT targets
            self.gt_usm = self.usm_sharpener(self.gt)

            # crop + queue
            if use_synthetic:
                gt_size = self.opt['gt_size']
                (self.gt, self.gt_usm), self.lq = paired_random_crop(
                    [self.gt, self.gt_usm], self.lq, gt_size, self.opt['scale'])

            self._dequeue_and_enqueue()
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()
        else:
            self.feed_data_train_pair(data, use_queue=False, apply_usm=False)
            if hasattr(self, 'gt'):
                self.gt_usm = self.usm_sharpener(self.gt)
            return

        self._apply_color_conversion()

    def feed_data_val(self, data):
        self.feed_data_val_pair(data)
        if (hasattr(self, 'gt_lines') and isinstance(self.gt_lines, list)
                and len(self.gt_lines) == 1 and isinstance(self.gt_lines[0], list)):
            self.gt_lines = self.gt_lines[0]

    # ----------------------------------------------------------------------- #
    # Optimization
    # ----------------------------------------------------------------------- #

    def optimize_parameters(self, current_iter):
        self._update_annealed_params(current_iter)

        # select GT targets for each loss according to USM flags
        l1_gt     = self.gt if self.opt.get('l1_gt_usm')     is False else self.gt_usm
        percep_gt = self.gt if self.opt.get('percep_gt_usm') is False else self.gt_usm
        gan_gt    = self.gt if self.opt.get('gan_gt_usm')    is False else self.gt_usm

        # ------------------------------------------------------------------- #
        # Generator update
        # ------------------------------------------------------------------- #
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if current_iter % self.net_g_iters == 0 and current_iter > self.net_d_init_iters:
            # pixel loss
            if self.cri_pix:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    l_g_pix = self.cri_pix(self.output, RGB2YUV(l1_gt, out_format='yuv'), mask=self.mask)
                else:
                    l_g_pix = self.cri_pix(self.output, l1_gt, mask=self.mask)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # FFT amplitude loss
            if self.cri_fft:
                output_rgb = self.cvtColor(self.output) if self.color in ('yuv', 'nv12', 'nv21') else self.output
                l1_gt_rgb  = self.gt_rgb if self.color in ('yuv', 'nv12', 'nv21') else l1_gt
                l_g_fft = self.cri_fft(output_rgb, l1_gt_rgb)
                l_g_total += l_g_fft
                loss_dict['l_g_fft'] = l_g_fft

            # second pixel loss
            if self.cri_pix2:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    l_g_pix2 = self.cri_pix2(self.output, RGB2YUV(l1_gt, out_format='yuv'), mask=self.mask)
                else:
                    l_g_pix2 = self.cri_pix2(self.output, l1_gt, mask=self.mask)
                l_g_total += l_g_pix2
                loss_dict['l_g_pix2'] = l_g_pix2

            # perceptual loss
            if self.cri_perceptual:
                output_rgb = self.cvtColor(self.output) if self.color in ('yuv', 'nv12', 'nv21') else self.output
                percep_gt_rgb = self.gt_rgb if self.color in ('yuv', 'nv12', 'nv21') else percep_gt
                l_g_percep, l_g_style = self.cri_perceptual(output_rgb, percep_gt_rgb)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # PPOCRv5 backbone perceptual loss
            if self.cri_ppocr_percep:
                output_rgb = self.cvtColor(self.output) if self.color in ('yuv', 'nv12', 'nv21') else self.output
                gt_rgb = self.gt_rgb if self.color in ('yuv', 'nv12', 'nv21') else l1_gt
                l_g_ppocr = self.cri_ppocr_percep(output_rgb, gt_rgb, mask=self.mask)
                l_g_total += l_g_ppocr
                loss_dict['l_g_ppocr_percep'] = l_g_ppocr

            # GAN loss
            # NOTE: call D(real) FIRST to match the SN-update order in the D step,
            # so the effective spectral-norm weights are consistent between the two steps.
            gan_type = self.opt['train']['gan_opt']['type']
            if gan_type == 'RelativisticSoftplusGANLoss':
                target = self.net_d(gan_gt).detach().clone()  # real first, same as D step
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(target, fake_g_pred, is_disc=False)
            else:
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            if self.cri_canny:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    l_percep, l_style = self.cri_canny(torch.cat([self.cvtColor(self.output), self.out_canny], dim=1), torch.cat([self.gt_rgb, self.gt_canny], dim=1), self.mask)
                else:
                    l_percep, l_style = self.cri_canny(torch.cat([self.output, self.out_canny], dim=1), torch.cat([self.gt, self.gt_canny], dim=1), self.mask)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

            # recognition loss
            if self.cri_rec:
                output_rgb = self.cvtColor(self.output) if self.color in ('yuv', 'nv12', 'nv21') else self.output
                gt_rgb = self.gt_rgb if self.color in ('yuv', 'nv12', 'nv21') else self.gt
                l_g_rec = self.cri_rec(output_rgb, gt_rgb)
                l_g_total += l_g_rec
                loss_dict['l_g_rec'] = l_g_rec

            # OCR loss
            if self.cri_ocr:
                output_rgb = self.cvtColor(self.output) if self.color in ('yuv', 'nv12', 'nv21') else self.output
                gt_rgb = self.gt_rgb if self.color in ('yuv', 'nv12', 'nv21') else self.gt
                l_g_ocr = self.cri_ocr(output_rgb, gt_rgb, self.gt_lines)
                l_g_total += l_g_ocr
                loss_dict['l_g_ocr'] = l_g_ocr

            # gradient loss
            if self.cri_grad:
                l_g_grad = self.cri_grad(self.output, self.gt)
                l_g_total += l_g_grad
                loss_dict['l_g_grad'] = l_g_grad

            # mask / patch GAN loss (generator side)
            masked_gan_cfg = self.opt['train'].get('masked_gan_opt', None)
            if masked_gan_cfg is not None and self.mask is not None:
                l_g_gan_mask = self._masked_gan_g_loss(masked_gan_cfg, gan_gt)
                if l_g_gan_mask is not None:
                    l_g_total += l_g_gan_mask
                    loss_dict['l_g_gan_mask'] = l_g_gan_mask

            l_g_total.backward()
            self.optimizer_g.step()

        # ------------------------------------------------------------------- #
        # Discriminator update
        # ------------------------------------------------------------------- #
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            gan_type = self.opt['train']['gan_opt']['type']
            if gan_type == 'RelativisticSoftplusGANLoss':
                real_d_pred = self.net_d(gan_gt)
                fake_d_pred = self.net_d(self.output.detach().clone())
                l_d = self.cri_gan(real_d_pred, fake_d_pred, is_disc=True)
                loss_dict['l_d'] = l_d
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d.backward()
            else:
                real_d_pred = self.net_d(gan_gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                l_d_real.backward()

                fake_d_pred = self.net_d(self.output.detach().clone())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_fake.backward()

            # mask / patch GAN loss (discriminator side)
            masked_gan_cfg = self.opt['train'].get('masked_gan_opt', None)
            if masked_gan_cfg is not None and self.mask is not None:
                self._masked_gan_d_loss(masked_gan_cfg, gan_gt, loss_dict)

            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    # ----------------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------------- #

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        save_recognition = self.opt['val'].get('save_recognition', False)
        comparison = self.opt['val'].get('comparison', None)

        metric_keys = set(self.opt['val']['metrics'].keys()) if with_metrics else set()
        use_ned = 'ned' in metric_keys
        use_ned_divide = 'ned_CHN' in metric_keys or 'ned_ENG' in metric_keys

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {m: 0 for m in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {m: 0 for m in self.metric_results}

        if use_ned:
            self.sr_rec_list, self.gt_rec_list = [], []
        if use_ned_divide:
            self.sr_rec = {'CHN': [], 'ENG': []}
            self.gt_rec = {'CHN': [], 'ENG': []}

        metric_data = {}
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_val(val_data)
            self.test()

            if self.color in ('yuv', 'nv12', 'nv21'):
                lq_h, lq_w = self.lq_rgb.shape[2:]
            else:
                lq_h, lq_w = self.lq.shape[2:]
            scale = self._extract_data_scale(gt=self.gt, lq=self.lq)

            visuals = self.get_current_visuals()
            self.lq = self.lq[..., :lq_h, :lq_w]
            self.gt = self.gt[..., :lq_h * scale, :lq_w * scale]
            visuals['result'] = visuals['result'][..., :lq_h * scale, :lq_w * scale]
            visuals['lq'] = visuals['lq'][..., :lq_h, :lq_w]

            sr_img = tensor2img([visuals['result']])
            H, W = sr_img.shape[:2]
            img_bicubic = F.interpolate(visuals['lq'], (H, W), mode='bicubic')
            lr_img = tensor2img([img_bicubic])

            if 'gt' in visuals:
                visuals['gt'] = visuals['gt'][..., :lq_h * scale, :lq_w * scale]
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img[..., :3]
                del self.gt
            if 'mask' in visuals:
                metric_data['mask'] = tensor2img([visuals['mask']])
            if 'gt_lines' in visuals:
                metric_data['gt_lines'] = visuals['gt_lines']

            if comparison == 'LR':
                sr_img = lr_img
            elif comparison == 'HR':
                sr_img = gt_img
            metric_data['img'] = sr_img[..., :3]

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                save_img_path = self._get_save_path(img_name, dataset_name, current_iter, suffix='.png')
                imwrite(sr_img, save_img_path)

            save_rec_path = None
            if save_recognition:
                save_rec_path = self._get_save_path(img_name, dataset_name, current_iter, suffix='.txt')

            if with_metrics:
                ocr_cache = None
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'recognition' in name:
                        continue
                    if 'ned' in name:
                        if ocr_cache is None:
                            sr_rec, gt_str = calculate_metric(metric_data, opt_)
                            ocr_cache = (sr_rec, gt_str)
                            if save_rec_path:
                                self._write_recognition(save_rec_path, sr_rec, gt_str)
                            if use_ned:
                                self._extend_rec_lists(sr_rec, gt_str)
                            if use_ned_divide and isinstance(sr_rec, dict):
                                for key in sr_rec:
                                    self.sr_rec[key].extend(sr_rec[key])
                                    self.gt_rec[key].extend(gt_str[key])
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                if metric == 'ned':
                    self.metric_results[metric] = self._calc_ned(self.sr_rec_list, self.gt_rec_list)
                elif metric in ('ned_CHN', 'ned_ENG'):
                    lang = metric.split('_')[-1]
                    self.metric_results[metric] = self._calc_ned(self.sr_rec[lang], self.gt_rec[lang])
                elif 'ned' in metric or 'recognition' in metric:
                    pass
                else:
                    self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            if use_ned_divide:
                self.sr_rec = {'CHN': [], 'ENG': []}
                self.gt_rec = {'CHN': [], 'ENG': []}

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.is_train = True

    # ----------------------------------------------------------------------- #
    # Visuals
    # ----------------------------------------------------------------------- #

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.color in ('yuv', 'nv12', 'nv21'):
            out_dict['lq'] = self.cvtColor(self.lq_rgb).detach().cpu()
            out_dict['result'] = self.cvtColor(self.output).detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt_rgb.detach().cpu()
        else:
            out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'gt_lines'):
            out_dict['gt_lines'] = self.gt_lines
        if hasattr(self, 'mask') and self.mask is not None:
            out_dict['mask'] = self.mask.detach().cpu()
        return out_dict

    # ----------------------------------------------------------------------- #
    # Private utilities
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    # Mask / patch GAN helpers
    # ----------------------------------------------------------------------- #

    def _sample_masked_patches(self, img_fake, img_real, mask, patch_size, n_patches):
        """Sample n_patches patches per image centred on mask-foreground pixels.

        Args:
            img_fake: (B, C, H, W) — attached or detached output
            img_real: (B, C, H, W) — GT reference at the same resolution
            mask:     (B, 1, H, W) float [0, 1] on GT/output space
            patch_size: side length P of square patches
            n_patches:  number of patches sampled per sample in the batch

        Returns:
            patches_fake, patches_real — each (B*n_patches, C, P, P)
        """
        B, _, H, W = img_fake.shape
        P = patch_size
        fg_thresh = self.opt['train'].get('masked_gan_opt', {}).get('fg_thresh', 0.5)

        fake_list, real_list = [], []
        for b in range(B):
            m = mask[b, 0]  # (H, W)
            fg_ys, fg_xs = torch.where(m > fg_thresh)

            for _ in range(n_patches):
                if len(fg_ys) > 0:
                    k = random.randint(0, len(fg_ys) - 1)
                    cy, cx = fg_ys[k].item(), fg_xs[k].item()
                    y0 = min(max(cy - P // 2, 0), H - P)
                    x0 = min(max(cx - P // 2, 0), W - P)
                else:
                    y0 = random.randint(0, H - P)
                    x0 = random.randint(0, W - P)

                fake_list.append(img_fake[b:b + 1, :, y0:y0 + P, x0:x0 + P])
                real_list.append(img_real[b:b + 1, :, y0:y0 + P, x0:x0 + P])

        return torch.cat(fake_list, dim=0), torch.cat(real_list, dim=0)

    def _make_composite(self, fake, real, mask):
        """Composite image: fake in masked region, real in background.

        Produces a tensor that looks like `real` everywhere except in the
        foreground mask region, where it uses `fake`.  Feeding this to the
        discriminator asks "is the foreground region realistic?".
        """
        return fake * mask + real * (1.0 - mask)

    def _masked_gan_g_loss(self, cfg, gan_gt):
        """Compute the masked / patch GAN loss for the generator.

        Config keys (under train.masked_gan_opt):
            mode        : 'patch' | 'composite'   (default 'patch')
            patch_size  : int   — side of each patch (default 128, patch mode only)
            n_patches   : int   — patches per sample (default 1, patch mode only)
            loss_weight : float — scalar multiplier   (default 1.0)

        Returns the loss tensor, or None if conditions aren't met (e.g. image
        too small for the requested patch size).
        """
        mode = cfg.get('mode', 'patch')
        w = cfg.get('loss_weight', 1.0)
        gan_type = self.opt['train']['gan_opt']['type']

        if mode == 'patch':
            P = cfg.get('patch_size', 128)
            N = cfg.get('n_patches', 1)
            if self.output.shape[2] < P or self.output.shape[3] < P:
                return None
            patches_fake, patches_real = self._sample_masked_patches(
                self.output, gan_gt, self.mask, P, N)
            if gan_type == 'RelativisticSoftplusGANLoss':
                real_pred = self.net_d(patches_real).detach()  # real first, match D step order
                fake_pred = self.net_d(patches_fake)
                return self.cri_gan(real_pred, fake_pred, is_disc=False) * w
            fake_pred = self.net_d(patches_fake)
            return self.cri_gan(fake_pred, True, is_disc=False) * w

        if mode == 'composite':
            composite = self._make_composite(self.output, gan_gt, self.mask)
            if gan_type == 'RelativisticSoftplusGANLoss':
                real_pred = self.net_d(gan_gt).detach()  # real first, match D step order
                fake_pred = self.net_d(composite)
                return self.cri_gan(real_pred, fake_pred, is_disc=False) * w
            fake_pred = self.net_d(composite)
            return self.cri_gan(fake_pred, True, is_disc=False) * w

        return None

    def _masked_gan_d_loss(self, cfg, gan_gt, loss_dict):
        """Compute the masked / patch GAN loss for the discriminator and
        immediately call backward() on each term (consistent with the standard
        D-update pattern in this model).
        """
        mode = cfg.get('mode', 'patch')
        w = cfg.get('loss_weight', 1.0)
        gan_type = self.opt['train']['gan_opt']['type']
        output_detach = self.output.detach()

        if mode == 'patch':
            P = cfg.get('patch_size', 128)
            N = cfg.get('n_patches', 1)
            if self.output.shape[2] < P or self.output.shape[3] < P:
                return
            patches_fake, patches_real = self._sample_masked_patches(
                output_detach, gan_gt, self.mask, P, N)

            if gan_type == 'RelativisticSoftplusGANLoss':
                real_pred = self.net_d(patches_real)
                fake_pred = self.net_d(patches_fake)
                l_d_mask = self.cri_gan(real_pred, fake_pred, is_disc=True) * w
                loss_dict['l_d_mask'] = l_d_mask
                l_d_mask.backward()
            else:
                real_pred = self.net_d(patches_real)
                l_d_mask_real = self.cri_gan(real_pred, True, is_disc=True) * w
                loss_dict['l_d_mask_real'] = l_d_mask_real
                loss_dict['out_d_mask_real'] = torch.mean(real_pred.detach())
                l_d_mask_real.backward()

                fake_pred = self.net_d(patches_fake)
                l_d_mask_fake = self.cri_gan(fake_pred, False, is_disc=True) * w
                loss_dict['l_d_mask_fake'] = l_d_mask_fake
                loss_dict['out_d_mask_fake'] = torch.mean(fake_pred.detach())
                l_d_mask_fake.backward()

        elif mode == 'composite':
            composite_fake = self._make_composite(output_detach, gan_gt, self.mask)

            if gan_type == 'RelativisticSoftplusGANLoss':
                real_pred = self.net_d(gan_gt)
                fake_pred = self.net_d(composite_fake)
                l_d_comp = self.cri_gan(real_pred, fake_pred, is_disc=True) * w
                loss_dict['l_d_comp'] = l_d_comp
                l_d_comp.backward()
            else:
                real_pred = self.net_d(gan_gt)
                l_d_comp_real = self.cri_gan(real_pred, True, is_disc=True) * w
                loss_dict['l_d_comp_real'] = l_d_comp_real
                loss_dict['out_d_comp_real'] = torch.mean(real_pred.detach())
                l_d_comp_real.backward()

                fake_pred = self.net_d(composite_fake)
                l_d_comp_fake = self.cri_gan(fake_pred, False, is_disc=True) * w
                loss_dict['l_d_comp_fake'] = l_d_comp_fake
                loss_dict['out_d_comp_fake'] = torch.mean(fake_pred.detach())
                l_d_comp_fake.backward()

    def _get_save_path(self, img_name, dataset_name, current_iter, suffix='.png'):
        if self.opt['is_train']:
            return osp.join(self.opt['path']['visualization'], img_name,
                            f'{img_name}_{current_iter}{suffix}')
        tag = self.opt['val'].get('suffix') or self.opt['name']
        return osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{tag}{suffix}')

    def _write_recognition(self, path, sr_rec, gt_str):
        with open(path, 'a+') as f:
            if isinstance(sr_rec, dict):
                for key in sr_rec:
                    for s, g in zip(sr_rec[key], gt_str[key]):
                        f.write(f'{s}\t{g}\n')
            else:
                for s, g in zip(sr_rec, gt_str):
                    f.write(f'{s}\t{g}\n')

    def _extend_rec_lists(self, sr_rec, gt_str):
        if isinstance(sr_rec, dict):
            for key in sr_rec:
                self.sr_rec_list.extend(sr_rec[key])
                self.gt_rec_list.extend(gt_str[key])
        else:
            self.sr_rec_list.extend(sr_rec)
            self.gt_rec_list.extend(gt_str)

    @staticmethod
    def _calc_ned(sr_list, gt_list):
        total = 0.0
        for pred, gt in zip(sr_list, gt_list):
            max_len = max(len(pred), len(gt))
            total += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
        return total / float(len(sr_list) + 1e-10)
