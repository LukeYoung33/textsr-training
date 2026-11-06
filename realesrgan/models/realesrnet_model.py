import numpy as np
import random
import copy
import torch
from torch.nn import functional as F
from tqdm import tqdm
import os.path as osp
import cv2
from collections import OrderedDict
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp, imwrite, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from realesrgan.utils import YUV2RGB, RGB2YUV, YUV2RGBConv, nv21_to_yuv444
import editdistance

def _sample_odd_ksize(ksize_range, default=(3, 11)):
    """@luna-comment
    Input: optional size range. Output: one odd blur size.
    Key Steps: normalize bounds and sample odd support.
    Key Params: default3..11. State/Dependencies: model settings, RNG and tensor utilities.
    """
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
    """@luna-comment
    Input: size, sigma, device and dtype. Output: normalized Gaussian tensor.
    Key Steps: form separable OpenCV kernel and transfer.
    Key Params: odd support. State/Dependencies: model settings, RNG and tensor utilities.
    """
    if k % 2 == 0:
        k += 1
    g1 = cv2.getGaussianKernel(k, sigma)
    kernel = np.dot(g1, g1.T).astype(np.float32)
    kernel = torch.from_numpy(kernel).to(dtype=dtype)
    if device is not None:
        kernel = kernel.to(device)
    return kernel


def _make_motion_kernel(k, length=None, angle_deg=None, device=None, dtype=torch.float32):
    """@luna-comment
    Input: size, optional length and direction. Output: normalized motion tensor.
    Key Steps: rasterize centered line and normalize.
    Key Params: random angle when omitted. State/Dependencies: model settings, RNG and tensor utilities.
    """
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
    """@luna-comment
    Input: batch, size and blur mode. Output: batched blur kernels.
    Key Steps: select Gaussian or motion kernel per mode.
    Key Params: normal/sim/rand. State/Dependencies: model settings, RNG and tensor utilities.
    """
    if mode == 'normal':
        kernel = _make_gaussian_kernel(k, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
    elif mode == 'sim':
        kernel = torch.stack(
            [_make_motion_kernel(k, device=device, dtype=dtype) for _ in range(batch)], dim=0)
    else:  # 'rand'
        g_kernel = _make_gaussian_kernel(k, device=device, dtype=dtype)
        kernels = []
        for _ in range(batch):
            if random.random() < 0.5:
                kernels.append(g_kernel)
            else:
                kernels.append(_make_motion_kernel(k, device=device, dtype=dtype))
        kernel = torch.stack(kernels, dim=0)
    return kernel


def _apply_degradation_blur(img, k, mode):
    """@luna-comment
    Input: BCHW image and blur settings. Output: filtered BCHW image.
    Key Steps: build matching kernels and call filter2D.
    Key Params: image device and dtype. State/Dependencies: model settings, RNG and tensor utilities.
    """
    b = img.size(0)
    kernel = _build_blur_kernels(b, k, mode, device=img.device, dtype=img.dtype)
    return filter2D(img, kernel)


@MODEL_REGISTRY.register()
class RealESRNetModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        """@luna-comment
        Input: model options. Output: initialized model and JPEG/USM helpers.
        Key Steps: initialize SR parent and degradation settings.
        Key Params: queue size and color. State/Dependencies: model settings, RNG and tensor utilities.
        """
        super(RealESRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        self.color = opt.get('color', 'rgb').lower()
        print(f'###################### COLOR is {self.color} ######################')
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.cvtColor = YUV2RGBConv().to(self.device)
        self.finetune_deg = opt.get('finetune_deg', None)

        if self.opt.get('use_feed_data_v2', False):
            print("Notice: Feed data V2 activated.")

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """Maintain an aligned training-pair queue for LQ, GT, mask, and OCR metadata.

        @shanlan-comment
        Input: current BCHW LQ/GT tensors and an optional BCHW text mask already placed on the model device.
        Output: when full, replaces the current pair with one dequeued from the same shuffled queue; otherwise
            advances the queue pointer. LQ, GT, mask, and ``gt_lines`` remain sample-aligned.
        Key Steps: initialize device-local queues, reset on mask-mode/shape changes, apply one permutation to
            every queued field, then dequeue/enqueue all fields together.
        Key Params: ``queue_size`` must be divisible by the current batch size; mask mode and spatial shape must
            remain stable while a queue is active.
        State/Dependencies: mutates ``queue_lr``, ``queue_gt``, ``queue_mask``, ``queue_gt_lines``, ``queue_ptr``,
            and the current ``lq``, ``gt``, ``mask``, and ``gt_lines`` fields.
        """
        batch_size, lq_channels, lq_height, lq_width = self.lq.size()
        current_mask = getattr(self, 'mask', None)
        current_mask_mode = current_mask is not None

        queue_ready = hasattr(self, 'queue_lr')
        queue_shape_matches = queue_ready and (
            self.queue_lr.shape[1:] == self.lq.shape[1:]
            and self.queue_gt.shape[1:] == self.gt.shape[1:]
        )
        queue_mask_matches = queue_ready and (
            getattr(self, 'queue_mask', None) is not None
        ) == current_mask_mode
        if queue_ready and current_mask_mode and queue_mask_matches:
            queue_mask_matches = self.queue_mask.shape[1:] == current_mask.shape[1:]
        current_lines_mode = hasattr(self, 'gt_lines')
        queue_lines_mode = queue_ready and hasattr(self, 'queue_gt_lines')

        if queue_ready and (not queue_shape_matches or not queue_mask_matches
                            or current_lines_mode != queue_lines_mode):
            del self.queue_lr, self.queue_gt
            if hasattr(self, 'queue_mask'):
                del self.queue_mask
            if hasattr(self, 'queue_gt_lines'):
                del self.queue_gt_lines
            queue_ready = False

        if not queue_ready:
            if self.queue_size <= 0:
                return
            assert self.queue_size % batch_size == 0, (
                f'queue size {self.queue_size} should be divisible by batch size {batch_size}')
            self.queue_lr = self.lq.new_zeros(self.queue_size, lq_channels, lq_height, lq_width)
            gt_channels, gt_height, gt_width = self.gt.shape[1:]
            self.queue_gt = self.gt.new_zeros(self.queue_size, gt_channels, gt_height, gt_width)
            if current_mask_mode:
                self.queue_mask = current_mask.new_zeros(
                    self.queue_size, *current_mask.shape[1:])
            if current_lines_mode:
                self.queue_gt_lines = [None] * self.queue_size
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:  # the pool is full
            idx = torch.randperm(self.queue_size, device=self.queue_lr.device)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            if current_mask_mode:
                self.queue_mask = self.queue_mask[idx]
            if current_lines_mode:
                idx_list = idx.tolist()
                self.queue_gt_lines = [self.queue_gt_lines[i] for i in idx_list]

            lq_dequeue = self.queue_lr[0:batch_size].clone()
            gt_dequeue = self.queue_gt[0:batch_size].clone()
            if current_mask_mode:
                mask_dequeue = self.queue_mask[0:batch_size].clone()
            if current_lines_mode:
                gt_lines_dequeue = copy.deepcopy(self.queue_gt_lines[0:batch_size])

            self.queue_lr[0:batch_size] = self.lq.clone()
            self.queue_gt[0:batch_size] = self.gt.clone()
            if current_mask_mode:
                self.queue_mask[0:batch_size] = current_mask.clone()
            if current_lines_mode:
                if not hasattr(self, 'queue_gt_lines'):
                    self.queue_gt_lines = [None] * self.queue_size
                self.queue_gt_lines[0:batch_size] = copy.deepcopy(self.gt_lines)

            self.lq = lq_dequeue
            self.gt = gt_dequeue
            if current_mask_mode:
                self.mask = mask_dequeue
            else:
                self.mask = None
            if current_lines_mode:
                self.gt_lines = gt_lines_dequeue
        else:
            queue_slice = slice(self.queue_ptr, self.queue_ptr + batch_size)
            self.queue_lr[queue_slice] = self.lq.clone()
            self.queue_gt[queue_slice] = self.gt.clone()
            if current_mask_mode:
                self.queue_mask[queue_slice] = current_mask.clone()
            if current_lines_mode:
                if not hasattr(self, 'queue_gt_lines'):
                    self.queue_gt_lines = [None] * self.queue_size
                self.queue_gt_lines[queue_slice] = copy.deepcopy(self.gt_lines)
            self.queue_ptr += batch_size

    def _extract_data_scale(self, data=None, gt=None, lq=None):
        """@luna-comment
        Input: optional batch and GT/LQ tensors. Output: positive integer scale.
        Key Steps: prefer batch scale, infer shape ratio, fall back to options.
        Key Params: minimum scale1. State/Dependencies: model settings, RNG and tensor utilities.
        """
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
        """@luna-comment
        Input: paired batch and sharpening switch. Output: device LQ/GT, mask and OCR metadata.
        Key Steps: transfer tensors, optionally sharpen GT, copy aligned metadata.
        Key Params: apply_usm defaultsfalse. State/Dependencies: model settings, RNG and tensor utilities.
        """
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
        self.current_data_scale = self._extract_data_scale(data=data, gt=getattr(self, 'gt', None), lq=self.lq)

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
            img,
            scale_range=poisson_scale_range,
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

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

    def _apply_synthetic_lq_degradation_v3(self, gt, kernel1, kernel2, sinc_kernel):
        """Apply the configured synthetic chain to a homogeneous GT subset."""
        out = filter2D(gt, kernel1)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            resize_scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            resize_scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            resize_scale = 1
        out = F.interpolate(
            out, scale_factor=resize_scale,
            mode=random.choice(['area', 'bilinear', 'bicubic']))
        out = self._apply_noise(
            out, self.opt['gaussian_noise_prob'], self.opt['noise_range'],
            self.opt['poisson_scale_range'], self.opt['gray_noise_prob'])
        out = self._apply_jpeg(out, self.opt['jpeg_range'])
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        out = self._apply_noise(
            out, self.opt['gaussian_noise_prob2'], self.opt['noise_range2'],
            self.opt['poisson_scale_range2'], self.opt['gray_noise_prob2'])
        if np.random.uniform() < 0.5:
            out = filter2D(out, sinc_kernel)
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
        else:
            out = self._apply_jpeg(out, self.opt['jpeg_range2'])
            out = filter2D(out, sinc_kernel)
        if out.shape[-2:] != gt.shape[-2:]:
            out = F.interpolate(out, size=gt.shape[-2:], mode='bicubic')
        return torch.clamp((out * 255.0).round(), 0, 255) / 255.

    def _feed_mixed_source_v3(self, data):
        """Build one batch from paired IQ samples and synthetic RealCE samples."""
        self._load_paired_data(data, apply_usm=True)
        modes = data['degradation_mode']
        outputs = []
        for index, mode in enumerate(modes):
            if mode == 'synthetic':
                outputs.append(self._apply_synthetic_lq_degradation_v3(
                    self.gt[index:index + 1],
                    data['kernel1'][index:index + 1].to(self.device),
                    data['kernel2'][index:index + 1].to(self.device),
                    data['sinc_kernel'][index:index + 1].to(self.device),
                ))
            else:
                outputs.append(self._apply_paired_lq_degradation_v3(
                    self.lq[index:index + 1]))
        self.lq = torch.cat(outputs, dim=0)
        self._dequeue_and_enqueue()
        self.lq = self.lq.contiguous()
        self._apply_color_conversion()

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
        down_rate = self.opt['datasets']['val']['down_rate']
        scale = self.current_data_scale
        down_rate_h = down_rate_w = down_rate
        pad_h = int(np.ceil(lq_h / down_rate_h) * down_rate_h - lq_h)
        pad_w = int(np.ceil(lq_w / down_rate_w) * down_rate_w - lq_w)
        self.lq = F.pad(self.lq, pad=(0, pad_w, 0, pad_h), mode='reflect')
        if hasattr(self, 'gt'):
            self.gt = F.pad(self.gt, pad=(0, pad_w * scale, 0, pad_h * scale), mode='reflect')
        if self.mask is not None:
            self.mask = F.pad(self.mask, pad=(0, pad_w * scale, 0, pad_h * scale), mode='reflect')
        self._apply_color_conversion()

    @torch.no_grad()
    def feed_data(self, data):
        """@shanlan-comment
        Input: paired or GT-plus-standard-kernel batch for RealESRNet training/validation.
        Output: device-resident LQ/GT tensors; high-order path emits a synthesized LQ.
        Key Steps: load GT, apply standard two-stage blur/resize/noise/JPEG/sinc
        degradation when configured, queue the pair, and optionally convert color.
        Key Params: high_order_degradation, finetune_deg, scale, and degradation ranges.
        State/Dependencies: jpeger, USM sharpener, queue buffers, kernels, and RNG state.
        """
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.opt.get('use_feed_data_v3', False):
            return self.feed_data_v3(data)
        if self.opt.get('use_feed_data_v2', False):
            return self.feed_data_v2(data)
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            if 'gt_lines' in data:
                self.gt_lines = copy.deepcopy(data['gt_lines'])
            elif hasattr(self, 'gt_lines'):
                del self.gt_lines
            # USM sharpen the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)
            if 'mask' in data and data['mask'] is not None:
                self.mask = data['mask'].to(self.device)
            else:
                self.mask = None
            if self.finetune_deg and random.random() > self.finetune_deg:
                self.kernel1 = data['kernel1'].to(self.device)
                self.kernel2 = data['kernel2'].to(self.device)
                self.sinc_kernel = data['sinc_kernel'].to(self.device)

                ori_h, ori_w = self.gt.size()[2:4]

                # ----------------------- The first degradation process ----------------------- #
                # blur
                out = filter2D(self.gt, self.kernel1)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                    out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

                # JPEG compression + the final sinc filter
                # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
                # as one operation.
                # We consider two orders:
                #   1. [resize back + sinc filter] + JPEG compression
                #   2. JPEG compression + [resize back + sinc filter]
                # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                else:
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    # resize back + the final sinc filter
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)

                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            else:
                self.feed_data_train_pair(data, use_queue=True, apply_usm=True)
                return

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            self.feed_data_train_pair(data, use_queue=False, apply_usm=False)
            return
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.gt_rgb = self.gt
            self.lq_rgb = self.lq
            self.lq = RGB2YUV(self.lq, out_format=self.color)
            self.gt = RGB2YUV(self.gt, out_format=self.color)
        if (not self.is_train or not self.opt.get('high_order_degradation', True)) and 'gt_lines' in data:
            self.gt_lines = data['gt_lines']

    @torch.no_grad()
    def feed_data_v2(self, data):
        """V2 degradation with gaussian + motion blur (prepare_screen_shot.make_dataV3 style)."""
        if self.is_train and self.opt.get('high_order_degradation', True):
            self.gt = data['gt'].to(self.device)
            if 'gt_lines' in data:
                self.gt_lines = copy.deepcopy(data['gt_lines'])
            elif hasattr(self, 'gt_lines'):
                del self.gt_lines
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)
            if 'mask' in data and data['mask'] is not None:
                self.mask = data['mask'].to(self.device)
            else:
                self.mask = None

            blur_mode = self.opt.get('blur_mode', 'rand')
            blur_ksize_range = self.opt.get('blur_ksize_range', (3, 11))
            blur_ksize_range2 = self.opt.get('blur_ksize_range2', blur_ksize_range)

            if self.finetune_deg is not None and random.random() > self.finetune_deg:
                self.kernel1 = data['kernel1'].to(self.device)
                self.kernel2 = data['kernel2'].to(self.device)
                self.sinc_kernel = data['sinc_kernel'].to(self.device)

                ori_h, ori_w = self.gt.size()[2:4]

                # ----------------------- The first degradation process ----------------------- #
                if 'kernel1' in data and data['kernel1'] is not None:
                    kernel1 = data['kernel1'].to(self.device)
                    out = filter2D(self.gt, kernel1)
                else:
                    k1 = _sample_odd_ksize(blur_ksize_range)
                    out = _apply_degradation_blur(self.gt, k1, blur_mode)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

                # ----------------------- The second degradation process ----------------------- #
                if np.random.uniform() < self.opt['second_blur_prob']:
                    if 'kernel2' in data and data['kernel2'] is not None:
                        kernel2 = data['kernel2'].to(self.device)
                        out = filter2D(out, kernel2)
                    else:
                        k2 = _sample_odd_ksize(blur_ksize_range2)
                        out = _apply_degradation_blur(out, k2, blur_mode)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                    out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

                # JPEG compression + the final sinc filter
                if np.random.uniform() < 0.5:
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                else:
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)

                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            else:
                self.lq = data['lq'].to(self.device)

            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
            if 'mask' in data and data['mask'] is not None:
                self.mask = data['mask'].to(self.device)
            else:
                self.mask = None
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.gt_rgb = self.gt
            self.lq_rgb = self.lq
            self.lq = RGB2YUV(self.lq, out_format=self.color)
            self.gt = RGB2YUV(self.gt, out_format=self.color)
        if (not self.is_train or not self.opt.get('high_order_degradation', True)) and 'gt_lines' in data:
            self.gt_lines = data['gt_lines']

    @torch.no_grad()
    def feed_data_v3(self, data):
        """V3 degradation: synthetic branch stays unchanged; paired branch only adds noise + JPEG without resize."""
        if self.is_train and self.opt.get('high_order_degradation', True):
            if 'degradation_mode' in data:
                self._feed_mixed_source_v3(data)
                return
            self.gt = data['gt'].to(self.device)
            if 'gt_lines' in data:
                self.gt_lines = copy.deepcopy(data['gt_lines'])
            elif hasattr(self, 'gt_lines'):
                del self.gt_lines
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)
            if 'mask' in data and data['mask'] is not None:
                self.mask = data['mask'].to(self.device)
            else:
                self.mask = None

            use_synthetic_degradation = self.finetune_deg is not None and random.random() > self.finetune_deg
            if use_synthetic_degradation:
                self.kernel1 = data['kernel1'].to(self.device)
                self.kernel2 = data['kernel2'].to(self.device)
                self.sinc_kernel = data['sinc_kernel'].to(self.device)

                ori_h, ori_w = self.gt.size()[2:4]

                out = filter2D(self.gt, self.kernel1)
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                out = self._apply_noise(
                    out,
                    gaussian_noise_prob=self.opt['gaussian_noise_prob'],
                    noise_range=self.opt['noise_range'],
                    poisson_scale_range=self.opt['poisson_scale_range'],
                    gray_noise_prob=self.opt['gray_noise_prob'])
                out = self._apply_jpeg(out, self.opt['jpeg_range'])

                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
                # updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                # if updown_type == 'up':
                #     scale = np.random.uniform(1, self.opt['resize_range2'][1])
                # elif updown_type == 'down':
                #     scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                # else:
                #     scale = 1
                # mode = random.choice(['area', 'bilinear', 'bicubic'])
                # out = F.interpolate(
                #     out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                out = self._apply_noise(
                    out,
                    gaussian_noise_prob=self.opt['gaussian_noise_prob2'],
                    noise_range=self.opt['noise_range2'],
                    poisson_scale_range=self.opt['poisson_scale_range2'],
                    gray_noise_prob=self.opt['gray_noise_prob2'])

                if np.random.uniform() < 0.5:
                    # mode = random.choice(['area', 'bilinear', 'bicubic'])
                    # out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
                    out = self._apply_jpeg(out, self.opt['jpeg_range2'])
                else:
                    out = self._apply_jpeg(out, self.opt['jpeg_range2'])
                    # mode = random.choice(['area', 'bilinear', 'bicubic'])
                    # out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)

                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            else:
                self._load_paired_data(data, apply_usm=True)
                self.lq = self._apply_paired_lq_degradation_v3(self.lq)

            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()
        else:
            self.feed_data_train_pair(data, use_queue=False, apply_usm=False)
            return
        self._apply_color_conversion()
        if (not self.is_train or not self.opt.get('high_order_degradation', True)) and 'gt_lines' in data:
            self.gt_lines = data['gt_lines']

    def feed_data_val(self, data):
        self.feed_data_val_pair(data)
        if hasattr(self, 'gt_lines') and isinstance(self.gt_lines, list) and len(self.gt_lines) == 1 and isinstance(self.gt_lines[0], list):
            self.gt_lines = self.gt_lines[0]

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
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        if use_ned:
            self.sr_rec_list = []
            self.gt_rec_list = []
        if use_ned_divide:
            self.sr_rec = {
                "CHN":[],
                "ENG":[]
            }
            self.gt_rec = {
                "CHN": [],
                "ENG": []
            }

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_val(val_data)

            self.test()

            # print("Total:", )

            # cv2.imwrite("mask_" + str(idx) + ".png", self.mask.data.cpu().numpy().astype(np.uint8)[0][0] * 255)

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
            img_bicubic = torch.nn.functional.interpolate(visuals['lq'], (H, W), mode="bicubic")
            lr_img = tensor2img([img_bicubic])
            #sr_img
            if 'gt' in visuals:
                visuals['gt'] = visuals['gt'][..., :lq_h * scale, :lq_w * scale]
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img[..., :3]
                del self.gt
            if 'mask' in visuals:
                # print("mask:", mask.shape)
                mask = tensor2img([visuals['mask']])
                metric_data['mask'] = mask
            if 'gt_lines' in visuals:
                # print("mask:", mask.shape)
                # mask = tensor2img([visuals['gt_lines']])
                metric_data['gt_lines'] = visuals['gt_lines']

            if comparison == "LR":
                sr_img = lr_img
            elif comparison == "HR":
                sr_img = gt_img
            metric_data['img'] = sr_img[..., :3] ###sr_img
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
            if save_recognition:
                if self.opt['is_train']:
                    save_rec_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.txt')
                else:
                    if self.opt['val']['suffix']:
                        save_rec_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.txt')
                    else:
                        save_rec_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.txt')

            if with_metrics:
                ocr_cache = None
                for name, opt_ in self.opt['val']['metrics'].items():
                    if "recognition" in name:
                        continue
                    elif "ned" in name:
                        if ocr_cache is None:
                            sr_rec, gt_str = calculate_metric(metric_data, opt_)
                            ocr_cache = (sr_rec, gt_str)
                            if save_recognition:
                                rec_f = open(save_rec_path, "a+")
                                if isinstance(sr_rec, dict):
                                    for key in sr_rec:
                                        for i in range(len(sr_rec[key])):
                                            rec_f.write(sr_rec[key][i] + "\t" + gt_str[key][i] + "\n")
                                else:
                                    for i in range(len(sr_rec)):
                                        rec_f.write(sr_rec[i] + "\t" + gt_str[i] + "\n")
                                rec_f.close()

                            if use_ned:
                                if isinstance(sr_rec, dict):
                                    for key in sr_rec:
                                        self.sr_rec_list.extend(sr_rec[key])
                                        self.gt_rec_list.extend(gt_str[key])
                                else:
                                    self.sr_rec_list.extend(sr_rec)
                                    self.gt_rec_list.extend(gt_str)
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
                if metric == "ned":
                    sr_list = self.sr_rec_list
                    gt_list = self.gt_rec_list
                    for i in range(len(sr_list)):
                        pred, gt = sr_list[i], gt_list[i]
                        max_len = max(len(pred), len(gt))
                        self.metric_results[metric] += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
                    self.metric_results[metric] /= float(len(sr_list) + 1e-10)
                elif metric in ("ned_CHN", "ned_ENG"):
                    lang = metric.split("_")[-1]
                    sr_list = self.sr_rec[lang]
                    gt_list = self.gt_rec[lang]
                    for i in range(len(sr_list)):
                        pred, gt = sr_list[i], gt_list[i]
                        max_len = max(len(pred), len(gt))
                        self.metric_results[metric] += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
                    self.metric_results[metric] /= float(len(sr_list) + 1e-10)

                elif "ned" in metric:
                    pass
                elif "recognition" in metric:
                    pass
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            if use_ned_divide:
                self.sr_rec = {
                    "CHN": [],
                    "ENG": []
                }
                self.gt_rec = {
                    "CHN": [],
                    "ENG": []
                }
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

            self.is_train = True

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.color in ('yuv', 'nv12', 'nv21'):
            # out_dict['lq'] = YUV2RGB(self.lq, format='RGB').detach().cpu()
            # out_dict['result'] = YUV2RGB(self.output, format='RGB').detach().cpu()
            # if hasattr(self, 'gt'):
            #     out_dict['gt'] = YUV2RGB(self.gt, format='RGB').detach().cpu()
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
        return out_dict

    def optimize_parameters(self, current_iter):
        # if current_iter % 2500 == 0:
        #     for i in range(self.lq.size()[0]):
        #         img_show = (self.lq[i][:3, ...].detach().cpu().permute(1, 2, 0).clamp(min=0, max=1).numpy() * 255).astype(np.uint8)[:, :, ::-1]
        #         # img_show = cv2.resize(img_show, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        #         cv2.imwrite(f'datasets/debug_train_vis/lq_train_{current_iter}_{i}.png', img_show)
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        # if current_iter % 2500 == 0:
        #     for i in range(self.lq.size()[0]):
        #         img_show = (self.output[i][:3, ...].detach().cpu().permute(1, 2, 0).clamp(min=0, max=1).numpy() * 255).astype(np.uint8)[:, :, ::-1]
        #         # img_show = cv2.resize(img_show, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        #         cv2.imwrite(f'datasets/debug_train_vis/lq_train_{current_iter}_{i}_out.png', img_show)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if self.color in ('yuv', 'nv12', 'nv21'):
                if isinstance(self.output, torch.Tensor):
                    # if self.opt.get('l1', 'rgb') == 'yuv':
                    l_pix = self.cri_pix(self.output, RGB2YUV(self.gt_rgb, out_format='yuv'), mask=self.mask)
                    # else:
                    #     l_pix = self.cri_pix(self.cvtColor(self.output), self.gt_rgb, mask=self.mask)
                else:
                    out_y, out_u, out_v = self.output[:]
                    gt_y, gt_u, gt_v = RGB2YUV(self.gt_rgb, out_format='NV21', split=True)
                    l_pix_y = self.cri_pix(out_y, gt_y, mask=self.mask)
                    l_pix_u = self.cri_pix(out_u, gt_u, mask=torch.nn.functional.avg_pool2d(self.mask, kernel_size=2, stride=2))
                    l_pix_v = self.cri_pix(out_v, gt_v, mask=torch.nn.functional.avg_pool2d(self.mask, kernel_size=2, stride=2))
                    l_pix = l_pix_y + l_pix_u + l_pix_v

            else:
                l_pix = self.cri_pix(self.output, self.gt, mask=self.mask)
            # l_pix = self.cri_pix(self.output, self.gt)

            ############################# debug grad #############################
            # def hook_fn(grad):
            #     print(f"Y grad norm: {grad[:, 0:1].norm().item():.6f}")
            #     print(f"U grad norm: {grad[:, 1:2].norm().item():.6f}")
            #     print(f"V grad norm: {grad[:, 2:3].norm().item():.6f}")

            # if current_iter % 100 == 1:
            #     self.output.register_hook(hook_fn)

            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_pix2 and isinstance(self.output, torch.Tensor):
            l_pix2 = self.cri_pix2(self.output, RGB2YUV(self.gt_rgb, out_format='yuv'), mask=self.mask)
            l_total += l_pix2
            loss_dict['l_pix_yuv'] = l_pix2

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.cvtColor(self.output), self.gt_rgb)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        if self.cri_rec:
            if current_iter < 25000:
                self.cri_rec.loss_weight = 1.0
            else:
                self.cri_rec.loss_weight = self.opt['train']['rec_opt']['loss_weight']
            if self.color in ('yuv', 'nv12', 'nv21'):
                l_rec = self.cri_rec(self.cvtColor(self.output), self.gt_rgb)
            else:
                l_rec = self.cri_rec(self.output, self.gt)
            if current_iter % 200 == 1:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    out_show = tensor2img(self.cvtColor(self.output))
                else:
                    out_show = tensor2img(self.output)
                cv2.imwrite(f'./datasets/train_vis_nv21/out_show_{current_iter-1}.png', out_show)
            l_total += l_rec
            loss_dict['l_rec'] = l_rec

        if self.cri_ocr:
            if self.color in ('yuv', 'nv12', 'nv21'):
                l_ocr = self.cri_ocr(self.cvtColor(self.output), self.gt_rgb, self.gt_lines)
            else:
                l_ocr = self.cri_ocr(self.output, self.gt, self.gt_lines)
            l_total += l_ocr
            loss_dict['l_ocr'] = l_ocr

        if self.cri_grad:
            l_grad = self.cri_grad(self.output, self.gt)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
