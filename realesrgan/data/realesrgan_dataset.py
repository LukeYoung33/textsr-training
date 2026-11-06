import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data


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


def _make_gaussian_kernel(k, sigma=0):
    if k % 2 == 0:
        k += 1
    g1 = cv2.getGaussianKernel(k, sigma)
    kernel = np.dot(g1, g1.T).astype(np.float32)
    return kernel


def _make_motion_kernel(k, length=None, angle_deg=None):
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
    return kernel


def _build_v2_blur_kernel(mode, k):
    if mode == 'normal':
        kernel = _make_gaussian_kernel(k)
    elif mode == 'sim':
        kernel = _make_motion_kernel(k)
    else:  # 'rand'
        if random.random() < 0.5:
            kernel = _make_gaussian_kernel(k)
        else:
            kernel = _make_motion_kernel(k)
    return kernel


def _pad_kernel_to(kernel, pad_to):
    k = kernel.shape[0]
    if k >= pad_to:
        return kernel
    pad = (pad_to - k) // 2
    return np.pad(kernel, ((pad, pad), (pad, pad)))


def _float01_to_uint8(img):
    return np.clip(img * 255.0, 0, 255).round().astype(np.uint8)


def _uint8_to_float01(img):
    return img.astype(np.float32) / 255.0


def _get_data_scale(opt):
    scale = int(opt.get('data_scale', opt.get('scale', 1)))
    return max(scale, 1)


def _align_crop_size(size, scale):
    size = int(size)
    return max(scale, int(math.ceil(size / float(scale))) * scale)


def _trim_paired_images_to_scale(img_gt, img_lq, scale, img_mask=None):
    h_lq, w_lq = img_lq.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    target_lq_h = min(h_lq, h_gt // scale)
    target_lq_w = min(w_lq, w_gt // scale)
    target_gt_h = target_lq_h * scale
    target_gt_w = target_lq_w * scale
    img_lq = img_lq[:target_lq_h, :target_lq_w, ...]
    img_gt = img_gt[:target_gt_h, :target_gt_w, ...]
    if img_mask is not None:
        img_mask = img_mask[:target_gt_h, :target_gt_w, ...]
    return img_gt, img_lq, img_mask


def _paired_random_crop(img_gt, img_lq, img_mask, gt_patch_size, scale):
    lq_patch_size = gt_patch_size // scale
    h_lq, w_lq = img_lq.shape[:2]
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}).')
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    top_gt = top * scale
    left_gt = left * scale
    img_lq = img_lq[top:top + lq_patch_size, left:left + lq_patch_size, ...]
    img_gt = img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
    img_mask = img_mask[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
    return img_gt, img_lq, img_mask


def adjust_contrast_yuv(img_bgr, contrast_factor=1.3, brightness=0.0):
    """Adjust contrast on Y channel in YUV space. Input/Output: uint8 BGR."""
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    y = y.astype(np.float32)
    mean = np.mean(y)
    y = contrast_factor * (y - mean) + mean + brightness
    y = np.clip(y, 0, 255).astype(np.uint8)
    yuv_adjusted = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_adjusted, cv2.COLOR_YUV2BGR)


def _resolve_text_size_value(text_size_info):
    """Convert various text-size descriptors to a single scalar size value."""
    if text_size_info is None:
        return None
    if isinstance(text_size_info, (int, float, np.integer, np.floating)):
        return float(text_size_info)
    if isinstance(text_size_info, dict):
        for key in ('text_height', 'font_size', 'size', 'height', 'fixed_font_size'):
            if key in text_size_info and text_size_info[key] is not None:
                return _resolve_text_size_value(text_size_info[key])
        if 'bbox' in text_size_info:
            return _resolve_text_size_value(text_size_info['bbox'])
        if 'box' in text_size_info:
            return _resolve_text_size_value(text_size_info['box'])
        min_font = text_size_info.get('min_font_size')
        max_font = text_size_info.get('max_font_size')
        if min_font is not None and max_font is not None:
            return 0.5 * (float(min_font) + float(max_font))
        return None
    if isinstance(text_size_info, (list, tuple)):
        if len(text_size_info) == 2:
            return float(min(text_size_info))
        if len(text_size_info) >= 4:
            x1, y1, x2, y2 = [float(v) for v in text_size_info[:4]]
            return min(abs(x2 - x1), abs(y2 - y1))
    return None


def adjust_contrast_yuv_v2(img_fg_bgr,
                           img_bg_bgr,
                           text_size_info,
                           min_contrast_factor=1.05,
                           max_contrast_factor=1.9,
                           small_text_size=10.0,
                           large_text_size=48.0,
                           diff_threshold=8.0,
                           target_delta=72.0,
                           eps=1e-6):
    """Enhance foreground contrast on Y channel while keeping background brightness unchanged.

    Args:
        img_fg_bgr: Foreground/composited text image in uint8 BGR.
        img_bg_bgr: Background-only image in uint8 BGR, same shape as ``img_fg_bgr``.
        text_size_info: Scalar/dict/tuple describing text size. Smaller text gets stronger enhancement.
        min_contrast_factor: Minimum foreground contrast gain.
        max_contrast_factor: Maximum foreground contrast gain.
        small_text_size: Reference size for strongest enhancement.
        large_text_size: Reference size for weakest enhancement.
        diff_threshold: Pixel difference threshold used to estimate foreground mask.
        target_delta: Desired foreground/background luminance separation.
        eps: Numerical stability term.

    Returns:
        uint8 BGR image with enhanced text foreground and unchanged background brightness.
    """
    if img_fg_bgr is None or img_bg_bgr is None:
        raise ValueError('img_fg_bgr and img_bg_bgr must not be None.')
    if img_fg_bgr.shape != img_bg_bgr.shape:
        raise ValueError(f'Foreground/background shape mismatch: {img_fg_bgr.shape} vs {img_bg_bgr.shape}')
    if img_fg_bgr.dtype != np.uint8 or img_bg_bgr.dtype != np.uint8:
        raise ValueError('adjust_contrast_yuv_v2 expects uint8 BGR inputs.')

    fg_diff = np.max(np.abs(img_fg_bgr.astype(np.int16) - img_bg_bgr.astype(np.int16)), axis=2)
    text_mask = fg_diff > float(diff_threshold)
    if not np.any(text_mask):
        return img_fg_bgr.copy()

    fg_yuv = cv2.cvtColor(img_fg_bgr, cv2.COLOR_BGR2YUV).astype(np.float32)
    bg_yuv = cv2.cvtColor(img_bg_bgr, cv2.COLOR_BGR2YUV).astype(np.float32)
    fg_y = fg_yuv[..., 0]
    bg_y = bg_yuv[..., 0]

    delta = fg_y - bg_y
    delta_text = delta[text_mask]
    abs_delta_mean = float(np.mean(np.abs(delta_text)))
    delta_std = float(np.std(delta_text))

    text_size = _resolve_text_size_value(text_size_info)
    if text_size is None:
        size_weight = 0.5
    else:
        size_norm = (float(text_size) - float(small_text_size)) / max(float(large_text_size - small_text_size), eps)
        size_weight = 1.0 - float(np.clip(size_norm, 0.0, 1.0))

    distribution_weight = 1.0 - float(np.clip((abs_delta_mean + 0.5 * delta_std) / max(float(target_delta), eps), 0.0, 1.0))
    gain_weight = 0.35 + 0.65 * distribution_weight
    contrast_factor = min_contrast_factor + (max_contrast_factor - min_contrast_factor) * size_weight * gain_weight

    enhanced_y = bg_y.copy()
    enhanced_y[text_mask] = bg_y[text_mask] + delta[text_mask] * contrast_factor
    enhanced_y = np.clip(enhanced_y, 0.0, 255.0)

    out_yuv = bg_yuv.copy()
    out_yuv[..., 0] = enhanced_y
    out_yuv[..., 1][text_mask] = fg_yuv[..., 1][text_mask]
    out_yuv[..., 2][text_mask] = fg_yuv[..., 2][text_mask]

    out_bgr = cv2.cvtColor(out_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    out_bgr[~text_mask] = img_bg_bgr[~text_mask]
    return out_bgr


def adjust_lab(image_bgr,
               a_shift=0.0,
               b_shift=0.0,
               a_scale=1.0,
               b_scale=1.0,
               clip=True):
    """Adjust a/b channels in Lab space. Input/Output: uint8 BGR."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    a = a.astype(np.float32) - 128.0
    b = b.astype(np.float32) - 128.0
    a = a * a_scale + a_shift
    b = b * b_scale + b_shift
    if clip:
        a = np.clip(a, -127, 127)
        b = np.clip(b, -127, 127)
    a = (a + 128.0).astype(np.uint8)
    b = (b + 128.0).astype(np.uint8)
    lab_adjusted = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_Lab2BGR)


@DATASET_REGISTRY.register()
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        """@shanlan-comment
        Input: standard Real-ESRGAN dataset options including GT storage and kernel ranges.
        Output: initialized GT reader and two-stage blur-kernel sampling state.
        Key Steps: resolve GT paths, validate I/O mode, and cache standard kernel settings.
        Key Params: kernel lists/probabilities and sinc probabilities control degradation draws.
        State/Dependencies: file-client options, torch pulse kernel, and random kernel helpers.
        """
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(', ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        """@shanlan-comment
        Input: integer GT-record index.
        Output: augmented GT RGB tensor plus standard kernel1/kernel2/final-sinc tensors.
        Key Steps: read/retry image, augment/crop it, sample both blur kernels and final
        sinc kernel, then convert image and kernels to tensors.
        Key Params: configured mixed-kernel and sinc probabilities define each draw.
        State/Dependencies: FileClient, numpy/random RNG state, and degradation helpers.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        """@shanlan-comment
        Input: none.
        Output: count of resolved GT records.
        Key Steps: return the cached path list length.
        Key Params: none.
        State/Dependencies: self.paths initialized from metadata.
        """
        return len(self.paths)


@DATASET_REGISTRY.register()
class RealESRGANFinetuneDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANFinetuneDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.data_scale = _get_data_scale(opt)

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(', ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        # color adjustments
        self.use_yuv_contrast = opt.get('use_yuv_contrast', False)
        self.yuv_contrast_factor = opt.get('yuv_contrast_factor', 1.3)
        self.yuv_brightness = opt.get('yuv_brightness', 0.0)
        self.use_adjust_lab = opt.get('use_adjust_lab', False)
        self.lab_a_shift = opt.get('lab_a_shift', 0.0)
        self.lab_b_shift = opt.get('lab_b_shift', 0.0)
        self.lab_a_scale = opt.get('lab_a_scale', 1.0)
        self.lab_b_scale = opt.get('lab_b_scale', 1.0)
        self.lab_clip = opt.get('lab_clip', True)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                mask_path = gt_path.replace('/gt/', '/mask/')  # .replace('_s', '_mask_s')
                try:
                    img_bytes = self.file_client.get(mask_path, 'gt')
                    img_mask = imfrombytes(img_bytes, float32=True)
                except:
                    img_mask = np.ones_like(img_gt, dtype=img_gt.dtype)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        lq_path = gt_path.replace('/gt/', '/lq/')
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_gt, img_lq, img_mask = _trim_paired_images_to_scale(img_gt, img_lq, self.data_scale, img_mask)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        # img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        crop_pad_size = _align_crop_size(self.opt.get('gt_size', 512), self.data_scale)
        lq_crop_pad_size = crop_pad_size // self.data_scale
        h_gt, w_gt = img_gt.shape[0:2]
        h_lq, w_lq = img_lq.shape[0:2]

        if h_gt < crop_pad_size or w_gt < crop_pad_size:
            pad_h_gt = max(0, crop_pad_size - h_gt)
            pad_w_gt = max(0, crop_pad_size - w_gt)
            pad_h_lq = max(0, lq_crop_pad_size - h_lq)
            pad_w_lq = max(0, lq_crop_pad_size - w_lq)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h_gt, 0, pad_w_gt, cv2.BORDER_REFLECT_101)
            img_mask = cv2.copyMakeBorder(img_mask, 0, pad_h_gt, 0, pad_w_gt, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h_lq, 0, pad_w_lq, cv2.BORDER_REFLECT_101)

        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            img_gt, img_lq, img_mask = _paired_random_crop(
                img_gt, img_lq, img_mask, crop_pad_size, self.data_scale)

        # -------------------- Optional color adjustments -------------------- #
        if self.use_yuv_contrast:
            img_gt_u8 = _float01_to_uint8(img_gt)
            img_gt = _uint8_to_float01(
                adjust_contrast_yuv(
                    img_gt_u8,
                    contrast_factor=self.yuv_contrast_factor,
                    brightness=self.yuv_brightness
                )
            )

        if self.use_adjust_lab:
            img_gt_u8 = _float01_to_uint8(img_gt)
            img_lq_u8 = _float01_to_uint8(img_lq)
            img_gt = _uint8_to_float01(
                adjust_lab(
                    img_gt_u8,
                    a_shift=self.lab_a_shift,
                    b_shift=self.lab_b_shift,
                    a_scale=self.lab_a_scale,
                    b_scale=self.lab_b_scale,
                    clip=self.lab_clip
                )
            )
            img_lq = _uint8_to_float01(
                adjust_lab(
                    img_lq_u8,
                    a_shift=self.lab_a_shift,
                    b_shift=self.lab_b_shift,
                    a_scale=self.lab_a_scale,
                    b_scale=self.lab_b_scale,
                    clip=self.lab_clip
                )
            )

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_mask = img2tensor([img_mask], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            'gt': img_gt,
            'lq': img_lq,
            'mask': img_mask,
            'kernel1': kernel,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path,
            'data_scale': self.data_scale
        }
        return return_d

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class RealESRGANV2BlurKernelDataset(data.Dataset):
    """Finetune-style dataset that generates v2 blur kernels (gaussian/motion) per-sample.

    It returns gt/lq/mask along with kernel1/kernel2/sinc_kernel. The blur kernels are
    created using the same gaussian/motion logic as feed_data_v2.
    """

    def __init__(self, opt):
        super(RealESRGANV2BlurKernelDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(', ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        self.blur_mode = opt.get('blur_mode', 'rand')
        self.blur_ksize_range = opt.get('blur_ksize_range', [3, 11])
        self.blur_ksize_range2 = opt.get('blur_ksize_range2', self.blur_ksize_range)
        self.blur_kernel_pad_size = int(opt.get('blur_kernel_pad_size', 21))

        self.final_sinc_prob = opt.get('final_sinc_prob', 0.0)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # 7..21
        self.pulse_tensor = torch.zeros(self.blur_kernel_pad_size, self.blur_kernel_pad_size).float()
        center = self.blur_kernel_pad_size // 2
        self.pulse_tensor[center, center] = 1

        # random resize augmentation (use same params as feed_data)
        self.resize_prob = opt.get('resize_prob', [0.2, 0.7, 0.1])
        self.resize_range = opt.get('resize_range', [0.15, 1.5])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                mask_path = gt_path.replace('/gt/', '/mask/')
                try:
                    img_bytes = self.file_client.get(mask_path, 'gt')
                    img_mask = imfrombytes(img_bytes, float32=True)
                except Exception:
                    img_mask = np.ones_like(img_gt, dtype=img_gt.dtype)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        lq_path = gt_path.replace('/gt/', '/lq/')
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # -------------------- Random resize augmentation (scale target size) -------------------- #
        crop_pad_size = 512
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        if scale != 1:
            h, w = img_gt.shape[0:2]
            if scale < 1:
                min_scale = max(crop_pad_size / float(h), crop_pad_size / float(w), self.resize_range[0])
                if scale < min_scale:
                    scale = min_scale
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            interp = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            img_gt = cv2.resize(img_gt, (new_w, new_h), interpolation=interp)
            img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=interp)
            img_mask = cv2.resize(img_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # crop or pad to 512
        h, w = img_gt.shape[0:2]
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_mask = cv2.copyMakeBorder(img_mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            img_lq = img_lq[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            img_mask = img_mask[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        k1 = _sample_odd_ksize(self.blur_ksize_range)
        kernel1 = _build_v2_blur_kernel(self.blur_mode, k1)
        kernel1 = _pad_kernel_to(kernel1, max(self.blur_kernel_pad_size, k1))

        k2 = _sample_odd_ksize(self.blur_ksize_range2)
        kernel2 = _build_v2_blur_kernel(self.blur_mode, k2)
        kernel2 = _pad_kernel_to(kernel2, max(self.blur_kernel_pad_size, k2))

        if np.random.uniform() < float(self.final_sinc_prob):
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_pad_size)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_mask = img2tensor([img_mask], bgr2rgb=True, float32=True)[0]
        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)

        return {
            'gt': img_gt,
            'lq': img_lq,
            'mask': img_mask,
            'kernel1': kernel1,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
