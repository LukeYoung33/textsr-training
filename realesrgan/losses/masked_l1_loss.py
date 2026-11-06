import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

sys.path.append('.')

import realesrgan.metrics.crnn as crnn
from basicsr.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY

_redcution_modes_ = ['none', 'mean', 'sum']

_CRNN_OPT = EasyDict({
    "Transformation": 'None',
    "FeatureExtraction": 'ResNet',
    "SequenceModeling": 'None',
    "Prediction": 'CTC',
    "num_fiducial": 20,
    "input_channel": 1,
    "output_channel": 512,
    "hidden_size": 256,
    "saved_model": "weights/scene_base_CRNN.pth",
    "saved_model_eng": "weights/crnn.pth",
    "character": "-0123456789abcdefghijklmnopqrstuvwxyz",
    "character_eng": "-0123456789abcdefghijklmnopqrstuvwxyz",
})


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


def _check_reduction(reduction):
    if reduction not in _redcution_modes_:
        raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_redcution_modes_}')


def _match_channels(tensor, ref_channels):
    if tensor.shape[1] == ref_channels:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, ref_channels, 1, 1)
    raise ValueError(f'Cannot broadcast tensor with {tensor.shape[1]} channels to {ref_channels} channels.')


def _ensure_mask(mask, ref):
    if mask is None:
        return torch.ones_like(ref[:, :1, :, :])
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != ref.shape[-2:]:
        mask = F.interpolate(mask.float(), size=ref.shape[-2:], mode='nearest')
    return mask.float()


def _reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


def _masked_average(loss, mask, reduction, eps=1e-6):
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom


def _build_sobel_kernels(device, dtype):
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype)
    return sobel_x.view(1, 1, 3, 3), sobel_y.view(1, 1, 3, 3)


def _compute_gradient(img, mode='sobel', eps=1e-6):
    channels = img.shape[1]
    if mode == 'sobel':
        kernel_x, kernel_y = _build_sobel_kernels(img.device, img.dtype)
        kernel_x = kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(img, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(img, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + eps)
    if mode == 'laplacian':
        kernel = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
            device=img.device,
            dtype=img.dtype).view(1, 1, 3, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(img, kernel, padding=1, groups=channels).abs()
    raise ValueError(f'Unsupported gradient mode: {mode}')


def _binary_erode(mask):
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)


def _distance_to_boundary(mask, max_iter=32):
    current = mask
    distance = torch.zeros_like(mask)
    for step in range(1, max_iter + 1):
        eroded = _binary_erode(current)
        shell = (current - eroded).clamp_min(0.0)
        distance = distance + shell * step
        current = eroded * mask
        if torch.count_nonzero(current).item() == 0:
            break
    return distance * mask

def _distance_to_boundary_v2(mask, max_iter=32):
    """
    更稳定的“宽度 proxy”，避免层级噪声
    """
    mask = (mask > 0).float()
    dist = torch.zeros_like(mask)
    current = mask.clone()

    for i in range(1, max_iter + 1):
        # min-pool ≈ erosion（更稳定）
        eroded = 1 - F.max_pool2d(1 - current, kernel_size=3, stride=1, padding=1)

        shell = (current - eroded).clamp(min=0)
        dist += shell * i

        current = eroded

        if current.max() == 0:
            break

    return dist * mask

def _build_width_proxy(mask, max_iter=32, eps=1e-6):
    dist = _distance_to_boundary_v2(mask, max_iter)
    width = dist * 2 + eps  # 直径近似
    inv_width = 1.0 / width

    # normalize（关键）
    inv_width = inv_width / (inv_width.mean(dim=[2,3], keepdim=True) + eps)

    return inv_width


def _distance_to_skeleton(mask, max_iter=32):
    radius_map = _distance_to_boundary(mask, max_iter=max_iter)
    local_max = F.max_pool2d(radius_map, kernel_size=3, stride=1, padding=1)
    skeleton = ((radius_map > 0) & (radius_map >= local_max)).float()
    if torch.count_nonzero(skeleton).item() == 0:
        skeleton = mask

    covered = skeleton.clone()
    distance = torch.zeros_like(mask)
    frontier = skeleton.clone()
    for step in range(1, max_iter + 1):
        dilated = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1) * mask
        new_region = (dilated > covered).float() * mask
        distance = distance + new_region * step
        covered = torch.maximum(covered, dilated)
        frontier = dilated
        if torch.all(covered >= mask):
            break
    return distance * mask


def _build_stroke_weight(mask, alpha=2.0, temperature=1.0, max_iter=32, outside_mask_weight=1.0):
    mask = (mask > 0).float()
    center_distance = _distance_to_skeleton(mask, max_iter=max_iter)
    center_score = torch.exp(-center_distance / max(temperature, 1e-6))
    return outside_mask_weight + mask * alpha * center_score


def _build_stroke_weight_v2(
    mask,
    alpha=2.0,
    beta=1.0,
    temperature=1.0,
    max_iter=32,
    outside_mask_weight=0.5,
    eps=1e-6
):
    """
    V2: width-aware + center-aware
    """

    mask = (mask > 0).float()

    # === 1️⃣ distance to boundary → width proxy ===
    radius_map = _distance_to_boundary(mask, max_iter=max_iter)
    width = radius_map + eps

    inv_width = 1.0 / width

    # normalize（防止极端值）
    inv_width = inv_width / (inv_width.mean(dim=[2, 3], keepdim=True) + eps)

    # === 2️⃣ center prior（辅助）===
    center_distance = _distance_to_skeleton(mask, max_iter=max_iter)
    center_score = torch.exp(-center_distance / max(temperature, eps))

    # normalize center_score
    center_score = center_score / (center_score.mean(dim=[2, 3], keepdim=True) + eps)

    # === 3️⃣ combine ===
    weight_inside = 1.0 + alpha * inv_width + beta * center_score

    # === 4️⃣ apply mask ===
    weight = outside_mask_weight + mask * weight_inside

    return weight

def _crnn_init(recognizer_path=None):
    alphabet = open("realesrgan/metrics/benchmark.txt", 'r', encoding='utf-8').readlines()[0].replace("\n", "")
    model = crnn.CRNN(3, 256, len(alphabet) + 1, 32)
    model_path = recognizer_path if recognizer_path is not None else _CRNN_OPT.saved_model
    stat_dict = torch.load(model_path, map_location='cpu')

    load_keys = stat_dict.keys()
    model_state = model.state_dict()
    for key in load_keys:
        mapped_key = key.replace("module.", "")
        if mapped_key in model_state:
            model_state[mapped_key] = stat_dict[key]
    model.load_state_dict(model_state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _parse_crnn_data(imgs_input, ratio_keep=True):
    in_width = 128
    if ratio_keep:
        real_height, real_width = imgs_input.shape[2:]
        ratio = real_width / float(real_height)
        in_width = max(min(int(ratio * 32), 1024), 16)
    return F.interpolate(imgs_input * 255.0, (32, in_width), mode='bicubic', align_corners=False)


@LOSS_REGISTRY.register()
class MaskedL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', fg_weight=1.0, bg_weight=0.5):
        super(MaskedL1Loss, self).__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]
        if mask is not None:
            mask = _ensure_mask(mask, pred)
            weight_mask = (mask > 0).float() * self.fg_weight + (mask <= 0).float() * self.bg_weight
        else:
            weight_mask = 1
        return self.loss_weight * l1_loss(pred * weight_mask, target * weight_mask, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MaskedL1Loss_v2(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', channel_weight=None, fg_weight=1.0, bg_weight=0.5):
        super(MaskedL1Loss_v2, self).__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        default_weight = [1.0, 1.0, 1.0]
        channel_weight = channel_weight if channel_weight is not None else default_weight
        self.register_buffer('channel_weight', torch.tensor(channel_weight, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]
        if mask is not None:
            mask = _ensure_mask(mask, pred)
            weight_mask = (mask > 0).float() * self.fg_weight + (mask <= 0).float() * self.bg_weight
        else:
            weight_mask = 1
        channel_weight = self.channel_weight.to(device=pred.device, dtype=pred.dtype)
        return self.loss_weight * l1_loss(
            pred * weight_mask * channel_weight,
            target * weight_mask * channel_weight,
            weight,
            reduction=self.reduction)


@LOSS_REGISTRY.register()
class MaskGuidedGradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', gradient_type='sobel', text_weight=1.0, bg_weight=0.0):
        super(MaskGuidedGradientLoss, self).__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.gradient_type = gradient_type
        self.text_weight = text_weight
        self.bg_weight = bg_weight

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]
        grad_pred = _compute_gradient(pred, mode=self.gradient_type)
        grad_target = _compute_gradient(target, mode=self.gradient_type)
        if mask is None:
            region_weight = torch.ones_like(grad_pred[:, :1])
        else:
            mask = _ensure_mask(mask, grad_pred)
            region_weight = (mask > 0).float() * self.text_weight + (mask <= 0).float() * self.bg_weight
        region_weight = _match_channels(region_weight, grad_pred.shape[1])
        loss = torch.abs(grad_pred - grad_target) * region_weight
        if weight is not None:
            loss = loss * _match_channels(weight, loss.shape[1])
        return self.loss_weight * _reduce_loss(loss, self.reduction)


@LOSS_REGISTRY.register()
class StrokeAwareMaskedL1Loss(nn.Module):
    def __init__(
            self,
            loss_weight=1.0,
            reduction='mean',
            alpha=2.0,
            temperature=1.0,
            max_iter=32,
            outside_mask_weight=1.0):
        super(StrokeAwareMaskedL1Loss, self).__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.alpha = alpha
        self.temperature = temperature
        self.max_iter = max_iter
        self.outside_mask_weight = outside_mask_weight

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]
        if mask is None:
            stroke_weight = torch.ones_like(pred[:, :1])
        else:
            mask = _ensure_mask(mask, pred)
            stroke_weight = _build_stroke_weight(
                mask,
                alpha=self.alpha,
                temperature=self.temperature,
                max_iter=self.max_iter,
                outside_mask_weight=self.outside_mask_weight)
        stroke_weight = _match_channels(stroke_weight, channels)
        loss = torch.abs(pred - target) * stroke_weight
        if weight is not None:
            loss = loss * _match_channels(weight, channels)
        return self.loss_weight * _reduce_loss(loss, self.reduction)


@LOSS_REGISTRY.register()
class StrokeAwareMaskedL1LossV2(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        alpha=2.0,
        beta=1.0,
        temperature=1.0,
        max_iter=32,
        outside_mask_weight=0.5,
        eps=1e-6,
        cache_size=256,
    ):
        super(StrokeAwareMaskedL1LossV2, self).__init__()
        _check_reduction(reduction)

        self.loss_weight = loss_weight
        self.reduction = reduction

        # weight control
        self.alpha = alpha  # width-aware strength
        self.beta = beta    # center-aware strength
        self.temperature = temperature

        self.max_iter = max_iter
        self.outside_mask_weight = outside_mask_weight
        self.eps = eps

        # Cache stroke weight maps keyed by mask tensor hash.
        # Stroke weight is a deterministic function of the mask, so the same
        # mask patch from the same GT image always produces the same weight map.
        # This avoids running ~96 max_pool2d ops per step when masks repeat
        # across iterations (which they do because the dataset recycles GTs).
        self._cache: dict = {}
        self._cache_size = max(1, int(cache_size))

    def _get_stroke_weight(self, mask: torch.Tensor) -> torch.Tensor:
        # Use a fast per-sample hash: quantise to uint8 and hash bytes.
        key = tuple(mask.detach().cpu().to(torch.uint8).reshape(-1).tolist())
        if key in self._cache:
            return self._cache[key].to(device=mask.device, dtype=mask.dtype)

        w = _build_stroke_weight_v2(
            mask,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
            max_iter=self.max_iter,
            outside_mask_weight=self.outside_mask_weight,
            eps=self.eps,
        )
        # Evict oldest entry when cache is full (simple FIFO)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = w.detach().cpu()
        return w

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]

        if mask is None:
            stroke_weight = torch.ones_like(pred[:, :1])
        else:
            mask = _ensure_mask(mask, pred)
            # Compute per-sample and concatenate to preserve batch dim.
            weights = [self._get_stroke_weight(mask[i:i+1]) for i in range(mask.shape[0])]
            stroke_weight = torch.cat(weights, dim=0)

        stroke_weight = _match_channels(stroke_weight, channels)

        loss = torch.abs(pred - target) * stroke_weight

        if weight is not None:
            loss = loss * _match_channels(weight, channels)

        return self.loss_weight * _reduce_loss(loss, self.reduction)


@LOSS_REGISTRY.register()
class TextRegionPerceptualLoss(nn.Module):
    def __init__(
            self,
            loss_weight=1.0,
            reduction='mean',
            recognizer_path=None,
            ratio_keep=True,
            detach_target=True,
            background_fill=0.0):
        super(TextRegionPerceptualLoss, self).__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ratio_keep = ratio_keep
        self.detach_target = detach_target
        self.background_fill = background_fill
        self.recognizer = _crnn_init(recognizer_path=recognizer_path)

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        channels = pred.shape[1]
        target = target[:, :channels]
        if mask is None:
            mask = torch.ones_like(pred[:, :1])
        else:
            mask = _ensure_mask(mask, pred)
        mask = _match_channels(mask, channels)

        pred_region = pred * mask + self.background_fill * (1.0 - mask)
        target_region = target * mask + self.background_fill * (1.0 - mask)

        pred_region = _parse_crnn_data(pred_region[:, :3], ratio_keep=self.ratio_keep)
        target_region = _parse_crnn_data(target_region[:, :3], ratio_keep=self.ratio_keep)

        recognizer = self.recognizer.to(pred.device)
        pred_feats = recognizer.get_feature(pred_region)
        target_feats = recognizer.get_feature(target_region)

        total_loss = 0.0
        for pred_feat, target_feat in zip(pred_feats, target_feats):
            feat_loss = torch.abs(pred_feat - (target_feat.detach() if self.detach_target else target_feat))
            total_loss = total_loss + _reduce_loss(feat_loss, self.reduction)
        return total_loss * self.loss_weight



if __name__ == "__main__":
    # Example usage
    import cv2
    import numpy as np

    loss_fn = StrokeAwareMaskedL1LossV2(loss_weight=1.0, reduction='mean')
    pred = cv2.imread(r"E:\workspace\AISR\data_pipeline\data\my_synthetic_dataset_v5\paired\lq\group04_img05.png")
    target = cv2.imread(r"C:\Users\Guang.Yang\Pictures\gtV3\group04_img05.png")

    pred = torch.from_numpy(pred / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch.float32).requires_grad_(True)
    target = torch.from_numpy(target / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

    mask = cv2.imread(r"E:\workspace\AISR\data_pipeline\data\my_synthetic_dataset_v5\paired\mask\group04_img05_mask.png", cv2.IMREAD_GRAYSCALE)
    mask = torch.from_numpy(mask / 255.0).unsqueeze(0).unsqueeze(0).to(torch.float32)

    text_mask = cv2.imread(r"E:\workspace\AISR\data_pipeline\data\my_synthetic_dataset_v5\paired\text_mask\group04_img05.png", cv2.IMREAD_GRAYSCALE)
    text_mask = torch.from_numpy(text_mask / 255.0).unsqueeze(0).unsqueeze(0).to(torch.float32)

    loss = loss_fn(pred, text_mask, mask)
    print(f'Loss: {loss.item()}')
