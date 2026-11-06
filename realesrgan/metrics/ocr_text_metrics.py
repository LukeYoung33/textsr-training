import math
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.metrics.metric_util import quantize
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from realesrgan.archs.ppocrv5_arch import PPOCRv5MobileRec, decode_ctc


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DICT_PATH = _REPO_ROOT / "assets" / "ppocrv5_dict.txt"
_DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "weights" / "ptocr_v5_mobile_rec.pth"


def rgb2ycbcr_pt(img, y_only=False):
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]], device=img.device, dtype=img.dtype)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor(
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]],
            device=img.device,
            dtype=img.dtype,
        )
        bias = torch.tensor([16, 128, 128], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    return out_img / 255.0


def calculate_psnr_batch_pytorch(img1, img2, crop_border=0, test_y_channel=False):
    assert img1.shape == img2.shape, f"Image shapes are different: {img1.shape}, {img2.shape}."
    img1, img2 = [quantize(img) for img in (img1, img2)]
    if crop_border > 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def _ssim_per_channel(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).to(device=img1.device, dtype=img1.dtype)
    window = window.expand(img1.size(1), 1, 11, 11)

    mu1 = F.conv2d(img1, window, stride=1, padding=0, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(dim=[1, 2, 3])


def calculate_ssim_batch_pytorch(img1, img2, crop_border=0, test_y_channel=False):
    assert img1.shape == img2.shape, f"Image shapes are different: {img1.shape}, {img2.shape}."
    img1, img2 = [quantize(img) for img in (img1, img2)]
    if crop_border > 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    return _ssim_per_channel(img1 * 255.0, img2 * 255.0)


def normalize_text_for_eval(text, language=None):
    text = "" if text is None else str(text).strip()
    lang = "" if language is None else str(language).lower()
    if lang.startswith("eng"):
        text = text.lower()
    return text


def normalized_edit_similarity(pred, gt):
    pred = "" if pred is None else str(pred)
    gt = "" if gt is None else str(gt)
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    return float((max_len - editdistance.eval(pred, gt)) / max_len)


def expand_and_clip_box(box, width, height, expand_pixels):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, int(math.floor(x1 - expand_pixels)))
    y1 = max(0, int(math.floor(y1 - expand_pixels)))
    x2 = min(width, int(math.ceil(x2 + expand_pixels)))
    y2 = min(height, int(math.ceil(y2 + expand_pixels)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def polygon_to_box(polygon):
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


def crop_bgr_region(image, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    return image[y1:y2, x1:x2].copy()


class PPOCRBatchRecognizer:
    def __init__(
        self,
        device,
        dict_path=None,
        weights_path=None,
        crop_height=48,
        min_width=16,
        max_width=512,
        use_space_char=True,
    ):
        self.device = device
        self.dict_path = str(Path(dict_path) if dict_path is not None else _DEFAULT_DICT_PATH)
        self.weights_path = str(Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS_PATH)
        self.crop_height = int(crop_height)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.use_space_char = use_space_char

        self.model = PPOCRv5MobileRec(
            character_dict_path=self.dict_path,
            use_space_char=self.use_space_char,
        )
        self.model.load_weights(self.weights_path, map_location="cpu", strict=True)
        self.model.eval()
        self.model.to(self.device)

    def _preprocess(self, img):
        if img is None or img.size == 0:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim != 3 or img.shape[2] != 3:
            return None

        height, width = img.shape[:2]
        if height <= 0 or width <= 0:
            return None

        scale = float(self.crop_height) / float(height)
        target_width = max(self.min_width, min(int(round(width * scale)), self.max_width))
        resized = cv2.resize(img, (target_width, self.crop_height), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1))
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def recognize(self, images, batch_size=32):
        results = [{"text": "", "confidence": 0.0} for _ in images]
        valid_items = []
        for idx, image in enumerate(images):
            tensor = self._preprocess(image)
            if tensor is not None:
                valid_items.append((idx, tensor))

        if not valid_items:
            return results

        for start in range(0, len(valid_items), batch_size):
            chunk = valid_items[start:start + batch_size]
            max_width = max(item[1].shape[2] for item in chunk)
            batch = torch.ones(len(chunk), 3, self.crop_height, max_width, dtype=torch.float32)
            for i, (_, tensor) in enumerate(chunk):
                width = tensor.shape[2]
                batch[i, :, :, :width] = tensor
            batch = batch.to(self.device)

            with torch.no_grad():
                preds = self.model(batch)
            decoded = decode_ctc(preds, self.dict_path, use_space_char=self.use_space_char)
            for (result_idx, _), (text, confidence) in zip(chunk, decoded):
                results[result_idx] = {"text": text, "confidence": float(confidence)}
        return results


class LocalVGGPerceptualMetric(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()
        self.layer_weights = layer_weights or {
            "relu1_2": 1.0 / 32.0,
            "relu2_2": 1.0 / 16.0,
            "relu3_4": 1.0 / 8.0,
            "relu4_4": 1.0 / 4.0,
            "relu5_4": 1.0,
        }
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type="vgg19",
            use_input_norm=True,
            range_norm=False,
            requires_grad=False,
            remove_pooling=False,
        )
        self.vgg.eval()

    def forward(self, x, y):
        feats_x = self.vgg(x)
        feats_y = self.vgg(y)
        distances = x.new_zeros((x.shape[0],), dtype=torch.float32)
        for layer_name, weight in self.layer_weights.items():
            feat_x = F.normalize(feats_x[layer_name], p=2, dim=1)
            feat_y = F.normalize(feats_y[layer_name], p=2, dim=1)
            layer_dist = (feat_x - feat_y).pow(2).mean(dim=[1, 2, 3])
            distances = distances + float(weight) * layer_dist
        return distances
