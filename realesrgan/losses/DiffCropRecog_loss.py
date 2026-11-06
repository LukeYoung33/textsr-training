import math
import torch
import kornia.geometry as K
from typing import List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import sys
sys.path.append('.')
from basicsr.utils.registry import LOSS_REGISTRY
from realesrgan.archs.ppocrv5_arch import PPOCRv5MobileRec


def load_char_dict(dict_path='assets/ppocr_keys_v1.txt', use_space_char: bool = True):
    """Load PP-OCR character rows without changing legal whitespace symbols.

    @shanlan-comment
    Input: UTF-8 dictionary path with one character or symbol per line and the
        PP-OCR ``use_space_char`` switch.
    Output: character-to-index map, index-to-character map, and CTC class count;
        index zero remains reserved for the blank class.
    Key Steps: remove only CR/LF line endings, preserve every dictionary symbol,
        then append the ASCII space class when requested and absent.
    Key Params: ``use_space_char`` controls the optional trailing ASCII space;
        fullwidth space and other dictionary whitespace are not stripped.
    State/Dependencies: dictionary ordering must match the frozen recognizer's
        output classes.
    """
    with open(dict_path, 'r', encoding='utf-8', newline='') as f:
        chars = [line.rstrip('\r\n') for line in f]
    if any(char == '' for char in chars):
        raise ValueError(f'Empty character row in PP-OCR dictionary: {dict_path}')
    char2idx = {char: i + 1 for i, char in enumerate(chars)}  # 0 留给 blank
    if use_space_char and ' ' not in char2idx:
        char2idx[' '] = len(char2idx) + 1
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(char2idx) + 1

def normalize_text_for_ctc(text: str, remove_spaces: bool = True) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-") \
               .replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-") \
               .replace("\uFF0D", "-")
    if remove_spaces:
        text = text.replace(" ", "").replace("\u3000", "")
    return text.strip()


def order_quad_points(points: torch.Tensor) -> torch.Tensor:
    """Normalize an arbitrary convex-quad permutation to TL, TR, BR, BL."""
    center = points.mean(dim=0, keepdim=True)
    angles = torch.atan2(points[:, 1] - center[:, 1], points[:, 0] - center[:, 0])
    ordered = points[torch.argsort(angles)]
    top_idx = torch.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = torch.roll(ordered, shifts=-int(top_idx.item()), dims=0)
    if ordered[1, 1] > ordered[3, 1]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def is_valid_quad(points: torch.Tensor, min_area: float = 1.0, min_edge: float = 1.0) -> bool:
    """Return whether ordered points define a finite, convex perspective region."""
    if points.shape != (4, 2) or not torch.isfinite(points).all():
        return False
    edges = torch.roll(points, shifts=-1, dims=0) - points
    if torch.any(torch.linalg.vector_norm(edges, dim=1) < min_edge):
        return False

    cross = edges[:, 0] * torch.roll(edges, -1, dims=0)[:, 1]
    cross -= edges[:, 1] * torch.roll(edges, -1, dims=0)[:, 0]
    if torch.any(cross.abs() < 1e-6):
        return False
    if not (torch.all(cross > 0) or torch.all(cross < 0)):
        return False

    x, y = points[:, 0], points[:, 1]
    area = 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, -1) - y * torch.roll(x, -1)))
    return bool(area.item() >= min_area)


def _audit_increment(audit, reason: str) -> None:
    if audit is not None:
        audit[reason] = int(audit.get(reason, 0)) + 1


def _validate_annotation_batch(annotations, batch_size: int) -> None:
    if not isinstance(annotations, (list, tuple)) or len(annotations) != batch_size:
        raise ValueError(
            f"Expected one annotation list per image ({batch_size}), got "
            f"{type(annotations).__name__} with length "
            f"{len(annotations) if isinstance(annotations, (list, tuple)) else 'n/a'}")
    for index, image_annotations in enumerate(annotations):
        if not isinstance(image_annotations, (list, tuple)) or isinstance(image_annotations, str):
            raise ValueError(
                f"annotations[{index}] must be a list/tuple of complete line records, "
                f"got {type(image_annotations).__name__}")


def _prepare_quad_transform(
    line_ann,
    img_h: int,
    img_w: int,
    crop_height: int,
    min_width: int,
    max_width: int,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    device,
    audit=None,
    require_text: bool = True,
):
    try:
        fields = line_ann.split(',') if isinstance(line_ann, str) else list(line_ann)
        if len(fields) < 9:
            _audit_increment(audit, 'invalid_annotation')
            return None
        coords = [float(value) for value in fields[:8]]
        text = ','.join(str(value) for value in fields[8:]).strip()
    except (TypeError, ValueError):
        _audit_increment(audit, 'invalid_annotation')
        return None

    if require_text and not text:
        _audit_increment(audit, 'empty_text')
        return None
    if crop_height < 2 or min_width < 2 or max_width < min_width:
        raise ValueError(
            f"Invalid OCR crop dimensions: height={crop_height}, "
            f"min_width={min_width}, max_width={max_width}")

    src_pts = torch.tensor(coords, dtype=torch.float32, device=device).view(4, 2)
    if not torch.isfinite(src_pts).all():
        _audit_increment(audit, 'nonfinite_coordinates')
        return None
    if (torch.any(src_pts[:, 0] < 0) or torch.any(src_pts[:, 0] > img_w - 1)
            or torch.any(src_pts[:, 1] < 0) or torch.any(src_pts[:, 1] > img_h - 1)):
        _audit_increment(audit, 'out_of_bounds')
        return None

    src_pts = order_quad_points(src_pts)
    if not is_valid_quad(src_pts):
        _audit_increment(audit, 'invalid_geometry')
        return None

    top_len = torch.linalg.vector_norm(src_pts[1] - src_pts[0])
    bottom_len = torch.linalg.vector_norm(src_pts[2] - src_pts[3])
    left_len = torch.linalg.vector_norm(src_pts[3] - src_pts[0])
    right_len = torch.linalg.vector_norm(src_pts[2] - src_pts[1])
    avg_width_src = (top_len + bottom_len) / 2.0
    avg_height_src = (left_len + right_len) / 2.0
    if avg_height_src < 1.0 or avg_width_src < 1.0:
        _audit_increment(audit, 'invalid_size')
        return None

    aspect_ratio = float((avg_width_src / avg_height_src).item())
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        _audit_increment(audit, 'invalid_aspect_ratio')
        return None

    target_width = int(float(avg_width_src.item()) * crop_height / float(avg_height_src.item()) + 0.5)
    target_width = max(min_width, min(target_width, max_width))
    if target_width < 2:
        _audit_increment(audit, 'invalid_size')
        return None

    dst_pts = src_pts.new_tensor([
        [0.0, 0.0],
        [float(target_width - 1), 0.0],
        [float(target_width - 1), float(crop_height - 1)],
        [0.0, float(crop_height - 1)],
    ])
    try:
        transform = K.get_perspective_transform(
            src_pts.unsqueeze(0), dst_pts.unsqueeze(0))
    except RuntimeError:
        _audit_increment(audit, 'homography_error')
        return None
    if not torch.isfinite(transform).all():
        _audit_increment(audit, 'nonfinite_homography')
        return None
    return transform, text, target_width


def _warp_crop(
    patch, transform, crop_height, target_width, padding_mode, audit=None,
    interpolation='bilinear'
):
    """Apply one differentiable perspective crop with a backward-compatible mode.

    @shanlan-comment
    Input: BCHW patch, shared homography, output size, padding mode, and interpolation mode.
    Output: CHW crop preserving autograd connectivity, or ``None`` after audited warp failure.
    Key Steps: call Kornia perspective warp, squeeze the singleton batch, reject non-finite output.
    Key Params: legacy callers retain bilinear; text-line CP may request bicubic.
    State/Dependencies: Kornia geometry; audit counters are mutated on recoverable failures.
    """
    try:
        crop = K.warp_perspective(
            patch,
            transform,
            dsize=(crop_height, target_width),
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=True,
        ).squeeze(0)
    except RuntimeError:
        _audit_increment(audit, 'warp_error')
        return None
    if not torch.isfinite(crop).all():
        _audit_increment(audit, 'nonfinite_crop')
        return None
    return crop


def prepare_text_crops(
    sr_patches: torch.Tensor,
    annotations: List[List[List]],
    crop_height: int = 48,
    device: torch.device = None,
    min_width: int = 16,
    max_width: int = 256,
    padding_mode: str = 'zeros',
    interpolation: str = 'bilinear',
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 20.0,
    audit=None,
    require_text: bool = True,
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """Warp variable-width text crops while preserving SR autograd connectivity.

    @shanlan-comment
    Input: BCHW image batch and nested polygon line annotations.
    Output: crop tensors, transcripts, and native widths; invalid lines are audited and skipped.
    Key Steps: validate each polygon, build its homography, and apply the selected interpolation mode.
    Key Params: crop geometry, width/aspect bounds, padding, and interpolation preserve legacy defaults.
    State/Dependencies: Kornia warp retains gradients to the input image.
    """
    if device is None:
        device = sr_patches.device
    sr_patches = sr_patches.to(device).contiguous()
    batch_size, _, img_h, img_w = sr_patches.shape
    _validate_annotation_batch(annotations, batch_size)

    all_crops: List[torch.Tensor] = []
    all_texts: List[str] = []
    all_widths: List[int] = []
    for batch_index in range(batch_size):
        patch = sr_patches[batch_index:batch_index + 1]
        for line_ann in annotations[batch_index]:
            prepared = _prepare_quad_transform(
                line_ann, img_h, img_w, crop_height, min_width, max_width,
                min_aspect_ratio, max_aspect_ratio, device, audit=audit,
                require_text=require_text)
            if prepared is None:
                continue
            transform, text, target_width = prepared
            crop = _warp_crop(
                patch, transform, crop_height, target_width, padding_mode,
                audit=audit, interpolation=interpolation)
            if crop is None:
                continue
            all_crops.append(crop)
            all_texts.append(text)
            all_widths.append(target_width)
            _audit_increment(audit, 'accepted')
    return all_crops, all_texts, all_widths

def encode_texts_for_ctc(
    texts: List[str],
    char2idx: dict,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    encoded_targets: List[torch.Tensor] = []
    target_lengths: List[int] = []

    for text in texts:
        encoded = [char2idx[c] for c in text if c in char2idx]
        if not encoded or len(encoded) != len(text):
            continue
        encoded_targets.append(torch.tensor(encoded, dtype=torch.long))
        target_lengths.append(len(encoded))

    if not encoded_targets:
        return None, None

    return torch.cat(encoded_targets, dim=0), torch.tensor(target_lengths, dtype=torch.long)


def normalize_rgb01_for_ppocrv5(images: torch.Tensor) -> torch.Tensor:
    """Convert model-produced RGB floats from [0, 1] to PP-OCRv5 BGR [-1, 1].

    @shanlan-comment
    Input: CHW or BCHW RGB floating tensors under the training image contract.
    Output: BGR tensor normalized to the PP-OCRv5 [-1, 1] input range.
    Key Steps: clamp each pixel independently, reorder channels, and normalize.
    Key Params: no range inference is performed; callers must provide [0, 1] data.
    State/Dependencies: differentiable PyTorch operations preserve unsaturated gradients.
    """
    images = images.float().clamp(0.0, 1.0)
    if images.dim() == 3:
        images = images[[2, 1, 0], ...]
    elif images.dim() == 4:
        images = images[:, [2, 1, 0], ...]
    else:
        raise ValueError(f"Expected CHW or BCHW RGB input, got {tuple(images.shape)}")
    return images * 2.0 - 1.0


def collate_crops_for_ppocrv5(
    crops: List[torch.Tensor],
    device,
    image_width: Optional[int] = None,
):
    """Pack variable-width RGB crops for PP-OCRv5.

    @shanlan-comment
    Input: non-empty CHW RGB tensors sharing height, a destination device, and
        an optional fixed recognizer width.
    Output: normalized BGR batch, effective unpadded widths, and padded pixel width.
    Key Steps: optionally resize over-wide crops with preserved height, clamp the
        [0, 1] contract, convert RGB to BGR, normalize, and right-pad with -1.
    Key Params: ``image_width=None`` retains legacy batch-max padding; a positive
        width (320 for PP-OCRv5) uses a fixed right-padded recognizer input.
    State/Dependencies: uses PyTorch padding and preserves input gradients.
    """
    if not crops:
        raise ValueError('PP-OCR crop collation requires at least one crop')

    if image_width is None:
        effective_widths = [int(c.shape[-1]) for c in crops]
        crops_norm = [normalize_rgb01_for_ppocrv5(c) for c in crops]
        crops_padded = pad_sequence(
            [c.permute(2, 0, 1) for c in crops_norm],
            batch_first=True,
            padding_value=-1.0
        ).permute(0, 2, 3, 1)
    else:
        image_width = int(image_width)
        if image_width < 1:
            raise ValueError(f'image_width must be positive, got {image_width}')
        effective_widths = []
        padded_crops = []
        for crop in crops:
            width = int(crop.shape[-1])
            if width > image_width:
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(int(crop.shape[-2]), image_width),
                    mode='bilinear',
                    align_corners=True,
                ).squeeze(0)
                width = image_width
            crop_norm = normalize_rgb01_for_ppocrv5(crop)
            if width < image_width:
                crop_norm = F.pad(
                    crop_norm,
                    (0, image_width - width, 0, 0),
                    mode='constant',
                    value=-1.0,
                )
            padded_crops.append(crop_norm)
            effective_widths.append(width)
        crops_padded = torch.stack(padded_crops, dim=0)

    return {
        'images': crops_padded.to(device),
        'orig_widths': torch.tensor(effective_widths, dtype=torch.float32, device=device),
        'padded_width': int(crops_padded.shape[-1])
    }


def prepare_paired_text_crops(
    out_images: torch.Tensor,
    gt_images: torch.Tensor,
    annotations: List[List[List]],
    crop_height: int = 48,
    device: torch.device = None,
    min_width: int = 16,
    max_width: int = 256,
    padding_mode: str = 'zeros',
    interpolation: str = 'bilinear',
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 20.0,
    audit=None,
    require_text: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[int]]:
    """Warp matching SR and GT text regions with one validated homography.

    @shanlan-comment
    Input: same-shaped BCHW SR/GT tensors and per-image quadrilateral OCR lines.
    Output: paired differentiable crops, normalized transcripts, and native crop
        widths; invalid polygons are skipped through the audit path.
    Key Steps: validate geometry, compute one perspective transform per line,
        and apply it identically to SR and GT.
    Key Params: crop height, width bounds, padding mode, interpolation, and aspect-ratio limits.
    State/Dependencies: Kornia homography; shared transforms preserve pair alignment.
    """
    if out_images.shape != gt_images.shape:
        raise ValueError(f"Shape mismatch: {tuple(out_images.shape)} vs {tuple(gt_images.shape)}")
    if device is None:
        device = out_images.device

    out_images = out_images.to(device).contiguous()
    gt_images = gt_images.to(device).contiguous()
    batch_size, _, img_h, img_w = out_images.shape
    _validate_annotation_batch(annotations, batch_size)

    out_crops: List[torch.Tensor] = []
    gt_crops: List[torch.Tensor] = []
    all_texts: List[str] = []
    all_widths: List[int] = []
    for batch_index in range(batch_size):
        out_patch = out_images[batch_index:batch_index + 1]
        gt_patch = gt_images[batch_index:batch_index + 1]
        for line_ann in annotations[batch_index]:
            prepared = _prepare_quad_transform(
                line_ann, img_h, img_w, crop_height, min_width, max_width,
                min_aspect_ratio, max_aspect_ratio, device, audit=audit,
                require_text=require_text)
            if prepared is None:
                continue
            transform, text, target_width = prepared
            out_crop = _warp_crop(
                out_patch, transform, crop_height, target_width, padding_mode,
                audit=audit, interpolation=interpolation)
            if out_crop is None:
                continue
            gt_crop = _warp_crop(
                gt_patch, transform, crop_height, target_width, padding_mode,
                audit=audit, interpolation=interpolation)
            if gt_crop is None:
                continue
            out_crops.append(out_crop)
            gt_crops.append(gt_crop)
            all_texts.append(text)
            all_widths.append(target_width)
            _audit_increment(audit, 'accepted')
    return out_crops, gt_crops, all_texts, all_widths


def build_valid_time_mask(orig_widths, padded_width: int, time_steps: int) -> torch.Tensor:
    """Map unpadded crop widths to valid OCR sequence positions.

    @shanlan-comment
    Input: positive per-crop pixel widths, shared padded width, and OCR length.
    Output: boolean ``[batch, time_steps]`` mask on ``orig_widths.device``.
    Key Steps: scale widths to sequence length with ceil, clamp bounds, compare
    each sequence index against its crop-specific valid length.
    Key Params: ``padded_width`` is the collated image width; ``time_steps`` is
    the OCR logit length.
    State/Dependencies: pure tensor logic with no module or global state.
    """
    if padded_width <= 0 or time_steps <= 0:
        raise ValueError(
            f"padded_width and time_steps must be positive, got "
            f"{padded_width} and {time_steps}")
    valid_lengths = torch.ceil(
        orig_widths.float() * (float(time_steps) / float(padded_width))
    ).to(dtype=torch.long).clamp_(min=1, max=time_steps)
    time_indices = torch.arange(time_steps, device=orig_widths.device)
    return time_indices.unsqueeze(0) < valid_lengths.unsqueeze(1)

@LOSS_REGISTRY.register()
class OCRCTCLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        dict_path='assets/ppocrv5_dict.txt',
        weights_path='weights/ptocr_v5_mobile_rec.pth',
        use_space_char=True,
        crop_height=48,
        min_width=16,
        max_width=256,
        padding_mode='zeros',
        normalize_text=True,
        remove_spaces=True,
        min_text_length=1,
        max_text_length=64,
        min_aspect_ratio=0.2,
        max_aspect_ratio=20.0,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.crop_height = int(crop_height)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.padding_mode = padding_mode
        self.normalize_text = bool(normalize_text)
        self.remove_spaces = bool(remove_spaces)
        self.min_text_length = int(min_text_length)
        self.max_text_length = int(max_text_length)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

        self.char2idx, self.idx2char, self.num_classes = load_char_dict(
            dict_path=dict_path,
            use_space_char=use_space_char
        )

        self.ocr_model = PPOCRv5MobileRec(
            character_dict_path=dict_path,
            use_space_char=use_space_char
        )
        self.ocr_model.load_weights(weights_path, map_location='cpu', strict=True)
        for p in self.ocr_model.parameters():
            p.requires_grad_(False)

        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self._device = None

    def _ensure_device(self, device):
        if self._device != device:
            self.ocr_model.to(device)
            self._device = device

    def forward(self, out_images, gt_lines, mask=None):
        device = out_images.device
        self._ensure_device(device)

        crops, texts, crop_widths = prepare_text_crops(
            out_images,
            gt_lines,
            crop_height=self.crop_height,
            device=device,
            min_width=self.min_width,
            max_width=self.max_width,
            padding_mode=self.padding_mode,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio
        )
        if not crops:
            return out_images.sum() * 0.0

        if self.normalize_text:
            texts = [normalize_text_for_ctc(t, remove_spaces=self.remove_spaces) for t in texts]
        filtered = [
            (crop, text, crop_width)
            for crop, text, crop_width in zip(crops, texts, crop_widths)
            if self.min_text_length <= len(text) <= self.max_text_length
        ]
        if not filtered:
            return out_images.sum() * 0.0

        crops, texts, crop_widths = zip(*filtered)
        crops = list(crops)
        texts = list(texts)

        valid_indices = []
        valid_texts = []
        for idx, text in enumerate(texts):
            if all(char in self.char2idx for char in text):
                valid_indices.append(idx)
                valid_texts.append(text)
        if not valid_indices:
            return out_images.sum() * 0.0

        crops = [crops[i] for i in valid_indices]
        targets, target_lengths = encode_texts_for_ctc(valid_texts, self.char2idx)
        if targets is None or target_lengths is None:
            return out_images.sum() * 0.0

        batch_crnn = collate_crops_for_ppocrv5(crops, device=device)

        # Ensure logits output (CTCHead only applies softmax when eval)
        self.ocr_model.eval()
        self.ocr_model.ctc_head.train()

        logits = self.ocr_model(batch_crnn['images'])
        log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
        time_steps = logits.shape[1]
        width_scale = time_steps / float(max(batch_crnn['padded_width'], 1))
        input_lengths = torch.clamp(
            torch.ceil(batch_crnn['orig_widths'] * width_scale).to(dtype=torch.long),
            min=1,
            max=time_steps
        )

        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        valid = input_lengths >= target_lengths
        if not torch.any(valid):
            return out_images.sum() * 0.0

        if not torch.all(valid):
            split_targets = torch.split(targets, target_lengths.tolist())
            kept_targets = [target for keep, target in zip(valid.tolist(), split_targets) if keep]
            targets = torch.cat(kept_targets, dim=0)
            target_lengths = target_lengths[valid]
            log_probs = log_probs[:, valid, :]
            input_lengths = input_lengths[valid]

        loss = self.ctc_loss_fn(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class OCRCTCSimpleLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        dict_path='assets/ppocrv5_dict.txt',
        weights_path='weights/ptocr_v5_mobile_rec.pth',
        use_space_char=True,
        normalize_text=True,
        remove_spaces=True,
        min_text_length=1,
        max_text_length=64,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.normalize_text = bool(normalize_text)
        self.remove_spaces = bool(remove_spaces)
        self.min_text_length = int(min_text_length)
        self.max_text_length = int(max_text_length)

        self.char2idx, self.idx2char, self.num_classes = load_char_dict(
            dict_path=dict_path,
            use_space_char=use_space_char
        )

        self.ocr_model = PPOCRv5MobileRec(
            character_dict_path=dict_path,
            use_space_char=use_space_char
        )
        self.ocr_model.load_weights(weights_path, map_location='cpu', strict=True)
        for p in self.ocr_model.parameters():
            p.requires_grad_(False)

        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self._device = None

    def _ensure_device(self, device):
        if self._device != device:
            self.ocr_model.to(device)
            self._device = device

    def _extract_text(self, line_ann):
        if line_ann is None:
            return ""
        if isinstance(line_ann, str):
            parts = line_ann.split(',')
        elif isinstance(line_ann, (list, tuple)):
            parts = list(line_ann)
        else:
            return ""
        if len(parts) < 9:
            return ""
        return ','.join([str(v) for v in parts[8:]]).strip()

    def forward(self, out_images, gt_lines, mask=None):
        device = out_images.device
        self._ensure_device(device)

        images = []
        texts = []
        for image, annos in zip(out_images, gt_lines):
            if not annos:
                continue
            text = self._extract_text(annos[0])
            if not text:
                continue
            if self.normalize_text:
                text = normalize_text_for_ctc(text, remove_spaces=self.remove_spaces)
            if not (self.min_text_length <= len(text) <= self.max_text_length):
                continue
            if not all(char in self.char2idx for char in text):
                continue
            images.append(image)
            texts.append(text)

        if not images:
            return out_images.sum() * 0.0

        targets, target_lengths = encode_texts_for_ctc(texts, self.char2idx)
        if targets is None or target_lengths is None:
            return out_images.sum() * 0.0

        batch_ocr = collate_crops_for_ppocrv5(images, device=device)

        self.ocr_model.eval()
        self.ocr_model.ctc_head.train()

        logits = self.ocr_model(batch_ocr['images'])
        log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
        time_steps = logits.shape[1]
        width_scale = time_steps / float(max(batch_ocr['padded_width'], 1))
        input_lengths = torch.clamp(
            torch.ceil(batch_ocr['orig_widths'] * width_scale).to(dtype=torch.long),
            min=1,
            max=time_steps
        )

        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        valid = input_lengths >= target_lengths
        if not torch.any(valid):
            return out_images.sum() * 0.0

        if not torch.all(valid):
            split_targets = torch.split(targets, target_lengths.tolist())
            kept_targets = [target for keep, target in zip(valid.tolist(), split_targets) if keep]
            targets = torch.cat(kept_targets, dim=0)
            target_lengths = target_lengths[valid]
            log_probs = log_probs[:, valid, :]
            input_lengths = input_lengths[valid]

        loss = self.ctc_loss_fn(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        return loss * self.loss_weight


def decode_ctc_labels(labels: torch.Tensor, idx2char: dict, valid_time_mask=None) -> List[str]:
    """Decode batched CTC argmax labels by removing blanks and repeated tokens."""
    decoded = []
    for batch_index, sequence in enumerate(labels):
        chars = []
        previous = None
        for time_index, label_tensor in enumerate(sequence):
            if valid_time_mask is not None and not bool(valid_time_mask[batch_index, time_index]):
                break
            label = int(label_tensor.item())
            if label != 0 and label != previous and label in idx2char:
                chars.append(idx2char[label])
            previous = label
        decoded.append(''.join(chars))
    return decoded


def token_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = 'hard_ce',
    temperature: float = 1.0,
    confidence_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce selected OCR tokens using hard CE or temperature-scaled soft KL."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher logit shape mismatch: {tuple(student_logits.shape)} "
            f"vs {tuple(teacher_logits.shape)}")
    if valid_mask.shape != student_logits.shape[:2]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} does not match "
            f"logits {tuple(student_logits.shape[:2])}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not torch.any(valid_mask):
        return student_logits.sum() * 0.0

    teacher_logits = teacher_logits.detach()
    if mode == 'hard_ce':
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        teacher_conf, teacher_labels = teacher_probs.max(dim=-1)
        token_losses = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            teacher_labels.reshape(-1),
            reduction='none',
        ).view_as(teacher_labels)
        weights = teacher_conf.detach() if confidence_weights is None else confidence_weights.detach()
    elif mode == 'soft_kl':
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        token_losses = F.kl_div(
            student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        token_losses = token_losses * (temperature ** 2)
        weights = torch.ones_like(token_losses) if confidence_weights is None else confidence_weights.detach()
    else:
        raise ValueError(f"Unsupported distillation_mode: {mode}")

    selected_weights = weights[valid_mask]
    return (token_losses[valid_mask] * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


@LOSS_REGISTRY.register()
class PPOCRTextPercepLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        dict_path='assets/ppocrv5_dict.txt',
        weights_path='weights/ptocr_v5_mobile_rec.pth',
        use_space_char=True,
        crop_height=48,
        min_width=16,
        max_width=320,
        image_width=320,
        padding_mode='zeros',
        normalize_text=True,
        remove_spaces=True,
        min_text_length=1,
        max_text_length=64,
        min_aspect_ratio=0.2,
        max_aspect_ratio=20.0,
        confidence_threshold=0.3,
        distillation_mode='hard_ce',
        temperature=1.0,
        require_transcript_match=False,
        enable_logging=False,
        log_prefix='PPOCRTextPercepLoss',
    ):
        """Configure the frozen PP-OCR recognizer and crop-supervision contract.

        @shanlan-comment
        Input: loss, dictionary, recognizer-weight, crop geometry, decoding, and
            distillation options supplied by the training registry.
        Output: initialized loss module with frozen OCR parameters and explicit
            dictionary/width contracts.
        Key Steps: validate scalar options, load the class map, construct the
            recognizer, and load its strict checkpoint.
        Key Params: ``max_width`` bounds geometry crops; ``image_width`` is the
            fixed right-padded PP-OCR input width.
        State/Dependencies: recognizer weights remain frozen; generator gradients
            are consumed by ``forward`` through differentiable crops.
        """
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.crop_height = int(crop_height)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.image_width = int(image_width)
        if self.image_width < 1:
            raise ValueError(f'image_width must be positive, got {image_width}')
        self.padding_mode = padding_mode
        self.normalize_text = bool(normalize_text)
        self.remove_spaces = bool(remove_spaces)
        self.min_text_length = int(min_text_length)
        self.max_text_length = int(max_text_length)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.confidence_threshold = float(confidence_threshold)
        self.distillation_mode = str(distillation_mode).lower()
        self.temperature = float(temperature)
        self.require_transcript_match = bool(require_transcript_match)
        if self.distillation_mode not in ('hard_ce', 'soft_kl'):
            raise ValueError(f"Unsupported distillation_mode: {distillation_mode}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.enable_logging = bool(enable_logging)
        self.log_prefix = str(log_prefix)

        self.char2idx, self.idx2char, self.num_classes = load_char_dict(
            dict_path=dict_path,
            use_space_char=use_space_char
        )

        self.ocr_model = PPOCRv5MobileRec(
            character_dict_path=dict_path,
            use_space_char=use_space_char
        )
        self.ocr_model.load_weights(weights_path, map_location='cpu', strict=True)
        for p in self.ocr_model.parameters():
            p.requires_grad_(False)

        self._device = None

    def _ensure_device(self, device):
        if self._device != device:
            self.ocr_model.to(device)
            self._device = device

    def _log(self, message: str) -> None:
        if self.enable_logging:
            print(f"[{self.log_prefix}] {message}")

    def forward(self, out_images, gt_images, gt_lines, mask=None):
        """Compute confidence-weighted OCR distillation on paired text crops.

        @shanlan-comment
        Input: BCHW RGB SR/GT tensors and polygon annotations in GT coordinates.
        Output: scalar weighted loss; zero when no usable crop/token exists.
        Key Steps: pair-warp crops, run frozen PP-OCRv5, select confident
        non-blank GT tokens inside each crop's valid width, and average CE.
        Key Params: confidence, text, crop-width, and aspect-ratio thresholds.
        State/Dependencies: frozen OCR parameters; gradients flow through SR.
        """
        device = out_images.device
        self._ensure_device(device)

        out_crops, gt_crops, texts, _ = prepare_paired_text_crops(
            out_images,
            gt_images,
            gt_lines,
            crop_height=self.crop_height,
            device=device,
            min_width=self.min_width,
            max_width=self.max_width,
            padding_mode=self.padding_mode,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )
        if not out_crops:
            self._log("skip: no valid paired crops")
            return out_images.sum() * 0.0

        if self.normalize_text:
            texts = [normalize_text_for_ctc(t, remove_spaces=self.remove_spaces) for t in texts]
        valid_indices = [
            idx for idx, text in enumerate(texts)
            if text.upper() != '[DET]'
            and self.min_text_length <= len(text) <= self.max_text_length
            and all(char in self.char2idx for char in text)
        ]
        if not valid_indices:
            self._log("skip: no crops remain after text filtering")
            return out_images.sum() * 0.0

        out_crops = [out_crops[i] for i in valid_indices]
        gt_crops = [gt_crops[i] for i in valid_indices]
        texts = [texts[i] for i in valid_indices]

        out_batch = collate_crops_for_ppocrv5(
            out_crops, device=device, image_width=self.image_width)
        gt_batch = collate_crops_for_ppocrv5(
            gt_crops, device=device, image_width=self.image_width)

        self.ocr_model.eval()

        out_logits = self.ocr_model(out_batch['images'])
        with torch.no_grad():
            gt_logits = self.ocr_model(gt_batch['images'])
            gt_probs = torch.softmax(gt_logits, dim=-1)
            gt_conf, gt_labels = gt_probs.max(dim=-1)

        valid_time_mask = build_valid_time_mask(
            gt_batch['orig_widths'],
            gt_batch['padded_width'],
            gt_logits.shape[1],
        )
        valid_mask = (
            (gt_conf >= self.confidence_threshold)
            & (gt_labels != 0)
            & torch.isfinite(gt_conf)
            & valid_time_mask
        )
        if self.require_transcript_match:
            decoded_texts = decode_ctc_labels(gt_labels, self.idx2char, valid_time_mask)
            transcript_matches = torch.tensor(
                [decoded == text for decoded, text in zip(decoded_texts, texts)],
                dtype=torch.bool,
                device=device,
            )
            valid_mask = valid_mask & transcript_matches.unsqueeze(1)

        if not torch.any(valid_mask):
            self._log(
                "skip: no valid OCR tokens "
                f"(crops={len(out_crops)}, threshold={self.confidence_threshold:.2f})"
            )
            return out_images.sum() * 0.0

        loss = token_distillation_loss(
            out_logits,
            gt_logits,
            valid_mask,
            mode=self.distillation_mode,
            temperature=self.temperature,
            confidence_weights=gt_conf,
        )
        valid_ratio = valid_mask.float().mean().item()
        mean_conf = gt_conf[valid_mask].mean().item()
        self._log(
            f"crops={len(out_crops)} valid_tokens={int(valid_mask.sum().item())}/{valid_mask.numel()} "
            f"ratio={valid_ratio:.4f} mean_conf={mean_conf:.4f} loss={loss.item():.6f}"
        )
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class PPOCRBackbonePercepLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        dict_path='assets/ppocrv5_dict.txt',
        weights_path='weights/ptocr_v5_mobile_rec.pth',
        use_space_char=True,
        layer_weights=(1.0, 1.0, 1.0, 1.0),
        criterion='l1',
        include_stem=False,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.include_stem = bool(include_stem)
        self.layer_weights = [float(v) for v in layer_weights]

        self.ocr_model = PPOCRv5MobileRec(
            character_dict_path=dict_path,
            use_space_char=use_space_char
        )
        self.ocr_model.load_weights(weights_path, map_location='cpu', strict=True)
        for p in self.ocr_model.parameters():
            p.requires_grad_(False)

        criterion = str(criterion).lower()
        if criterion == 'l1':
            self.criterion = 'l1'
        elif criterion in ('l2', 'mse'):
            self.criterion = 'mse'
        elif criterion in ('smooth_l1', 'smoothl1', 'sl1'):
            self.criterion = 'smooth_l1'
        else:
            raise ValueError(f'Unsupported criterion: {criterion}')

        self._device = None

    def _ensure_device(self, device):
        if self._device != device:
            self.ocr_model.to(device)
            self._device = device

    def _normalize_for_ocr(self, images):
        return normalize_rgb01_for_ppocrv5(images)

    def _extract_backbone_feats(self, images):
        x = self._normalize_for_ocr(images)
        feats = []
        backbone = self.ocr_model.backbone
        x = backbone.conv1(x)
        if self.include_stem:
            feats.append(x)
        x = backbone.blocks2(x)
        x = backbone.blocks3(x)
        feats.append(x)
        x = backbone.blocks4(x)
        feats.append(x)
        x = backbone.blocks5(x)
        feats.append(x)
        x = backbone.blocks6(x)
        feats.append(x)
        return feats

    def _reduce_feature_loss(self, pred_feat, gt_feat, mask=None):
        if self.criterion == 'l1':
            diff = (pred_feat - gt_feat).abs()
        elif self.criterion == 'mse':
            diff = (pred_feat - gt_feat).pow(2)
        else:
            diff = F.smooth_l1_loss(pred_feat, gt_feat, reduction='none')

        if mask is None:
            return diff.mean()

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] > 1:
            mask = mask[:, :1, ...]
        feat_mask = F.interpolate(mask.float(), size=pred_feat.shape[-2:], mode='nearest')
        denom = feat_mask.sum() * pred_feat.shape[1]
        if denom.item() <= 0:
            return diff.mean()
        return (diff * feat_mask).sum() / denom.clamp_min(1e-6)

    def forward(self, out_images, gt_images, mask=None):
        if out_images.shape != gt_images.shape:
            raise ValueError(f'Shape mismatch: {tuple(out_images.shape)} vs {tuple(gt_images.shape)}')

        device = out_images.device
        self._ensure_device(device)
        self.ocr_model.eval()

        pred_feats = self._extract_backbone_feats(out_images)
        with torch.no_grad():
            gt_feats = self._extract_backbone_feats(gt_images)

        if len(pred_feats) != len(gt_feats):
            raise RuntimeError('Backbone feature count mismatch.')

        layer_weights = self.layer_weights
        if len(layer_weights) < len(pred_feats):
            layer_weights = layer_weights + [1.0] * (len(pred_feats) - len(layer_weights))

        total = out_images.new_tensor(0.0)
        weight_sum = 0.0
        for idx, (pred_feat, gt_feat) in enumerate(zip(pred_feats, gt_feats)):
            weight = float(layer_weights[idx])
            if weight <= 0:
                continue
            total = total + weight * self._reduce_feature_loss(pred_feat, gt_feat, mask=mask)
            weight_sum += weight

        if weight_sum <= 0:
            return out_images.sum() * 0.0
        return total / weight_sum * self.loss_weight


if __name__ == "__main__":
    # 这里添加一些简单的测试代码，验证 OCRCTCLoss 的前向计算是否正常
    import json
    import cv2
    import numpy as np

    img_out = r"E:\workspace\AISR\data_pipeline\data\my_synthetic_dataset_v3\annotation_align_test\aligned\paired\lq\group00_img05.png"

    img_gt = r"E:\workspace\AISR\data_pipeline\data\my_synthetic_dataset_v3\annotation_align_test\aligned\paired\gt\group00_img05.png"

    anno = r"E:\workspace\AISR\data_pipeline\data\my_synthetic_dataset_v3\annotation_align_test\aligned\paired\annotations\group00_img05.json"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(anno, "r", encoding="utf-8") as f:
        data = json.load(f)

    rec_texts = data.get("rec_texts", [])
    rec_boxes = data.get("rec_boxes", [])
    if len(rec_texts) != len(rec_boxes):
        min_len = min(len(rec_texts), len(rec_boxes))
        print(f"Warning: rec_texts({len(rec_texts)}) != rec_boxes({len(rec_boxes)}), use first {min_len}")
        rec_texts = rec_texts[:min_len]
        rec_boxes = rec_boxes[:min_len]

    img = cv2.imread(img_out)
    if img is None:
        raise FileNotFoundError(img_out)

    img_gt = cv2.imread(img_gt)
    if img_gt is None:
        raise FileNotFoundError(img_gt)

    h, w = img.shape[:2]
    target_h = 512
    target_w = 512
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    out_images = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    img_gt_resized = cv2.resize(img_gt, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_gt_rgb = cv2.cvtColor(img_gt_resized, cv2.COLOR_BGR2RGB)
    gt_images = torch.from_numpy(img_gt_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    scale_x = target_w / float(w)
    scale_y = target_h / float(h)

    lines = []
    for box, text in zip(rec_boxes, rec_texts):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        x1 = int(round(x1 * scale_x))
        x2 = int(round(x2 * scale_x))
        y1 = int(round(y1 * scale_y))
        y2 = int(round(y2 * scale_y))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = max(0, min(target_w - 1, x1))
        x2 = max(0, min(target_w - 1, x2))
        y1 = max(0, min(target_h - 1, y1))
        y2 = max(0, min(target_h - 1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append([x1, y1, x2, y1, x2, y2, x1, y2, str(text)])

    gt_lines = [lines]

    # loss_fn = OCRCTCLoss(loss_weight=1.0)
    loss_fn = PPOCRTextPercepLoss(loss_weight=1.0, confidence_threshold=0.0, enable_logging=True)
    out_images = out_images.to(device).requires_grad_(True)
    loss = loss_fn(out_images, gt_images, gt_lines)
    print("OCRCTCLoss:", float(loss))

    loss.backward()
    grad = out_images.grad
    if grad is None:
        print("Grad: None (no backprop)")
    else:
        print(
            "Grad stats:",
            "mean_abs={:.6f}".format(grad.abs().mean().item()),
            "max_abs={:.6f}".format(grad.abs().max().item()),
            "nan={}".format(torch.isnan(grad).any().item()),
            "inf={}".format(torch.isinf(grad).any().item()),
        )
