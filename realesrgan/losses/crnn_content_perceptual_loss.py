"""Text-line CRNN content perceptual loss.

The loss follows the five-tap feature MSE used by TATSR, but obtains the
paired inputs from the same polygon homography instead of resizing a complete
training image.  The recognizer is deliberately kept in this training-only
loss module; it is never part of the exported restoration network.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from realesrgan.losses.DiffCropRecog_loss import prepare_paired_text_crops
from realesrgan.metrics.crnn.crnn import CRNN


_DEFAULT_LAYER_NAMES = ("pooling0", "pooling1", "pooling2", "pooling3", "relu6")
_DEFAULT_LAYER_WEIGHTS = (1.4, 1.4, 1.4, 0.4, 0.4)
# YUV->RGB remains unclamped during CP training. The generator has no output
# activation, so finite RGB overshoot is part of the prediction signal and is
# monitored rather than clipped or rejected.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str) -> Path:
    """Resolve a checkpoint/dictionary path without depending on cwd.

    @shanlan-comment
    Input: absolute or repository-relative filesystem path.
    Output: normalized ``Path``; relative paths are rooted at Real-ESRGAN.
    Key Steps: inspect path form and prepend the repository root when needed.
    Key Params: ``path_value`` identifies a checkpoint or alphabet file.
    State/Dependencies: uses the source file location as the repository root.
    """
    path = Path(path_value)
    return path if path.is_absolute() else _REPO_ROOT / path


def _load_alphabet_size(alphabet_path: str) -> int:
    """Read the single-line CRNN alphabet and return its character count.

    @shanlan-comment
    Input: UTF-8 alphabet path whose first line contains the ordered alphabet.
    Output: positive class count excluding the CTC blank class.
    Key Steps: resolve the path, reject missing/empty files, count first-line characters.
    Key Params: ``alphabet_path`` must match the frozen checkpoint vocabulary.
    State/Dependencies: performs filesystem I/O only.
    """
    path = _resolve_path(alphabet_path)
    if not path.is_file():
        raise FileNotFoundError(f"CRNN alphabet file does not exist: {path}")
    text = path.read_text(encoding="utf-8").splitlines()
    if not text or not text[0]:
        raise ValueError(f"CRNN alphabet file is empty: {path}")
    return len(text[0])


def _load_strict_state_dict(model: nn.Module, checkpoint_path: str) -> None:
    """Load a parameter-only CRNN checkpoint and reject key mismatches.

    @shanlan-comment
    Input: CRNN module and checkpoint path containing its state dictionary.
    Output: model populated in place; missing or unexpected keys raise an error.
    Key Steps: load on CPU, unwrap an optional state_dict container, strip one DataParallel prefix, validate keys.
    Key Params: strict key equality protects the five-tap architecture contract.
    State/Dependencies: filesystem checkpoint I/O; no optimizer state is retained.
    """
    path = _resolve_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CRNN checkpoint does not exist: {path}")
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected a CRNN state dict, got {type(state).__name__}")

    cleaned = {str(key).replace("module.", "", 1): value for key, value in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            "CRNN checkpoint keys do not match the expected architecture: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.load_state_dict(cleaned, strict=True)


class FrozenCRNNFeatureExtractor(nn.Module):
    """Extract frozen CNN taps while preserving gradients to the input image."""

    def __init__(
        self,
        recognizer_path: str,
        alphabet_path: str,
        layer_names: Sequence[str] = _DEFAULT_LAYER_NAMES,
    ) -> None:
        """Construct and freeze the CNN feature extractor.

        @shanlan-comment
        Input: checkpoint/alphabet paths and the required ordered CNN tap names.
        Output: a frozen CRNN submodule whose forward keeps gradients to image inputs.
        Key Steps: build the known H=32 CRNN, strict-load parameters, disable parameter gradients, force eval mode.
        Key Params: five taps are pooling0/1/2/3 and relu6.
        State/Dependencies: registered recognizer parameters move with the parent loss but are never optimized.
        """
        super().__init__()
        self.layer_names = tuple(layer_names)
        if not self.layer_names:
            raise ValueError("At least one CRNN feature tap is required")
        alphabet_size = _load_alphabet_size(alphabet_path)
        recognizer = CRNN(3, 256, alphabet_size + 1, 32)
        _load_strict_state_dict(recognizer, recognizer_path)
        self.recognizer = recognizer
        for parameter in self.recognizer.parameters():
            parameter.requires_grad_(False)
        self.recognizer.eval()

    def train(self, mode: bool = True):
        """Keep the frozen CRNN in eval mode even if its parent loss trains.

        @shanlan-comment
        Input: requested parent-module training flag.
        Output: this extractor, with the recognizer forced to ``training=False``.
        Key Steps: apply normal module state propagation, then restore recognizer eval mode.
        Key Params: ``mode`` is intentionally ignored for the frozen recognizer.
        State/Dependencies: mutates module training flags only.
        """
        super().train(mode)
        self.recognizer.eval()
        return self

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run the CRNN CNN prefix and return the configured feature taps.

        @shanlan-comment
        Input: RGB float BCHW crops with three channels and height 32.
        Output: ordered feature tensors at the five tap locations, retaining input autograd.
        Key Steps: validate shape, explicitly traverse named CNN children, capture requested outputs.
        Key Params: tap names must exist in the known CRNN CNN sequence.
        State/Dependencies: frozen recognizer weights are read; no recognizer parameter gradients are created.
        """
        if images.dim() != 4 or images.shape[1] != 3 or images.shape[2] != 32:
            raise ValueError(
                "CRNN expects RGB BCHW crops with H=32, got "
                f"{tuple(images.shape)}"
            )
        captures: Dict[str, torch.Tensor] = {}
        features = images
        requested = set(self.layer_names)
        for name, module in self.recognizer.cnn.named_children():
            features = module(features)
            if name in requested:
                captures[name] = features
        missing = [name for name in self.layer_names if name not in captures]
        if missing:
            raise KeyError(f"CRNN feature taps were not produced: {missing}")
        return tuple(captures[name] for name in self.layer_names)


def _normalize_annotation_batch(gt_lines, batch_size: int) -> List[List]:
    """Normalize the dataset's B=1 flat annotation form to a batch form.

    @shanlan-comment
    Input: flat or nested polygon-line records and the image batch size.
    Output: ``List[List[line]]`` with one line list per image.
    Key Steps: accept the legacy B=1 form, reject ambiguous B>1 flat forms, validate nesting.
    Key Params: ``batch_size`` determines whether flat compatibility is legal.
    State/Dependencies: no global state; line records remain unchanged.
    """
    if gt_lines is None:
        raise ValueError("CRNN content loss requires gt_lines; missing annotations are not allowed")
    if not isinstance(gt_lines, (list, tuple)):
        raise ValueError(f"gt_lines must be a list/tuple, got {type(gt_lines).__name__}")

    def is_line_record(value) -> bool:
        if isinstance(value, str):
            return True
        if not isinstance(value, (list, tuple)) or len(value) < 9:
            return False
        return all(not isinstance(item, (list, tuple, dict)) for item in value[:8])

    # Existing B=1 collate paths return [line_record, ...].
    if batch_size == 1 and (not gt_lines or is_line_record(gt_lines[0])):
        return [list(gt_lines)]
    if batch_size > 1 and any(is_line_record(item) for item in gt_lines):
        raise ValueError("B>1 gt_lines must be nested per image; flat annotations are ambiguous")
    if len(gt_lines) != batch_size:
        raise ValueError(
            f"Expected one gt_lines list per image ({batch_size}), got {len(gt_lines)}"
        )
    normalized = []
    for image_index, image_lines in enumerate(gt_lines):
        if not isinstance(image_lines, (list, tuple)) or isinstance(image_lines, str):
            raise ValueError(
                f"gt_lines[{image_index}] must be a list/tuple of polygon lines, "
                f"got {type(image_lines).__name__}"
            )
        normalized.append(list(image_lines))
    return normalized


@LOSS_REGISTRY.register()
class CRNNTextLineContentLoss(nn.Module):
    """TATSR-style five-layer CRNN MSE over paired text-line polygon crops."""

    def __init__(
        self,
        recognizer_path: str = "weights/scene_base_CRNN.pth",
        alphabet_path: str = "realesrgan/metrics/benchmark.txt",
        crop_height: int = 32,
        min_width: int = 16,
        max_width: int = 1024,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 32.0,
        layer_names: Sequence[str] = _DEFAULT_LAYER_NAMES,
        layer_weights: Sequence[float] = _DEFAULT_LAYER_WEIGHTS,
        micro_batch_size: int = 16,
        loss_weight: float = 1.0,
    ) -> None:
        """Construct the strict TATSR-style text-line CRNN loss.

        @shanlan-comment
        Input: frozen CRNN assets, crop geometry, fixed tap weights, micro-batch size, and outer loss weight.
        Output: initialized training-only loss module with a frozen feature extractor.
        Key Steps: validate H=32 and exact tap contract, store geometry, construct the extractor and audit state.
        Key Params: crop width [16,1024], aspect ratio [0.5,32], weights [1.4,1.4,1.4,0.4,0.4].
        State/Dependencies: loads the checkpoint once; no inference/export path is modified.
        """
        super().__init__()
        if int(crop_height) != 32:
            raise ValueError(f"CRNN contract requires crop_height=32, got {crop_height}")
        layer_names = tuple(layer_names)
        if tuple(layer_names) != _DEFAULT_LAYER_NAMES:
            raise ValueError(
                "CRNN content loss requires taps "
                f"{list(_DEFAULT_LAYER_NAMES)}, got {list(layer_names)}"
            )
        if len(layer_names) != len(layer_weights):
            raise ValueError("layer_names and layer_weights must have equal lengths")
        if any(float(weight) <= 0 for weight in layer_weights):
            raise ValueError("CRNN layer weights must be positive")
        if int(micro_batch_size) <= 0:
            raise ValueError("micro_batch_size must be positive")
        self.crop_height = 32
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.layer_names = layer_names
        self.layer_weights = tuple(float(weight) for weight in layer_weights)
        self.micro_batch_size = int(micro_batch_size)
        self.loss_weight = float(loss_weight)
        self.extractor = FrozenCRNNFeatureExtractor(
            recognizer_path=recognizer_path,
            alphabet_path=alphabet_path,
            layer_names=layer_names,
        )
        self.extractor.eval()
        self.last_stats = {
            "accepted_lines": 0,
            "skipped_lines": 0,
            "overshoot_ratio": 0.0,
            "clamp_ratio": 0.0,
            "width_p50": 0.0,
            "width_p95": 0.0,
            "width_max": 0.0,
            "layer_losses": {name: 0.0 for name in self.layer_names},
        }

    def train(self, mode: bool = True):
        """Keep the nested CRNN extractor in eval mode during loss training.

        @shanlan-comment
        Input: requested loss training flag.
        Output: loss module with frozen extractor state and unchanged parameter freeze.
        Key Steps: propagate the parent flag, then force the extractor back to eval mode.
        Key Params: ``mode`` affects only ordinary loss-module state.
        State/Dependencies: mutates PyTorch training flags; recognizer weights remain frozen.
        """
        super().train(mode)
        self.extractor.eval()
        return self

    @staticmethod
    def _validate_rgb01(images: torch.Tensor, name: str, allow_mild_overshoot: bool = False):
        """Validate RGB tensors before differentiable cropping.

        @shanlan-comment
        Input: BCHW tensor, field name, and whether small restoration overshoot is allowed.
        Output: no return on valid finite RGB data; raises on shape, non-finite,
            or strict target range errors.
        Key Steps: validate layout and finiteness; enforce [0,1] only for the
            detached target, while generator output remains unconstrained.
        Key Params: strict target range is [0,1].
        State/Dependencies: reads detached extrema and does not alter the input.
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"{name} must be RGB BCHW, got {tuple(images.shape)}")
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains NaN or Inf")
        lower = float(images.detach().amin().item())
        upper = float(images.detach().amax().item())
        if not allow_mild_overshoot and (lower < -1e-4 or upper > 1.0001):
            raise ValueError(f"{name} is outside the RGB [0,1] contract: [{lower}, {upper}]")

    def _feature_loss_for_bucket(
        self,
        output_crops: List[torch.Tensor],
        target_crops: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted five-layer MSE for one exact-width crop bucket.

        @shanlan-comment
        Input: equal-height, equal-width differentiable output/GT crop lists.
        Output: one weighted CP scalar per text line, preserving output-crop gradients.
        Key Steps: split into micro-batches, extract five taps, compute per-line effective-area MSE.
        Key Params: ``micro_batch_size`` limits memory; layer weights follow TATSR.
        State/Dependencies: target extraction is no-grad; extractor parameters stay frozen.
        """
        per_line_losses = []
        layer_sums = None
        for start in range(0, len(output_crops), self.micro_batch_size):
            output_batch = torch.stack(output_crops[start:start + self.micro_batch_size], dim=0)
            target_batch = torch.stack(target_crops[start:start + self.micro_batch_size], dim=0)
            output_features = self.extractor(output_batch)
            with torch.no_grad():
                target_features = self.extractor(target_batch)
            line_loss = output_batch.new_zeros(output_batch.shape[0])
            if layer_sums is None:
                layer_sums = output_batch.new_zeros(len(self.layer_weights))
            for layer_index, (output_feature, target_feature, weight) in enumerate(zip(
                output_features, target_features, self.layer_weights
            )):
                # Exact-width buckets mean there is no padded sequence.  Mean
                # over every feature element is the TATSR effective-area MSE.
                layer_loss = F.mse_loss(
                    output_feature,
                    target_feature.detach(),
                    reduction="none",
                ).flatten(1).mean(dim=1)
                line_loss = line_loss + float(weight) * layer_loss
                layer_sums[layer_index] += layer_loss.detach().sum()
            per_line_losses.append(line_loss)
        return torch.cat(per_line_losses, dim=0), layer_sums

    def forward(self, output_rgb: torch.Tensor, target_rgb: torch.Tensor, gt_lines) -> torch.Tensor:
        """Compute text-line CP loss from paired polygon crops.

        @shanlan-comment
        Input: same-shaped RGB BCHW output/GT tensors and per-image polygon line annotations.
        Output: finite scalar, equal-weighted across accepted lines, connected to output for no-text zero.
        Key Steps: validate finite values, preserve output overshoot, shared-homography bicubic crop, width bucket, weighted MSE.
        Key Params: H=32, width [16,1024], aspect [0.5,32], and five fixed layer weights.
        State/Dependencies: updates ``last_stats``; GT branch is detached and annotations are required.
        """
        if output_rgb.shape != target_rgb.shape:
            raise ValueError(
                f"CRNN content loss requires equal output/target shapes, got "
                f"{tuple(output_rgb.shape)} vs {tuple(target_rgb.shape)}"
            )
        self._validate_rgb01(output_rgb, "output_rgb", allow_mild_overshoot=True)
        self._validate_rgb01(target_rgb, "target_rgb")
        annotations = _normalize_annotation_batch(gt_lines, output_rgb.shape[0])

        overshoot_mask = (output_rgb < 0.0) | (output_rgb > 1.0)
        self.last_stats["overshoot_ratio"] = float(overshoot_mask.float().mean().detach().item())
        audit = {}
        output_crops, target_crops, _, widths = prepare_paired_text_crops(
            output_rgb,
            target_rgb,
            annotations,
            crop_height=self.crop_height,
            device=output_rgb.device,
            min_width=self.min_width,
            max_width=self.max_width,
            interpolation="bicubic",
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            audit=audit,
            require_text=False,
        )
        self.last_stats["accepted_lines"] = len(output_crops)
        total_lines = sum(len(lines) for lines in annotations)
        self.last_stats["skipped_lines"] = total_lines - len(output_crops)
        if not output_crops:
            self.last_stats["width_p50"] = 0.0
            self.last_stats["width_p95"] = 0.0
            self.last_stats["width_max"] = 0.0
            self.last_stats["overshoot_ratio"] = 0.0
            self.last_stats["layer_losses"] = {
                name: 0.0 for name in self.layer_names
            }
            return output_rgb.sum() * 0.0

        sorted_widths = sorted(int(width) for width in widths)
        self.last_stats["width_p50"] = float(sorted_widths[(len(sorted_widths) - 1) // 2])
        p95_index = min(len(sorted_widths) - 1, int(round(0.95 * (len(sorted_widths) - 1))))
        self.last_stats["width_p95"] = float(sorted_widths[p95_index])
        self.last_stats["width_max"] = float(sorted_widths[-1])

        buckets: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        for output_crop, target_crop, width in zip(output_crops, target_crops, widths):
            bucket = buckets.setdefault(int(width), ([], []))
            bucket[0].append(output_crop)
            bucket[1].append(target_crop)
        line_losses = []
        layer_sums = output_rgb.new_zeros(len(self.layer_weights))
        for bucket in buckets.values():
            bucket_losses, bucket_layer_sums = self._feature_loss_for_bucket(*bucket)
            line_losses.append(bucket_losses)
            layer_sums += bucket_layer_sums
        accepted = float(len(output_crops))
        self.last_stats["layer_losses"] = {
            name: float((layer_sums[index] / accepted).detach().item())
            for index, name in enumerate(self.layer_names)
        }
        return torch.cat(line_losses, dim=0).mean() * self.loss_weight


def _segment_metadata_list(
    metadata: Mapping[str, Any], key: str, batch_size: int, default: Any,
) -> List[Any]:
    """Normalize collated fixed-canvas segment metadata to one value per image.

    @shanlan-comment
    Input: BasicTextLineSegmentDatasetV3 collated metadata, field name, batch size,
        and fallback value.
    Output: batch-length Python list without changing the image tensors.
    Key Steps: accept the normal collate form and the convenient B=1 scalar form;
        reject ambiguous B>1 metadata.
    Key Params: ``valid_hw`` is normally a list containing one ``[H, W]`` pair.
    State/Dependencies: pure metadata normalization.
    """
    values = metadata.get(key)
    if values is None:
        return [default for _ in range(batch_size)]
    if batch_size == 1 and key == "valid_hw":
        if isinstance(values, (list, tuple)) and len(values) == 2 and all(
            isinstance(item, (int, float)) for item in values
        ):
            return [values]
    if isinstance(values, (list, tuple)):
        if len(values) != batch_size:
            raise ValueError(
                f"segment_metadata[{key!r}] length {len(values)} != batch size {batch_size}"
            )
        return list(values)
    if batch_size == 1:
        return [values]
    raise ValueError(
        f"segment_metadata[{key!r}] must be batch-length metadata for batch size {batch_size}"
    )


@LOSS_REGISTRY.register()
class CRNNTextLineTileContentLoss(CRNNTextLineContentLoss):
    """Direct CRNN feature loss over already-cropped BASIC V2 text-line tiles.

    Unlike :class:`CRNNTextLineContentLoss`, this loss does not interpret a
    polygon or call a perspective cropper.  BASIC V2 has already selected one
    text-line AABB/tile before the restoration network, so the loss uses the
    valid ``valid_hw`` region directly, rescales only to the frozen CRNN H=32
    contract, and compares paired output/GT features.  Horizontal and oblique
    tiles may be used even when ``tile_count > 1``; the manifest keeps vertical
    samples out of this CRNN path for the current experiment.
    """

    expects_segment_metadata = True

    def _make_direct_pair(
        self,
        output_image: torch.Tensor,
        target_image: torch.Tensor,
        valid_hw: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, int] | None:
        """@luna-comment
        Input: paired BCHW singleton RGB and valid height/width metadata.
        Output: paired CHW CRNN crops plus width, or None for invalid extents.
        Key Steps: clamp extents, crop valid pixels, resize jointly to CRNN height.
        Key Params: crop_height/min_width/max_width. State/Dependencies: torch interpolation.
        """
        try:
            valid_h, valid_w = (int(valid_hw[0]), int(valid_hw[1]))
        except (TypeError, ValueError, IndexError):
            return None
        image_h, image_w = output_image.shape[-2:]
        valid_h = min(valid_h, int(image_h))
        valid_w = min(valid_w, int(image_w))
        if valid_h <= 0 or valid_w <= 0:
            return None
        output_crop = output_image[:, :, :valid_h, :valid_w]
        target_crop = target_image[:, :, :valid_h, :valid_w]
        target_width = int(round(float(valid_w) * self.crop_height / float(valid_h)))
        target_width = max(self.min_width, min(self.max_width, target_width))
        output_crop = F.interpolate(
            output_crop,
            size=(self.crop_height, target_width),
            mode="bicubic",
            align_corners=True,
        )
        target_crop = F.interpolate(
            target_crop,
            size=(self.crop_height, target_width),
            mode="bicubic",
            align_corners=True,
        )
        return output_crop.squeeze(0), target_crop.squeeze(0), target_width

    def forward(
        self,
        output_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        segment_metadata: Mapping[str, Any],
    ) -> torch.Tensor:
        """@luna-comment
        Input: matching RGB batches and per-sample fixed-canvas metadata.
        Output: differentiable paired CRNN feature loss and updated diagnostic stats.
        Key Steps: validate RGB, reject vertical/ineligible tiles, bucket crops, compare features.
        Key Params: orientation/cp_eligible/valid_hw gate valid pairs.
        State/Dependencies: frozen recognizer, inherited feature loss, last_stats.
        """
        if output_rgb.shape != target_rgb.shape:
            raise ValueError(
                f"CRNN tile content loss requires equal output/target shapes, got "
                f"{tuple(output_rgb.shape)} vs {tuple(target_rgb.shape)}"
            )
        if not isinstance(segment_metadata, Mapping):
            raise TypeError(
                "CRNN tile content loss requires segment metadata with valid_hw/orientation"
            )
        self._validate_rgb01(output_rgb, "output_rgb", allow_mild_overshoot=True)
        self._validate_rgb01(target_rgb, "target_rgb")
        batch_size = int(output_rgb.shape[0])
        valid_hws = _segment_metadata_list(
            segment_metadata, "valid_hw", batch_size,
            [int(output_rgb.shape[-2]), int(output_rgb.shape[-1])],
        )
        orientations = _segment_metadata_list(segment_metadata, "orientation", batch_size, "")
        eligible_values = _segment_metadata_list(segment_metadata, "cp_eligible", batch_size, None)

        overshoot_mask = (output_rgb < 0.0) | (output_rgb > 1.0)
        self.last_stats["overshoot_ratio"] = float(overshoot_mask.float().mean().detach().item())
        eligible_count = 0
        vertical_skipped = 0
        output_crops: List[torch.Tensor] = []
        target_crops: List[torch.Tensor] = []
        widths: List[int] = []
        for index in range(batch_size):
            orientation = str(orientations[index]).upper()
            if orientation == "VERTICAL":
                vertical_skipped += 1
                continue
            if eligible_values[index] is not None and not bool(eligible_values[index]):
                continue
            if orientation not in ("HORIZONTAL", "OBLIQUE"):
                continue
            eligible_count += 1
            pair = self._make_direct_pair(
                output_rgb[index:index + 1],
                target_rgb[index:index + 1],
                valid_hws[index],
            )
            if pair is None:
                continue
            output_crop, target_crop, width = pair
            output_crops.append(output_crop)
            target_crops.append(target_crop)
            widths.append(width)

        self.last_stats["eligible_lines"] = eligible_count
        self.last_stats["vertical_skipped"] = vertical_skipped
        self.last_stats["accepted_lines"] = len(output_crops)
        self.last_stats["skipped_lines"] = batch_size - len(output_crops)
        if not output_crops:
            self.last_stats["width_p50"] = 0.0
            self.last_stats["width_p95"] = 0.0
            self.last_stats["width_max"] = 0.0
            self.last_stats["overshoot_ratio"] = 0.0
            self.last_stats["layer_losses"] = {
                name: 0.0 for name in self.layer_names
            }
            return output_rgb.sum() * 0.0

        sorted_widths = sorted(int(width) for width in widths)
        self.last_stats["width_p50"] = float(sorted_widths[(len(sorted_widths) - 1) // 2])
        p95_index = min(len(sorted_widths) - 1, int(round(0.95 * (len(sorted_widths) - 1))))
        self.last_stats["width_p95"] = float(sorted_widths[p95_index])
        self.last_stats["width_max"] = float(sorted_widths[-1])

        buckets: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        for output_crop, target_crop, width in zip(output_crops, target_crops, widths):
            bucket = buckets.setdefault(int(width), ([], []))
            bucket[0].append(output_crop)
            bucket[1].append(target_crop)
        line_losses = []
        layer_sums = output_rgb.new_zeros(len(self.layer_weights))
        for bucket in buckets.values():
            bucket_losses, bucket_layer_sums = self._feature_loss_for_bucket(*bucket)
            line_losses.append(bucket_losses)
            layer_sums += bucket_layer_sums
        accepted = float(len(output_crops))
        self.last_stats["layer_losses"] = {
            name: float((layer_sums[index] / accepted).detach().item())
            for index, name in enumerate(self.layer_names)
        }
        return torch.cat(line_losses, dim=0).mean() * self.loss_weight
