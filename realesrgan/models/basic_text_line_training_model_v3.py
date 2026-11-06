"""Training wrapper for fixed-canvas BASIC V2 text-line segment batches."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

from realesrgan.data.text_line_crop_v3 import polygon_to_quad
from realesrgan.utils import YUV2RGB


@MODEL_REGISTRY.register()
class BasicTextLineTrainingModelV3(SRModel):
    """Train the R4 Student with configurable baseline, CP, and Y-gradient losses."""

    def init_training_settings(self):
        """@shanlan-comment
        Input: train options containing pixel/CP/gradient loss and optimizer settings.
        Output: initialized train-mode generator, losses, optimizer, and schedulers.
        Key Steps: require pixel loss, build optional auxiliaries, then set optimizers/schedulers.
        Key Params: cp_opt and grad_opt selectively enable auxiliary objectives.
        State/Dependencies: mutates net_g, cri_* members, and SRModel optimizer state.
        """
        self.net_g.train()
        train_opt = self.opt["train"]
        pixel_opt = train_opt.get("pixel_opt")
        if not pixel_opt:
            raise ValueError("train.pixel_opt is required for BASIC text-line training")
        self.cri_pix = build_loss(pixel_opt).to(self.device)
        self.cri_cp = build_loss(train_opt["cp_opt"]).to(self.device) if train_opt.get("cp_opt") else None
        self.cri_grad = build_loss(train_opt["grad_opt"]).to(self.device) if train_opt.get("grad_opt") else None
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        """@shanlan-comment
        Input: fixed-canvas packed LQ, YUV444 GT, masks, RGB copies, and tile metadata.
        Output: device-resident training tensors plus retained CPU-side segment metadata.
        Key Steps: transfer generator/loss tensors while leaving metadata accessible for CP.
        Key Params: lq/gt/mask keys must match the BASIC V2 Dataset contract.
        State/Dependencies: self.device and self.segment_metadata are updated.
        """
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)
        self.valid_mask = data["valid_mask"].to(self.device)
        self.text_mask = data["text_mask"].to(self.device)
        self.gt_rgb = data.get("gt_rgb")
        if self.gt_rgb is not None:
            self.gt_rgb = self.gt_rgb.to(self.device)
        self.lq_rgb = data.get("lq_rgb")
        if self.lq_rgb is not None:
            self.lq_rgb = self.lq_rgb.to(self.device)
        self.segment_metadata = data

    def _schedule(self, target: float, current_iter: int) -> float:
        """@shanlan-comment
        Input: full auxiliary-loss weight and zero-based training iteration.
        Output: scheduled scalar weight after an optional delayed linear ramp.
        Key Steps: apply start gate, zero-ramp shortcut, or clipped linear interpolation.
        Key Params: aux_start_iter and aux_ramp_iters define the schedule.
        State/Dependencies: reads immutable train options only.
        """
        train_opt = self.opt["train"]
        start = int(train_opt.get("aux_start_iter", 0))
        ramp = int(train_opt.get("aux_ramp_iters", 0))
        if current_iter < start:
            return 0.0
        if ramp <= 0:
            return float(target)
        return float(target) * min(1.0, (current_iter - start + 1) / float(ramp))

    @staticmethod
    def _metadata_list(metadata: Mapping[str, Any], key: str, batch_size: int, default: Any) -> list[Any]:
        """@shanlan-comment
        Input: collated metadata mapping, key, batch size, and absent-key default.
        Output: exact batch-length Python metadata list or a contract error.
        Key Steps: choose default, validate list cardinality, and normalize singleton batches.
        Key Params: batch_size distinguishes scalar singleton metadata from invalid batches.
        State/Dependencies: no shared state; used by CP annotation construction.
        """
        values = metadata.get(key)
        if values is None:
            return [default for _ in range(batch_size)]
        if isinstance(values, (list, tuple)):
            if len(values) != batch_size:
                raise ValueError(f"metadata[{key!r}] length {len(values)} != batch size {batch_size}")
            return list(values)
        if batch_size == 1:
            return [values]
        raise ValueError(f"metadata[{key!r}] must be a batch-length list for batch size {batch_size}")

    @classmethod
    def _build_cp_annotations(cls, metadata: Mapping[str, Any], batch_size: int) -> list[list[list[Any]]]:
        """@shanlan-comment
        Input: collated BASIC V2 metadata and batch size.
        Output: one CRNN quadrilateral annotation list per batch item.
        Key Steps: normalize eligibility/polygon fields, reject invalid or truncated tiles,
        convert valid polygons to quads, and attach optional transcript placeholders.
        Key Params: cp_eligible and polygon_valid gate CP contribution.
        State/Dependencies: polygon_to_quad and collated Dataset metadata.
        """
        eligible = cls._metadata_list(metadata, "cp_eligible", batch_size, False)
        polygons = cls._metadata_list(metadata, "polygon_local", batch_size, [])
        polygon_valid = cls._metadata_list(metadata, "polygon_valid", batch_size, True)
        truncated = cls._metadata_list(metadata, "is_crop_truncated", batch_size, False)
        transcriptions = cls._metadata_list(metadata, "transcription", batch_size, "")
        annotations: list[list[list[Any]]] = []
        for is_eligible, polygon, is_valid, is_truncated, transcription in zip(
            eligible, polygons, polygon_valid, truncated, transcriptions
        ):
            if not bool(is_eligible) or not bool(is_valid) or bool(is_truncated):
                annotations.append([])
                continue
            try:
                quad = polygon_to_quad(polygon)
            except (TypeError, ValueError):
                annotations.append([])
                continue
            flat = [float(value) for value in quad.reshape(-1).tolist()]
            # The CP cropper uses the ninth field only for legacy transcript
            # bookkeeping. CRNN feature loss sets require_text=False.
            annotations.append([flat + [str(transcription)]])
        return annotations

    @staticmethod
    def _append_cp_stats(logs: OrderedDict, reference: torch.Tensor, stats: Mapping[str, Any]) -> None:
        """@shanlan-comment
        Input: mutable training log, reference tensor/device, and CP statistics mapping.
        Output: tensor-backed CP acceptance, width, and per-layer log entries.
        Key Steps: copy recognized scalar diagnostics and flatten feature-layer losses.
        Key Params: recognized names define the stable logging schema.
        State/Dependencies: mutates logs and uses reference dtype/device.
        """
        for name in ("eligible_lines", "accepted_lines", "skipped_lines", "vertical_skipped", "overshoot_ratio", "clamp_ratio", "width_p50", "width_p95", "width_max"):
            if name in stats:
                logs[f"cp_{name}"] = reference.new_tensor(float(stats[name]))
        for name, value in stats.get("layer_losses", {}).items():
            logs[f"cp_layer_{name}"] = reference.new_tensor(float(value))

    def optimize_parameters(self, current_iter):
        """@shanlan-comment
        Input: current iteration after feed_data populated BASIC V2 tensors/metadata.
        Output: one optimizer update and scalar reconstruction/CP/gradient log entries.
        Key Steps: forward generator, compute scheduled configured losses, backpropagate,
        step optimizer, and append CP diagnostics.
        Key Params: cp_weight/grad_weight are ramped by _schedule at current_iter.
        State/Dependencies: mutates output, optimizer, and log_dict using cri_* losses.
        """
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq)
        total = self.output.sum() * 0.0
        logs = OrderedDict()

        base = self.cri_pix(
            self.output,
            self.gt,
            valid_mask=self.valid_mask,
            text_mask=self.text_mask,
        )
        total = total + base
        logs["l_base"] = base.detach()

        train_opt = self.opt["train"]
        cp_weight = self._schedule(float(train_opt.get("cp_weight", 0.0)), current_iter)
        grad_weight = self._schedule(float(train_opt.get("grad_weight", 0.0)), current_iter)
        logs["cp_scale"] = self.output.new_tensor(cp_weight)
        logs["grad_scale"] = self.output.new_tensor(grad_weight)

        if self.cri_cp is not None and cp_weight > 0.0:
            if self.gt_rgb is None:
                raise RuntimeError("CP loss is enabled but the segment batch has no gt_rgb target")
            # BASIC output uses normalized YUV: U/V are encoded with a 0.5
            # neutral offset.  Decode that contract explicitly before the
            # CRNN loss; the conversion no longer infers the range from data.
            cp_output = YUV2RGB(self.output, format="RGB", is_train=True, uv_normalized=True)
            if bool(getattr(self.cri_cp, "expects_segment_metadata", False)):
                # BASIC V2 already provides one text-line tile per model input.
                # The direct tile CP path consumes valid_hw/orientation metadata
                # and does not re-crop the tile with a polygon homography.
                cp_raw = self.cri_cp(cp_output, self.gt_rgb, self.segment_metadata)
            else:
                gt_lines = self._build_cp_annotations(self.segment_metadata, self.output.shape[0])
                cp_raw = self.cri_cp(cp_output, self.gt_rgb, gt_lines)
            total = total + cp_weight * cp_raw
            logs["l_cp_raw"] = cp_raw.detach()
            logs["l_cp"] = (cp_weight * cp_raw).detach()
            self._append_cp_stats(logs, self.output, getattr(self.cri_cp, "last_stats", {}))
        elif self.cri_cp is not None:
            logs["l_cp_raw"] = self.output.sum() * 0.0
            logs["l_cp"] = self.output.sum() * 0.0

        if self.cri_grad is not None and grad_weight > 0.0:
            raw_grad = self.cri_grad(
                self.output,
                self.gt,
                valid_mask=self.valid_mask,
                text_mask=self.text_mask,
            )
            total = total + grad_weight * raw_grad
            logs["l_grad_raw"] = raw_grad.detach()
            logs["l_grad"] = (grad_weight * raw_grad).detach()
        elif self.cri_grad is not None:
            logs["l_grad_raw"] = self.output.sum() * 0.0
            logs["l_grad"] = self.output.sum() * 0.0

        total.backward()
        self.optimizer_g.step()
        logs["l_total"] = total.detach()
        self.log_dict = self.reduce_loss_dict(logs)
