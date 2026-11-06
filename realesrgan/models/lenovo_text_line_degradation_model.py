"""V4 Y4UV trainer that synthesizes LQ only through Lenovo degradation."""

from __future__ import annotations

from typing import Any

import torch

from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, yuv420_mask_to_y4uv
from realesrgan.lenovo_degradation import LenovoDegradationPipeline

from .basic_text_line_y4uv_training_model_v4 import BasicTextLineY4UVTrainingModelV4


@MODEL_REGISTRY.register()
class LenovoTextLineDegradationModel(BasicTextLineY4UVTrainingModelV4):
    """Train V4 from clean RGB GT without invoking Real-ESRGAN's legacy feed path."""

    def init_training_settings(self):
        super().init_training_settings()
        self._lenovo_seed = int(self.opt.get("manual_seed", 0))
        self._lenovo_calls = 0
        config = self.opt.get("lenovo_degradation")
        if not isinstance(config, dict):
            raise ValueError("LenovoTextLineDegradationModel requires a lenovo_degradation mapping")

        self.lenovo_degradation = LenovoDegradationPipeline(config)

    def _generator(self, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self._lenovo_seed + self._lenovo_calls)
        self._lenovo_calls += 1
        return generator

    @torch.no_grad()
    def feed_data(self, data: dict[str, Any]):
        if "gt_rgb" not in data:
            raise ValueError("Lenovo training requires clean gt_rgb from its clean-GT adapter")
        gt_rgb = data["gt_rgb"].to(self.device).float().clamp(0., 1.)
        if gt_rgb.ndim != 4 or gt_rgb.shape[1] != 3:
            raise ValueError("gt_rgb must be RGB BCHW")
        if self.is_train:
            lq_rgb, metadata = self.lenovo_degradation(gt_rgb, self._generator(gt_rgb.device))
        else:
            lq_rgb = data["lq_rgb"].to(self.device).float()
            metadata = data.get("lenovo_degradation", [])
        self.lq_rgb, self.gt_rgb = lq_rgb, gt_rgb
        self.lq, self.gt = rgb_to_y4uv(lq_rgb), rgb_to_y4uv(gt_rgb)
        valid_rgb = data["valid_mask_rgb"].to(self.device)
        text_rgb = data["text_mask_rgb"].to(self.device)
        self.valid_mask = yuv420_mask_to_y4uv(valid_rgb)
        self.text_mask = yuv420_mask_to_y4uv(text_rgb)
        self.valid_mask_rgb, self.text_mask_rgb = valid_rgb, text_rgb
        self.structure_available = list(data.get("structure_available", []))
        self.segment_metadata = dict(data)
        self.segment_metadata["lenovo_degradation"] = metadata


__all__ = ["LenovoTextLineDegradationModel"]
