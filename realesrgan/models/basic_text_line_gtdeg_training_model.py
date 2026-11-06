"""BASIC V2 model that invokes the standard RealESRNet GPU degradation path."""

from __future__ import annotations

from typing import Any

import torch

from basicsr.utils import DiffJPEG
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrnet_model import RealESRNetModel
from realesrgan.utils import RGB2YUV

from .basic_text_line_training_model_v3 import BasicTextLineTrainingModelV3


@MODEL_REGISTRY.register()
class BasicTextLineGTDegradationTrainingModel(BasicTextLineTrainingModelV3):
    """Run the canonical RealESRNet two-stage pipeline before BASIC V2 optimization."""

    def init_training_settings(self):
        """@shanlan-comment
        Input: BASIC V2 train options plus standard high-order degradation options.
        Output: initialized BASIC losses/optimizer and non-differentiable standard JPEG module.
        Key Steps: delegate normal BASIC setup, then create the same DiffJPEG helper
        required by RealESRNetModel.feed_data.
        Key Params: degradation options are consumed by the inherited standard feed call.
        State/Dependencies: mutates jpeger, finetune_deg, queue_size, and color members.
        """
        super().init_training_settings()
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.finetune_deg = -1.0
        self.queue_size = 0
        self.color = "nv21"

    def _dequeue_and_enqueue(self) -> None:
        """@shanlan-comment
        Input: current synthetic LQ/GT tensors prepared by the standard feed method.
        Output: no operation; current tensors remain aligned and unchanged.
        Key Steps: intentionally bypass RealESRNet's cross-batch queue for fixed text tiles.
        Key Params: queue_size is fixed at zero for deterministic fixed-canvas behavior.
        State/Dependencies: called by RealESRNetModel.feed_data through this model instance.
        """
        return None

    @torch.no_grad()
    def feed_data(self, data: dict[str, Any]):
        """@shanlan-comment
        Input: GT-only RGB batch with standard kernel tensors and BASIC V2 masks/metadata.
        Output: packed NV21 LQ [B,1,144,320], YUV444 GT [B,3,96,320], and CP metadata.
        Key Steps: invoke RealESRNetModel.feed_data directly for the canonical two-stage
        GPU degradation, retain its RGB copies, convert clean GT to YUV444, and restore
        BASIC V2 masks/metadata.
        Key Params: model scale must be one; standard x4plus degradation knobs remain active.
        State/Dependencies: RealESRNetModel.feed_data, DiffJPEG, RNG state, and data kernels.
        """
        if int(self.opt.get("scale", 1)) != 1:
            raise ValueError("BasicTextLineGTDegradationTrainingModel requires scale=1")
        RealESRNetModel.feed_data(self, data)
        self.gt = RGB2YUV(self.gt_rgb, out_format="YUV")
        self.valid_mask = data["valid_mask"].to(self.device)
        self.text_mask = data["text_mask"].to(self.device)
        self.segment_metadata = data


__all__ = ["BasicTextLineGTDegradationTrainingModel"]
