"""Native Y4UV V4 training with the standard RGB Real-ESRGAN degradation."""

from __future__ import annotations

import torch

from basicsr.utils import DiffJPEG
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, yuv420_mask_to_y4uv
from realesrgan.models.realesrnet_model import RealESRNetModel
from realesrgan.lenovo_p17_preprocess import normalize_lq_preprocess_config, preprocess_lq_rgb

from .basic_text_line_y4uv_training_model_v4 import BasicTextLineY4UVTrainingModelV4


@MODEL_REGISTRY.register()
class BasicTextLineY4UVV4GTDegradationTrainingModel(BasicTextLineY4UVTrainingModelV4):
    """Apply canonical Real-ESRGAN degradation in RGB, then pack native Y4UV."""

    def init_training_settings(self):
        """@luna-comment
        Input: historical training options and device. Output: initialized losses/JPEG adapter.
        Key Steps: initialize parent state, disable pair queue, retain native color packing.
        Key Params: queue_size=0; finetune_deg=-1. State/Dependencies: parent model, DiffJPEG.
        """
        super().init_training_settings()
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.finetune_deg = -1.0
        self.queue_size = 0
        # RealESRNetModel only converts these legacy formats.  Keeping this
        # value native lets this wrapper pack RGB tensors exactly once below.
        self.color = "y4uv420"

    def _dequeue_and_enqueue(self):
        """@luna-comment
        Input: current model batch. Output: None, with no queue mutation.
        Key Steps: intentionally bypass legacy pair shuffling to retain text metadata.
        Key Params: none. State/Dependencies: model queue is disabled.
        """
        return None

    @torch.no_grad()
    def feed_data(self, data):
        """@luna-comment
        Input: matching RGB BCHW pairs, masks, metadata and optional degradation kernels.
        Output: device RGB/Y4UV model inputs, targets, masks and segment metadata.
        Key Steps: optionally synthesize LQ; apply configured preprocessing; pack once.
        Key Params: kernel1 selects synthesis; lq_preprocess controls legacy adapter.
        State/Dependencies: parent device/state, RealESRNetModel, RGB/Y4UV conversion.
        """
        if self.is_train and "kernel1" in data:
            # BasicTextLineGTDegradeDataset supplies clean RGB as ``gt`` plus
            # the standard two-stage Real-ESRGAN kernel triplet.
            RealESRNetModel.feed_data(self, data)
            lq_rgb, gt_rgb = self.lq, self.gt
        else:
            lq_rgb = data["lq_rgb"].to(self.device)
            gt_rgb = data["gt_rgb"].to(self.device)
        if lq_rgb.ndim != 4 or lq_rgb.shape[1] != 3 or lq_rgb.shape != gt_rgb.shape:
            raise ValueError("V4 GT-deg requires matching RGB BCHW pairs")
        preprocess_config = normalize_lq_preprocess_config(self.opt.get("lq_preprocess"))
        original_lq_y4uv = None
        if preprocess_config["enabled"] and preprocess_config["residual_input_source"] == "original":
            original_lq_y4uv = rgb_to_y4uv(lq_rgb)
        # The final RGB LQ is common to online degradation and validation;
        # apply the opt-in P17-B adapter once before packing Y4UV.
        lq_rgb = preprocess_lq_rgb(lq_rgb, self.opt.get("lq_preprocess"))
        self.lq_rgb, self.gt_rgb = lq_rgb, gt_rgb
        self.lq, self.gt = rgb_to_y4uv(lq_rgb), rgb_to_y4uv(gt_rgb)
        self.original_lq = original_lq_y4uv
        self.valid_mask = yuv420_mask_to_y4uv(data["valid_mask"].to(self.device))
        # @luna-comment Preserve the dataset text membership independently of
        # valid-area coverage. Native reconstruction may assign equal
        # foreground/background weights, but R1 and UV strong supervision use
        # this mask to build their text, edge-ring, and far-background regions.
        self.text_mask = yuv420_mask_to_y4uv(data["text_mask"].to(self.device))
        self.valid_mask_rgb = data["valid_mask_rgb"].to(self.device)
        self.text_mask_rgb = data["text_mask_rgb"].to(self.device)
        self.structure_available = list(data.get("structure_available", []))
        self.segment_metadata = data


__all__ = ["BasicTextLineY4UVV4GTDegradationTrainingModel"]
