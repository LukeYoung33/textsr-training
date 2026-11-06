"""Training registry wrapper for the isolated V4 Y4UV model and dataset."""

from __future__ import annotations

from basicsr.utils.registry import MODEL_REGISTRY

from .basic_text_line_y4uv_training_model_v3 import BasicTextLineY4UVTrainingModelV3


@MODEL_REGISTRY.register()
class BasicTextLineY4UVTrainingModelV4(BasicTextLineY4UVTrainingModelV3):
    """@shanlan-comment Reuse the dual native/reference loss contract for the new V4 graph only."""


__all__ = ["BasicTextLineY4UVTrainingModelV4"]
