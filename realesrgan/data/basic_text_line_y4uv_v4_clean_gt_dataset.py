"""Clean-GT Remote58 adapter for V4 Lenovo synthetic degradation training."""

from __future__ import annotations

from basicsr.utils.registry import DATASET_REGISTRY

from .basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset


@DATASET_REGISTRY.register()
class BasicTextLineY4UVV4CleanGTDataset(BasicTextLineRemote58Dataset):
    """Follow the existing ``sources``/``root`` contract without sampling kernels."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # Parent reads paired files for its established geometry contract, but
        # Lenovo synthesis only consumes clean GT after collation.
        sample["lq"] = sample["gt"].clone()
        sample["lq_rgb"] = sample["gt_rgb"].clone()
        sample["clean_gt_adapter"] = True
        return sample


@DATASET_REGISTRY.register()
class LenovoCleanTextLineDataset(BasicTextLineY4UVV4CleanGTDataset):
    """Configuration-facing alias for the Lenovo V4 experiment suite."""


__all__ = ["BasicTextLineY4UVV4CleanGTDataset", "LenovoCleanTextLineDataset"]
