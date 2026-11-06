"""Reuse immutable HR blocks while reducing only real LQ to half resolution."""
from torch.nn import functional as F
from basicsr.utils.registry import DATASET_REGISTRY
from .offline_augmented_region_dataset import OfflineAugmentedRegionDataset


@DATASET_REGISTRY.register()
class OfflineSR2uDataset(OfflineAugmentedRegionDataset):
    def __getitem__(self, index):
        """@luna-comment
        Input: unchanged balanced index into native256 paired bank. Output: RGB128 feature LQ, original HR LQ/GT.
        Key Steps: retain original LQ, exact2x area-average feature input before independent Y4UV packing.
        Key Params: fixed2x; no GT-dependent operation, preprocessing, random degradation or extra crop.
        State/Dependencies: parent immutable bank/cache, CPU torch pooling; HR masks/CP coordinates unchanged.
        """
        sample = super().__getitem__(index)
        if sample['lq_rgb'].shape != (3, 256, 256) or sample['gt_rgb'].shape != (3, 256, 256):
            raise ValueError('SR2u requires original RGB256 blocks')
        sample['original_lq_rgb'] = sample['lq_rgb']
        sample['lq_rgb'] = F.avg_pool2d(sample['lq_rgb'].unsqueeze(0), 2, 2).squeeze(0)
        sample['sr_scale'] = 2
        return sample
