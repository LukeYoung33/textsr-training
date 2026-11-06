"""Opt-in sidecar dataset and area-adaptive base loss; historical adapters unchanged."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from basicsr.utils.registry import DATASET_REGISTRY, LOSS_REGISTRY, MODEL_REGISTRY
from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
from realesrgan.data.basic_text_line_y4uv_v3 import yuv420_mask_to_y4uv
from realesrgan.models.block256_region_model import Block256RegionModel


@DATASET_REGISTRY.register()
class V3PreAreaDataset(OfflineAugmentedRegionDataset):
    def __getitem__(self, index):
        """@luna-comment Input:bank draw index. Output:unchanged sample plus line-region mask.
        Key Steps:bounded read-only bitpacked sidecar LRU; unpack aligned source sample.
        Key Params:256native; no online geometry. State/Dependencies:verified sidecars.
        """
        sample = super().__getitem__(index)
        si, ri = map(int, self.index[index])
        if not hasattr(self, 'region_cache'):
            self.region_cache = OrderedDict()
        if si not in self.region_cache:
            path = self.root / 'line_regions' / (self.shards[si]['array'] + '.bits.npy')
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            assert values.shape == (self.shards[si]['count'], 8192) and values.dtype == np.uint8
            self.region_cache[si] = values
        self.region_cache.move_to_end(si)
        values = self.region_cache[si]
        while len(self.region_cache) > self.cache_shards:
            self.region_cache.popitem(last=False)
        mask = np.unpackbits(values[ri]).reshape(1, 256, 256).copy()
        sample['line_region_mask'] = torch.from_numpy(mask).float()
        return sample

    def __getstate__(self):
        """@luna-comment Input:dataset spawn. Output:state without either mmap cache.
        Key Steps:delegate existing safe state then clear region LRU. Key Params:none.
        State/Dependencies:worker handles remain process-local.
        """
        return {**super().__getstate__(), 'region_cache': OrderedDict()}


@LOSS_REGISTRY.register()
class AreaAdaptiveNativeL1(torch.nn.Module):
    def __init__(self, cap=4., loss_weight=1., eps=1e-6, **options):
        """@luna-comment Input:cap/coefficient/epsilon. Output:opt-in base L1 module.
        Key Steps:validate cap and store constants. Key Params:1control/4selected.
        State/Dependencies:per-batch region/valid tensors supplied by opt-in model.
        """
        super().__init__()
        assert cap >= 1 and loss_weight == 1.
        self.cap, self.eps = float(cap), float(eps)

    def forward(self, pred, target, **unused):
        """@luna-comment Input:native six-channel prediction/target and bound masks.
        Output:mean of nonempty per-sample normalized adaptive L1. Key Steps:compute
        full-resolution r/a, pack W, apply c=(1,1,1,1,.25,.25); keep auxiliary masks.
        Key Params:cap4/eps1e-6; no mask gradients. State/Dependencies:latest feed masks.
        """
        valid = self.raw_valid.to(pred)
        region = self.line_region.to(pred) * valid
        n = valid.sum((1, 2, 3))
        r = region.sum((1, 2, 3)) / n.clamp_min(self.eps)
        a = torch.where(r > 0, ((1-r)/(r+self.eps)).clamp(1, self.cap), torch.ones_like(r)).detach()
        weight = valid * (1 + (a[:, None, None, None]-1) * region)
        channels = pred.new_tensor([1, 1, 1, 1, .25, .25])[None, :, None, None]
        # The binary-mask helper clamps values to1; adaptive weights must retain
        # their1..cap magnitudes while using exactly the same spatial packing.
        y4 = torch.nn.functional.pixel_unshuffle(weight, 2)
        uv = torch.nn.functional.avg_pool2d(weight, 2)
        packed = torch.cat((y4, uv, uv), dim=1) * channels
        error = (pred-target).abs()
        loss = (error*packed).sum((1, 2, 3)) / packed.sum((1, 2, 3)).clamp_min(self.eps)
        base = yuv420_mask_to_y4uv(valid) * channels
        unweighted = (error*base).sum((1, 2, 3)) / base.sum((1, 2, 3)).clamp_min(self.eps)
        self.stats = dict(area_r=r.detach().mean(), area_a=a.detach().mean(),
                          area_unweighted_l1=unweighted.detach().mean(), area_weighted_l1=loss.detach().mean())
        keep = (n > 0).to(pred)
        return (loss*keep).sum() / keep.sum().clamp_min(1)


@MODEL_REGISTRY.register()
class V3PreAreaModel(Block256RegionModel):
    def feed_data(self, data):
        """@luna-comment Input:ordinary paired batch plus line mask. Output:bound base-loss masks.
        Key Steps:unchanged parent feed, then bind extra line and valid support only
        to adaptive native criterion. Key Params:no auxiliary input edits.
        State/Dependencies:latest feed overwrites masks, including validation.
        """
        super().feed_data(data)
        self.cri_y4uv_native.line_region = data['line_region_mask'].to(self.device)
        self.cri_y4uv_native.raw_valid = data['valid_mask_rgb'].to(self.device)

    def optimize_parameters(self, current_iter):
        """@luna-comment Input:iteration. Output:ordinary full-loss update plus area diagnostics.
        Key Steps:delegate all auxiliary/schedule logic; append actual adaptive stats.
        Key Params:no extra optimizer operations. State/Dependencies:existing parent.
        """
        super().optimize_parameters(current_iter)
        self.log_dict.update({key: float(value) for key, value in self.cri_y4uv_native.stats.items()})
