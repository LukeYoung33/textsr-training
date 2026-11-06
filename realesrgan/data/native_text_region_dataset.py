"""Native source-coordinate text patches for the isolated V3 comparison."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY
from .basic_text_line_gtdeg_dataset import STANDARD_KERNEL_DEFAULTS, sample_standard_realesrgan_kernels


def native_crop(array, xy, hw, *, mask=False):
    """@luna-comment
    Input: source HWC/HW array, nonnegative even xy, requested canvas hw.
    Output: copied native crop, padded at bottom/right if necessary.
    Key Steps: slice without resizing, zero-pad masks or replicate image edges.
    Key Params: mask excludes artificial support. State/Dependencies: numpy only.
    """
    x, y = xy
    h, w = hw
    crop = array[y:y+h, x:x+w].copy()
    if not crop.size:
        raise ValueError('Crop outside image')
    pads = [(0, h-crop.shape[0]), (0, w-crop.shape[1])] + ([(0, 0)] if crop.ndim == 3 else [])
    return np.pad(crop, pads, mode='constant' if mask else 'edge')


@DATASET_REGISTRY.register()
class NativeTextRegionDataset(Dataset):
    def __init__(self, opt):
        """@luna-comment
        Input: frozen image/patch manifest and historical kernel/augmentation options.
        Output: indexed native-patch dataset with bounded worker-local decode cache.
        Key Steps: load image records, flatten crop references, merge kernel defaults.
        Key Params: canvas_hw=96x480, cache_images bounds RAM; phase gates augmentation.
        State/Dependencies: JSON input, numpy/torch RNG, OpenCV image decoding.
        """
        self.opt = opt
        cv2.setNumThreads(1)
        self.records = [json.loads(x) for x in Path(opt['manifest']).read_text().splitlines() if x.strip()]
        self.items = [(i, j) for i, r in enumerate(self.records) for j in range(len(r['patches']))]
        if not self.items:
            raise ValueError('Empty native patch dataset')
        self.hw = tuple(opt.get('canvas_hw', [96, 480]))
        if self.hw != (96, 480):
            raise ValueError('This frozen comparison uses native 96x480 patches')
        self.training = opt.get('phase', opt.get('split')) == 'train'
        self.cache = OrderedDict()
        self.cache_images = int(opt.get('cache_images', 2))
        self.degradation_opt = dict(STANDARD_KERNEL_DEFAULTS)
        self.degradation_opt.update({k: v for k, v in opt.items() if k in STANDARD_KERNEL_DEFAULTS})

    def __len__(self):
        """@luna-comment Input: dataset. Output: patch count. Key Steps: count frozen index.
        Key Params: none. State/Dependencies: immutable item list.
        """
        return len(self.items)

    def _load(self, image_index):
        """@luna-comment
        Input: source image index. Output: aligned RGB uint arrays and binary masks.
        Key Steps: read unchanged bit depth, validate geometry, retain bounded LRU.
        Key Params: cache_images; all four assets required. State/Dependencies: disk/cache.
        """
        if image_index in self.cache:
            self.cache.move_to_end(image_index)
            return self.cache[image_index]
        record = self.records[image_index]
        arrays = {}
        for key in ('lq', 'gt', 'text_mask', 'valid_mask'):
            arr = cv2.imread(record['files'][key], cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(record['files'][key])
            if key in ('lq', 'gt'):
                if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype not in (np.uint8, np.uint16):
                    raise ValueError('Expected uint8/uint16 three-channel source')
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                if arr.ndim == 3:
                    arr = arr[..., 0]
                arr = (arr > (np.iinfo(arr.dtype).max * .5)).astype(np.uint8)
            arrays[key] = arr
        if len({a.shape[:2] for a in arrays.values()}) != 1:
            raise ValueError('Source pair/mask geometry mismatch')
        self.cache[image_index] = arrays
        while len(self.cache) > self.cache_images:
            self.cache.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        """@luna-comment
        Input: native crop index. Output: aligned RGB/masks, kernels and historical CP metadata.
        Key Steps: crop same source coordinates, normalize, synchronously warp if enabled,
        and generate standard kernel triplet. No height normalization occurs.
        Key Params: historical homography probability/jitter; RGB scale follows bit depth.
        State/Dependencies: bounded source cache and worker numpy/kernel random states.
        """
        image_index, tile_index = self.items[index]
        record = self.records[image_index]
        patch = record['patches'][tile_index]
        xy = patch['xy']
        arrays = self._load(image_index)
        crops = {}
        for key, arr in arrays.items():
            crop = native_crop(arr, xy, self.hw, mask=key.endswith('mask'))
            crops[key] = crop.astype(np.float32) / (np.iinfo(arr.dtype).max if key in ('lq', 'gt') else 1.)
        h, w = self.hw
        prob = float(self.opt.get('train_homography_prob', 0)) if self.training else 0
        if np.random.random() < prob:
            src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            jitter = float(self.opt.get('train_homography_jitter', .015)) * min(h, w)
            dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            for key, arr in crops.items():
                mask = key.endswith('mask')
                crops[key] = cv2.warpPerspective(arr, matrix, (w, h),
                    flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT if mask else cv2.BORDER_REFLECT_101)
        crops['text_mask'] *= crops['valid_mask']
        sample = {}
        for key, arr in crops.items():
            arr = arr.transpose(2, 0, 1) if arr.ndim == 3 else arr[None]
            sample[key + '_rgb'] = torch.from_numpy(np.ascontiguousarray(arr))
        sample['gt'] = sample['gt_rgb']
        sample['valid_mask'] = sample['valid_mask_rgb']
        sample['text_mask'] = sample['text_mask_rgb']
        if self.training:
            for key, kernel in zip(('kernel1', 'kernel2', 'sinc_kernel'), sample_standard_realesrgan_kernels(self.degradation_opt)):
                sample[key] = kernel
        source_h, source_w = arrays['gt'].shape[:2]
        sample.update(structure_available=True, valid_hw=[min(h, source_h-xy[1]), min(w, source_w-xy[0])],
                      orientation='HORIZONTAL', cp_eligible=bool(crops['text_mask'].sum() > 0),
                      line_id=f"{record['sample_id']}:{tile_index}", frame_id=record['sample_id'],
                      lq_path=record['files']['lq'], gt_path=record['files']['gt'],
                      source_crop_xyxy=[xy[0], xy[1], xy[0]+w, xy[1]+h], resize_scale=1.0,
                      tile_index=tile_index, tile_count=len(record['patches']))
        return sample

    def collate_fn(self, batch):
        """@luna-comment Input: homogeneous samples. Output: tensor stacks/metadata lists.
        Key Steps: stack tensor fields only. Key Params: batch schema must match.
        State/Dependencies: torch; preserves per-sample crop coordinates.
        """
        return {key: torch.stack([s[key] for s in batch]) if torch.is_tensor(batch[0][key])
                else [s[key] for s in batch] for key in batch[0]}
