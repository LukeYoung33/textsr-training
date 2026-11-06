"""Read immutable paired native uint8 shards without online image transforms."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class OfflineAugmentedRegionDataset(Dataset):
    def __init__(self, opt):
        """@luna-comment
        Input: offline manifest, optional root override and cache bounds.
        Output: verified compact index and bounded per-worker shard caches.
        Key Steps: require complete bank; check shape and paired-only preprocessing contract.
        Key Params: canvas_hw explicitly selects geometry; default96x480; unique_index audits assets.
        State/Dependencies: immutable JSON/NPY bank, filesystem; no kernels/warps/estimator.
        """
        self.opt = opt
        path = Path(opt['manifest'])
        self.root = Path(opt.get('root', path.parent))
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        complete = json.loads((self.root / 'COMPLETE.json').read_text(encoding='utf-8'))
        if not complete.get('complete') or complete['build_id'] != self.manifest['build_id']:
            raise ValueError('Offline dataset has no matching completion receipt')
        self.hw = tuple(opt.get('canvas_hw', [96, 480]))
        if len(self.hw) != 2 or any(not isinstance(v, int) or v <= 0 or v % 2 for v in self.hw):
            raise ValueError('Expected positive even canvas dimensions')
        if self.manifest['shape_hw'] != list(self.hw) or not self.manifest['paired_only']:
            raise ValueError('Offline bank geometry does not match configured canvas')
        if self.manifest['runtime_preprocessing'] or self.manifest['runtime_geometry']:
            raise ValueError('Offline reader forbids online image transforms')
        key = 'unique_index' if opt.get('unique_index') else 'index'
        self.index = np.load(self.root / self.manifest[key], allow_pickle=False)
        if self.index.dtype != np.uint32 or self.index.ndim != 2 or self.index.shape[1] != 2:
            raise ValueError('Invalid offline sampling index')
        self.shards = self.manifest['shards']
        self.cache_shards = int(opt.get('cache_shards', 8))
        if self.cache_shards < 1:
            raise ValueError('cache_shards must be positive')
        self.cache = OrderedDict()

    def __len__(self):
        """@luna-comment
        Input: dataset. Output: indexed draw count. Key Steps: count compact index rows.
        Key Params: index may repeat baseline assets. State/Dependencies: immutable index.
        """
        return len(self.index)

    def _load(self, shard_index):
        """@luna-comment
        Input: shard ordinal. Output: read-only mmap and per-sample provenance.
        Key Steps: open lazily, validate byte shape and metadata, retain bounded LRU.
        Key Params: explicit canvas and cache_shards bound shapes/handles. State/Dependencies: worker cache.
        """
        if shard_index in self.cache:
            self.cache.move_to_end(shard_index)
            return self.cache[shard_index]
        shard = self.shards[shard_index]
        array = np.load(self.root / shard['array'], mmap_mode='r', allow_pickle=False)
        metadata = json.loads((self.root / shard['metadata']).read_text(encoding='utf-8'))
        if array.dtype != np.uint8 or array.shape != (shard['count'], *self.hw, 8) or len(metadata) != len(array):
            raise ValueError('Shard data/metadata shape mismatch')
        self.cache[shard_index] = (array, metadata)
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return array, metadata

    def __getitem__(self, index):
        """@luna-comment
        Input: compact draw index. Output: historical paired RGB/mask tensors and CP metadata.
        Key Steps: copy one immutable sample, reorder BGR/RGB, normalize, attach eligibility.
        Key Params: configured native canvas; optional block CP lines; no online image transforms.
        State/Dependencies: worker mmap cache,numpy/torch; no tensor aliases mmap storage; GT/mask keys intentionally alias tensors.
        """
        shard_index, sample_index = map(int, self.index[index])
        array, metadata = self._load(shard_index)
        value = array[sample_index]
        meta = metadata[sample_index]
        shard = self.shards[shard_index]
        lq = torch.from_numpy(np.ascontiguousarray(value[..., [2, 1, 0]].transpose(2, 0, 1))).float().div_(255)
        gt = torch.from_numpy(np.ascontiguousarray(value[..., [5, 4, 3]].transpose(2, 0, 1))).float().div_(255)
        text = torch.from_numpy(np.ascontiguousarray(value[..., 6][None])).float().div_(255)
        valid = torch.from_numpy(np.ascontiguousarray(value[..., 7][None])).float().div_(255)
        return dict(lq_rgb=lq, gt_rgb=gt, gt=gt, valid_mask=valid, text_mask=text,
                    valid_mask_rgb=valid, text_mask_rgb=text, structure_available=True,
                    valid_hw=list(self.hw), orientation=meta['orientation'], cp_eligible=meta['cp_eligible'],
                    cp_lines=meta.get('cp_lines', []),
                    line_id=f"{shard['sample_id']}:{sample_index}", frame_id=shard['sample_id'],
                    lq_path=str(self.root / shard['array']), gt_path=str(self.root / shard['array']),
                    source_crop_xyxy=[*meta['xy'], meta['xy'][0] + self.hw[1], meta['xy'][1] + self.hw[0]],
                    resize_scale=1., tile_index=sample_index, tile_count=shard['count'],
                    augmentation_category=meta['category'], source_identity=shard['source_identity'])

    def collate_fn(self, batch):
        """@luna-comment
        Input: homogeneous sample dictionaries. Output: tensor stacks and unchanged metadata lists.
        Key Steps: stack tensor fields only. Key Params: schema matches native paired model.
        State/Dependencies: torch; no image transforms.
        """
        return {key: torch.stack([item[key] for item in batch]) if torch.is_tensor(batch[0][key])
                else [item[key] for item in batch] for key in batch[0]}

    def __getstate__(self):
        """@luna-comment
        Input: dataset about to spawn worker. Output: pickle state without mmap handles.
        Key Steps: shallow-copy state and clear cache. Key Params: none.
        State/Dependencies: original parent cache is unchanged.
        """
        return {**self.__dict__, 'cache': OrderedDict()}
