"""Native full-frame paired tiles; annotations are supervision only."""
import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.utils.registry import DATASET_REGISTRY


def _read(path, rgb=False):
    """@luna-comment
    Input: Local image path; rgb selects three-channel [0,1] data.
    Output: HWC RGB or HW float array; missing files raise.
    Key Steps: Decode unchanged, validate depth, convert BGR and normalize.
    Key Params: rgb distinguishes images from annotation masks.
    State/Dependencies: OpenCV; no shared mutable state.
    """
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if rgb:
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f'Expected three-channel RGB image: {path}')
        value = value[..., ::-1]
    elif value.ndim != 2:
        raise ValueError(f'Expected single-channel mask: {path}')
    maximum = np.iinfo(value.dtype).max if np.issubdtype(value.dtype, np.integer) else 1.0
    value = value.astype(np.float32) / maximum
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError(f'Invalid image range: {path}')
    return value


def _extract(array, x, y, width, height, image=False):
    """@luna-comment
    Input: HW/HWC array and integer native crop geometry, possibly outside image.
    Output: Exact requested canvas; image edge replication or mask zero padding.
    Key Steps: Compute clipped overlap and pad deterministically.
    Key Params: image selects replication; masks never supervise external padding.
    State/Dependencies: NumPy only; input is not mutated.
    """
    h, w = array.shape[:2]
    xs = np.clip(np.arange(x, x + width), 0, w - 1)
    ys = np.clip(np.arange(y, y + height), 0, h - 1)
    result = array[ys[:, None], xs[None, :]].copy()
    if not image:
        valid = ((np.arange(y, y + height) >= 0) & (np.arange(y, y + height) < h))[:, None]
        valid = valid & ((np.arange(x, x + width) >= 0) & (np.arange(x, x + width) < w))[None, :]
        result *= valid[..., None] if result.ndim == 3 else valid
    return result


@DATASET_REGISTRY.register()
class FullFrameTextPriorityDataset(Dataset):
    """Separate registration leaves all legacy dataset defaults unchanged."""

    def __init__(self, opt):
        """@luna-comment
        Input: BasicSR options with root, manifest, canvas_hw, halo and split.
        Output: Indexed native tiles and an epoch-reconstructible sampling plan.
        Key Steps: Validate options, read pairs/labels, allocate unique ownership, enumerate grid.
        Key Params: Sampling ratios choose text/hard/other; source_balance equalizes sources.
        State/Dependencies: Reads local manifest/images; retains metadata and a bounded
            worker-local decoded-record cache; set_epoch refreshes the plan.
        """
        self.opt = opt
        self.root = Path(opt.get('root', '.'))
        raw_canvas = opt.get('canvas_hw', (256, 256))
        if len(raw_canvas) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in raw_canvas):
            raise ValueError('canvas_hw dimensions must be integers')
        if not isinstance(opt.get('halo', 32), (int, np.integer)) or isinstance(opt.get('halo', 32), bool):
            raise ValueError('halo must be an integer')
        self.canvas = tuple(int(v) for v in opt.get('canvas_hw', (256, 256)))
        self.halo = int(opt.get('halo', 32))
        self.roi_context = int(opt.get('roi_context', 8))
        if len(self.canvas) != 2 or any(v <= 0 or v % 8 for v in self.canvas):
            raise ValueError('canvas_hw must contain positive multiples of 8')
        if self.halo < 0 or self.halo % 8 or min(self.canvas) <= 2 * self.halo:
            raise ValueError('halo must be nonnegative, phase aligned, and smaller than half canvas')
        if self.roi_context < 0 or opt.get('scale', 1) != 1:
            raise ValueError('roi_context must be nonnegative and scale must be 1')
        for key in ('augmentations', 'augmentation'):
            if any(bool(v) for v in opt.get(key, {}).values()):
                raise ValueError('Full-frame augmentation is not implemented; disable all augmentation options')
        for key in ('use_hflip', 'use_rot', 'perspective_prob', 'resize', 'gt_size'):
            if opt.get(key):
                raise ValueError(f'Unsupported native-tile option: {key}')
        if opt.get('persistent_workers', False):
            raise ValueError('persistent_workers must be false for refreshed epoch plans')
        for key in ('train_homography_prob', 'ocr_line_crop', 'text_line_crop', 'height_crop_pad',
                    'height_crop', 'target_height', 'max_upscale', 'min_height', 'max_height'):
            if opt.get(key):
                raise ValueError(f'Unsupported legacy text-line option: {key}')
        if opt.get('resize_mode', 'native') != 'native':
            raise ValueError('Only resize_mode=native is supported')
        manifest = Path(opt['manifest'])
        if not manifest.is_absolute():
            manifest = self.root / manifest
        text = manifest.read_text(encoding='utf-8-sig')
        rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(v) for v in text.splitlines() if v.strip()]
        if not isinstance(rows, list):
            raise ValueError('Manifest must be a JSON list or JSONL')
        self.records, self.tiles = [], []
        self.tile_instances = []
        self.cache = OrderedDict()
        self.cache_records = int(opt.get('cache_records', 2))
        if self.cache_records < 0:
            raise ValueError('cache_records must be nonnegative')
        split = opt.get('split')
        for row in rows:
            if split and row.get('split', split) != split:
                continue
            record = self._load_record(row)
            idx = len(self.records)
            self.records.append(dict(row=row, shape=record['gt'].shape))
            h, w = record['gt'].shape[:2]
            sh, sw = (v - 2 * self.halo for v in self.canvas)
            for cy in range(0, h, sh):
                for cx in range(0, w, sw):
                    visible = (record['text'][cy:min(cy + sh, h), cx:min(cx + sw, w)] *
                               record['valid'][cy:min(cy + sh, h), cx:min(cx + sw, w)])
                    stratum = row.get('stratum', 'text' if np.any(visible) else 'other')
                    if stratum not in ('text', 'hard', 'other'):
                        raise ValueError(f'Unknown stratum: {stratum}')
                    if stratum in ('text', 'hard') and not np.any(visible):
                        stratum = 'other'
                    self.tiles.append((idx, cx - self.halo, cy - self.halo, stratum))
                    ids = np.unique(record['ownership'][cy:min(cy + sh, h), cx:min(cx + sw, w)] *
                                    (record['valid'][cy:min(cy + sh, h), cx:min(cx + sw, w)] > 0))
                    self.tile_instances.append([int(i) for i in ids if i])
        if not self.tiles:
            raise ValueError('No paired tiles selected by manifest/split')
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """@luna-comment
        Input: Nonnegative integer sampler epoch, including resumed epoch.
        Output: Refreshes fixed-length sampling plan and scheduled exposure counters.
        Key Steps: Seed epoch plan deterministically, recount tile and instance appearances.
        Key Params: seed+epoch refreshes sampled subsets without changing epoch length.
        State/Dependencies: Dataset-local plan; call before creating nonpersistent workers.
        """
        if int(epoch) != epoch or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer')
        self.epoch = int(epoch)
        self.plan = self._sampling_plan(self.opt.get('sampling', {}))
        self.realized_strata = {name: sum(self.tiles[i][3] == name for i in self.plan)
                               for name in ('text', 'hard', 'other')}
        self.tile_exposures = {}
        for tile_idx in self.plan:
            self.tile_exposures[tile_idx] = self.tile_exposures.get(tile_idx, 0) + 1
        self.exposures = {}
        for tile_idx in self.plan:
            record_idx = self.tiles[tile_idx][0]
            for instance_id in self.tile_instances[tile_idx]:
                if instance_id:
                    key = (record_idx, int(instance_id))
                    self.exposures[key] = self.exposures.get(key, 0) + 1

    def _record(self, index):
        """@luna-comment
        Input: Manifest record index.
        Output: Decoded arrays held by at most cache_records resident records.
        Key Steps: Reuse LRU hit or lazily decode, evict oldest cached records.
        Key Params: cache_records=0 disables retained image caching.
        State/Dependencies: Worker-local bounded LRU; metadata remains immutable.
        """
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self._load_record(self.records[index]['row'])
        if self.cache_records:
            self.cache[index] = record
            while len(self.cache) > self.cache_records:
                self.cache.popitem(last=False)
        return record

    def _load_record(self, row):
        """@luna-comment
        Input: Manifest row with aligned image paths and optional stroke/instance annotations.
        Output: Validated decoded record with positive integer ownership map.
        Key Steps: Check pair shapes, intersect geometry with stroke mask, deduplicate IDs/geometries.
        Key Params: annotation_status preserves unknown versus known no-text semantics.
        State/Dependencies: OpenCV geometry filling; annotations never become generator input.
        """
        def resolve(name):
            """@luna-comment
            Input: Manifest path field name.
            Output: Absolute or root-relative local path.
            Key Steps: Resolve declared field relative to dataset root.
            Key Params: name selects one manifest asset.
            State/Dependencies: Enclosing immutable row and root.
            """
            path = Path(row[name])
            return path if path.is_absolute() else self.root / path
        lq, gt = _read(resolve('lq'), True), _read(resolve('gt'), True)
        if lq.shape != gt.shape:
            raise ValueError('LQ and GT must have identical native geometry')
        shape = gt.shape[:2]
        valid = _read(resolve('valid_mask')) if row.get('valid_mask') else np.ones(shape, np.float32)
        stroke = _read(resolve('text_mask')) if row.get('text_mask') else np.zeros(shape, np.float32)
        if valid.shape != shape or stroke.shape != shape:
            raise ValueError('Annotation mask shape differs from aligned image')
        status = row.get('annotation_status', 'annotated' if row.get('text_mask') else 'unknown')
        if status not in ('unknown', 'no_text', 'annotated'):
            raise ValueError('Invalid annotation_status')
        if status == 'no_text' and (np.any(stroke) or row.get('instances')):
            raise ValueError('no_text contradicts supplied text annotations')
        if status == 'unknown':
            stroke = np.zeros(shape, np.float32)
        ownership = np.zeros(shape, np.int32)
        instances, seen_ids, seen_geometry = [], set(), set()
        for item in row.get('instances', []):
            original_id = str(item['id'])
            region = np.zeros(shape, np.uint8)
            if 'polygon' in item:
                points = np.asarray(item['polygon'], np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                    raise ValueError('Invalid instance polygon')
                cv2.fillPoly(region, [np.round(points).astype(np.int32)], 1)
                bbox = [int(math.floor(points[:, 0].min())), int(math.floor(points[:, 1].min())),
                        int(math.ceil(points[:, 0].max())) + 1, int(math.ceil(points[:, 1].max())) + 1]
            else:
                bbox = list(item['bbox'])
                if len(bbox) != 4 or not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError('Invalid half-open instance bbox')
                bbox = [int(math.floor(bbox[0])), int(math.floor(bbox[1])), int(math.ceil(bbox[2])), int(math.ceil(bbox[3]))]
                x0, y0, x1, y1 = bbox
                if min(y1, shape[0]) > max(y0, 0) and min(x1, shape[1]) > max(x0, 0):
                    region[max(y0, 0):min(y1, shape[0]), max(x0, 0):min(x1, shape[1])] = 1
            confidence = float(item.get('confidence', 1.0))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Instance confidence must be in [0,1]')
            geometry = region.tobytes()
            if original_id in seen_ids or geometry in seen_geometry:
                continue
            seen_ids.add(original_id)
            seen_geometry.add(geometry)
            instance_id = len(instances) + 1
            ownership[(region > 0) & (stroke > 0) & (ownership == 0)] = instance_id
            instances.append(dict(item, bbox=bbox, instance_id=instance_id, original_id=original_id,
                                  confidence=confidence, orientation=item.get('orientation', 'horizontal')))
        return dict(lq=lq, gt=gt, valid=valid, text=stroke, ownership=ownership,
                    instances=instances, row=row, annotation_status=status)

    def _sampling_plan(self, sampling):
        """@luna-comment
        Input: Optional ratios, source_balance and num_samples.
        Output: Deterministic epoch tile-index list with explicit stratum quotas.
        Key Steps: Allocate largest-remainder quotas; cycle sources and shuffled source tiles.
        Key Params: seed controls reproducibility; unavailable requested strata raise.
        State/Dependencies: Local NumPy generator; plan is compatible with BasicSR samplers.
        """
        if not sampling or self.opt.get('phase', 'train') != 'train':
            return list(range(len(self.tiles)))
        ratios = sampling.get('ratios', {})
        if not ratios:
            ratios = {name: sum(t[3] == name for t in self.tiles) for name in ('text', 'hard', 'other')}
        if any(k not in ('text', 'hard', 'other') or not math.isfinite(float(v)) or v < 0 for k, v in ratios.items()):
            raise ValueError('Invalid sampling ratios')
        total = sum(ratios.values())
        n = int(sampling.get('num_samples', len(self.tiles)))
        if total <= 0 or n <= 0:
            raise ValueError('Sampling budget must be positive')
        exact = {k: n * v / total for k, v in ratios.items()}
        quotas = {k: int(v) for k, v in exact.items()}
        for k in sorted(exact, key=lambda k: exact[k] - quotas[k], reverse=True)[:n - sum(quotas.values())]:
            quotas[k] += 1
        rng = np.random.default_rng(int(self.opt.get('seed', 0)) + self.epoch)
        result = []
        for name, quota in quotas.items():
            pool = [i for i, tile in enumerate(self.tiles) if tile[3] == name]
            if quota and not pool:
                raise ValueError(f'Requested sampling stratum has no tiles: {name}')
            if not quota:
                continue
            if not sampling.get('source_balance', True):
                result.extend(rng.choice(pool, quota, replace=True).tolist())
                continue
            groups = {}
            for i in pool:
                row = self.records[self.tiles[i][0]]['row']
                source = str(row.get('source_id', row['lq']))
                groups.setdefault(source, []).append(i)
            sources = list(groups)
            rng.shuffle(sources)
            for j in range(quota):
                result.append(int(rng.choice(groups[sources[j % len(sources)]])))
        rng.shuffle(result)
        return result

    def __len__(self):
        """@luna-comment
        Input: None.
        Output: Number of scheduled native tiles.
        Key Steps: Return plan length.
        Key Params: None.
        State/Dependencies: Immutable sampling plan.
        """
        return len(self.plan)

    def __getitem__(self, index):
        """@luna-comment
        Input: Scheduled tile index.
        Output: CHW tensors, per-instance ROIs and auditable exposure metadata.
        Key Steps: Crop synchronized arrays, intersect center validity, clip contextual ROIs.
        Key Params: halo excludes unreliable edges; complete gates optional single-line CP.
        State/Dependencies: Read-only records/plan; no mutable worker exposure counters.
        """
        tile_index = self.plan[index]
        record_idx, x, y, stratum = self.tiles[tile_index]
        r = self._record(record_idx)
        h, w = self.canvas
        context_valid = _extract(r['valid'], x, y, w, h)
        valid = context_valid.copy()
        center = np.zeros((h, w), np.float32)
        center[self.halo:h - self.halo, self.halo:w - self.halo] = 1
        valid *= center
        instance_map = _extract(r['ownership'], x, y, w, h)
        rois = []
        for item in r['instances']:
            iid = item['instance_id']
            if not np.any((instance_map == iid) & (valid > 0)):
                continue
            x0, y0, x1, y1 = item['bbox']
            complete = x0 >= x and y0 >= y and x1 <= x + w and y1 <= y + h
            complete = complete and x0 >= 0 and y0 >= 0 and x1 <= r['gt'].shape[1] and y1 <= r['gt'].shape[0]
            c = self.roi_context
            bbox = [max(0, x0 - x - c), max(0, y0 - y - c), min(w, x1 - x + c), min(h, y1 - y + c)]
            rois.append(dict(bbox=bbox, instance_id=iid, original_id=item['original_id'],
                             tight_bbox=[max(0, x0 - x), max(0, y0 - y), min(w, x1 - x), min(h, y1 - y)],
                             confidence=item['confidence'], orientation=item['orientation'], complete=complete,
                             cp_eligible=bool(complete and item.get('single_line', False) and item['orientation'] == 'horizontal'),
                             original_bbox=item['bbox'], scheduled_exposure=self.exposures[(record_idx, iid)]))
            for key in ('weight', 'loss_confidence', 'char_height', 'stroke_width'):
                if key in item:
                    rois[-1][key] = item[key]
        def tensor(value):
            """@luna-comment
            Input: HW or HWC native NumPy array.
            Output: Contiguous CHW torch tensor retaining numeric dtype.
            Key Steps: Add channel axis or transpose RGB.
            Key Params: None.
            State/Dependencies: PyTorch; creates independent contiguous storage.
            """
            return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1) if value.ndim == 3 else value[None]))
        row = r['row']
        return dict(lq_rgb=tensor(_extract(r['lq'], x, y, w, h, True)),
                    gt_rgb=tensor(_extract(r['gt'], x, y, w, h, True)),
                    context_valid_mask_rgb=tensor(context_valid), valid_mask_rgb=tensor(valid),
                    text_mask_rgb=tensor(_extract(r['text'], x, y, w, h)),
                    instance_map=tensor(instance_map).long(), text_rois=rois,
                    structure_available=r['annotation_status'] == 'annotated' and bool(row.get('text_mask')),
                    annotation_status=r['annotation_status'], lq_path=str(row['lq']), gt_path=str(row['gt']),
                    metadata=dict(tile_index=tile_index, origin=[x, y], source_id=row.get('source_id', row['lq']),
                                  stratum=stratum, scheduled_tile_exposure=self.tile_exposures[tile_index],
                                  realized_strata=dict(self.realized_strata),
                                  stroke_fraction=float((_extract(r['text'], x, y, w, h) * valid).sum() / max(valid.sum(), 1))))

    @staticmethod
    def collate_fn(batch):
        """@luna-comment
        Input: Nonempty list of same-canvas sample dictionaries.
        Output: Stacked tensor batch and untruncated per-image ROI/metadata lists.
        Key Steps: Stack tensors; preserve variable-length annotation lists verbatim.
        Key Params: None.
        State/Dependencies: PyTorch; never modifies input samples.
        """
        return {key: torch.stack([sample[key] for sample in batch]) if torch.is_tensor(batch[0][key])
                else [sample[key] for sample in batch] for key in batch[0]}
