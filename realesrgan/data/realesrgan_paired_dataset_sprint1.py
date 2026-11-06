"""Sprint-1 RealCE text-centred paired degradation dataset.

This version deliberately keeps TXT and JSON OCR annotations on the same
training contract.  Regardless of source format, ``_load_ocr_lines`` returns
only line-level records:

    {"poly": np.ndarray(shape=(4, 2), dtype=float32), "text": str}

The base dataset treats JSON records containing ``word_boxes`` differently
from TXT records during text-centred crop selection (word crop vs line crop).
Dropping those JSON-only fields makes crop selection and all downstream
geometric transforms equivalent across the two annotation formats.
"""

import copy
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import get_worker_info

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.data_util import paired_paths_from_folder
from torchvision.transforms.functional import normalize

from .realesrgan_dataset import (
    _align_crop_size,
    _float01_to_uint8,
    _uint8_to_float01,
    adjust_contrast_yuv,
    adjust_lab,
)
from .realesrgan_paired_dataset import (
    RealESRGANPairedTextCenterDegradeDataset,
    _trim_paired_to_data_scale,
)
from .paired_color_augmentation import sample_safe_paired_color_augmentation


@DATASET_REGISTRY.register()
class RealESRGANPairedTextCenterDegradeDatasetSprint1(
        RealESRGANPairedTextCenterDegradeDataset):
    """RealCE dataset with canonical TXT/JSON line annotations.

    Additional options:
        text_crop_required (bool): If true, reject samples without a valid
            text-centred crop.  Default false: OCR annotations are a sampling
            preference; no-text scenes and failed text-crop attempts use the
            standard paired random crop instead.
    """

    def __init__(self, opt):
        """Initialize Sprint-1 text-center options and controlled color transforms.

        @shanlan-comment
        Input: dataset options containing inherited paired-data settings and optional Sprint-1 augmentation ranges.
        Output: initialized dataset state with OCR, mask, color, and contrast controls available to ``__getitem__``.
        Key Steps: initialize the parent dataset, normalize boolean/range options, and prepare worker-local RNG state.
        Key Params: ``use_yuv_contrast`` controls the optional GT contrast transform; range values accept scalar or pair forms.
        State/Dependencies: parent dataset paths/file-client state and cached paired-color RNG fields.
        """
        super().__init__(opt)
        self.text_crop_required = bool(opt.get('text_crop_required', False))
        self.use_adjust_lab = bool(opt.get('use_adjust_lab', False))
        self.lab_a_shift_range = opt.get('lab_a_shift', [0.0, 0.0])
        self.lab_b_shift_range = opt.get('lab_b_shift', [0.0, 0.0])
        self.lab_a_scale_range = opt.get('lab_a_scale', [1.0, 1.0])
        self.lab_b_scale_range = opt.get('lab_b_scale', [1.0, 1.0])
        self.lab_clip = bool(opt.get('lab_clip', True))
        self.use_yuv_contrast = bool(opt.get('use_yuv_contrast', False))
        contrast_range = opt.get('yuv_contrast_factor_range', opt.get('yuv_contrast_factor', [1.25, 1.25]))
        brightness_range = opt.get('yuv_brightness_range', opt.get('yuv_brightness', [0.0, 0.0]))
        if not isinstance(contrast_range, (list, tuple)):
            contrast_range = [contrast_range, contrast_range]
        if not isinstance(brightness_range, (list, tuple)):
            brightness_range = [brightness_range, brightness_range]
        self.yuv_contrast_factor_range = tuple(float(value) for value in contrast_range)
        self.yuv_brightness_range = tuple(float(value) for value in brightness_range)
        self.mask_folder = opt.get('mask_folder', None)
        self.mask_prefer_suffix = str(opt.get('mask_prefer_suffix', '_mask.png'))
        self.use_paired_color_augmentation = bool(
            opt.get('use_paired_color_augmentation', False))
        self.paired_color_probability = float(
            opt.get('paired_color_probability', 0.85))
        self.paired_color_max_projection_ratio = float(
            opt.get('paired_color_max_projection_ratio', 0.05))
        self.paired_color_max_attempts = int(
            opt.get('paired_color_max_attempts', 8))
        self.paired_color_temperature_range = tuple(
            opt.get('paired_color_temperature_range', [-0.15, 0.03]))
        self.paired_color_tint_range = tuple(
            opt.get('paired_color_tint_range', [-0.12, 0.02]))
        self.paired_color_saturation_range = tuple(
            opt.get('paired_color_saturation_range', [0.85, 1.05]))
        self._paired_color_rng = None
        self._paired_color_worker_seed = None

    def _get_paired_color_rng(self):
        """Create one deterministic NumPy generator per DataLoader worker.

        @shanlan-comment
        Input: the current DataLoader worker context and dataset instance state.
        Output: a stable NumPy generator for the worker's paired color augmentation.
        Key Steps: derive the worker seed, recreate the generator when the worker changes, and cache it.
        Key Params: the worker seed is reduced to the NumPy uint32 range.
        State/Dependencies: mutates ``_paired_color_rng`` and ``_paired_color_worker_seed``.
        """
        worker_info = get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        if self._paired_color_rng is None or self._paired_color_worker_seed != worker_seed:
            self._paired_color_rng = np.random.default_rng(worker_seed % (2 ** 32))
            self._paired_color_worker_seed = worker_seed
        return self._paired_color_rng

    def _maybe_apply_yuv_contrast(self, img_gt_u8):
        """Apply the optional GT-only YUV contrast transform.

        @shanlan-comment
        Input: uint8 HWC GT image and dataset options initialized by ``__init__``.
        Output: uint8 HWC image; returns the input unchanged when the switch is disabled or outside training.
        Key Steps: sample contrast/brightness ranges only for the enabled training branch, then call the shared YUV helper.
        Key Params: ``use_yuv_contrast``, ``yuv_contrast_factor_range``, and ``yuv_brightness_range`` control the transform.
        State/Dependencies: reads dataset options and NumPy RNG; does not mutate the source array in the disabled path.
        """
        if self.opt['phase'] != 'train' or not self.use_yuv_contrast:
            return img_gt_u8
        contrast_factor = float(np.random.uniform(*self.yuv_contrast_factor_range))
        brightness = float(np.random.uniform(*self.yuv_brightness_range))
        return adjust_contrast_yuv(
            img_gt_u8,
            contrast_factor=contrast_factor,
            brightness=brightness)

    def _annotation_candidates(self, gt_path):
        """Return ordered TXT/JSON candidates for a GT path.

        Supports both historical ``res_<stem>.txt`` naming used by RealCE TXT
        labels and the current ``<stem>[_res].json`` convention.  The same
        canonical dataset can therefore switch formats through
        ``read_OCR_json`` without a crop-policy change.
        """
        if self.anno_folder:
            relative_path = os.path.relpath(gt_path, self.gt_folder)
            anno_file = os.path.join(self.anno_folder, relative_path)
        else:
            # Keep legacy fallback behaviour when no explicit annotation root
            # is supplied.
            anno_file = (gt_path.replace('/gt_new/', '/annotations/')
                         .replace('/gt/', '/annotations/')
                         .replace('/52mm/', '/trans_annos_52mm_renov/'))

        anno_base = os.path.splitext(anno_file)[0]
        parent, stem = os.path.split(anno_base)

        json_candidates = [
            anno_base + '_res.json',
            anno_base + '.json',
            os.path.join(parent, 'res_' + stem + '.json'),
        ]
        txt_candidates = [
            anno_base + '_res.txt',
            anno_base + '.txt',
            os.path.join(parent, 'res_' + stem + '.txt'),
        ]
        return json_candidates, txt_candidates

    @staticmethod
    def _canonicalise_lines(items):
        """Canonicalize OCR records and discard malformed or empty lines.

        Input: an iterable of dictionaries containing polygon and text fields.
        Output: independent four-point float32 polygons with nonempty text.
        Key Steps: reshape coordinates, reject non-finite/degenerate polygons,
        and normalize text whitespace.
        Key Params: one-pixel minimum width and height reject unusable boxes.
        State/Dependencies: no mutable dataset state; NumPy performs validation.
        """
        canonical = []
        for item in items or []:
            poly = item.get('poly') if isinstance(item, dict) else None
            text = item.get('text', '') if isinstance(item, dict) else ''
            if poly is None:
                continue
            try:
                poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if poly.shape != (4, 2) or not np.isfinite(poly).all():
                continue
            width = float(poly[:, 0].max() - poly[:, 0].min())
            height = float(poly[:, 1].max() - poly[:, 1].min())
            if width < 1.0 or height < 1.0:
                continue
            text = str(text).strip()
            if not text:
                continue
            canonical.append({'poly': poly.copy(), 'text': text})
        return canonical

    def _load_ocr_lines(self, gt_path):
        """Load canonical OCR lines without crashing mixed no-text workers.

        Input: a GT image path whose sibling TXT/JSON candidates may be absent
        or malformed. Output: validated canonical line dictionaries, or an
        empty list in best-effort mode. Key Steps: try preferred and fallback
        candidates, normalize records, and enforce strict policy when enabled.
        Key Params: ``read_OCR_json`` selects candidate priority and
        ``text_crop_required`` controls missing/invalid annotation handling.
        State/Dependencies: inherited TXT/JSON readers and filesystem state.
        """
        if not self.read_OCR:
            return []

        json_candidates, txt_candidates = self._annotation_candidates(gt_path)
        preferred = json_candidates if self.read_OCR_json else txt_candidates
        fallback = txt_candidates if self.read_OCR_json else json_candidates

        for anno_path in preferred + fallback:
            if not os.path.isfile(anno_path):
                continue
            try:
                if anno_path.lower().endswith('.json'):
                    items = self.read_OCR_annos_json(anno_path)
                else:
                    items = self._parse_ocr_lines(self.read_OCR_annos(anno_path))
            except (AttributeError, OSError, TypeError, ValueError, UnicodeDecodeError):
                continue
            canonical = self._canonicalise_lines(items)
            if canonical:
                return canonical

        if self.text_crop_required:
            expected = preferred[0]
            raise FileNotFoundError(
                f'No valid OCR annotation for {gt_path}. Expected one of: '
                f'{preferred + fallback}; primary expected path: {expected}')
        return []

    def _mask_candidates(self, gt_path):
        """Return deterministic RealCE mask candidates for a GT image."""
        if not self.mask_folder:
            return []
        relative_path = os.path.relpath(gt_path, self.gt_folder)
        base = os.path.splitext(os.path.join(self.mask_folder, relative_path))[0]
        parent, stem = os.path.split(base)
        return [
            base + self.mask_prefer_suffix,
            base + '_mask.png',
            base + '_label_mask.npy',
            os.path.join(parent, stem + '.png'),
            os.path.join(parent, stem + '.npy'),
        ]

    def _load_realce_mask(self, gt_path, gt_shape):
        """Load PNG/NPY text mask and return float32 HWC mask in [0, 1]."""
        candidates = self._mask_candidates(gt_path)
        for mask_path in candidates:
            if not os.path.isfile(mask_path):
                continue
            if mask_path.lower().endswith('.npy'):
                try:
                    mask = np.load(mask_path)
                except (OSError, ValueError):
                    continue
                if mask.ndim == 2:
                    mask = mask[..., None]
                if mask.ndim != 3:
                    continue
                mask = mask.astype(np.float32)
                if mask.max() > 1.0:
                    mask /= 255.0
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                if mask.ndim == 2:
                    mask = mask[..., None]
                elif mask.shape[2] > 1:
                    mask = mask[..., :1]
                mask = mask.astype(np.float32) / 255.0
            if mask.shape[:2] != gt_shape[:2]:
                mask = cv2.resize(mask, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)
                if mask.ndim == 2:
                    mask = mask[..., None]
            # Keep a single-channel text mask: loss helpers broadcast it to
            # YUV/RGB channels and text-centred crop needs only spatial support.
            return np.clip(mask[..., :1], 0.0, 1.0), mask_path
        raise FileNotFoundError(
            f'No RealCE mask found for {gt_path}. Checked: {candidates}')

    def _sample_annotation_path(self, gt_path):
        """Use the configured TXT/JSON annotation root for blur filtering."""
        json_candidates, _ = self._annotation_candidates(gt_path)
        return next((path for path in json_candidates if os.path.isfile(path)), None)

    def __getitem__(self, index):
        """Load one paired sample with best-effort OCR-centred cropping.

        Input: a dataset index resolving to GT/LQ paths, plus optional OCR and mask files.
        Output: augmented tensor dictionaries with empty ``gt_lines`` for no-text crops.
        Key Steps: load and validate assets, prefer an OCR crop, fall back to paired cropping,
        then apply shared augmentation and online degradation metadata.
        Key Params: ``text_crop_required`` selects strict text-only versus mixed sampling;
        ``gt_size`` and ``data_scale`` define crop geometry.
        State/Dependencies: the per-worker ``FileClient`` and parent OCR geometry helpers.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.data_scale
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        retry = 3
        while retry > 0:
            try:
                img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
                img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)
                if img_gt is None or img_lq is None:
                    raise IOError(f'Unable to decode paired image for {gt_path}')
                img_mask, _ = self._load_realce_mask(gt_path, img_gt.shape)
                gt_lines = self._load_ocr_lines(gt_path)
                if self.text_crop_required and not gt_lines:
                    raise ValueError(f'No usable canonical OCR line for {gt_path}')
            except (IOError, OSError, ValueError, FileNotFoundError) as exc:
                logger = get_root_logger()
                logger.warning(f'RealCE sample error: {exc}; retry={retry - 1}')
                index = random.randrange(len(self.paths))
                gt_path = self.paths[index]['gt_path']
                lq_path = self.paths[index]['lq_path']
                retry -= 1
                continue
            break
        else:
            raise RuntimeError('Unable to load a valid RealCE sample after retries.')

        if self.read_OCR and gt_lines:
            gt_lines = self._scale_ocr_annotations_to_gt(gt_lines, scale)

        if (img_gt.shape[0] != img_lq.shape[0] * scale or
                img_gt.shape[1] != img_lq.shape[1] * scale):
            img_gt, img_lq, img_mask = _trim_paired_to_data_scale(
                img_gt, img_lq, scale, img_mask)
            if self.read_OCR and gt_lines:
                gt_lines = self._crop_ocr_lines(
                    gt_lines, 0, 0, img_gt.shape[1], img_gt.shape[0],
                    crop_text=bool(self.opt.get('ocr_crop_text', False)))

        gt_size = _align_crop_size(self.opt['gt_size'], scale)
        img_gt, img_lq, img_mask, gt_lines = self._maybe_random_resize_pair(
            img_gt, img_lq, img_mask, gt_lines, scale,
            min_gt_h=gt_size, min_gt_w=gt_size)

        crop_success = False
        if self.read_OCR and gt_lines:
            max_retry = int(self.opt.get('ocr_center_max_retry', 5))
            jitter = float(self.opt.get('text_center_jitter', 0.25))
            for _ in range(max_retry):
                box = self._pick_text_box(gt_lines)
                if box is None:
                    break
                h_lq, w_lq = img_lq.shape[:2]
                left, top, left_gt, top_gt, lq_patch = self._center_crop_coords(
                    box, h_lq, w_lq, gt_size, scale, jitter)
                candidate_lines = self._crop_ocr_lines(
                    copy.deepcopy(gt_lines), left_gt, top_gt, gt_size, gt_size,
                    crop_text=bool(self.opt.get('ocr_crop_text', False)))
                if candidate_lines:
                    img_lq = img_lq[top:top + lq_patch, left:left + lq_patch, ...]
                    img_gt = img_gt[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    img_mask = img_mask[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    gt_lines = candidate_lines
                    crop_success = True
                    break

        if not crop_success:
            if self.text_crop_required:
                raise RuntimeError(f'Unable to make a text-centred crop for {gt_path}')
            # No-text scenes and failed text-centred attempts are valid image
            # restoration samples.  Use the standard paired crop, preserving
            # annotations only for lines actually visible in the fallback crop.
            img_gt, img_lq, crop_info = paired_random_crop(
                [img_gt, img_mask], img_lq, gt_size, scale, gt_path, return_coords=True)
            img_gt, img_mask = img_gt
            if self.read_OCR and gt_lines:
                gt_lines = self._crop_ocr_lines(
                    gt_lines,
                    crop_info['left_gt'],
                    crop_info['top_gt'],
                    crop_info['gt_patch_size'],
                    crop_info['gt_patch_size'],
                    crop_text=bool(self.opt.get('ocr_crop_text', False)))

        h, w = img_gt.shape[:2]
        (img_gt, img_lq, img_mask), (hflip, vflip, rot90) = augment(
            [img_gt, img_lq, img_mask], self.opt['use_hflip'], self.opt['use_rot'], return_status=True)
        if self.read_OCR and gt_lines:
            gt_lines = self._augment_ocr_lines(gt_lines, h, w, hflip, vflip, rot90)

        if self.opt['phase'] == 'train' and self.use_paired_color_augmentation:
            img_lq_u8, img_gt_u8, _, _ = sample_safe_paired_color_augmentation(
                _float01_to_uint8(img_lq),
                _float01_to_uint8(img_gt),
                rng=self._get_paired_color_rng(),
                probability=self.paired_color_probability,
                max_projection_ratio=self.paired_color_max_projection_ratio,
                max_attempts=self.paired_color_max_attempts,
                temperature_range=self.paired_color_temperature_range,
                tint_range=self.paired_color_tint_range,
                saturation_range=self.paired_color_saturation_range,
            )
            img_lq = _uint8_to_float01(img_lq_u8)
            img_gt = _uint8_to_float01(img_gt_u8)

        if self.use_adjust_lab:
            lab_a_shift = float(np.random.uniform(*self.lab_a_shift_range))
            lab_b_shift = float(np.random.uniform(*self.lab_b_shift_range))
            lab_a_scale = float(np.random.uniform(*self.lab_a_scale_range))
            lab_b_scale = float(np.random.uniform(*self.lab_b_scale_range))
            img_gt = _uint8_to_float01(adjust_lab(
                _float01_to_uint8(img_gt), lab_a_shift, lab_b_shift,
                lab_a_scale, lab_b_scale, clip=self.lab_clip))
            img_lq = _uint8_to_float01(adjust_lab(
                _float01_to_uint8(img_lq), lab_a_shift, lab_b_shift,
                lab_a_scale, lab_b_scale, clip=self.lab_clip))

        # OpenCV resize/crop can collapse a HxWx1 mask to HxW.  Restore the
        # channel axis before img2tensor, which expects HWC inputs.
        if img_mask.ndim == 2:
            img_mask = img_mask[..., None]

        img_gt_u8 = _float01_to_uint8(img_gt)
        img_gt = _uint8_to_float01(self._maybe_apply_yuv_contrast(img_gt_u8))

        canny_map = gt_canny = None
        img_gt, img_lq, img_mask = img2tensor(
            [img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_mask, self.mean, self.std, inplace=True)

        if self.read_OCR:
            gt_lines = self._format_ocr_lines(gt_lines)

        result = {
            'lq': img_lq,
            'gt': img_gt,
            'mask': img_mask,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'gt_lines': gt_lines,
            'data_scale': scale,
            'canny_map': canny_map,
            'gt_canny': gt_canny,
        }
        if self.opt['phase'] == 'train':
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob:
                omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list, self.kernel_prob, kernel_size,
                    self.blur_sigma, self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range, self.betap_range, noise_range=None)
            pad_size = (21 - kernel_size) // 2
            result['kernel1'] = torch.FloatTensor(np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size))))

            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob2:
                omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2, self.kernel_prob2, kernel_size,
                    self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2, self.betap_range2, noise_range=None)
            pad_size = (21 - kernel_size) // 2
            result['kernel2'] = torch.FloatTensor(np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size))))

            if np.random.uniform() < self.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                sinc_kernel = circular_lowpass_kernel(
                    np.random.uniform(np.pi / 3, np.pi), kernel_size, pad_to=21)
                result['sinc_kernel'] = torch.FloatTensor(sinc_kernel)
            else:
                result['sinc_kernel'] = self.pulse_tensor
        return result


@DATASET_REGISTRY.register()
class RealESRGANPairedTextCenterMixedSourceDatasetSprint1(
        RealESRGANPairedTextCenterDegradeDatasetSprint1):
    """Sprint1 dataset supporting rooted sources or explicit JSONL records."""

    def __init__(self, opt):
        sources = opt.get('sources', [])
        manifest_jsonl = opt.get('manifest_jsonl')
        manifest_records = self._load_manifest_records(manifest_jsonl) if manifest_jsonl else []
        if manifest_records:
            first_record = manifest_records[0]
            sources = [{
                'name': 'jsonl_manifest',
                'dataroot_gt': os.path.dirname(first_record['gt']),
                'dataroot_lq': os.path.dirname(first_record['lq']),
                'degradation_mode': 'paired',
            }]
        if not sources:
            raise ValueError('sources or manifest_jsonl must define at least one dataset source')
        first = dict(opt)
        first.update(sources[0])
        first.pop('sources', None)
        first.pop('manifest_jsonl', None)
        if manifest_records:
            first['_explicit_paths'] = [
                {'gt_path': record['gt'], 'lq_path': record['lq']}
                for record in manifest_records
            ]
        super().__init__(first)
        self.opt.update(opt)
        self.sources = []
        self.paths = []
        self._explicit_records = {}
        if manifest_records:
            source = dict(sources[0])
            source['index'] = 0
            self.sources.append(source)
            for record in manifest_records:
                path = {
                    'gt_path': record['gt'],
                    'lq_path': record['lq'],
                    'mask_path': record['text_mask'],
                    'anno_path': record['ocr'],
                    'valid_mask_path': record.get('valid_mask'),
                    'dataset': record.get('dataset', ''),
                    'stem': record.get('stem', ''),
                    'source_index': 0,
                    'degradation_mode': 'paired',
                }
                self.paths.append(path)
                self._explicit_records[os.path.normpath(record['gt'])] = path
        else:
            self._build_rooted_source_paths(sources)
        self.gt_folder = self.sources[0]['dataroot_gt']
        self.lq_folder = self.sources[0]['dataroot_lq']
        self._active_source = self.sources[0]

    @staticmethod
    def _load_manifest_records(manifest_path):
        """Parse explicit paired assets from a JSONL manifest.

        Each non-empty record must provide LQ, GT, text-mask, and OCR paths.
        Invalid records fail immediately with their manifest line number so a
        partially loaded split cannot silently change experiment membership.
        """
        required = ('lq', 'gt', 'text_mask', 'ocr')
        records = []
        with open(manifest_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSON in {manifest_path}:{line_number}: {exc}') from exc
                missing = [key for key in required if not record.get(key)]
                if missing:
                    raise ValueError(
                        f'Missing keys {missing} in {manifest_path}:{line_number}')
                records.append(record)
        if not records:
            raise ValueError(f'No records found in JSONL manifest: {manifest_path}')
        return records

    def _build_rooted_source_paths(self, sources):
        """Build historical directory-derived paths for rooted data sources."""
        for source_index, source in enumerate(sources):
            source = dict(source)
            gt_root = source['dataroot_gt']
            lq_root = source['dataroot_lq']
            meta_info = source.get('meta_info')
            if meta_info:
                with open(meta_info, encoding='utf-8') as file:
                    source_paths = []
                    for line in file:
                        gt_name, lq_name = line.strip().split(', ')
                        source_paths.append({
                            'gt_path': os.path.join(gt_root, gt_name),
                            'lq_path': os.path.join(lq_root, lq_name),
                        })
            else:
                source_paths = paired_paths_from_folder(
                    [lq_root, gt_root], ['lq', 'gt'], source.get('filename_tmpl', '{}'))
            source['index'] = source_index
            self.sources.append(source)
            for path in source_paths:
                path['source_index'] = source_index
                path['degradation_mode'] = source.get('degradation_mode', 'paired')
                self.paths.append(path)

    def _source_for_gt(self, gt_path):
        """Resolve rooted-source metadata, falling back for explicit records."""
        return next(
            (source for source in self.sources
             if os.path.commonpath([gt_path, source['dataroot_gt']]) == source['dataroot_gt']),
            self._active_source,
        )

    def _annotation_candidates(self, gt_path):
        """Prefer a JSONL record's explicit OCR path over rooted conventions."""
        record = self._explicit_records.get(os.path.normpath(gt_path))
        if record:
            anno_path = record['anno_path']
            if anno_path.lower().endswith('.json'):
                return [anno_path], []
            return [], [anno_path]
        source = self._source_for_gt(gt_path)
        anno_root = source.get('anno_folder')
        if not anno_root:
            return super()._annotation_candidates(gt_path)
        relative = os.path.relpath(gt_path, source['dataroot_gt'])
        base = os.path.splitext(os.path.join(anno_root, relative))[0]
        parent, stem = os.path.split(base)
        return (
            [base + '.json', base + '_res.json', os.path.join(parent, 'res_' + stem + '.json')],
            [base + '.txt', base + '_res.txt', os.path.join(parent, 'res_' + stem + '.txt')],
        )

    def _mask_candidates(self, gt_path):
        """Prefer a JSONL record's explicit text-mask path."""
        record = self._explicit_records.get(os.path.normpath(gt_path))
        if record:
            return [record['mask_path']]
        source = self._source_for_gt(gt_path)
        mask_root = source.get('mask_folder')
        if not mask_root:
            return super()._mask_candidates(gt_path)
        relative = os.path.relpath(gt_path, source['dataroot_gt'])
        base = os.path.splitext(os.path.join(mask_root, relative))[0]
        return [base + '.png', base + '_mask.png', base + '.npy']

    def read_OCR_annos_json(self, anno_path):
        with open(anno_path, 'r', encoding='utf-8') as file:
            data = __import__('json').load(file)
        polys = data.get('rec_polys') or data.get('dt_polys') or []
        texts = data.get('rec_texts') or []
        parsed = []
        for index, poly in enumerate(polys):
            text = str(texts[index]).strip() if index < len(texts) else ''
            # Detection-only IQ annotations still need to drive text-centred crops.
            parsed.append({'poly': poly, 'text': text or '[DET]'})
        if parsed:
            return parsed
        return super().read_OCR_annos_json(anno_path)

    def __getitem__(self, index):
        self._active_source = self.sources[self.paths[index]['source_index']]
        result = super().__getitem__(index)
        result['degradation_mode'] = self.paths[index]['degradation_mode']
        result['source_index'] = self.paths[index]['source_index']
        return result

    def collate_fn(self, batch):
        collated = super().collate_fn(batch)
        collated['degradation_mode'] = [item['degradation_mode'] for item in batch]
        collated['source_index'] = torch.tensor(
            [item['source_index'] for item in batch], dtype=torch.long)
        return collated


@DATASET_REGISTRY.register()
class JSONLPairedImageDataset(torch.utils.data.Dataset):
    """Read validation LQ/GT pairs from explicit JSONL asset paths."""

    def __init__(self, opt):
        self.opt = opt
        self.paths = RealESRGANPairedTextCenterMixedSourceDatasetSprint1._load_manifest_records(
            opt['manifest_jsonl'])
        self.file_client = None
        self.io_backend_opt = dict(opt['io_backend'])

    def __getitem__(self, index):
        """Decode one manifest pair and return aligned RGB tensors."""
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        record = self.paths[index]
        gt = imfrombytes(self.file_client.get(record['gt'], 'gt'), float32=True)
        lq = imfrombytes(self.file_client.get(record['lq'], 'lq'), float32=True)
        scale = int(self.opt.get('scale', 1))
        gt = gt[:lq.shape[0] * scale, :lq.shape[1] * scale, ...]
        gt, lq = img2tensor([gt, lq], bgr2rgb=True, float32=True)
        return {
            'lq': lq,
            'gt': gt,
            'lq_path': record['lq'],
            'gt_path': record['gt'],
        }

    def __len__(self):
        """Return the exact JSONL split size."""
        return len(self.paths)
