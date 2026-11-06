import os
import json
import copy
import cv2
import math
import numpy as np
import random
import time
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import pdb
from realesrgan.data.realesrgan_dataset import (
    adjust_contrast_yuv,
    adjust_lab,
    _align_crop_size,
    _float01_to_uint8,
    _get_data_scale,
    _uint8_to_float01,
)


def _build_mask_path(gt_path):
    mask_path = gt_path.replace('/gt/', '/mask/').replace('\\gt\\', '\\mask\\')
    if os.path.isfile(mask_path):
        return mask_path
    stem, ext = os.path.splitext(mask_path)
    mask_path_alt = stem + '_mask' + ext
    if os.path.isfile(mask_path_alt):
        return mask_path_alt
    return mask_path


def _trim_paired_to_data_scale(img_gt, img_lq, scale, img_mask=None):
    h_lq, w_lq = img_lq.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    target_lq_h = min(h_lq, h_gt // scale)
    target_lq_w = min(w_lq, w_gt // scale)
    target_gt_h = target_lq_h * scale
    target_gt_w = target_lq_w * scale
    img_lq = img_lq[:target_lq_h, :target_lq_w, ...]
    img_gt = img_gt[:target_gt_h, :target_gt_w, ...]
    if img_mask is not None:
        img_mask = img_mask[:target_gt_h, :target_gt_w, ...]
    return img_gt, img_lq, img_mask

@DATASET_REGISTRY.register()
class RealESRGANPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RealESRGANPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.data_scale = _get_data_scale(opt)
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.down_rate = opt.get('down_rate', 1)

        # Setting OCR-related options
        self.read_OCR = opt.get('read_OCR', False)
        self.anno_folder = opt.get('anno_folder', False)
        self.read_OCR_json = opt.get('read_OCR_json', False)

    def read_OCR_annos(self, anno_path):
        try:
            anno_lines = open(anno_path, "r", encoding="gbk").readlines()
        except:
            anno_lines = open(anno_path, "r", encoding="utf-8").readlines()
        # str_lines = []
        # for i in range(len(anno_lines)):
        #     line = anno_lines[i]
        #     items = line.replace("\n", "").split(",")
        #     str_lines.append(items[-1])
        return anno_lines

    def read_OCR_annos_json(self, anno_path):
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        word_boxes = data.get("text_word_boxes", [])
        words = data.get("text_word", [])
        if not word_boxes or not words:
            return []
        parsed = []
        for line_boxes, line_words in zip(word_boxes, words):
            if not isinstance(line_boxes, list):
                continue
            if not isinstance(line_words, list):
                line_words = [str(line_words)]
            count = min(len(line_boxes), len(line_words))
            if count == 0:
                continue
            xs = []
            ys = []
            kept_boxes = []
            kept_texts = []
            for i in range(count):
                box = line_boxes[i]
                if not box or len(box) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                except ValueError:
                    continue
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                kept_boxes.append([x1, y1, x2, y2])
                kept_texts.append(str(line_words[i]))
            if not xs or not ys or not kept_boxes:
                continue
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            poly = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
            text = "".join(kept_texts)
            parsed.append({"poly": poly, "text": text, "word_boxes": kept_boxes, "word_texts": kept_texts})
        return parsed

    def _parse_ocr_lines(self, anno_lines):
        parsed = []
        for line in anno_lines:
            s = line.strip()
            if not s:
                continue
            if s[0] == "\ufeff":
                s = s.lstrip("\ufeff")
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
            except ValueError:
                continue
            text = ",".join(parts[8:]).strip()
            poly = np.array(coords, dtype=np.float32).reshape(4, 2)
            parsed.append({"poly": poly, "text": text})
        return parsed

    def _format_ocr_lines(self, items):
        lines = []
        for item in items:
            poly = item["poly"].reshape(-1)
            coords = [str(int(round(x))) for x in poly.tolist()]
            text = item["text"]
            lines.append(",".join(coords + [text]))
        return lines

    def _scale_ocr_lines(self, items, sx, sy):
        if not items or (sx == 1.0 and sy == 1.0):
            return items
        for item in items:
            item["poly"][:, 0] *= sx
            item["poly"][:, 1] *= sy
            if "word_boxes" in item and item["word_boxes"]:
                scaled_boxes = []
                for box in item["word_boxes"]:
                    if not box or len(box) < 4:
                        continue
                    scaled_boxes.append([
                        float(box[0]) * sx,
                        float(box[1]) * sy,
                        float(box[2]) * sx,
                        float(box[3]) * sy
                    ])
                item["word_boxes"] = scaled_boxes
        return items

    def _scale_ocr_annotations_to_gt(self, items, scale):
        if not items:
            return items
        scale = float(scale)
        if scale == 1.0:
            return items
        return self._scale_ocr_lines(items, scale, scale)

    def _text_box_min_size(self, box):
        if box is None or len(box) < 4:
            return None
        return min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))

    def _maybe_upscale_text_crop(self, img_gt, img_lq, img_mask, gt_lines, box):
        min_size = self._text_box_min_size(box)
        threshold = float(self.opt.get('ocr_small_text_min_size', 0))
        upscale = float(self.opt.get('ocr_small_text_upscale', 2.0))
        if min_size is None or threshold <= 0 or min_size >= threshold or upscale <= 1.0:
            return img_gt, img_lq, img_mask, gt_lines, box

        h_lq, w_lq = img_lq.shape[:2]
        h_gt, w_gt = img_gt.shape[:2]
        new_w_lq = max(1, int(round(w_lq * upscale)))
        new_h_lq = max(1, int(round(h_lq * upscale)))
        new_w_gt = max(1, int(round(w_gt * upscale)))
        new_h_gt = max(1, int(round(h_gt * upscale)))
        img_lq = cv2.resize(img_lq, (new_w_lq, new_h_lq), interpolation=cv2.INTER_CUBIC)
        img_gt = cv2.resize(img_gt, (new_w_gt, new_h_gt), interpolation=cv2.INTER_CUBIC)
        img_mask = cv2.resize(img_mask, (new_w_gt, new_h_gt), interpolation=cv2.INTER_CUBIC)
        if gt_lines:
            gt_lines = self._scale_ocr_lines(gt_lines, upscale, upscale)
        box = [float(v) * upscale for v in box]
        return img_gt, img_lq, img_mask, gt_lines, box

    def _poly_intersects_rect(self, poly, left, top, crop_w, crop_h):
        x_min = float(poly[:, 0].min())
        x_max = float(poly[:, 0].max())
        y_min = float(poly[:, 1].min())
        y_max = float(poly[:, 1].max())
        if x_max < left or y_max < top:
            return False
        if x_min >= (left + crop_w) or y_min >= (top + crop_h):
            return False
        return True

    def _crop_ocr_lines(self, items, left, top, crop_w, crop_h, crop_text=False):
        if not items:
            return items
        kept = []
        for item in items:
            if "poly" in item and item["poly"] is not None:
                # Early skip using line-level box overlap with crop region
                if not self._poly_intersects_rect(item["poly"], left, top, crop_w, crop_h):
                    continue
            if "word_boxes" in item and "word_texts" in item:
                line_boxes = item["word_boxes"]
                line_texts = item["word_texts"]
                kept_boxes = []
                kept_texts = []
                for box, wtxt in zip(line_boxes, line_texts):
                    x1, y1, x2, y2 = box
                    x1 -= left
                    x2 -= left
                    y1 -= top
                    y2 -= top
                    if x2 < 0 or y2 < 0 or x1 >= crop_w or y1 >= crop_h:
                        continue
                    orig_w = max((x2 - x1), 1.0)
                    x1 = float(np.clip(x1, 0, crop_w - 1))
                    x2 = float(np.clip(x2, 0, crop_w - 1))
                    y1 = float(np.clip(y1, 0, crop_h - 1))
                    y2 = float(np.clip(y2, 0, crop_h - 1))
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue
                    if crop_text and wtxt:
                        vis_w = max(x2 - x1, 0.0)
                        vis_ratio = min(max(vis_w / orig_w, 0.0), 1.0)
                        # Chinese single-character: drop if mostly cut off
                        if any('\u4e00' <= ch <= '\u9fff' for ch in str(wtxt)):
                            if vis_ratio < 0.6:
                                continue
                        else:
                            if vis_ratio < 0.999:
                                left_trim = max(0.0, - (box[0] - left))
                                start_ratio = min(max(left_trim / orig_w, 0.0), 1.0)
                                n = len(str(wtxt))
                                keep = max(1, int(round(n * vis_ratio)))
                                start = int(round(n * start_ratio))
                                if start + keep > n:
                                    start = max(0, n - keep)
                                wtxt = str(wtxt)[start:start + keep]
                    kept_boxes.append([x1, y1, x2, y2])
                    kept_texts.append(wtxt)
                if not kept_boxes:
                    continue
                xs = [b[0] for b in kept_boxes] + [b[2] for b in kept_boxes]
                ys = [b[1] for b in kept_boxes] + [b[3] for b in kept_boxes]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                poly = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
                text = "".join([str(t) for t in kept_texts])
                if (y_max - y_min) < 10:
                    continue
                kept.append({"poly": poly, "text": text, "word_boxes": kept_boxes, "word_texts": kept_texts})
            else:
                poly = item["poly"].copy()
                poly[:, 0] -= left
                poly[:, 1] -= top
                x_min, x_max = poly[:, 0].min(), poly[:, 0].max()
                y_min, y_max = poly[:, 1].min(), poly[:, 1].max()
                if x_max < 0 or y_max < 0 or x_min >= crop_w or y_min >= crop_h:
                    continue
                poly[:, 0] = np.clip(poly[:, 0], 0, crop_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, crop_h - 1)
                if (poly[:, 0].max() - poly[:, 0].min()) < 1 or (poly[:, 1].max() - poly[:, 1].min()) < 1:
                    continue
                if (poly[:, 1].max() - poly[:, 1].min()) < 10:
                    continue
                text = item["text"]
                if crop_text and text:
                    full_w = max(x_max - x_min, 1.0)
                    vis_w = max(poly[:, 0].max() - poly[:, 0].min(), 0.0)
                    vis_ratio = min(max(vis_w / full_w, 0.0), 1.0)
                    if vis_ratio < 0.999:
                        left_trim = max(0.0, -x_min)
                        start_ratio = min(max(left_trim / full_w, 0.0), 1.0)
                        n = len(text)
                        keep = max(1, int(round(n * vis_ratio)))
                        start = int(round(n * start_ratio))
                        if start + keep > n:
                            start = max(0, n - keep)
                        text = text[start:start + keep]
                kept.append({"poly": poly, "text": text})
        return kept

    def _augment_ocr_lines(self, items, h, w, hflip, vflip, rot90):
        if not items:
            return items
        for item in items:
            poly = item["poly"]
            if hflip:
                poly[:, 0] = (w - 1) - poly[:, 0]
            if vflip:
                poly[:, 1] = (h - 1) - poly[:, 1]
            if rot90:
                poly = poly[:, [1, 0]]
            item["poly"] = poly
        return items

    def _pick_text_box(self, gt_lines):
        boxes = []
        for item in gt_lines:
            if "word_boxes" in item and item["word_boxes"]:
                for box in item["word_boxes"]:
                    if not box or len(box) < 4:
                        continue
                    boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            else:
                poly = item["poly"]
                x_min, x_max = float(poly[:, 0].min()), float(poly[:, 0].max())
                y_min, y_max = float(poly[:, 1].min()), float(poly[:, 1].max())
                boxes.append([x_min, y_min, x_max, y_max])
        if not boxes:
            return None
        return random.choice(boxes)

    def _center_crop_coords(self, box, h_lq, w_lq, gt_size, scale, jitter_ratio):
        lq_patch = gt_size // scale
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        jitter_px = max(0.0, float(jitter_ratio)) * float(gt_size)
        cx += random.uniform(-jitter_px, jitter_px)
        cy += random.uniform(-jitter_px, jitter_px)
        left = int(round((cx - gt_size / 2.0) / scale))
        top = int(round((cy - gt_size / 2.0) / scale))
        left = max(0, min(left, w_lq - lq_patch))
        top = max(0, min(top, h_lq - lq_patch))
        left_gt = left * scale
        top_gt = top * scale
        return left, top, left_gt, top_gt, lq_patch

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.data_scale

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                mask_path = _build_mask_path(gt_path)
                img_bytes = self.file_client.get(mask_path, 'gt')
                img_mask = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]['gt_path']
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        # img_bytes = self.file_client.get(gt_path, 'gt')


        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.read_OCR:
            anno_base = gt_path
            if self.anno_folder:
                anno_base = gt_path.replace(self.gt_folder, self.anno_folder)
            anno_base = anno_base.rsplit('.', 1)[0]
            if self.read_OCR_json:
                anno_candidates = [
                    anno_base + '_res.json',
                    anno_base + '.json',
                    anno_base + '_res.txt',
                    anno_base + '.txt'
                ]
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                if anno_path.endswith('.json'):
                    gt_lines = self.read_OCR_annos_json(anno_path)
                else:
                    gt_lines = self.read_OCR_annos(anno_path)
            else:
                anno_candidates = [anno_base + '_res.txt', anno_base + '.txt']
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                gt_lines = self.read_OCR_annos(anno_path)
            if gt_lines and isinstance(gt_lines[0], dict):
                pass
            else:
                gt_lines = self._parse_ocr_lines(gt_lines)
        else:
            gt_lines = []

        if self.read_OCR and gt_lines:
            gt_lines = self._scale_ocr_annotations_to_gt(gt_lines, scale)

        h_lq, w_lq = img_lq.shape[0:2]
        h_gt, w_gt = img_gt.shape[0:2]

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            img_gt, img_lq, img_mask = _trim_paired_to_data_scale(img_gt, img_lq, scale, img_mask)
            if self.read_OCR and gt_lines:
                gt_lines = self._crop_ocr_lines(
                    gt_lines,
                    0,
                    0,
                    img_gt.shape[1],
                    img_gt.shape[0],
                    crop_text=bool(self.opt.get('ocr_crop_text', False))
                )

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = _align_crop_size(self.opt['gt_size'], scale)
            # random crop
            if self.read_OCR and gt_lines:
                max_retry = int(self.opt.get('ocr_crop_max_retry', 10))
                img_gt_src = img_gt
                img_lq_src = img_lq
                img_mask_src = img_mask
                gt_lines_src = copy.deepcopy(gt_lines)
                crop_info = None
                for _ in range(max_retry):
                    img_gt, img_lq, crop_info = paired_random_crop(
                        [img_gt_src, img_mask_src],
                        img_lq_src,
                        gt_size,
                        scale,
                        gt_path,
                        return_coords=True
                    )
                    img_gt, img_mask = img_gt[:]
                    gt_lines_try = self._crop_ocr_lines(
                        copy.deepcopy(gt_lines_src),
                        crop_info['left_gt'],
                        crop_info['top_gt'],
                        crop_info['gt_patch_size'],
                        crop_info['gt_patch_size'],
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )
                    if gt_lines_try:
                        gt_lines = gt_lines_try
                        break
                else:
                    # deterministic fallback: center crop around first text line
                    h_gt_src, w_gt_src = img_gt_src.shape[0:2]
                    h_lq_src, w_lq_src = img_lq_src.shape[0:2]
                    lq_patch = gt_size // scale
                    first_poly = gt_lines_src[0]["poly"]
                    x_min = float(first_poly[:, 0].min())
                    x_max = float(first_poly[:, 0].max())
                    y_min = float(first_poly[:, 1].min())
                    y_max = float(first_poly[:, 1].max())
                    cx = 0.5 * (x_min + x_max)
                    cy = 0.5 * (y_min + y_max)
                    left_gt = int(round(cx - gt_size / 2.0))
                    top_gt = int(round(cy - gt_size / 2.0))
                    left_gt = max(0, min(left_gt, w_gt_src - gt_size))
                    top_gt = max(0, min(top_gt, h_gt_src - gt_size))
                    left = int(round(left_gt / scale))
                    top = int(round(top_gt / scale))
                    left = max(0, min(left, w_lq_src - lq_patch))
                    top = max(0, min(top, h_lq_src - lq_patch))
                    img_lq = img_lq_src[top:top + lq_patch, left:left + lq_patch, ...]
                    img_gt = img_gt_src[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    img_mask = img_mask_src[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    gt_lines = self._crop_ocr_lines(
                        copy.deepcopy(gt_lines_src),
                        left_gt,
                        top_gt,
                        gt_size,
                        gt_size,
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )
            else:
                img_gt, img_lq, crop_info = paired_random_crop([img_gt, img_mask], img_lq, gt_size, scale, gt_path, return_coords=True)
                img_gt, img_mask = img_gt[:]
            # flip, rotation
            h, w = img_gt.shape[0:2]
            (img_gt, img_lq, img_mask), (hflip, vflip, rot90) = augment(
                [img_gt, img_lq, img_mask],
                self.opt['use_hflip'],
                self.opt['use_rot'],
                return_status=True
            )
            if self.read_OCR and gt_lines:
                gt_lines = self._augment_ocr_lines(gt_lines, h, w, hflip, vflip, rot90)
            # brightness & contrast augmentation (applied consistently to GT and LQ; mask unchanged)
            # opt['brightness'] and opt['contrast'] are relative ranges, e.g. 0.2 -> factor in [0.8, 1.2]
            br_range = float(self.opt.get('brightness', 0.0))
            ct_range = float(self.opt.get('contrast', 0.0))
            if (br_range > 0.0) or (ct_range > 0.0):
                # sample factors
                if br_range > 0.0:
                    br_min, br_max = 1.0 - br_range, 1.0 + br_range
                    br_factor = random.uniform(br_min, br_max)
                else:
                    br_factor = 1.0
                if ct_range > 0.0:
                    ct_min, ct_max = 1.0 - ct_range, 1.0 + ct_range
                    ct_factor = random.uniform(ct_min, ct_max)
                else:
                    ct_factor = 1.0

                def _apply_bc(img_np):
                    # img_np: HWC, float32, range [0,1]
                    if ct_factor != 1.0:
                        mean = img_np.mean(axis=(0, 1), keepdims=True)
                        img_np = (img_np - mean) * ct_factor + mean
                    if br_factor != 1.0:
                        img_np = img_np * br_factor
                    return np.clip(img_np, 0.0, 1.0).astype(np.uint8)

                img_gt = _apply_bc(img_gt)
                img_lq = _apply_bc(img_lq)
        else:
            crop_h, crop_w , c= img_lq.shape
            if min(crop_h, crop_w) < self.down_rate:
                crop_h_ = np.ceil(crop_h / self.down_rate) * self.down_rate
                crop_w_ = np.ceil(crop_w / self.down_rate) * self.down_rate
                temp_lq = np.zeros((int(crop_h_), int(crop_w_), c), dtype=img_lq.dtype)
                temp_gt = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_mask = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_lq[0:crop_h, 0:crop_w, :] = img_lq
                temp_gt[0:crop_h*scale, 0:crop_w*scale, :] = img_gt
                temp_mask[0:crop_h*scale, 0:crop_w*scale, :] = img_mask
                img_lq = temp_lq
                img_gt = temp_gt
                img_mask = temp_mask
            else:
                crop_h = np.floor(crop_h / self.down_rate) * self.down_rate
                crop_w = np.floor(crop_w / self.down_rate) * self.down_rate
                img_lq = img_lq[:int(crop_h), :int(crop_w), :]
                img_gt = img_gt[:int(crop_h * scale), :int(crop_w * scale), :]
                img_mask = img_mask[:int(crop_h * scale), :int(crop_w * scale), :]
                if self.read_OCR and gt_lines:
                    gt_lines = self._crop_ocr_lines(
                        gt_lines,
                        0,
                        0,
                        int(crop_w * scale),
                        int(crop_h * scale),
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_mask, self.mean, self.std, inplace=True)

        if self.read_OCR:
            gt_lines = self._format_ocr_lines(gt_lines)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'mask': img_mask,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'gt_lines': gt_lines,
            'data_scale': scale
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class RealESRGANPairedTextCenterDataset(RealESRGANPairedDataset):
    """Paired dataset that crops around a randomly chosen text region."""

    def collate_fn(self, batch):
        # Stack tensors for lq/gt/mask; keep metadata fields as lists
        lq = torch.stack([item['lq'] for item in batch], dim=0)
        gt = torch.stack([item['gt'] for item in batch], dim=0)
        mask = torch.stack([item['mask'] for item in batch], dim=0)
        return {
            'lq': lq,
            'gt': gt,
            'mask': mask,
            'lq_path': [item['lq_path'] for item in batch],
            'gt_path': [item['gt_path'] for item in batch],
            'gt_lines': [item['gt_lines'] for item in batch],
            'data_scale': batch[0].get('data_scale', self.data_scale)
        }

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.data_scale

        gt_path = self.paths[index]['gt_path']
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                mask_path = _build_mask_path(gt_path)
                img_bytes = self.file_client.get(mask_path, 'gt')
                img_mask = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]['gt_path']
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.read_OCR:
            anno_base = gt_path
            if self.anno_folder:
                anno_base = gt_path.replace(self.gt_folder, self.anno_folder)
            anno_base = anno_base.rsplit('.', 1)[0]
            if self.read_OCR_json:
                anno_candidates = [
                    anno_base + '_res.json',
                    anno_base + '.json',
                    anno_base + '_res.txt',
                    anno_base + '.txt'
                ]
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                if anno_path.endswith('.json'):
                    gt_lines = self.read_OCR_annos_json(anno_path)
                else:
                    gt_lines = self.read_OCR_annos(anno_path)
            else:
                anno_candidates = [anno_base + '_res.txt', anno_base + '.txt']
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                gt_lines = self.read_OCR_annos(anno_path)
            if gt_lines and isinstance(gt_lines[0], dict):
                pass
            else:
                gt_lines = self._parse_ocr_lines(gt_lines)
        else:
            gt_lines = []

        if self.read_OCR and gt_lines:
            gt_lines = self._scale_ocr_annotations_to_gt(gt_lines, scale)

        h_lq, w_lq = img_lq.shape[0:2]
        h_gt, w_gt = img_gt.shape[0:2]

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            img_gt, img_lq, img_mask = _trim_paired_to_data_scale(img_gt, img_lq, scale, img_mask)
            if self.read_OCR and gt_lines:
                gt_lines = self._crop_ocr_lines(
                    gt_lines,
                    0,
                    0,
                    img_gt.shape[1],
                    img_gt.shape[0],
                    crop_text=bool(self.opt.get('ocr_crop_text', False))
                )

        if self.opt['phase'] == 'train':
            gt_size = _align_crop_size(self.opt['gt_size'], scale)
            if self.read_OCR and gt_lines:
                max_retry = int(self.opt.get('ocr_center_max_retry', 5))
                jitter = float(self.opt.get('text_center_jitter', 0.25))
                success = False
                for _ in range(max_retry):
                    box = self._pick_text_box(gt_lines)
                    if box is None:
                        print(f"[WARNING] No text box found in {gt_path}")
                        break
                    else:
                        print(box)
                    left, top, left_gt, top_gt, lq_patch = self._center_crop_coords(
                        box, h_lq, w_lq, gt_size, scale, jitter
                    )
                    img_lq = img_lq[top:top + lq_patch, left:left + lq_patch, ...]
                    img_gt = img_gt[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    img_mask = img_mask[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    gt_lines_try = self._crop_ocr_lines(
                        copy.deepcopy(gt_lines),
                        left_gt,
                        top_gt,
                        gt_size,
                        gt_size,
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )
                    if gt_lines_try:
                        gt_lines = gt_lines_try
                        success = True
                        break
                if not success:
                    img_gt, img_lq, crop_info = paired_random_crop(
                        [img_gt, img_mask], img_lq, gt_size, scale, gt_path, return_coords=True
                    )
                    img_gt, img_mask = img_gt[:]
                    if self.read_OCR and gt_lines:
                        gt_lines = self._crop_ocr_lines(
                            gt_lines,
                            crop_info['left_gt'],
                            crop_info['top_gt'],
                            crop_info['gt_patch_size'],
                            crop_info['gt_patch_size'],
                            crop_text=bool(self.opt.get('ocr_crop_text', False))
                        )
            else:
                img_gt, img_lq, crop_info = paired_random_crop(
                    [img_gt, img_mask], img_lq, gt_size, scale, gt_path, return_coords=True
                )
                img_gt, img_mask = img_gt[:]

            h, w = img_gt.shape[0:2]
            (img_gt, img_lq, img_mask), (hflip, vflip, rot90) = augment(
                [img_gt, img_lq, img_mask],
                self.opt['use_hflip'],
                self.opt['use_rot'],
                return_status=True
            )
            if self.read_OCR and gt_lines:
                gt_lines = self._augment_ocr_lines(gt_lines, h, w, hflip, vflip, rot90)

            br_range = float(self.opt.get('brightness', 0.0))
            ct_range = float(self.opt.get('contrast', 0.0))
            if (br_range > 0.0) or (ct_range > 0.0):
                if br_range > 0.0:
                    br_min, br_max = 1.0 - br_range, 1.0 + br_range
                    br_factor = random.uniform(br_min, br_max)
                else:
                    br_factor = 1.0
                if ct_range > 0.0:
                    ct_min, ct_max = 1.0 - ct_range, 1.0 + ct_range
                    ct_factor = random.uniform(ct_min, ct_max)
                else:
                    ct_factor = 1.0

                def _apply_bc(img_np):
                    if ct_factor != 1.0:
                        mean = img_np.mean(axis=(0, 1), keepdims=True)
                        img_np = (img_np - mean) * ct_factor + mean
                    if br_factor != 1.0:
                        img_np = img_np * br_factor
                    return np.clip(img_np, 0.0, 1.0).astype(np.uint8)

                img_gt = _apply_bc(img_gt)
                img_lq = _apply_bc(img_lq)
        else:
            crop_h, crop_w, c = img_lq.shape
            if min(crop_h, crop_w) < self.down_rate:
                crop_h_ = np.ceil(crop_h / self.down_rate) * self.down_rate
                crop_w_ = np.ceil(crop_w / self.down_rate) * self.down_rate
                temp_lq = np.zeros((int(crop_h_), int(crop_w_), c), dtype=img_lq.dtype)
                temp_gt = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_mask = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_lq[0:crop_h, 0:crop_w, :] = img_lq
                temp_gt[0:crop_h * scale, 0:crop_w * scale, :] = img_gt
                temp_mask[0:crop_h * scale, 0:crop_w * scale, :] = img_mask
                img_lq = temp_lq
                img_gt = temp_gt
                img_mask = temp_mask
            else:
                crop_h = np.floor(crop_h / self.down_rate) * self.down_rate
                crop_w = np.floor(crop_w / self.down_rate) * self.down_rate
                img_lq = img_lq[:int(crop_h), :int(crop_w), :]
                img_gt = img_gt[:int(crop_h * scale), :int(crop_w * scale), :]
                img_mask = img_mask[:int(crop_h * scale), :int(crop_w * scale), :]
                if self.read_OCR and gt_lines:
                    gt_lines = self._crop_ocr_lines(
                        gt_lines,
                        0,
                        0,
                        int(crop_w * scale),
                        int(crop_h * scale),
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )

        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_mask, self.mean, self.std, inplace=True)

        if self.read_OCR:
            gt_lines = self._format_ocr_lines(gt_lines)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'mask': img_mask,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'gt_lines': gt_lines,
            'data_scale': scale
        }


@DATASET_REGISTRY.register()
class RealESRGANPairedTextCenterDegradeDataset(RealESRGANPairedTextCenterDataset):
    """Paired text-center dataset with finetune-style augmentation and degradation kernels."""

    def __init__(self, opt):
        super(RealESRGANPairedTextCenterDegradeDataset, self).__init__(opt)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        # color adjustments (range-based)
        self.use_yuv_contrast = opt.get('use_yuv_contrast', False)
        self.yuv_contrast_factor_range = opt.get('yuv_contrast_factor_range', [1.3, 1.3])
        self.yuv_brightness_range = opt.get('yuv_brightness_range', [0.0, 0.0])
        self.use_adjust_lab = opt.get('use_adjust_lab', False)
        self.lab_a_shift_range = opt.get('lab_a_shift_range', [0.0, 0.0])
        self.lab_b_shift_range = opt.get('lab_b_shift_range', [0.0, 0.0])
        self.lab_a_scale_range = opt.get('lab_a_scale_range', [1.0, 1.0])
        self.lab_b_scale_range = opt.get('lab_b_scale_range', [1.0, 1.0])
        self.lab_clip = opt.get('lab_clip', True)

    def collate_fn(self, batch):
        # Stack tensors for lq/gt/mask; keep metadata fields as lists
        lq = torch.stack([item['lq'] for item in batch], dim=0)
        gt = torch.stack([item['gt'] for item in batch], dim=0)
        mask = torch.stack([item['mask'] for item in batch], dim=0)
        collated = {
            'lq': lq,
            'gt': gt,
            'mask': mask,
            'lq_path': [item['lq_path'] for item in batch],
            'gt_path': [item['gt_path'] for item in batch],
            'gt_lines': batch[0]['gt_lines'] if len(batch) == 1 else [item['gt_lines'] for item in batch],
            'data_scale': batch[0].get('data_scale', self.data_scale)
        }
        if 'kernel1' in batch[0]:
            collated['kernel1'] = torch.stack([item['kernel1'] for item in batch], dim=0)
            collated['kernel2'] = torch.stack([item['kernel2'] for item in batch], dim=0)
            collated['sinc_kernel'] = torch.stack([item['sinc_kernel'] for item in batch], dim=0)
        return collated

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.data_scale

        gt_path = self.paths[index]['gt_path']
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                mask_path = _build_mask_path(gt_path)
                img_bytes = self.file_client.get(mask_path, 'gt')
                img_mask = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]['gt_path']
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.read_OCR:
            anno_base = gt_path
            if self.anno_folder:
                anno_base = gt_path.replace(self.gt_folder, self.anno_folder)
            anno_base = anno_base.rsplit('.', 1)[0]
            if self.read_OCR_json:
                anno_candidates = [
                    anno_base + '_res.json',
                    anno_base + '.json',
                    anno_base + '_res.txt',
                    anno_base + '.txt'
                ]
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                if anno_path.endswith('.json'):
                    gt_lines = self.read_OCR_annos_json(anno_path)
                else:
                    gt_lines = self.read_OCR_annos(anno_path)
            else:
                anno_candidates = [anno_base + '_res.txt', anno_base + '.txt']
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                gt_lines = self.read_OCR_annos(anno_path)
            if gt_lines and isinstance(gt_lines[0], dict):
                pass
            else:
                gt_lines = self._parse_ocr_lines(gt_lines)
        else:
            gt_lines = []

        if self.read_OCR and gt_lines:
            gt_lines = self._scale_ocr_annotations_to_gt(gt_lines, scale)

        h_lq, w_lq = img_lq.shape[0:2]
        h_gt, w_gt = img_gt.shape[0:2]

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            img_gt, img_lq, img_mask = _trim_paired_to_data_scale(img_gt, img_lq, scale, img_mask)
            if self.read_OCR and gt_lines:
                gt_lines = self._crop_ocr_lines(
                    gt_lines,
                    0,
                    0,
                    img_gt.shape[1],
                    img_gt.shape[0],
                    crop_text=bool(self.opt.get('ocr_crop_text', False))
                )
            h_lq, w_lq = img_lq.shape[0:2]

        if self.opt['phase'] == 'train':
            gt_size = _align_crop_size(self.opt['gt_size'], scale)
            if self.read_OCR and gt_lines:
                max_retry = int(self.opt.get('ocr_center_max_retry', 5))
                jitter = float(self.opt.get('text_center_jitter', 0.25))
                success = False
                img_lq_src = img_lq
                img_gt_src = img_gt
                img_mask_src = img_mask
                gt_lines_src = gt_lines
                for _ in range(max_retry):
                    box = self._pick_text_box(gt_lines_src)
                    if box is None:
                        print(f"[WARNING] No text box found in {gt_path}")
                        break
                    img_gt_try = img_gt_src
                    img_lq_try = img_lq_src
                    img_mask_try = img_mask_src
                    gt_lines_try = copy.deepcopy(gt_lines_src)
                    img_gt_try, img_lq_try, img_mask_try, gt_lines_try, box = self._maybe_upscale_text_crop(
                        img_gt_try,
                        img_lq_try,
                        img_mask_try,
                        gt_lines_try,
                        box
                    )
                    h_lq_try, w_lq_try = img_lq_try.shape[0:2]
                    left, top, left_gt, top_gt, lq_patch = self._center_crop_coords(
                        box, h_lq_try, w_lq_try, gt_size, scale, jitter
                    )
                    img_lq_crop = img_lq_try[top:top + lq_patch, left:left + lq_patch, ...]
                    img_gt_crop = img_gt_try[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    img_mask_crop = img_mask_try[top_gt:top_gt + gt_size, left_gt:left_gt + gt_size, ...]
                    gt_lines_crop = self._crop_ocr_lines(
                        gt_lines_try,
                        left_gt,
                        top_gt,
                        gt_size,
                        gt_size,
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )
                    if gt_lines_crop:
                        img_lq = img_lq_crop
                        img_gt = img_gt_crop
                        img_mask = img_mask_crop
                        gt_lines = gt_lines_crop
                        success = True
                        break
                if not success:
                    img_gt, img_lq, crop_info = paired_random_crop(
                        [img_gt, img_mask], img_lq, gt_size, scale, gt_path, return_coords=True
                    )
                    img_gt, img_mask = img_gt[:]
                    if self.read_OCR and gt_lines:
                        gt_lines = self._crop_ocr_lines(
                            gt_lines,
                            crop_info['left_gt'],
                            crop_info['top_gt'],
                            crop_info['gt_patch_size'],
                            crop_info['gt_patch_size'],
                            crop_text=bool(self.opt.get('ocr_crop_text', False))
                        )
            else:
                img_gt, img_lq, crop_info = paired_random_crop(
                    [img_gt, img_mask], img_lq, gt_size, scale, gt_path, return_coords=True
                )
                img_gt, img_mask = img_gt[:]

            h, w = img_gt.shape[0:2]
            (img_gt, img_lq, img_mask), (hflip, vflip, rot90) = augment(
                [img_gt, img_lq, img_mask],
                self.opt['use_hflip'],
                self.opt['use_rot'],
                return_status=True
            )
            if self.read_OCR and gt_lines:
                gt_lines = self._augment_ocr_lines(gt_lines, h, w, hflip, vflip, rot90)

            if self.use_yuv_contrast:
                yuv_contrast = float(np.random.uniform(*self.yuv_contrast_factor_range))
                yuv_brightness = float(np.random.uniform(*self.yuv_brightness_range))
                img_gt_u8 = _float01_to_uint8(img_gt)
                img_lq_u8 = _float01_to_uint8(img_lq)
                img_gt = _uint8_to_float01(
                    adjust_contrast_yuv(
                        img_gt_u8,
                        contrast_factor=yuv_contrast,
                        brightness=yuv_brightness
                    )
                )

            if self.use_adjust_lab:
                lab_a_shift = float(np.random.uniform(*self.lab_a_shift_range))
                lab_b_shift = float(np.random.uniform(*self.lab_b_shift_range))
                lab_a_scale = float(np.random.uniform(*self.lab_a_scale_range))
                lab_b_scale = float(np.random.uniform(*self.lab_b_scale_range))
                img_gt_u8 = _float01_to_uint8(img_gt)
                img_lq_u8 = _float01_to_uint8(img_lq)
                img_gt = _uint8_to_float01(
                    adjust_lab(
                        img_gt_u8,
                        a_shift=lab_a_shift,
                        b_shift=lab_b_shift,
                        a_scale=lab_a_scale,
                        b_scale=lab_b_scale,
                        clip=self.lab_clip
                    )
                )
                img_lq = _uint8_to_float01(
                    adjust_lab(
                        img_lq_u8,
                        a_shift=lab_a_shift,
                        b_shift=lab_b_shift,
                        a_scale=lab_a_scale,
                        b_scale=lab_b_scale,
                        clip=self.lab_clip
                    )
                )
        else:
            crop_h, crop_w, c = img_lq.shape
            if min(crop_h, crop_w) < self.down_rate:
                crop_h_ = np.ceil(crop_h / self.down_rate) * self.down_rate
                crop_w_ = np.ceil(crop_w / self.down_rate) * self.down_rate
                temp_lq = np.zeros((int(crop_h_), int(crop_w_), c), dtype=img_lq.dtype)
                temp_gt = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_mask = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_lq[0:crop_h, 0:crop_w, :] = img_lq
                temp_gt[0:crop_h * scale, 0:crop_w * scale, :] = img_gt
                temp_mask[0:crop_h * scale, 0:crop_w * scale, :] = img_mask
                img_lq = temp_lq
                img_gt = temp_gt
                img_mask = temp_mask
            else:
                crop_h = np.floor(crop_h / self.down_rate) * self.down_rate
                crop_w = np.floor(crop_w / self.down_rate) * self.down_rate
                img_lq = img_lq[:int(crop_h), :int(crop_w), :]
                img_gt = img_gt[:int(crop_h * scale), :int(crop_w * scale), :]
                img_mask = img_mask[:int(crop_h * scale), :int(crop_w * scale), :]
                if self.read_OCR and gt_lines:
                    gt_lines = self._crop_ocr_lines(
                        gt_lines,
                        0,
                        0,
                        int(crop_w * scale),
                        int(crop_h * scale),
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )

        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)

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
            'data_scale': scale
        }
        if self.opt['phase'] == 'train':
            # Validation does not consume degradation kernels.
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob2:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            if np.random.uniform() < self.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            result['kernel1'] = torch.FloatTensor(kernel)
            result['kernel2'] = torch.FloatTensor(kernel2)
            result['sinc_kernel'] = sinc_kernel

        return result

@DATASET_REGISTRY.register()
class RealESRGANPairedTextCenterStripDegradeDataset(RealESRGANPairedTextCenterDegradeDataset):
    """Paired text-center dataset that outputs a single-line strip crop."""

    def __init__(self, opt):
        super(RealESRGANPairedTextCenterStripDegradeDataset, self).__init__(opt)
        self.text_strip_gt_h = int(opt.get('text_strip_gt_h', 64))
        self.text_strip_gt_w = int(opt.get('text_strip_gt_w', 256))
        self.text_strip_center_min_h = float(opt.get('text_strip_center_min_h', 48))
        self.text_strip_shrink_trigger_h = float(opt.get('text_strip_shrink_trigger_h', 64))
        self.text_strip_target_h_range = opt.get('text_strip_target_h_range', [32, 48])

    def _get_line_box(self, item):
        if item is None:
            return None
        if "word_boxes" in item and item["word_boxes"]:
            xs = []
            ys = []
            for box in item["word_boxes"]:
                if not box or len(box) < 4:
                    continue
                xs.extend([float(box[0]), float(box[2])])
                ys.extend([float(box[1]), float(box[3])])
            if xs and ys:
                return [min(xs), min(ys), max(xs), max(ys)]
        poly = item.get("poly")
        if poly is None or len(poly) == 0:
            return None
        return [
            float(poly[:, 0].min()),
            float(poly[:, 1].min()),
            float(poly[:, 0].max()),
            float(poly[:, 1].max())
        ]

    def _resize_pair_and_annotations(self, img_gt, img_lq, img_mask, gt_lines, scale_factor):
        if scale_factor == 1.0:
            return img_gt, img_lq, img_mask, gt_lines
        h_lq, w_lq = img_lq.shape[:2]
        h_gt, w_gt = img_gt.shape[:2]
        new_w_lq = max(1, int(round(w_lq * scale_factor)))
        new_h_lq = max(1, int(round(h_lq * scale_factor)))
        new_w_gt = max(1, int(round(w_gt * scale_factor)))
        new_h_gt = max(1, int(round(h_gt * scale_factor)))
        img_lq = cv2.resize(img_lq, (new_w_lq, new_h_lq), interpolation=cv2.INTER_CUBIC)
        img_gt = cv2.resize(img_gt, (new_w_gt, new_h_gt), interpolation=cv2.INTER_CUBIC)
        img_mask = cv2.resize(img_mask, (new_w_gt, new_h_gt), interpolation=cv2.INTER_CUBIC)
        if gt_lines:
            gt_lines = self._scale_ocr_lines(gt_lines, scale_factor, scale_factor)
        return img_gt, img_lq, img_mask, gt_lines

    def _crop_with_pad(self, img, left, top, crop_w, crop_h):
        if img.ndim == 2:
            out = np.zeros((crop_h, crop_w), dtype=img.dtype)
        else:
            out = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)
        h, w = img.shape[:2]
        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(w, left + crop_w)
        src_bottom = min(h, top + crop_h)
        if src_right <= src_left or src_bottom <= src_top:
            return out
        dst_left = src_left - left
        dst_top = src_top - top
        dst_right = dst_left + (src_right - src_left)
        dst_bottom = dst_top + (src_bottom - src_top)
        out[dst_top:dst_bottom, dst_left:dst_right, ...] = img[src_top:src_bottom, src_left:src_right, ...]
        return out

    def _crop_selected_text_strip(self, img_gt, img_lq, img_mask, gt_lines, scale):
        if not gt_lines:
            return None
        selected = copy.deepcopy(random.choice(gt_lines))
        box = self._get_line_box(selected)
        if box is None:
            return None

        x1, y1, x2, y2 = box
        line_h = max(1.0, y2 - y1)
        target_min_h = min(float(self.text_strip_target_h_range[0]), float(self.text_strip_target_h_range[-1]))
        target_max_h = max(float(self.text_strip_target_h_range[0]), float(self.text_strip_target_h_range[-1]))
        if line_h > self.text_strip_shrink_trigger_h:
            target_h = random.uniform(target_min_h, target_max_h)
            scale_factor = target_h / line_h
            img_gt, img_lq, img_mask, gt_lines_scaled = self._resize_pair_and_annotations(
                img_gt, img_lq, img_mask, [selected], scale_factor
            )
            selected = gt_lines_scaled[0]
            box = self._get_line_box(selected)
            x1, y1, x2, y2 = box
            line_h = max(1.0, y2 - y1)

        line_cx = 0.5 * (x1 + x2)
        line_cy = 0.5 * (y1 + y2)
        top_gt = int(round(line_cy - self.text_strip_gt_h / 2.0))
        if line_h < self.text_strip_center_min_h:
            top_gt = int(round(line_cy - self.text_strip_gt_h / 2.0))
        left_gt = int(round(line_cx - self.text_strip_gt_w / 2.0))

        img_gt_crop = self._crop_with_pad(img_gt, left_gt, top_gt, self.text_strip_gt_w, self.text_strip_gt_h)
        img_mask_crop = self._crop_with_pad(img_mask, left_gt, top_gt, self.text_strip_gt_w, self.text_strip_gt_h)
        img_lq_crop = self._crop_with_pad(
            img_lq,
            int(math.floor(left_gt / scale)),
            int(math.floor(top_gt / scale)),
            max(1, int(round(self.text_strip_gt_w / scale))),
            max(1, int(round(self.text_strip_gt_h / scale)))
        )
        gt_lines_crop = self._crop_ocr_lines(
            [selected],
            left_gt,
            top_gt,
            self.text_strip_gt_w,
            self.text_strip_gt_h,
            crop_text=True
        )
        if not gt_lines_crop:
            return None
        return img_gt_crop, img_lq_crop, img_mask_crop, gt_lines_crop

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.data_scale

        gt_path = self.paths[index]['gt_path']
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                mask_path = _build_mask_path(gt_path)
                img_bytes = self.file_client.get(mask_path, 'gt')
                img_mask = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]['gt_path']
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.read_OCR:
            anno_base = gt_path
            if self.anno_folder:
                anno_base = gt_path.replace(self.gt_folder, self.anno_folder)
            anno_base = anno_base.rsplit('.', 1)[0]
            if self.read_OCR_json:
                anno_candidates = [
                    anno_base + '_res.json',
                    anno_base + '.json',
                    anno_base + '_res.txt',
                    anno_base + '.txt'
                ]
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                if anno_path.endswith('.json'):
                    gt_lines = self.read_OCR_annos_json(anno_path)
                else:
                    gt_lines = self.read_OCR_annos(anno_path)
            else:
                anno_candidates = [anno_base + '_res.txt', anno_base + '.txt']
                anno_path = next((p for p in anno_candidates if os.path.isfile(p)), anno_candidates[0])
                gt_lines = self.read_OCR_annos(anno_path)
            if gt_lines and isinstance(gt_lines[0], dict):
                pass
            else:
                gt_lines = self._parse_ocr_lines(gt_lines)
        else:
            gt_lines = []

        if self.read_OCR and gt_lines:
            gt_lines = self._scale_ocr_annotations_to_gt(gt_lines, scale)

        h_lq, w_lq = img_lq.shape[0:2]
        h_gt, w_gt = img_gt.shape[0:2]

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            img_gt, img_lq, img_mask = _trim_paired_to_data_scale(img_gt, img_lq, scale, img_mask)
            if self.read_OCR and gt_lines:
                gt_lines = self._crop_ocr_lines(
                    gt_lines,
                    0,
                    0,
                    img_gt.shape[1],
                    img_gt.shape[0],
                    crop_text=bool(self.opt.get('ocr_crop_text', False))
                )

        if self.opt['phase'] == 'train':
            if self.read_OCR and gt_lines:
                success = False
                max_retry = int(self.opt.get('ocr_center_max_retry', 5))
                for _ in range(max_retry):
                    cropped = self._crop_selected_text_strip(img_gt, img_lq, img_mask, gt_lines, scale)
                    if cropped is not None:
                        img_gt, img_lq, img_mask, gt_lines = cropped
                        success = True
                        break
                if not success:
                    gt_h = self.text_strip_gt_h
                    gt_w = self.text_strip_gt_w
                    left_gt = max(0, (img_gt.shape[1] - gt_w) // 2)
                    top_gt = max(0, (img_gt.shape[0] - gt_h) // 2)
                    img_gt = self._crop_with_pad(img_gt, left_gt, top_gt, gt_w, gt_h)
                    img_mask = self._crop_with_pad(img_mask, left_gt, top_gt, gt_w, gt_h)
                    img_lq = self._crop_with_pad(
                        img_lq,
                        int(math.floor(left_gt / scale)),
                        int(math.floor(top_gt / scale)),
                        max(1, int(round(gt_w / scale))),
                        max(1, int(round(gt_h / scale)))
                    )
                    gt_lines = []
            else:
                gt_h = self.text_strip_gt_h
                gt_w = self.text_strip_gt_w
                left_gt = max(0, (img_gt.shape[1] - gt_w) // 2)
                top_gt = max(0, (img_gt.shape[0] - gt_h) // 2)
                img_gt = self._crop_with_pad(img_gt, left_gt, top_gt, gt_w, gt_h)
                img_mask = self._crop_with_pad(img_mask, left_gt, top_gt, gt_w, gt_h)
                img_lq = self._crop_with_pad(
                    img_lq,
                    int(math.floor(left_gt / scale)),
                    int(math.floor(top_gt / scale)),
                    max(1, int(round(gt_w / scale))),
                    max(1, int(round(gt_h / scale)))
                )

            h, w = img_gt.shape[0:2]
            (img_gt, img_lq, img_mask), (hflip, vflip, rot90) = augment(
                [img_gt, img_lq, img_mask],
                self.opt['use_hflip'],
                False,
                return_status=True
            )
            if self.read_OCR and gt_lines:
                gt_lines = self._augment_ocr_lines(gt_lines, h, w, hflip, vflip, rot90)

            if self.use_yuv_contrast:
                yuv_contrast = float(np.random.uniform(*self.yuv_contrast_factor_range))
                yuv_brightness = float(np.random.uniform(*self.yuv_brightness_range))
                img_gt_u8 = _float01_to_uint8(img_gt)
                img_lq_u8 = _float01_to_uint8(img_lq)
                img_gt = _uint8_to_float01(
                    adjust_contrast_yuv(
                        img_gt_u8,
                        contrast_factor=yuv_contrast,
                        brightness=yuv_brightness
                    )
                )

            if self.use_adjust_lab:
                lab_a_shift = float(np.random.uniform(*self.lab_a_shift_range))
                lab_b_shift = float(np.random.uniform(*self.lab_b_shift_range))
                lab_a_scale = float(np.random.uniform(*self.lab_a_scale_range))
                lab_b_scale = float(np.random.uniform(*self.lab_b_scale_range))
                img_gt_u8 = _float01_to_uint8(img_gt)
                img_lq_u8 = _float01_to_uint8(img_lq)
                img_gt = _uint8_to_float01(
                    adjust_lab(
                        img_gt_u8,
                        a_shift=lab_a_shift,
                        b_shift=lab_b_shift,
                        a_scale=lab_a_scale,
                        b_scale=lab_b_scale,
                        clip=self.lab_clip
                    )
                )
                img_lq = _uint8_to_float01(
                    adjust_lab(
                        img_lq_u8,
                        a_shift=lab_a_shift,
                        b_shift=lab_b_shift,
                        a_scale=lab_a_scale,
                        b_scale=lab_b_scale,
                        clip=self.lab_clip
                    )
                )
        else:
            crop_h, crop_w, c = img_lq.shape
            if min(crop_h, crop_w) < self.down_rate:
                crop_h_ = np.ceil(crop_h / self.down_rate) * self.down_rate
                crop_w_ = np.ceil(crop_w / self.down_rate) * self.down_rate
                temp_lq = np.zeros((int(crop_h_), int(crop_w_), c), dtype=img_lq.dtype)
                temp_gt = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_mask = np.zeros((int(crop_h_ * scale), int(crop_w_ * scale), c), dtype=img_lq.dtype)
                temp_lq[0:crop_h, 0:crop_w, :] = img_lq
                temp_gt[0:crop_h * scale, 0:crop_w * scale, :] = img_gt
                temp_mask[0:crop_h * scale, 0:crop_w * scale, :] = img_mask
                img_lq = temp_lq
                img_gt = temp_gt
                img_mask = temp_mask
            else:
                crop_h = np.floor(crop_h / self.down_rate) * self.down_rate
                crop_w = np.floor(crop_w / self.down_rate) * self.down_rate
                img_lq = img_lq[:int(crop_h), :int(crop_w), :]
                img_gt = img_gt[:int(crop_h * scale), :int(crop_w * scale), :]
                img_mask = img_mask[:int(crop_h * scale), :int(crop_w * scale), :]
                if self.read_OCR and gt_lines:
                    gt_lines = self._crop_ocr_lines(
                        gt_lines,
                        0,
                        0,
                        int(crop_w * scale),
                        int(crop_h * scale),
                        crop_text=bool(self.opt.get('ocr_crop_text', False))
                    )

        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)

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
            'data_scale': scale
        }
        if self.opt['phase'] == 'train':
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob2:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            if np.random.uniform() < self.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            result['kernel1'] = torch.FloatTensor(kernel)
            result['kernel2'] = torch.FloatTensor(kernel2)
            result['sinc_kernel'] = sinc_kernel

        return result
