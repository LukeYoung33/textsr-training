import json
import os

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from realesrgan.data.realesrgan_paired_dataset import RealESRGANPairedDataset


@DATASET_REGISTRY.register()
class RealESRGANOCRTestDataset(RealESRGANPairedDataset):
    def __init__(self, opt):
        super(RealESRGANOCRTestDataset, self).__init__(opt)
        self.annotation_folder = opt.get("dataroot_anno") or opt.get("anno_folder")
        if not self.annotation_folder:
            raise ValueError("RealESRGANOCRTestDataset requires `dataroot_anno` or `anno_folder`.")
        if not os.path.isdir(self.annotation_folder):
            raise FileNotFoundError(f"Annotation folder not found: {self.annotation_folder}")

        self.strict_shape = bool(opt.get("strict_shape", True))
        self.expected_gt_size = None
        for item in self.paths:
            stem = os.path.splitext(os.path.basename(item["gt_path"]))[0]
            annotation_path = os.path.join(self.annotation_folder, f"{stem}.json")
            if not os.path.isfile(annotation_path):
                raise FileNotFoundError(f"Missing annotation for `{stem}`: {annotation_path}")
            item["annotation_path"] = annotation_path

    def _load_annotation(self, annotation_path):
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        line_texts = data.get("annotations", []) or []
        line_polys = data.get("dt_polys", []) or []
        line_count = min(len(line_texts), len(line_polys))
        line_annotations = []
        for idx in range(line_count):
            poly = line_polys[idx]
            if not poly:
                continue
            try:
                poly = [[float(pt[0]), float(pt[1])] for pt in poly[:4]]
            except (TypeError, ValueError, IndexError):
                continue
            line_annotations.append(
                {
                    "line_index": idx,
                    "polygon": poly,
                    "text": str(line_texts[idx]),
                }
            )

        word_boxes_all = data.get("text_word_boxes", []) or []
        word_texts_all = data.get("text_word", []) or []
        word_annotations = []
        line_group_count = min(len(word_boxes_all), len(word_texts_all))
        for line_idx in range(line_group_count):
            boxes = word_boxes_all[line_idx] if isinstance(word_boxes_all[line_idx], list) else []
            texts = word_texts_all[line_idx] if isinstance(word_texts_all[line_idx], list) else []
            for word_idx in range(min(len(boxes), len(texts))):
                box = boxes[word_idx]
                if not box or len(box) < 4:
                    continue
                try:
                    parsed_box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                except (TypeError, ValueError):
                    continue
                word_annotations.append(
                    {
                        "line_index": line_idx,
                        "word_index": word_idx,
                        "box": parsed_box,
                        "text": str(texts[word_idx]),
                    }
                )

        return {
            "language": data.get("language", ""),
            "line_annotations": line_annotations,
            "word_annotations": word_annotations,
            "blur": data.get("blur", {}),
            "noise": data.get("noise", {}),
            "text_meta": {
                "min_font_size": data.get("min_font_size"),
                "max_font_size": data.get("max_font_size"),
                "fixed_font_size": data.get("fixed_font_size"),
                "paragraph_sizes": data.get("paragraph_sizes", []),
                "paragraph_count": data.get("paragraph_count"),
            },
        }

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img

    def collate_fn(self, batch):
        lq = torch.stack([item["lq"] for item in batch], dim=0)
        gt = torch.stack([item["gt"] for item in batch], dim=0)
        return {
            "lq": lq,
            "gt": gt,
            "lq_path": [item["lq_path"] for item in batch],
            "gt_path": [item["gt_path"] for item in batch],
            "annotation_path": [item["annotation_path"] for item in batch],
            "image_name": [item["image_name"] for item in batch],
            "language": [item["language"] for item in batch],
            "line_annotations": [item["line_annotations"] for item in batch],
            "word_annotations": [item["word_annotations"] for item in batch],
            "blur": [item["blur"] for item in batch],
            "noise": [item["noise"] for item in batch],
            "text_meta": [item["text_meta"] for item in batch],
        }

    def __getitem__(self, index):
        path_info = self.paths[index]
        gt_path = path_info["gt_path"]
        lq_path = path_info["lq_path"]
        annotation_path = path_info["annotation_path"]

        img_gt = self._read_image(gt_path)
        img_lq = self._read_image(lq_path)
        if img_gt.shape != img_lq.shape:
            raise ValueError(f"GT/LQ shape mismatch for {gt_path}: {img_gt.shape} vs {img_lq.shape}")

        if self.strict_shape:
            current_size = tuple(img_gt.shape[:2])
            if self.expected_gt_size is None:
                self.expected_gt_size = current_size
            elif self.expected_gt_size != current_size:
                raise ValueError(
                    f"Inconsistent image shape for batched test: expected {self.expected_gt_size}, got {current_size}."
                )

        annotation = self._load_annotation(annotation_path)

        img_gt = img_gt.astype(np.float32) / 255.0
        img_lq = img_lq.astype(np.float32) / 255.0
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "annotation_path": annotation_path,
            "image_name": os.path.splitext(os.path.basename(lq_path))[0],
            "language": annotation["language"],
            "line_annotations": annotation["line_annotations"],
            "word_annotations": annotation["word_annotations"],
            "blur": annotation["blur"],
            "noise": annotation["noise"],
            "text_meta": annotation["text_meta"],
        }
