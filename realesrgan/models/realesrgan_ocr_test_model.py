import json
import os
from collections import OrderedDict
from os import path as osp

import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from tqdm import tqdm

from realesrgan.metrics.ocr_text_metrics import (
    LocalVGGPerceptualMetric,
    PPOCRBatchRecognizer,
    calculate_psnr_batch_pytorch,
    calculate_ssim_batch_pytorch,
    crop_bgr_region,
    expand_and_clip_box,
    normalize_text_for_eval,
    normalized_edit_similarity,
    polygon_to_box,
)


@MODEL_REGISTRY.register()
class RealESRGANOCRTestModel(SRModel):
    def __init__(self, opt):
        super(RealESRGANOCRTestModel, self).__init__(opt)
        self.metric_opts = opt.get("val", {}).get("metrics", {}) or {}
        self.ocr_expand_pixels = float(opt.get("val", {}).get("ocr_expand_pixels", 2))
        self.ocr_batch_size = int(opt.get("val", {}).get("ocr_batch_size", 32))
        self.save_ocr = bool(opt.get("val", {}).get("save_ocr", True))
        self.save_summary = bool(opt.get("val", {}).get("save_summary", True))
        self.recognizer = PPOCRBatchRecognizer(
            device=self.device,
            dict_path=opt.get("val", {}).get("ocr_dict_path"),
            weights_path=opt.get("val", {}).get("ocr_weights_path"),
            crop_height=opt.get("val", {}).get("ocr_crop_height", 48),
            min_width=opt.get("val", {}).get("ocr_min_width", 16),
            max_width=opt.get("val", {}).get("ocr_max_width", 512),
            use_space_char=opt.get("val", {}).get("ocr_use_space_char", True),
        )

        self.lpips_metric = None
        if "lpips" in self.metric_opts:
            lpips_mode = opt.get("val", {}).get("lpips_mode", "local_vgg")
            if lpips_mode == "pyiqa":
                import pyiqa

                metric_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.lpips_metric = pyiqa.create_metric("lpips", device=metric_device, as_loss=False)
            else:
                self.lpips_metric = LocalVGGPerceptualMetric().to(self.device)
                self.lpips_metric.eval()

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)
        self.image_names = data["image_name"]
        self.lq_paths = data["lq_path"]
        self.gt_paths = data["gt_path"]
        self.annotation_paths = data["annotation_path"]
        self.languages = data["language"]
        self.line_annotations = data["line_annotations"]
        self.word_annotations = data["word_annotations"]
        self.blur_infos = data["blur"]
        self.noise_infos = data["noise"]
        self.text_meta_infos = data["text_meta"]

    def _make_output_dirs(self, dataset_name):
        base_dir = osp.join(self.opt["path"]["visualization"], dataset_name)
        image_dir = osp.join(base_dir, "images")
        ocr_dir = osp.join(base_dir, "ocr")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(ocr_dir, exist_ok=True)
        return base_dir, image_dir, ocr_dir

    def _get_suffix(self):
        suffix = self.opt.get("val", {}).get("suffix")
        return suffix if suffix else self.opt["name"]

    def _save_sr_images(self, sr_images, image_dir):
        suffix = self._get_suffix()
        for image_name, sr_img in zip(self.image_names, sr_images):
            imwrite(sr_img, osp.join(image_dir, f"{image_name}_{suffix}.png"))

    def _metric_value_list(self, tensor):
        if tensor is None:
            return []
        if isinstance(tensor, torch.Tensor):
            values = tensor.detach().float().cpu().view(-1).tolist()
        else:
            values = list(tensor)
        return [float(v) for v in values]

    def _compute_image_metrics(self, sr, gt):
        metric_values = {}

        if "psnr" in self.metric_opts:
            opt_ = self.metric_opts["psnr"]
            metric_values["psnr"] = self._metric_value_list(
                calculate_psnr_batch_pytorch(
                    sr,
                    gt,
                    crop_border=int(opt_.get("crop_border", 0)),
                    test_y_channel=bool(opt_.get("test_y_channel", False)),
                )
            )

        if "ssim" in self.metric_opts:
            opt_ = self.metric_opts["ssim"]
            metric_values["ssim"] = self._metric_value_list(
                calculate_ssim_batch_pytorch(
                    sr,
                    gt,
                    crop_border=int(opt_.get("crop_border", 0)),
                    test_y_channel=bool(opt_.get("test_y_channel", False)),
                )
            )

        if "lpips" in self.metric_opts and self.lpips_metric is not None:
            opt_ = self.metric_opts["lpips"]
            crop_border = int(opt_.get("crop_border", 0))
            sr_eval = sr
            gt_eval = gt
            if crop_border > 0:
                sr_eval = sr_eval[:, :, crop_border:-crop_border, crop_border:-crop_border]
                gt_eval = gt_eval[:, :, crop_border:-crop_border, crop_border:-crop_border]
            with torch.no_grad():
                lpips_vals = self.lpips_metric(sr_eval, gt_eval).view(-1)
            metric_values["lpips"] = self._metric_value_list(lpips_vals)

        return metric_values

    def _build_ocr_requests(self, sr_images, annotations_per_image, annotation_type):
        requests = []
        for sample_idx, annotations in enumerate(annotations_per_image):
            image = sr_images[sample_idx]
            height, width = image.shape[:2]
            language = self.languages[sample_idx]
            for anno in annotations:
                if annotation_type == "line":
                    raw_box = polygon_to_box(anno["polygon"])
                else:
                    raw_box = anno["box"]
                expanded_box = expand_and_clip_box(raw_box, width, height, self.ocr_expand_pixels)
                crop = crop_bgr_region(image, expanded_box)
                requests.append(
                    {
                        "sample_idx": sample_idx,
                        "annotation_type": annotation_type,
                        "crop": crop,
                        "gt_text": anno["text"],
                        "language": language,
                        "raw_box": raw_box,
                        "expanded_box": expanded_box,
                        "line_index": int(anno["line_index"]),
                        "word_index": None if annotation_type == "line" else int(anno["word_index"]),
                        "polygon": anno.get("polygon"),
                    }
                )
        return requests

    def _prepare_ocr_results(self, sr_images):
        line_requests = self._build_ocr_requests(sr_images, self.line_annotations, "line")
        word_requests = self._build_ocr_requests(sr_images, self.word_annotations, "word")
        all_requests = line_requests + word_requests
        predictions = self.recognizer.recognize([item["crop"] for item in all_requests], batch_size=self.ocr_batch_size)
        return all_requests, predictions

    def _init_image_result(self, batch_idx, image_metrics):
        result = {
            "image_name": self.image_names[batch_idx],
            "lq_path": self.lq_paths[batch_idx],
            "gt_path": self.gt_paths[batch_idx],
            "annotation_path": self.annotation_paths[batch_idx],
            "language": self.languages[batch_idx],
            "blur": self.blur_infos[batch_idx],
            "noise": self.noise_infos[batch_idx],
            "text_meta": self.text_meta_infos[batch_idx],
            "text_stats": self._compute_text_stats(
                self.line_annotations[batch_idx],
                self.word_annotations[batch_idx],
                self.text_meta_infos[batch_idx],
            ),
            "metrics": {},
            "ocr_summary": {},
            "ocr": {
                "lines": [],
                "words": [],
            },
        }
        for metric_name, values in image_metrics.items():
            if batch_idx < len(values):
                result["metrics"][metric_name] = float(values[batch_idx])
        return result

    def _compute_text_stats(self, line_annotations, word_annotations, text_meta):
        line_heights = []
        line_widths = []
        for anno in line_annotations:
            box = polygon_to_box(anno["polygon"])
            line_widths.append(float(box[2] - box[0]))
            line_heights.append(float(box[3] - box[1]))

        word_heights = []
        word_widths = []
        for anno in word_annotations:
            box = anno["box"]
            word_widths.append(float(box[2] - box[0]))
            word_heights.append(float(box[3] - box[1]))

        def _summary(values):
            if not values:
                return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
            values_tensor = torch.tensor(values, dtype=torch.float32)
            return {
                "count": int(values_tensor.numel()),
                "min": float(values_tensor.min().item()),
                "max": float(values_tensor.max().item()),
                "mean": float(values_tensor.mean().item()),
                "median": float(values_tensor.median().item()),
            }

        paragraph_sizes = text_meta.get("paragraph_sizes", []) if isinstance(text_meta, dict) else []
        return {
            "line_height": _summary(line_heights),
            "line_width": _summary(line_widths),
            "word_height": _summary(word_heights),
            "word_width": _summary(word_widths),
            "font_size": _summary(paragraph_sizes),
            "representative_size": {
                "avg_word_height": _summary(word_heights)["mean"],
                "avg_line_height": _summary(line_heights)["mean"],
                "fixed_font_size": text_meta.get("fixed_font_size") if isinstance(text_meta, dict) else None,
            },
        }

    def _accumulate_ocr(self, per_image_results, requests, predictions):
        ocr_totals = {
            "line_total": 0,
            "line_correct": 0,
            "line_ned_sum": 0.0,
            "word_total": 0,
            "word_correct": 0,
        }

        for request, pred in zip(requests, predictions):
            sample_idx = request["sample_idx"]
            annotation_type = request["annotation_type"]
            normalized_gt = normalize_text_for_eval(request["gt_text"], request["language"])
            normalized_pred = normalize_text_for_eval(pred["text"], request["language"])
            similarity = normalized_edit_similarity(normalized_pred, normalized_gt)
            is_correct = normalized_pred == normalized_gt

            item = {
                "line_index": request["line_index"],
                "word_index": request["word_index"],
                "raw_box": [float(v) for v in request["raw_box"]],
                "expanded_box": [int(v) for v in request["expanded_box"]],
                "polygon": request["polygon"],
                "gt_text": request["gt_text"],
                "pred_text": pred["text"],
                "gt_text_normalized": normalized_gt,
                "pred_text_normalized": normalized_pred,
                "confidence": float(pred["confidence"]),
                "ned": float(similarity),
                "correct": bool(is_correct),
            }

            if annotation_type == "line":
                per_image_results[sample_idx]["ocr"]["lines"].append(item)
                ocr_totals["line_total"] += 1
                ocr_totals["line_correct"] += int(is_correct)
                ocr_totals["line_ned_sum"] += similarity
            else:
                per_image_results[sample_idx]["ocr"]["words"].append(item)
                ocr_totals["word_total"] += 1
                ocr_totals["word_correct"] += int(is_correct)

        for result in per_image_results:
            lines = result["ocr"]["lines"]
            words = result["ocr"]["words"]
            result["metrics"]["line_ned"] = (
                float(sum(item["ned"] for item in lines) / len(lines)) if lines else None
            )
            result["metrics"]["line_accuracy"] = (
                float(sum(int(item["correct"]) for item in lines) / len(lines)) if lines else None
            )
            result["metrics"]["word_accuracy"] = (
                float(sum(int(item["correct"]) for item in words) / len(words)) if words else None
            )
            result["ocr_summary"] = {
                "line_count": len(lines),
                "line_correct_count": int(sum(int(item["correct"]) for item in lines)),
                "word_count": len(words),
                "word_correct_count": int(sum(int(item["correct"]) for item in words)),
            }

        return ocr_totals

    def _save_ocr_results(self, per_image_results, ocr_dir):
        for result in per_image_results:
            output_path = osp.join(ocr_dir, f"{result['image_name']}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                f.write("\n")

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        use_pbar = self.opt["val"].get("pbar", False)
        with_metrics = bool(self.metric_opts)
        base_dir, image_dir, ocr_dir = self._make_output_dirs(dataset_name)

        if with_metrics:
            self.metric_results = {metric: 0.0 for metric in self.metric_opts.keys()}
            self.metric_results.setdefault("line_ned", 0.0)
            self.metric_results.setdefault("line_accuracy", 0.0)
            self.metric_results.setdefault("word_accuracy", 0.0)
            self._initialize_best_metric_results(dataset_name)
            for metric_name in self.metric_results.keys():
                if metric_name not in self.best_metric_results[dataset_name]:
                    self.best_metric_results[dataset_name][metric_name] = {
                        "better": "higher",
                        "val": float("-inf"),
                        "iter": -1,
                    }

        dataset_results = []
        num_images = 0
        ocr_totals = {
            "line_total": 0,
            "line_correct": 0,
            "line_ned_sum": 0.0,
            "word_total": 0,
            "word_correct": 0,
        }

        pbar = tqdm(total=len(dataloader), unit="batch") if use_pbar else None
        for batch_idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            sr = self.output.detach().clamp_(0, 1)
            gt = self.gt.detach().clamp_(0, 1)
            image_metrics = self._compute_image_metrics(sr, gt)
            sr_images = [tensor2img(sr[i]) for i in range(sr.shape[0])]

            if save_img:
                self._save_sr_images(sr_images, image_dir)

            per_image_results = [
                self._init_image_result(i, image_metrics) for i in range(sr.shape[0])
            ]
            requests, predictions = self._prepare_ocr_results(sr_images)
            batch_ocr_totals = self._accumulate_ocr(per_image_results, requests, predictions)

            if self.save_ocr:
                self._save_ocr_results(per_image_results, ocr_dir)

            dataset_results.extend(per_image_results)
            num_images += sr.shape[0]
            for key in ("line_total", "line_correct", "word_total", "word_correct"):
                ocr_totals[key] += batch_ocr_totals[key]
            ocr_totals["line_ned_sum"] += batch_ocr_totals["line_ned_sum"]

            for metric_name, values in image_metrics.items():
                self.metric_results[metric_name] += float(sum(values))

            del self.lq
            del self.gt
            del self.output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Test {self.image_names[0]}")

        if pbar is not None:
            pbar.close()

        if with_metrics and num_images > 0:
            for metric_name in list(self.metric_results.keys()):
                if metric_name in {"line_ned", "line_accuracy", "word_accuracy"}:
                    continue
                self.metric_results[metric_name] /= float(num_images)

            self.metric_results["line_ned"] = (
                float(ocr_totals["line_ned_sum"] / ocr_totals["line_total"]) if ocr_totals["line_total"] > 0 else 0.0
            )
            self.metric_results["line_accuracy"] = (
                float(ocr_totals["line_correct"] / ocr_totals["line_total"]) if ocr_totals["line_total"] > 0 else 0.0
            )
            self.metric_results["word_accuracy"] = (
                float(ocr_totals["word_correct"] / ocr_totals["word_total"]) if ocr_totals["word_total"] > 0 else 0.0
            )

            for metric_name, value in self.metric_results.items():
                self._update_best_metric_result(dataset_name, metric_name, value, current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if self.save_summary:
            summary = OrderedDict(
                dataset=dataset_name,
                num_images=num_images,
                metrics={key: float(value) for key, value in getattr(self, "metric_results", {}).items()},
                ocr_totals=ocr_totals,
                results=dataset_results,
            )
            with open(osp.join(base_dir, f"summary_{self._get_suffix()}.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                f.write("\n")
