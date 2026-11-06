"""Create a fixed three-variant Lenovo-degradation validation set from real_finetune_v2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from basicsr.data import build_dataset
from basicsr.utils import DiffJPEG
from realesrgan.lenovo_degradation import DEFAULT_LENOVO_DEGRADATION_CONFIG, LenovoDegradationPipeline


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("validation dataset is empty")
    if not 1 <= count <= length:
        raise ValueError(f"sample_count must be in [1, {length}], got {count}")
    return [round(index * (length - 1) / max(count - 1, 1)) for index in range(count)]


def _save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(path)


def _height_fit_pad_crop(source_path: str, canvas_hw: tuple[int, int], seed: int,
                         horizontal_stretch: tuple[float, float] = (.96, 1.04)) -> tuple[torch.Tensor, dict]:
    """Fit a source line by height, with only a small horizontal perturbation.

    Width is then center-cropped or edge-padded to the requested canvas.  This
    keeps glyph aspect ratios intact apart from the explicitly bounded stretch.
    """
    target_h, target_w = canvas_hw
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        source_w, source_h = source.size
        stretch = float(np.random.default_rng(seed).uniform(*horizontal_stretch))
        resized_w = max(2, int(round(source_w * target_h / source_h * stretch)))
        if resized_w % 2:
            resized_w += 1
        resized = source.resize((resized_w, target_h), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if resized_w >= target_w:
        crop_left = (resized_w - target_w) // 2
        output = array[:, crop_left:crop_left + target_w]
        pad_ltrb = [0, 0, 0, 0]
    else:
        crop_left = 0
        pad_left = (target_w - resized_w) // 2
        pad_right = target_w - resized_w - pad_left
        output = np.empty((target_h, target_w, 3), dtype=np.float32)
        output[:, pad_left:pad_left + resized_w] = array
        # Edge padding follows the local background instead of adding an
        # artificial black or white band around the source line.
        output[:, :pad_left] = array[:, :1]
        output[:, pad_left + resized_w:] = array[:, -1:]
        pad_ltrb = [pad_left, 0, pad_right, 0]
    metadata = {
        "mode": "height_fit_small_horizontal_stretch_pad_or_crop",
        "source_hw": [source_h, source_w],
        "output_hw": [target_h, target_w],
        "height_scale": target_h / float(source_h),
        "horizontal_stretch": stretch,
        "resized_hw": [target_h, resized_w],
        "crop_left": crop_left,
        "pad_ltrb": pad_ltrb,
    }
    return torch.from_numpy(np.ascontiguousarray(output)).permute(2, 0, 1), metadata


def _save_contact_sheet(gt_path: Path, lq_paths: list[Path], labels: list[str], path: Path) -> None:
    """Save a compact, directly reviewable GT/LQ comparison for one source line."""
    images = [Image.open(gt_path).convert("RGB")] + [Image.open(item).convert("RGB") for item in lq_paths]
    width = max(image.width for image in images)
    label_height = 24
    sheet = Image.new("RGB", (width, sum(image.height + label_height for image in images)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for image, label in zip(images, ["GT"] + labels):
        sheet.paste(image, (0, y + label_height))
        draw.text((4, y + 4), label, fill="black")
        y += image.height + label_height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _seed(source_id: str, variant: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{source_id}|{variant}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _variant_degradation(base: dict, variant: int, variants: int, profile_set: str,
                         selection_seed: int | None = None) -> tuple[str, dict]:
    """Return four target-aligned validation profiles rather than random operator samples."""
    resolved = copy.deepcopy(DEFAULT_LENOVO_DEGRADATION_CONFIG)
    for key, value in base.items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key].update(value)
        else:
            resolved[key] = value
    if profile_set == "test_composed":
        if variants != 3:
            raise ValueError("test_composed requires exactly three variants")
        config = resolved
        composition = config["composition"]
        composition.update({"secondary_capture_probability": 0.0, "max_post_branches": 2})
        if variant == 0:
            # Bright text on a dark background: no dark-stroke chroma noise is meaningful here.
            composition.update({
                "capture_weights": {"c1": 0.0, "c2": 0.0, "c1_c2": 1.0, "zoom_c2": 0.0},
                "post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 1.0, "c7": 1.0},
                "forced_c1_subbranch": "defocus",
            })
            config["c1"]["defocus_radius"] = [2, 3]
            config["c2"].update({"psf_sigma": [.65, .95], "sample_scale": [.50, .62], "phase_px": .50})
            config["c7"].update({"shift_x": [.7, 1.5], "shift_y": [.2, .9], "weight": [.10, .20]})
            config["c5"].update({"usm_sigma": [2.0, 3.2], "usm_amount": [.45, .70], "jpeg_quality": [84, 92]})
            return "endpoint_fade_sampling_ghost_halo", config
        if variant == 1:
            # Dark Chinese strokes: retain a clean plane and place chroma noise only inside strokes.
            composition.update({
                "capture_weights": {"c1": 0.0, "c2": 0.0, "c1_c2": 1.0, "zoom_c2": 0.0},
                "post_probabilities": {"c3": 1.0, "c4": 0.0, "c5": 0.0, "c7": 0.0},
                "forced_c1_subbranch": "defocus",
            })
            config["c1"]["defocus_radius"] = [2, 3]
            config["c2"].update({"psf_sigma": [.55, .85], "sample_scale": [.54, .66], "phase_px": .48})
            config["c3"]["strength"] = [6.0, 10.0]
            return "dark_stroke_midfreq_uv_residue_sampling", config
        # A high-chroma text/background pair: blur UV after capture to spread existing chroma.
        composition.update({
            "capture_weights": {"c1": 0.0, "c2": 0.0, "c1_c2": 1.0, "zoom_c2": 0.0},
            "post_probabilities": {"c3": 0.0, "c4": 1.0, "c5": 0.0, "c7": 0.0},
            "forced_c1_subbranch": "gaussian",
        })
        # RGB blur spreads white or dark foreground luminance; C4 then spreads only its chroma.
        config["c1"]["gaussian_sigma"] = [.85, 1.20]
        config["c2"].update({"psf_sigma": [.60, .90], "sample_scale": [.56, .66], "phase_px": .40})
        config["c4"].update({"uv_blur_sigma": [3.5, 5.5], "midfreq_strength": [.5, 1.5]})
        return "foreground_chroma_spread_sampling", config
    if profile_set == "random_strong":
        # The profile choice is a deterministic hash of this LQ item.  Parameter
        # values remain independently random inside each profile through its
        # torch Generator, so the set is fixed yet not visually templated.
        config = resolved
        composition = config["composition"]
        composition.update({"secondary_capture_probability": 0.0})
        choice = int(selection_seed or 0) % 8
        if choice == 0:
            subbranch = ("gaussian", "motion", "zoom", "defocus")[int(selection_seed or 0) // 8 % 4]
            composition.update({"capture_weights": {"c1": 1.0, "c2": 0.0, "c1_c2": 0.0, "zoom_c2": 0.0},
                                "post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 0.0, "c7": 0.0},
                                "max_post_branches": 0, "forced_c1_subbranch": subbranch})
            config["c1"].update({"gaussian_sigma": [1.0, 1.8], "motion_length": [7, 13],
                                 "defocus_radius": [3, 5], "zoom_source_scale": [.32, .52]})
            return f"C1__strong_{subbranch}_blur", config
        composition.update({"capture_weights": {"c1": 0.0, "c2": 0.0, "c1_c2": 1.0, "zoom_c2": 0.0}})
        if choice == 1:
            composition.update({"post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 0.0, "c7": 0.0},
                                "max_post_branches": 0, "forced_c1_subbranch": "defocus"})
            config["c1"]["defocus_radius"] = [3, 5]
            config["c2"].update({"psf_sigma": [1.0, 1.45], "sample_scale": [.38, .54], "phase_px": .70})
            return "C1+C2__strong_endpoint_fade_sampling", config
        if choice == 2:
            composition.update({"post_probabilities": {"c3": 1.0, "c4": 0.0, "c5": 0.0, "c7": 0.0},
                                "max_post_branches": 1, "forced_c1_subbranch": "defocus"})
            config["c1"]["defocus_radius"] = [3, 4]
            config["c2"].update({"psf_sigma": [.85, 1.25], "sample_scale": [.42, .58], "phase_px": .62})
            config["c3"].update({"strength": [10.0, 15.0], "sigmas": [1.5, 4.0, 9.0], "scale": .5, "rho": -.75})
            return "C1+C2+C3__dark_stroke_edge_protected_uv_blotches", config
        if choice == 3:
            composition.update({"post_probabilities": {"c3": 0.0, "c4": 1.0, "c5": 0.0, "c7": 0.0},
                                "max_post_branches": 1, "forced_c1_subbranch": "gaussian"})
            config["c1"]["gaussian_sigma"] = [1.0, 1.5]
            config["c2"].update({"psf_sigma": [.75, 1.20], "sample_scale": [.44, .60], "phase_px": .60})
            config["c4"].update({"uv_blur_sigma": [5.0, 8.0], "midfreq_strength": [1.0, 2.4]})
            return "C1+C2+C4__wide_foreground_chroma_spread", config
        if choice == 4:
            composition.update({"post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 1.0, "c7": 0.0},
                                "max_post_branches": 1, "forced_c1_subbranch": "gaussian"})
            config["c1"]["gaussian_sigma"] = [.8, 1.4]
            config["c2"].update({"psf_sigma": [.75, 1.15], "sample_scale": [.45, .60], "phase_px": .55})
            config["c5"].update({"usm_sigma": [2.5, 4.2], "usm_amount": [.60, .95], "jpeg_quality": [76, 88]})
            return "C1+C2+C5__wide_edge_band_and_codec", config
        if choice == 5:
            composition.update({"post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 0.0, "c7": 1.0},
                                "max_post_branches": 1, "forced_c1_subbranch": "defocus"})
            config["c1"]["defocus_radius"] = [2, 4]
            config["c2"].update({"psf_sigma": [.70, 1.10], "sample_scale": [.48, .62], "phase_px": .52})
            config["c7"].update({"shift_x": [1.2, 2.8], "shift_y": [.4, 1.4], "weight": [.16, .30]})
            return "C1+C2+C7__directional_chroma_ghosting", config
        if choice == 6:
            composition.update({"post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 0.0, "c7": 0.0},
                                "max_post_branches": 0, "secondary_capture_probability": 1.0,
                                "forced_c1_subbranch": "gaussian"})
            config["c1"]["gaussian_sigma"] = [.75, 1.25]
            config["c2"].update({"psf_sigma": [.70, 1.05], "sample_scale": [.48, .62], "phase_px": .50})
            config["c8"].update({"display_scale": [.42, .58], "display_jpeg_quality": [82, 91],
                                 "camera_blur_sigma": [.7, 1.2], "final_jpeg_quality": [76, 87],
                                 "perspective_fraction": [.002, .010]})
            return "C1+C2+C8__secondary_display_capture", config
        composition.update({"post_probabilities": {"c3": 1.0, "c4": 0.0, "c5": 1.0, "c7": 0.0},
                            "max_post_branches": 2, "forced_c1_subbranch": "defocus"})
        config["c1"]["defocus_radius"] = [2, 4]
        config["c2"].update({"psf_sigma": [.85, 1.25], "sample_scale": [.42, .58], "phase_px": .60})
        config["c3"].update({"strength": [8.0, 13.0], "sigmas": [1.5, 4.0, 9.0], "scale": .5, "rho": -.75})
        config["c5"].update({"usm_sigma": [2.2, 3.8], "usm_amount": [.45, .75], "jpeg_quality": [78, 89]})
        return "C1+C2+C3+C5__uv_blotches_with_codec_edge_band", config
    if profile_set == "random_2_to_5":
        # One LQ item receives an independently sampled 2--5 atom chain.
        # C1/C2 capture contributes one or two atoms, up to two post stages are
        # selected at random, and the secondary capture can contribute C8.
        # The pipeline's enforcement makes a single-atom capture acquire one
        # post stage when its random post draw is empty.
        config = resolved
        composition = config["composition"]
        composition.update({
            # All samples pass C2 so the set has a visible optical/sampling
            # loss floor; most also receive a randomly selected C1 blur.
            "capture_weights": {"c1": 0.0, "c2": .16, "c1_c2": .54, "zoom_c2": .30},
            # C5 is deliberately common: wide dark/bright edge bands are a
            # recurring property of the target test images.
            "post_probabilities": {"c3": .30, "c4": .26, "c5": .85, "c7": .30},
            "max_post_branches": 2,
            "secondary_capture_probability": .42,
            "enforce_problem_count": True,
            "forced_c1_subbranch": None,
        })
        config["c1"].update({"gaussian_sigma": [1.10, 2.15], "motion_length": [8, 16],
                             "defocus_radius": [3, 6], "zoom_source_scale": [.26, .48]})
        config["c2"].update({"psf_sigma": [1.15, 1.90], "sample_scale": [.30, .50], "phase_px": .78})
        config["c3"].update({"strength": [8.0, 14.0], "sigmas": [1.5, 4.0, 9.0], "scale": .5, "rho": -.75})
        config["c4"].update({"uv_blur_sigma": [5.0, 9.0], "midfreq_strength": [1.0, 2.8]})
        config["c5"].update({"usm_sigma": [3.5, 6.0], "usm_amount": [.80, 1.35], "jpeg_quality": [68, 84]})
        config["c7"].update({"shift_x": [1.4, 3.4], "shift_y": [.5, 1.8], "weight": [.18, .34]})
        config["c8"].update({"display_scale": [.34, .54], "display_jpeg_quality": [78, 90],
                             "camera_blur_sigma": [.80, 1.45], "final_jpeg_quality": [72, 86],
                             "perspective_fraction": [.001, .009]})
        return "random_2_to_5_strong", config
    if profile_set == "fixed_strong":
        if variants != 3:
            raise ValueError("fixed_strong requires exactly three variants")
        config = resolved
        composition = config["composition"]
        composition.update({
            "capture_weights": {"c1": 0.0, "c2": 0.0, "c1_c2": 1.0, "zoom_c2": 0.0},
            "secondary_capture_probability": 0.0,
        })
        if variant == 0:
            # A strong but still physically motivated thin-stroke failure: defocus, sampling and a mild capture halo.
            composition.update({
                "post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 1.0, "c7": 1.0},
                "max_post_branches": 2,
                "forced_c1_subbranch": "defocus",
            })
            config["c1"]["defocus_radius"] = [3, 4]
            config["c2"].update({"psf_sigma": [1.00, 1.35], "sample_scale": [.40, .52], "phase_px": .65})
            config["c7"].update({"shift_x": [1.0, 2.0], "shift_y": [.3, 1.1], "weight": [.14, .24]})
            config["c5"].update({"usm_sigma": [2.4, 3.8], "usm_amount": [.55, .85], "jpeg_quality": [80, 89]})
            return "strong_endpoint_fade_sampling_ghost_halo", config
        if variant == 1:
            # Edge-protected UV blotches appear on dark strokes while clean flat background stays neutral.
            composition.update({
                "post_probabilities": {"c3": 1.0, "c4": 0.0, "c5": 0.0, "c7": 0.0},
                "max_post_branches": 1,
                "forced_c1_subbranch": "defocus",
            })
            config["c1"]["defocus_radius"] = [3, 4]
            config["c2"].update({"psf_sigma": [.90, 1.20], "sample_scale": [.44, .56], "phase_px": .58})
            config["c3"].update({"strength": [11.0, 15.0], "sigmas": [1.5, 4.0, 9.0], "scale": .5, "rho": -.75})
            return "strong_dark_stroke_edge_protected_uv_blotches", config
        # Deliberately wide UV spreading.  It only becomes pronounced where the source has chroma,
        # so it remains an ISP-like chroma process rather than tinting neutral paper.
        composition.update({
            "post_probabilities": {"c3": 0.0, "c4": 1.0, "c5": 1.0, "c7": 0.0},
            "max_post_branches": 2,
            "forced_c1_subbranch": "gaussian",
        })
        config["c1"]["gaussian_sigma"] = [1.05, 1.45]
        config["c2"].update({"psf_sigma": [.85, 1.20], "sample_scale": [.46, .58], "phase_px": .55})
        config["c4"].update({"uv_blur_sigma": [5.5, 8.0], "midfreq_strength": [1.2, 2.4]})
        config["c5"].update({"usm_sigma": [2.2, 3.4], "usm_amount": [.35, .60], "jpeg_quality": [78, 88]})
        return "strong_wide_foreground_chroma_spread", config
    if variants != 4:
        return "random_composed", resolved
    config = resolved
    composition = config["composition"]
    composition.update({
        "capture_weights": {"c1": 0.0, "c2": 0.0, "c1_c2": 1.0, "zoom_c2": 0.0},
        "secondary_capture_probability": 0.0,
    })
    if variant == 0:
        composition.update({"post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 0.0, "c7": 0.0}, "max_post_branches": 0,
                            "forced_c1_subbranch": "defocus"})
        config["c1"]["defocus_radius"] = [2, 3]
        config["c2"].update({"psf_sigma": [.70, 1.00], "sample_scale": [.48, .60], "phase_px": .55})
        return "thin_stroke_endpoint_fade", config
    if variant == 1:
        composition.update({"post_probabilities": {"c3": 1.0, "c4": 0.0, "c5": 0.0, "c7": 0.0}, "max_post_branches": 1,
                            "forced_c1_subbranch": "gaussian"})
        config["c1"]["gaussian_sigma"] = [.55, .95]
        config["c2"].update({"psf_sigma": [.50, .80], "sample_scale": [.58, .70], "phase_px": .45})
        config["c3"]["strength"] = [6.0, 6.0]
        return "dark_stroke_chroma_noise", config
    if variant == 2:
        composition.update({"post_probabilities": {"c3": 0.0, "c4": 1.0, "c5": 0.0, "c7": 0.0}, "max_post_branches": 1,
                            "forced_c1_subbranch": "gaussian"})
        config["c1"]["gaussian_sigma"] = [.35, .65]
        config["c2"].update({"psf_sigma": [.40, .70], "sample_scale": [.62, .74], "phase_px": .35})
        config["c4"].update({"uv_blur_sigma": [3.5, 5.5], "midfreq_strength": [0., 1.5]})
        return "colored_text_chroma_spread", config
    composition.update({
        "post_probabilities": {"c3": 0.0, "c4": 0.0, "c5": 1.0, "c7": 0.0}, "max_post_branches": 1,
        "forced_c1_subbranch": "gaussian",
    })
    config["c1"]["gaussian_sigma"] = [.25, .45]
    config["c2"].update({"psf_sigma": [.35, .60], "sample_scale": [.66, .76], "phase_px": .30})
    config["c5"].update({"usm_sigma": [2.2, 3.8], "usm_amount": [.70, 1.05], "jpeg_quality": [82, 92]})
    return "wide_black_white_edge_halo", config


def run(config_path: Path, output_root: Path, sample_count: int, variants: int, seed: int, device: str,
        source_indices: list[int] | None = None, profile_set: str = "isolated") -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # The generator reads the source GT directly below; keep the adapter in
    # aspect-fit mode as well so no validation path silently stretches lines.
    config["datasets"]["val"]["resize_mode"] = "aspect_fit"
    degradation = config["lenovo_degradation"]
    dataset = build_dataset(config["datasets"]["val"])
    indices = source_indices if source_indices is not None else _uniform_indices(len(dataset), sample_count)
    target = torch.device(device)
    jpeger = DiffJPEG(differentiable=False).to(target)

    def jpeg(image: torch.Tensor, quality: float) -> torch.Tensor:
        return jpeger(image, quality=image.new_full((image.shape[0],), quality))

    records = []
    for output_index, source_index in enumerate(indices):
        sample = dataset[source_index]
        source_id = str(sample.get("segment_id", sample.get("source_line_id", source_index)))
        safe_id = f"{output_index:03d}_{source_index:06d}"
        geometry_seed = _seed(source_id, -1, seed)
        gt_cpu, geometry = _height_fit_pad_crop(sample["gt_path"], (96, 480), geometry_seed)
        gt = gt_cpu.unsqueeze(0).to(target).float().clamp(0, 1)
        gt_path = output_root / "gt" / f"{safe_id}.png"
        _save_rgb(gt[0], gt_path)
        variants_metadata = []
        lq_paths: list[Path] = []
        labels: list[str] = []
        for variant in range(variants):
            item_seed = _seed(source_id, variant, seed)
            generator = torch.Generator(device=target).manual_seed(item_seed)
            variant_name, variant_config = _variant_degradation(degradation, variant, variants, profile_set, item_seed)
            pipeline = LenovoDegradationPipeline(variant_config, jpeg=jpeg)
            lq, metadata = pipeline(gt, generator)
            lq_path = output_root / "lq" / f"variant_{variant + 1}" / f"{safe_id}.png"
            _save_rgb(lq[0], lq_path)
            lq_paths.append(lq_path)
            category_prefix, separator, display_profile = variant_name.partition("__")
            profile = display_profile if separator else variant_name
            categories = [str(stage["branch"]).upper() for stage in metadata[0]["stages"]
                          if stage.get("branch") != "always"]
            labels.append(f"V{variant + 1}: {'+'.join(categories)} | {profile}")
            variants_metadata.append({
                "variant": variant + 1,
                "profile": profile,
                "problem_categories": categories,
                "seed": item_seed,
                "lq": str(lq_path),
                "stages": metadata[0],
            })
        contact_path = output_root / "contact_sheets" / f"{safe_id}.png"
        _save_contact_sheet(gt_path, lq_paths, labels, contact_path)
        records.append({"uniform_rank": output_index, "dataset_index": source_index, "source_id": source_id,
                        "geometry_seed": geometry_seed, "geometry": geometry, "gt": str(gt_path),
                        "contact_sheet": str(contact_path), "variants": variants_metadata})
    manifest = {
        "schema_version": "lenovo_validation_degradation_v1",
        "config": str(config_path),
        "degradation": degradation,
        "dataset_size": len(dataset),
        "uniform_sample_count": sample_count,
        "variants_per_gt": variants,
        "seed": seed,
        "geometry_policy": "height_fit_to_96; horizontal_stretch_uniform_[0.96,1.04]; center_pad_or_crop_to_480",
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    fixed_pairs = []
    for record in records:
        for variant in record["variants"]:
            fixed_pairs.append({"image": f"{record['uniform_rank']:03d}_v{variant['variant']}.png",
                                "lq": variant["lq"], "gt": record["gt"]})
    (output_root / "validation_pairs.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in fixed_pairs), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-indices", type=int, nargs="+", help="Generate specified validation indices for visual review.")
    parser.add_argument("--profile-set", choices=("isolated", "test_composed", "fixed_strong", "random_strong", "random_2_to_5"), default="isolated")
    args = parser.parse_args()
    print(run(args.config, args.output_root, args.sample_count, args.variants, args.seed, args.device,
              args.dataset_indices, args.profile_set))


if __name__ == "__main__":
    main()
