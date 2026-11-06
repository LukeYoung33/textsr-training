"""Create a deterministic, stratified visual audit set for real_finetune_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CANVAS_HW = (96, 480)


def read_rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def prepare_triplet(lq: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Use the same V4 resize/common-crop policy as the overfit experiments."""
    height, width = lq.shape[:2]
    target_w = max(2, int(round(width * CANVAS_HW[0] / height))); target_w += target_w % 2
    lq = cv2.resize(lq, (target_w, 96), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, (target_w, 96), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (target_w, 96), interpolation=cv2.INTER_NEAREST).astype(bool)
    crop_x = 0
    if target_w > 480:
        edge = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150).astype(np.float32) / 255.0
        scores = np.convolve((mask.astype(np.float32) + edge).sum(0), np.ones(480, np.float32), mode="valid")
        crop_x = int(np.argmax(scores)); crop_x -= crop_x % 2; crop_x = min(crop_x, target_w - 480); crop_x -= crop_x % 2
        lq, gt, mask = lq[:, crop_x:crop_x + 480], gt[:, crop_x:crop_x + 480], mask[:, crop_x:crop_x + 480]
    return lq, gt, mask, min(target_w, 480), crop_x, target_w


def ring(mask: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return dilated & ~mask


def stats(root: Path, stem: str) -> dict:
    lq, gt = read_rgb(root / "lq/train" / f"{stem}.png"), read_rgb(root / "gt/train" / f"{stem}.png")
    raw_mask = cv2.imread(str(root / "structure/train" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None: raise FileNotFoundError(stem)
    mask = raw_mask > 127
    y = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY); outer = ring(mask)
    fg = float(y[mask].mean()) if mask.any() else float("nan")
    bg = float(y[outer].mean()) if outer.any() else float("nan")
    return {"id": stem, "source_hw": [int(gt.shape[0]), int(gt.shape[1])], "mask_fraction": float(mask.mean()), "fg_y": fg, "ring_bg_y": bg, "contrast_bg_minus_fg": bg - fg, "long_after_v4_resize": int(round(gt.shape[1] * 96 / gt.shape[0])) > 480}


def choose(records: list[dict], count: int) -> list[dict]:
    selected: dict[str, str] = {"002652": "fixed_weak_stroke_case"}
    for key, qs, prefix in (("mask_fraction", (.05, .25, .5, .75, .95), "stroke_density"), ("contrast_bg_minus_fg", (.05, .3, .7, .95), "contrast")):
        ordered = sorted(records, key=lambda item: item[key])
        for q in qs:
            item = ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
            selected.setdefault(item["id"], f"{prefix}_q{q:.2f}")
    long = sorted((item for item in records if item["long_after_v4_resize"]), key=lambda item: item["mask_fraction"])
    for q in (.1, .5, .9):
        if long:
            item = long[min(len(long) - 1, int(q * (len(long) - 1)))]
            selected.setdefault(item["id"], f"long_crop_density_q{q:.1f}")
    result = [next(item for item in records if item["id"] == stem) | {"selection_reason": reason} for stem, reason in selected.items()]
    return result[:count]


def panel(root: Path, record: dict, out: Path) -> dict:
    stem = record["id"]; lq = read_rgb(root / "lq/train" / f"{stem}.png"); gt = read_rgb(root / "gt/train" / f"{stem}.png")
    raw_mask = cv2.imread(str(root / "structure/train" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
    lq, gt, mask, valid_w, crop_x, resized_w = prepare_triplet(lq, gt, raw_mask)
    overlay = gt.copy(); boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool); overlay[boundary] = (1.0, 0.0, 1.0)
    rows = [lq, gt, np.repeat(mask[..., None], 3, axis=2).astype(np.float32), overlay]
    labels = ["LQ (V4 crop)", "GT", "structure mask", "GT + mask boundary"]
    title = f"{stem}  raw={record['source_hw'][0]}x{record['source_hw'][1]}  valid_w={valid_w} crop_x={crop_x} resized_w={resized_w}  mask={record['mask_fraction']:.3f}  fg={record['fg_y']:.3f} bg={record['ring_bg_y']:.3f} contrast={record['contrast_bg_minus_fg']:.3f}"
    rendered = []
    for image, label in zip(rows, labels):
        bgr = cv2.cvtColor((image.clip(0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 255), 1, cv2.LINE_AA); rendered.append(bgr)
    board = np.vstack(rendered); board = cv2.copyMakeBorder(board, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)); cv2.putText(board, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / f"{stem}_audit.png"), board)
    return record | {"valid_width": valid_w, "crop_x_after_resize": crop_x, "resized_width": resized_w}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(path.stem for path in (args.root / "lq/train").glob("*.png") if (args.root / "gt/train" / path.name).is_file() and (args.root / "structure/train" / path.name).is_file())
    records = [stats(args.root, stem) for stem in stems]
    chosen = choose(records, args.count); audited = [panel(args.root, record, args.out) for record in chosen]
    report = {"dataset_root": str(args.root), "count": len(audited), "selection": audited, "note": "Mask foreground is source structure grayscale >127. Panels use the exact V4 resize/common-crop policy; no model inference is involved."}
    (args.out / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
