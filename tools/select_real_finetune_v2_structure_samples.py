"""Select diverse text-structure-mask samples and render an LQ/GT/mask sheet."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def metrics(mask_path: Path):
    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Cannot read {mask_path}")
    fg = (gray > 127).astype(np.uint8)
    edge = cv2.Canny(fg * 255, 50, 150) > 0
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nontrivial = [c for c in contours if cv2.contourArea(c) >= 2]
    return {
        "foreground_fraction": float(fg.mean()),
        "edge_fraction": float(edge.mean()),
        "components": len(nontrivial),
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
    }


def pick_diverse(items, count):
    # Use three foreground-density bands so the sheet does not contain only bold text.
    items = sorted(items, key=lambda x: (x["edge_fraction"], x["components"]), reverse=True)
    bands = [
        [x for x in items if x["foreground_fraction"] < 0.12],
        [x for x in items if 0.12 <= x["foreground_fraction"] < 0.24],
        [x for x in items if x["foreground_fraction"] >= 0.24],
    ]
    chosen, seen = [], set()
    targets = [count // 3, count // 3, count - 2 * (count // 3)]
    for band, target in zip(bands, targets):
        for item in band[:target]:
            chosen.append(item)
            seen.add(item["stem"])
    for item in items:
        if len(chosen) >= count:
            break
        if item["stem"] not in seen:
            chosen.append(item)
            seen.add(item["stem"])
    return chosen


def load_panel(path: Path, width, height):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-resized-width", type=int, default=480)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    structure_dir = args.root / "structure" / args.split
    lq_dir, gt_dir = args.root / "lq" / args.split, args.root / "gt" / args.split
    candidates = []
    for mask in sorted(structure_dir.glob("*.png")):
        if not (lq_dir / mask.name).is_file() or not (gt_dir / mask.name).is_file():
            continue
        item = metrics(mask)
        resized_width = max(2, int(round(item["width"] * 96 / item["height"])))
        resized_width += resized_width % 2
        if resized_width > args.max_resized_width:
            continue
        item["stem"] = mask.stem
        item["resized_width"] = resized_width
        candidates.append(item)
    selected = pick_diverse(candidates, args.count)
    args.out.mkdir(parents=True, exist_ok=True)
    selected_dir = args.out / "selected"
    selected_dir.mkdir(exist_ok=True)

    panel_w, panel_h, header_h = 380, 96, 25
    sheet = np.full((len(selected) * (panel_h + header_h), panel_w * 3, 3), 255, dtype=np.uint8)
    labels = [("LQ", lq_dir), ("GT", gt_dir), ("MASK", structure_dir)]
    for row, item in enumerate(selected):
        y0 = row * (panel_h + header_h)
        label = f"{item['stem']}  fg={item['foreground_fraction']:.3f} edge={item['edge_fraction']:.3f} cc={item['components']}"
        cv2.putText(sheet, label, (5, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        for col, (name, directory) in enumerate(labels):
            source = directory / f"{item['stem']}.png"
            panel = load_panel(source, panel_w, panel_h)
            cv2.putText(panel, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
            x0 = col * panel_w
            sheet[y0 + header_h:y0 + header_h + panel_h, x0:x0 + panel_w] = panel
            shutil.copy2(source, selected_dir / f"{item['stem']}_{name.lower()}.png")
    cv2.imwrite(str(args.out / "contact_sheet.png"), sheet)
    (args.out / "selection.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
