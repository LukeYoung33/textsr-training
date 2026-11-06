"""Run a fixed V4 checkpoint on real_finetune_v2 samples with stroke-mask regions.

This is an inference-only diagnostic.  Each original line is transformed with
the V4 training geometry (aspect resize to height 96 plus right edge padding),
then LQ, prediction and GT are compared in exact mask-derived regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB


CANVAS_HW = (96, 480)


def read_rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def prepare_rgb(value: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = value.shape[:2]
    resized_width = max(2, int(round(width * CANVAS_HW[0] / height)))
    resized_width += resized_width % 2
    if resized_width > CANVAS_HW[1]:
        raise ValueError(f"sample becomes {resized_width}px wide; tiled inference is not implemented")
    resized = cv2.resize(value, (resized_width, CANVAS_HW[0]), interpolation=cv2.INTER_AREA)
    result = np.pad(resized, ((0, 0), (0, CANVAS_HW[1] - resized_width), (0, 0)), mode="edge")
    return np.ascontiguousarray(result, dtype=np.float32), resized_width


def prepare_mask(path: Path, valid_width: int) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    # Dataset masks encode strokes in white on black. Nearest resize preserves topology.
    mask = (value > 127).astype(np.uint8)
    resized = cv2.resize(mask, (valid_width, CANVAS_HW[0]), interpolation=cv2.INTER_NEAREST)
    result = np.zeros(CANVAS_HW, dtype=np.uint8)
    result[:, :valid_width] = resized
    return result


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint has no state mapping")
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model = BasicTextLineR4StudentY4UVV4(haar_switch=True).to(device).eval()
    model.load_state_dict(state, strict=True)
    return model


def to_rgb(native: torch.Tensor) -> np.ndarray:
    value = YUV2RGB(y4uv_to_yuv444(native), format="RGB", is_train=True, uv_normalized=True)
    return value[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def luma(value: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(value.astype(np.float32), cv2.COLOR_RGB2GRAY)


def highpass(value: np.ndarray) -> np.ndarray:
    y = luma(value)
    return y - cv2.GaussianBlur(y, (0, 0), 1.0)


def region_masks(strokes: np.ndarray, valid_width: int) -> dict[str, np.ndarray]:
    valid = np.zeros(CANVAS_HW, dtype=bool)
    valid[:, :valid_width] = True
    kernel = np.ones((3, 3), dtype=np.uint8)
    interior = cv2.erode(strokes, kernel, iterations=1).astype(bool)
    boundary = strokes.astype(bool) & ~interior
    near_gap = cv2.dilate(strokes, kernel, iterations=2).astype(bool) & ~strokes.astype(bool)
    return {
        "stroke_interior": interior & valid,
        "stroke_boundary": boundary & valid,
        "near_stroke_gap": near_gap & valid,
        "far_background": valid & ~cv2.dilate(strokes, kernel, iterations=2).astype(bool),
    }


def metrics(lq: np.ndarray, pred: np.ndarray, gt: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    lq_y, pred_y, gt_y = luma(lq), luma(pred), luma(gt)
    lq_hp, pred_hp, gt_hp = highpass(lq), highpass(pred), highpass(gt)
    result: dict[str, dict[str, float]] = {}
    for name, mask in regions.items():
        if not mask.any():
            continue
        lq_mae = float(np.abs(lq_y[mask] - gt_y[mask]).mean())
        pred_mae = float(np.abs(pred_y[mask] - gt_y[mask]).mean())
        lq_hf_mae = float(np.abs(lq_hp[mask] - gt_hp[mask]).mean())
        pred_hf_mae = float(np.abs(pred_hp[mask] - gt_hp[mask]).mean())
        result[name] = {
            "pixels": int(mask.sum()),
            "lq_y_mae": lq_mae,
            "pred_y_mae": pred_mae,
            "y_error_reduction": float((lq_mae - pred_mae) / max(lq_mae, 1e-8)),
            "lq_highpass_mae": lq_hf_mae,
            "pred_highpass_mae": pred_hf_mae,
            "highpass_error_reduction": float((lq_hf_mae - pred_hf_mae) / max(lq_hf_mae, 1e-8)),
        }
    return result


def write_montage(path: Path, lq: np.ndarray, pred: np.ndarray, gt: np.ndarray, strokes: np.ndarray, valid_width: int) -> None:
    # Inference/deployment output is constrained to the displayable RGB range.
    # Keep training losses unconstrained elsewhere; this function is evaluation-only.
    lq, pred, gt = (np.clip(value, 0.0, 1.0) for value in (lq, pred, gt))
    # Right-side V4 padding is deliberately excluded by valid_mask during
    # training. Crop it from diagnostic figures so unconstrained pixels are
    # not mistaken for restored text.
    lq, pred, gt = (value[:, :valid_width] for value in (lq, pred, gt))
    strokes = strokes[:, :valid_width]
    panels = []
    for name, image in (("LQ", lq), ("PRED", pred), ("GT", gt)):
        panel = cv2.cvtColor((image * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(panel, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    mask_panel = np.zeros((96, valid_width, 3), dtype=np.uint8)
    mask_panel[strokes.astype(bool)] = (255, 255, 255)
    cv2.putText(mask_panel, "STRUCTURE MASK", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1, cv2.LINE_AA)
    panels.append(mask_panel)
    error = np.abs(luma(pred) - luma(gt))
    error = cv2.applyColorMap(np.clip(error / max(float(np.percentile(error, 99)), 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.putText(error, "PRED-GT Y ERROR", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(error)
    cv2.imwrite(str(path), np.vstack(panels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    all_results = {}
    for stem in args.ids:
        lq, valid_width = prepare_rgb(read_rgb(args.root / "lq" / args.split / f"{stem}.png"))
        gt, gt_width = prepare_rgb(read_rgb(args.root / "gt" / args.split / f"{stem}.png"))
        if valid_width != gt_width:
            raise RuntimeError(f"LQ/GT geometry differs for {stem}")
        strokes = prepare_mask(args.root / "structure" / args.split / f"{stem}.png", valid_width)
        native = rgb_to_y4uv(torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).to(device))
        with torch.inference_mode():
            pred = to_rgb(model(native))
        regions = region_masks(strokes, valid_width)
        item = {"valid_width": valid_width, "regions": metrics(lq, pred, gt, regions)}
        all_results[stem] = item
        write_montage(args.out / f"{stem}_montage.png", lq, pred, gt, strokes, valid_width)
    (args.out / "summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
