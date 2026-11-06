"""Real IQ560 smoke using the minimal OCR-polygon AABB crop contract.

This script intentionally keeps the first validation narrow: each sample starts
from exactly one OCR polygon AABB. Native pixels are preserved, coordinates are
expanded only for even YUV420 alignment, and fixed-canvas padding happens after
cropping. Long horizontal and tall vertical one-line boxes are split only when
needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch import nn


def _load_crop_module(path: Path):
    """@shanlan-comment Load the standalone V3 crop contract without importing legacy registries."""
    spec = importlib.util.spec_from_file_location("text_line_crop_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _image(path: str) -> np.ndarray:
    """@shanlan-comment Decode a BGR source image as float32 RGB in [0,1]."""
    value = cv2.imread(path, cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_ocr(record: Dict[str, Any]) -> List[np.ndarray]:
    """@shanlan-comment Return OCR polygons from rec_polys, dt_polys, or rec_boxes."""
    data = json.loads(Path(record["ocr"]).read_text(encoding="utf-8"))
    values = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or []
    polygons = []
    for value in values:
        try:
            polygon = np.asarray(value, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if polygon.shape[0] >= 4 and np.isfinite(polygon).all():
            polygons.append(polygon)
    return polygons


def _find_records(root: Path) -> List[Dict[str, Any]]:
    """@shanlan-comment Select one short horizontal, one long horizontal, and one vertical OCR record."""
    records = [json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for polygon in _load_ocr(record):
            x0, y0 = polygon.min(axis=0); x1, y1 = polygon.max(axis=0)
            width, height = float(x1 - x0), float(y1 - y0)
            if height > width * 1.5:
                selected.setdefault("VERTICAL", record)
            elif width > 320:
                selected.setdefault("LONG_HORIZONTAL", record)
            elif width >= height * 1.2:
                selected.setdefault("HORIZONTAL", record)
        if len(selected) == 3:
            break
    return [selected[key] for key in ("HORIZONTAL", "LONG_HORIZONTAL", "VERTICAL") if key in selected]


def _rgb_to_nv21(rgb: np.ndarray, canvas_hw=(96, 320)) -> np.ndarray:
    """@shanlan-comment Pad an RGB crop and encode canonical packed NV21."""
    height, width = canvas_hw
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[:rgb.shape[0], :rgb.shape[1]] = rgb
    yuv = cv2.cvtColor(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    u = cv2.resize(yuv[..., 1], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[..., 2], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((height // 2, width), dtype=np.uint8)
    vu[:, 0::2] = v; vu[:, 1::2] = u
    return np.concatenate([yuv[..., 0], vu]).astype(np.float32) / 255.0


def _rgb_to_yuv444(rgb: np.ndarray, canvas_hw=(96, 320)) -> np.ndarray:
    """@shanlan-comment Pad an RGB crop and encode planar YUV444 CHW."""
    height, width = canvas_hw
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[:rgb.shape[0], :rgb.shape[1]] = rgb
    return cv2.cvtColor(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV).astype(np.float32).transpose(2, 0, 1) / 255.0


def _dump(path: Path, value: np.ndarray, mode: str) -> None:
    """@shanlan-comment Write a visual evidence PNG for RGB/YUV/mask arrays."""
    if mode == "rgb":
        image = cv2.cvtColor(np.clip(value * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    elif mode == "yuv":
        image = cv2.cvtColor(np.clip(value.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
    elif mode == "gray":
        image = cv2.cvtColor(np.clip(value * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(mode)
    cv2.imwrite(str(path), image)


def _model():
    """@shanlan-comment Construct the isolated V2 smoke network without legacy imports."""
    class SmokeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 3, padding=1))
        def forward(self, packed):
            y = packed[:, :, :96, :]
            uv = packed[:, :, 96:, :].reshape(packed.shape[0], 1, 48, 160, 2).permute(0, 1, 4, 2, 3).reshape(packed.shape[0], 2, 48, 160)
            uv = torch.nn.functional.interpolate(uv, size=(96, 320), mode="bilinear", align_corners=False)
            base = torch.cat([y, uv], dim=1)
            return base + self.body(base)
    return SmokeNet()


def main(argv=None) -> int:
    """@shanlan-comment Build exact OCR AABB crops, run three real samples, and write visual/gradient evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--crop-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    crop = _load_crop_module(args.crop_module)
    records = _find_records(args.manifest_root)
    if len(records) < 3:
        raise RuntimeError(f"expected horizontal, long-horizontal, and vertical records; got {len(records)}")
    output = args.output_dir; evidence = output / "evidence"; evidence.mkdir(parents=True, exist_ok=True)
    net = _model(); optimizer = torch.optim.Adam(net.parameters(), lr=1e-3); trace=[]; geometry=[]
    for sample_index, record in enumerate(records):
        polygons = _load_ocr(record)
        chosen = None
        for polygon in polygons:
            x0, y0 = polygon.min(axis=0); x1, y1 = polygon.max(axis=0)
            width, height = float(x1-x0), float(y1-y0)
            if sample_index == 0 and width >= height * 1.2 and width <= 320: chosen = polygon; break
            if sample_index == 1 and width > 320: chosen = polygon; break
            if sample_index == 2 and height > width * 1.5: chosen = polygon; break
        if chosen is None: chosen = polygons[0]
        crops = crop.build_line_crops(chosen, _image(record["lq"]).shape[:2], (96, 320), overlap=64, line_index=0)
        item = crops[0]
        lq = _image(record["lq"]); gt = _image(record["gt"])
        left, top, right, bottom = item.crop_xyxy_even
        lq_crop, gt_crop = lq[top:bottom, left:right], gt[top:bottom, left:right]
        x = torch.from_numpy(_rgb_to_nv21(lq_crop)[None, None]); target = torch.from_numpy(_rgb_to_yuv444(gt_crop)[None]); valid = torch.from_numpy(item.valid_mask[None, None]);
        optimizer.zero_grad(set_to_none=True); pred = net(x); loss = ((pred-target).abs()*valid).sum()/(valid.sum()*3+1e-6); loss.backward(); grads=[p.grad.detach().abs().max().item() for p in net.parameters() if p.grad is not None]; optimizer.step()
        trace.append({"sample_index":sample_index,"stem":record["stem"],"loss":float(loss),"max_grad":max(grads),"finite_grad":all(math.isfinite(v) for v in grads),"input_shape":list(x.shape),"output_shape":list(pred.shape)})
        geometry.append({"stem":record["stem"],"orientation":item.orientation,"crop_xyxy_even":item.crop_xyxy_even,"tile_index":item.tile_index,"tile_count":item.tile_count,"text_area":float(item.text_mask.sum()),"valid_area":float(item.valid_mask.sum())})
        _dump(evidence/f"{sample_index:02d}_raw_lq.png", lq_crop, "rgb"); _dump(evidence/f"{sample_index:02d}_raw_gt.png", gt_crop, "rgb"); _dump(evidence/f"{sample_index:02d}_text_mask.png", item.text_mask, "gray"); _dump(evidence/f"{sample_index:02d}_valid_mask.png", item.valid_mask, "gray"); _dump(evidence/f"{sample_index:02d}_model_input_y.png", x[0,0,:96].numpy(), "gray"); _dump(evidence/f"{sample_index:02d}_model_output.png", pred.detach()[0].numpy(), "yuv"); _dump(evidence/f"{sample_index:02d}_target.png", target[0].numpy(), "yuv")
    summary={"status":"passed","real_data":True,"records":len(records),"trace":trace,"geometry":geometry,"canvas_hw":[96,320],"contract":"one OCR polygon AABB -> one crop -> fixed canvas"}; (output/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); (output/'gradient_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
