"""Render P17 validation as full-frame, OCR-annotation-guided paste-backs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.data.basic_text_line_full_frame import parse_p17_ocr_annotations, rasterize_polygon_mask
from tools.basic_v2_eval_network import build_eval_network
from tools.infer_basic_text_line_full_image import NativeY4UVTileRunner, infer_full_image, load_lq_preprocess_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bgr(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _write(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), value):
        raise IOError(path)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _records(manifest: Path, root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        files = row.get("files", {})
        sample_id = str(row.get("sample_id") or row.get("image") or Path(str(files.get("lq", ""))).stem)
        if sample_id in seen:
            continue
        required = {key: files.get(key) or row.get(key) for key in ("lq", "gt", "ocr")}
        if not all(required.values()):
            raise ValueError(f"{sample_id}: missing lq/gt/ocr manifest paths")
        seen.add(sample_id)
        result.append({
            "sample_id": sample_id,
            **{key: _resolve(root, str(value)) for key, value in required.items()},
            "valid_mask": _resolve(root, str(files["valid_mask"])) if files.get("valid_mask") else None,
            "text_mask": _resolve(root, str(files["text_mask"])) if files.get("text_mask") else None,
        })
    if not result:
        raise ValueError(f"no records in {manifest}")
    return result


def _regions(ocr_path: Path, shape: tuple[int, int], valid_path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(ocr_path.read_text(encoding="utf-8"))
    core = np.zeros(shape, dtype=np.uint8)
    for line in parse_p17_ocr_annotations(data, shape):
        core = np.maximum(core, rasterize_polygon_mask(line.polygon_xy, shape).astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    ring = (cv2.dilate(core, kernel) > 0) & ~(core > 0)
    if valid_path is None:
        valid = np.ones(shape, dtype=bool)
    else:
        loaded = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
        if loaded is None or loaded.shape != shape:
            raise ValueError(f"invalid valid mask: {valid_path}")
        valid = loaded > 127
    return core > 0, ring, valid


def _region_metrics(output: np.ndarray, gt: np.ndarray, core: np.ndarray, ring: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, float]]:
    diff = output.astype(np.float32) - gt.astype(np.float32)
    result: dict[str, dict[str, float]] = {}
    for name, mask in (("text_core", core), ("boundary_ring", ring), ("far_background", ~(core | ring))):
        selected = valid & mask
        if not selected.any():
            result[name] = {"pixels": 0, "mae_rgb": float("nan"), "mae_y": float("nan")}
            continue
        rgb = np.abs(diff[selected]).mean()
        y = np.abs(0.299 * diff[..., 2] + 0.587 * diff[..., 1] + 0.114 * diff[..., 0])[selected].mean()
        result[name] = {"pixels": int(selected.sum()), "mae_rgb": float(rgb), "mae_y": float(y)}
    return result


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for region in ("text_core", "boundary_ring", "far_background"):
        values = [row["metrics"][region] for row in rows if row["metrics"][region]["pixels"] > 0]
        result[region] = {
            "samples": len(values),
            "mae_rgb_mean": float(np.mean([item["mae_rgb"] for item in values])) if values else float("nan"),
            "mae_y_mean": float(np.mean([item["mae_y"] for item in values])) if values else float("nan"),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--debug-tiles", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    records = _records(args.manifest, args.data_root)
    if args.max_images is not None:
        records = records[:max(0, args.max_images)]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    network, checkpoint_hash = build_eval_network(args.checkpoint, device, args.network_config)
    # The checkpoint's P17 branch was trained on this transformed input.  The
    # runner must receive the same YAML configuration; otherwise preprocess
    # arms silently evaluate on raw LQ and are not comparable to original.
    runner = NativeY4UVTileRunner(network, device, load_lq_preprocess_config(args.network_config))
    args.output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        sample_root = args.output_root / record["sample_id"]
        try:
            output_path = sample_root / "output.png"
            trace = infer_full_image(
                record["lq"], record["ocr"], args.checkpoint, output_path, device_name=args.device,
                overlap=64, blend_radius=8, network=network, checkpoint_sha256=checkpoint_hash,
                network_config=args.network_config, tile_runner=runner, resize_mode="p17_height96_tile_pad",
                ocr_format="p17_annotations", p17_vertical_context=.12, p17_horizontal_context=.03,
                text_mask_path=record["text_mask"],
                debug_dir=(sample_root / "tiles") if args.debug_tiles else None,
            )
            lq, output, gt = _read_bgr(record["lq"]), _read_bgr(output_path), _read_bgr(record["gt"])
            if output.shape != lq.shape or gt.shape != lq.shape:
                raise ValueError(f"full-frame shape mismatch: lq={lq.shape}, output={output.shape}, gt={gt.shape}")
            core, ring, valid = _regions(record["ocr"], lq.shape[:2], record["valid_mask"])
            _write(sample_root / "lq.png", lq); _write(sample_root / "gt.png", gt)
            _write(sample_root / "comparison_lq_output_gt.png", np.hstack((lq, output, gt)))
            _write(sample_root / "ocr_core.png", (core.astype(np.uint8) * 255))
            _write(sample_root / "ocr_boundary_ring.png", (ring.astype(np.uint8) * 255))
            _write(sample_root / "changed_pixels.png", (np.any(output != lq, axis=2).astype(np.uint8) * 255))
            row = {"sample_id": record["sample_id"], "status": "passed", "trace": trace,
                   "metrics": _region_metrics(output, gt, core, ring, valid)}
        except Exception as exc:
            row = {"sample_id": record["sample_id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        (args.output_root / "progress.json").write_text(json.dumps({"completed": index, "total": len(records), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"index": index, "total": len(records), **row}, ensure_ascii=False), flush=True)
    passed = [row for row in rows if row["status"] == "passed"]
    report = {"status": "COMPLETE" if len(passed) == len(rows) else "COMPLETED_WITH_FAILURES", "arm": args.arm,
              "samples": len(rows), "passed": len(passed), "failed": len(rows) - len(passed),
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
              "config": str(args.network_config), "config_sha256": _sha256(args.network_config),
              "pipeline": "P17 OCR annotations -> context crop -> height96 tiled Y4UV -> polygon feather paste-back",
              "metrics": _mean_metrics(passed), "native_calls": runner.calls, "rows": rows}
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "arm", "samples", "passed", "failed", "native_calls")}, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
