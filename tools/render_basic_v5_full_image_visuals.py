"""Render pasted-back V5 validation and HX-testset comparisons for one V4 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from infer_basic_v2_deployment_pipeline import DetectorONNXRunner, RestoreGeometry, run_bgr_adapter_frame
from infer_basic_v2_y4uv_v4_checkpoint_adapter import (
    V4CheckpointRunner,
    V4_CANVAS_HW,
    V4_RESIZE_INTERPOLATION,
    _rewrite_trace_for_v4,
    _sha256,
    load_v4_checkpoint,
)
from tools.basic_v2_eval_network import build_eval_network


def _unique_validation_inputs(manifest: Path) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    result: list[tuple[str, Path]] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        identity = str(row.get("dataset", "")) + "__" + str(row.get("stem", ""))
        image = Path(str(row["lq"]))
        if identity not in seen:
            seen.add(identity)
            result.append((identity, image))
    return result


def _hx_inputs(root: Path) -> list[tuple[str, Path]]:
    accepted = {".png", ".jpg", ".jpeg"}
    return [(path.stem, path) for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in accepted]


def _write_compare(source, output, target: Path) -> None:
    header = 34
    left = source.copy()
    right = output.copy()
    cv2.putText(left, "LQ original", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "V4 pasted-back output", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(target), cv2.hconcat((left, right))):
        raise IOError(target)


def render_set(
    items: list[tuple[str, Path]],
    label: str,
    output_root: Path,
    detector: DetectorONNXRunner,
    runner: V4CheckpointRunner,
    geometry: RestoreGeometry,
    checkpoint: Path,
    checkpoint_sha256: str,
    haar_switch: bool,
    stage_repeat: int,
    bottleneck_repeat: int,
) -> dict:
    target_root = output_root / label
    target_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures: list[dict] = []
    for index, (sample_id, source_path) in enumerate(items, 1):
        try:
            source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source is None:
                raise FileNotFoundError(source_path)
            _, output, trace = run_bgr_adapter_frame(source, detector, runner, debug_dir=None, restore_geometry=geometry)
            _rewrite_trace_for_v4(
                trace, runner, checkpoint, checkpoint_sha256, haar_switch,
                (6, 24, 96, 384), 16, stage_repeat, bottleneck_repeat,
            )
            trace.update({"input": str(source_path), "sample_id": sample_id, "output_is_full_frame_pasteback": True})
            output_path = target_root / f"{sample_id}.png"
            if output.shape != source.shape:
                raise RuntimeError(f"paste-back shape mismatch: {output.shape} != {source.shape}")
            if not cv2.imwrite(str(output_path), output):
                raise IOError(output_path)
            _write_compare(source, output, target_root / f"{sample_id}__compare.jpg")
            (target_root / f"{sample_id}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            completed += 1
        except Exception as exc:  # retain batch progress and exact per-image failure
            failures.append({"sample_id": sample_id, "input": str(source_path), "error": repr(exc)})
        if index % 25 == 0:
            (target_root / "progress.json").write_text(json.dumps({"seen": index, "completed": completed, "failures": failures}, indent=2), encoding="utf-8")
    return {"inputs": len(items), "completed": completed, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--network-config", type=Path,
                        help="Training YAML used to build an exact native Y4UV architecture (V4 capacity or V5).")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--hx-lq-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--haar-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage-repeat", type=int, default=1)
    parser.add_argument("--bottleneck-repeat", type=int, default=2)
    parser.add_argument("--preserve-lq-uv", action="store_true", help="use checkpoint luma and original LQ chroma")
    parser.add_argument("--only-hx", action="store_true", help="render HX only; skip the validation manifest")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite visual root: {args.output_root}")
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    if args.network_config is not None:
        model, _ = build_eval_network(args.checkpoint, device, args.network_config)
    else:
        model = load_v4_checkpoint(args.checkpoint, haar_switch=args.haar_switch, device=device,
                                   stage_repeat=args.stage_repeat, bottleneck_repeat=args.bottleneck_repeat)
    detector = DetectorONNXRunner(str(args.detector))
    runner = V4CheckpointRunner(model, device, preserve_lq_uv=bool(args.preserve_lq_uv))
    geometry = RestoreGeometry(content_h=V4_CANVAS_HW[0], overlap=64,
                               interpolation=V4_RESIZE_INTERPOLATION, force_even_width=True)
    args.output_root.mkdir(parents=True)
    checkpoint_sha256 = _sha256(args.checkpoint)
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "architecture": {"haar_switch": args.haar_switch, "stage_repeat": args.stage_repeat, "bottleneck_repeat": args.bottleneck_repeat},
        "pasteback_contract": "all PNG outputs retain input full-frame dimensions; only accepted OCR ROIs are replaced",
    }
    if not args.only_hx:
        report["validation"] = render_set(_unique_validation_inputs(args.validation_manifest), "validation", args.output_root, detector, runner, geometry, args.checkpoint, checkpoint_sha256, args.haar_switch, args.stage_repeat, args.bottleneck_repeat)
    report["hx_testset"] = render_set(_hx_inputs(args.hx_lq_root), "hx_testset", args.output_root, detector, runner, geometry, args.checkpoint, checkpoint_sha256, args.haar_switch, args.stage_repeat, args.bottleneck_repeat)
    report["native_calls"] = runner.calls
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
