"""Run a fixed-GT, C1-motion-only control experiment for Lenovo V5 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from basicsr.utils import DiffJPEG
from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import V5_CONTEXT_SCALE_LIMIT
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv420, y4uv_to_yuv444
from realesrgan.lenovo_degradation import LenovoDegradationPipeline
from realesrgan import lenovo_atomic_torch as torch_atoms
from tools.basic_v2_eval_network import build_eval_network
from tools.create_lenovo_validation_degradation_set import _variant_degradation
from tools.dump_basic_v2_validation_rgb_pairs import _yuv444_to_rgb


VARIANTS = ("motion_original", "motion_off", "motion_short7", "motion_rot90")


def _args() -> argparse.Namespace:
    """@luna-comment
    Input: command-line paths for one manifest sample, two frozen checkpoints, and an empty output root.
    Output: parsed bounded control-experiment arguments. Key Steps: require explicit artifacts and expose only
    the fixed source/variant selector. Key Params: default selector is 076_000446/v2. State/Dependencies: argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--historical-lq", required=True, type=Path)
    parser.add_argument("--c0-config", required=True, type=Path)
    parser.add_argument("--c0-checkpoint", required=True, type=Path)
    parser.add_argument("--r1-config", required=True, type=Path)
    parser.add_argument("--r1-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-id", default="446")
    parser.add_argument("--variant", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revised", action="store_true", help="Run the matched gaussian/zoom and alpha diagnostic matrix.")
    parser.add_argument("--localize", action="store_true", help="Export first-appearance stage tensors and a Y4UV round trip.")
    parser.add_argument("--localize-control", default="motion_original", help="One fixed C1 control to localize; for example zoom:0.35.")
    parser.add_argument("--c2-controls", action="store_true", help="Run C2 bypass, vertical-phase, and sample-scale controls.")
    parser.add_argument("--r1-necessity", action="store_true", help="Run clean-GT and one-float-Gaussian C0/R1 necessity controls.")
    parser.add_argument("--branch-dump", action="store_true", help="Dump V5 local-skip and enc-dec context branch outputs.")
    parser.add_argument("--context-ablation", action="store_true", help="Localize historical-v2 context depth with frozen decoder interventions.")
    parser.add_argument("--stage2-ablation", action="store_true", help="Localize cumulative V5 stage2 blocks on historical v2.")
    parser.add_argument("--stage2-probe", action="store_true", help="Probe block1 increment versus matched isotropic sharpening.")
    parser.add_argument("--stage2-frequency-probe", action="store_true", help="Test stage2 block1 vertical Nyquist component.")
    parser.add_argument("--cross-context-probe", action="store_true", help="Cross-test block1 and decoder roles for C0/R1.")
    parser.add_argument("--stage2-branch-probe", action="store_true", help="Localize frozen R1 stage2[1] branch and PReLU contributions on shared C0 Z0.")
    parser.add_argument("--subband-dump", action="store_true", help="Dump four inverse-Haar local subband contributions.")
    parser.add_argument("--raw-subband-dump", action="store_true", help="Dump native-resolution local Haar input and predicted bands.")
    parser.add_argument("--hl-ablation", action="store_true", help="Export diagnostic R1 HL/LH/HH local-subband subtractions.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """@luna-comment
    Input: an existing binary or text path. Output: hexadecimal SHA-256 digest. Key Steps: stream 1 MiB blocks.
    Key Params: block size bounds memory use. State/Dependencies: local filesystem only.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    """@luna-comment
    Input: RGB PNG path. Output: contiguous HWC float32 RGB in [0,1]. Key Steps: force RGB decoding and normalize.
    Key Params: none. State/Dependencies: Pillow image decoder.
    """
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    """@luna-comment
    Input: HWC RGB float image in nominal [0,1] and destination path. Output: a lossless RGB PNG.
    Key Steps: clamp, quantize, create parents, and save. Key Params: 255 quantization scale. State/Dependencies: Pillow.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(value, "RGB").save(path)


def _find_variant(manifest: dict[str, Any], source_id: str, variant: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """@luna-comment
    Input: parsed validation manifest, source identifier, and one-based variant number. Output: matching record/variant.
    Key Steps: exact string/int match and fail on absence. Key Params: variant is one-based like the manifest. State/Dependencies: none.
    """
    for record in manifest["records"]:
        if str(record["source_id"]) != str(source_id):
            continue
        for item in record["variants"]:
            if int(item["variant"]) == variant:
                return record, item
    raise KeyError(f"missing source_id={source_id!r}, variant={variant}")


def _make_lq(
    gt: torch.Tensor, degradation: dict[str, Any], item: dict[str, Any], device: torch.device, control: str,
    c2_control: str = "original"
) -> tuple[torch.Tensor, dict[str, Any]]:
    """@luna-comment
    Input: one GT BCHW RGB tensor, manifest degradation/config item, device, and a named C1 control.
    Output: final LQ tensor and path-hit stage metadata. Key Steps: recreate the original composed pipeline from its
    stored seed, consume the native C1 branch, and substitute only its output for off/short/rotated controls.
    Key Params: short length=7 and rotated angle=original+90 degrees. State/Dependencies: temporary instance-method
    wrappers only; no checkpoint, config, or global source state is mutated.
    """
    variant_index = int(item["variant"]) - 1
    profile, config = _variant_degradation(
        degradation, variant_index, 3, "random_2_to_5", int(item["seed"])
    )
    jpeger = DiffJPEG(differentiable=False).to(device)

    def jpeg(image: torch.Tensor, quality: float) -> torch.Tensor:
        return jpeger(image, quality=image.new_full((image.shape[0],), quality))

    pipeline = LenovoDegradationPipeline(config, jpeg=jpeg)
    original_always = pipeline._always_capture
    original_c1 = pipeline._c1
    captured: dict[str, torch.Tensor] = {}

    def captured_stage(label: str, method):
        def wrapper(*method_args, **method_kwargs):
            result, stage_record = method(*method_args, **method_kwargs)
            captured[label] = result.detach().clone()
            return result, stage_record
        return wrapper

    pipeline._always_capture = captured_stage("after_always", original_always)  # type: ignore[method-assign]

    def controlled_c1(image: torch.Tensor, generator: torch.Generator, subbranch: str | None = None):
        native, record = original_c1(image, generator, subbranch)
        if record.get("subbranch") != "motion":
            raise RuntimeError(f"expected fixed sample C1 motion, got {record.get('subbranch')!r}")
        captured["before_c1"] = image.detach().clone()
        captured["native_c1"] = native.detach().clone()
        params = dict(record["parameters"])
        if control == "motion_original":
            result = native
        elif control == "motion_off":
            result = image
        else:
            if control == "motion_short7":
                params["length"] = 7
            elif control == "motion_rot90":
                params["angle"] = float(params["angle"]) + 90.0
            elif control.startswith("gaussian:"):
                result, replacement = pipeline.torch_atomic(
                    image, "gaussian", generator, {"sigma": float(control.split(":", 1)[1])}
                )
                replacement["control"] = control
                replacement["effective_length"] = None
                captured["after_c1"] = result.detach().clone()
                return result, replacement
            elif control.startswith("zoom:"):
                result, replacement = pipeline.torch_atomic(
                    image, "zoom", generator, {"source_scale": float(control.split(":", 1)[1])}
                )
                replacement["control"] = control
                replacement["effective_length"] = None
                captured["after_c1"] = result.detach().clone()
                return result, replacement
            elif control.startswith("alpha:"):
                alpha = float(control.split(":", 1)[1])
                result = (1.0 - alpha) * image + alpha * native
                record["alpha"] = alpha
                captured["after_c1"] = result.detach().clone()
                return result, record
            else:
                raise ValueError(control)
            # The native call above has already consumed all C1 parameter RNG. Torch motion with explicit
            # parameters consumes none, so downstream C2/C7/C8 random draws remain aligned.
            result, replacement = pipeline.torch_atomic(image, "motion", generator, params)
            record = replacement
        captured["after_c1"] = result.detach().clone()
        record = dict(record)
        record["control"] = control
        record["effective_length"] = int(float(record["parameters"].get("length", item["stages"]["stages"][1]["length"])))
        return result, record

    pipeline._c1 = controlled_c1  # type: ignore[method-assign]
    original_c2 = pipeline._c2

    def controlled_c2(image: torch.Tensor, generator: torch.Generator):
        native, record = original_c2(image, generator)
        captured["native_c2"] = native.detach().clone()
        if c2_control == "original":
            result, replacement = native, dict(record)
        elif c2_control == "bypass":
            result, replacement = image, dict(record)
        else:
            parameters = dict(record["parameters"])
            if c2_control.startswith("phase_y:"):
                parameters["phase_xy"] = [0.0, float(c2_control.split(":", 1)[1])]
            elif c2_control.startswith("scale:"):
                parameters["sample_scale"] = float(c2_control.split(":", 1)[1])
                parameters["phase_xy"] = [0.0, 0.0]
            else:
                raise ValueError(f"unsupported C2 control {c2_control!r}")
            result, replacement = pipeline.torch_atomic(image, "c2", generator, parameters)
        captured["after_c2"] = result.detach().clone()
        replacement = dict(replacement)
        replacement["control"] = c2_control
        return result, replacement

    pipeline._c2 = controlled_c2  # type: ignore[method-assign]
    for stage_name in ("c3", "c4", "c5", "c7", "c8"):
        method_name = f"_{stage_name}"
        setattr(pipeline, method_name, captured_stage(f"after_{stage_name}", getattr(pipeline, method_name)))
    generator = torch.Generator(device=device).manual_seed(int(item["seed"]))
    lq, metadata = pipeline(gt, generator)
    return lq, {"profile": profile, "metadata": metadata[0], "captured": captured}


def _infer(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[np.ndarray, str]:
    """@luna-comment
    Input: one RGB BCHW LQ tensor and an existing V5 training YAML/checkpoint. Output: RGB HWC prediction and SHA-256.
    Key Steps: apply the direct-RGB to Y4UV inference contract, run eval-mode network, and decode normalized YUV444.
    Key Params: none. State/Dependencies: frozen checkpoint loaded through the shared evaluator builder.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        output = _yuv444_to_rgb(y4uv_to_yuv444(network(packed)))
    return output, checkpoint_sha256


def _metrics(c0: np.ndarray, r1: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """@luna-comment
    Input: matched C0/R1/GT RGB images. Output: factual residual, luma, and row-oriented stripe proxy metrics.
    Key Steps: form luma residuals and compare vertical versus horizontal finite differences. Key Params: RGB luma weights.
    State/Dependencies: NumPy only; values are diagnostics, not a visual-quality verdict.
    """
    luma = lambda x: 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    delta = luma(r1) - luma(c0)
    residual = luma(r1) - luma(gt)
    dy = float(np.mean(np.abs(np.diff(delta, axis=0))))
    dx = float(np.mean(np.abs(np.diff(delta, axis=1))))
    return {
        "r1_minus_c0_luma_mean": float(delta.mean()),
        "r1_gt_l1_y": float(np.abs(residual).mean()),
        "stripe_proxy_vertical_over_horizontal": dy / max(dx, 1e-12),
        "stripe_proxy_vertical_difference": dy,
    }


def _high_frequency_energy(image: np.ndarray) -> float:
    """@luna-comment
    Input: RGB HWC image in [0,1]. Output: mean squared high-pass luma energy. Key Steps: subtract a 5x5 Gaussian
    low-pass luma estimate. Key Params: fixed kernel prevents output-dependent matching. State/Dependencies: OpenCV.
    """
    luma = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    high = luma - cv2.GaussianBlur(luma, (5, 5), 0)
    return float(np.mean(np.square(high)))


def _localization_profile(image: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """@luna-comment
    Input: one stage RGB image and fixed GT RGB image. Output: dark-glyph/background detrended row profiles and
    non-DC vertical spectral candidates. Key Steps: use GT-derived masks, remove a broad vertical trend, and rank
    frequencies separately per mask. Key Params: glyph/background luma thresholds are diagnostics only. State/Dependencies: NumPy/OpenCV.
    """
    luma = lambda value: 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]
    value, gt_luma = luma(image), luma(gt)
    masks = {"glyph": gt_luma < 0.65, "background": gt_luma > 0.90}
    profiles: dict[str, Any] = {}
    for name, mask in masks.items():
        weight = mask.astype(np.float32)
        row = (value * weight).sum(axis=1) / np.maximum(weight.sum(axis=1), 1.0)
        valid = weight.sum(axis=1) > 0
        trend = cv2.GaussianBlur(row[None, :], (0, 0), sigmaX=5)[0]
        detrended = (row - trend) * valid
        power = np.abs(np.fft.rfft(detrended)) ** 2
        candidates = np.argsort(power[2:])[-8:][::-1] + 2
        profiles[name] = {"valid_rows": int(valid.sum()), "row": row.tolist(), "detrended": detrended.tolist(),
                          "top_vertical_frequency_bins": [{"bin": int(index), "power": float(power[index]),
                                                            "period_rows": float(len(row) / index)} for index in candidates]}
    return profiles


def _write_raw_stage(path: Path, tensor: torch.Tensor) -> None:
    """@luna-comment
    Input: BCHW float RGB stage tensor and append-only destination. Output: float32 HWC NumPy array without display quantization.
    Key Steps: detach, move to CPU, reorder channels, and save. Key Params: preserves nominal [0,1] values. State/Dependencies: NumPy filesystem output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor[0].detach().float().cpu().permute(1, 2, 0).numpy())


def _tensor_summary(tensor: torch.Tensor, phase_packed: bool = False) -> dict[str, Any]:
    """@luna-comment
    Input: one debug BCHW tensor and a phase-pack flag. Output: shape, energy, activation, and vertical-frequency proxy.
    Key Steps: inverse-repack four phases before spatial metrics when requested. Key Params: phase order TL/TR/BL/BR.
    State/Dependencies: detached Torch/NumPy tensors only.
    """
    value = tensor.detach().float()
    if phase_packed:
        _, channels, height, width = value.shape
        if channels % 4:
            raise ValueError(f"phase-packed channels must divide by 4: {channels}")
        phases = value.reshape(1, channels // 4, 4, height, width)
        unpacked = value.new_empty((1, channels // 4, height * 2, width * 2))
        unpacked[:, :, 0::2, 0::2] = phases[:, :, 0]
        unpacked[:, :, 0::2, 1::2] = phases[:, :, 1]
        unpacked[:, :, 1::2, 0::2] = phases[:, :, 2]
        unpacked[:, :, 1::2, 1::2] = phases[:, :, 3]
        value = unpacked
    plane = value.mean(1)[0].cpu().numpy()
    plane = plane - plane.mean()
    row_power = np.abs(np.fft.rfft(plane.mean(1))) ** 2
    return {"shape": list(value.shape), "mean": float(value.mean()), "std": float(value.std()),
            "positive_ratio": float((value > 0).float().mean()), "row_frequency_peak": int(np.argmax(row_power[1:]) + 1),
            "row_frequency_peak_power": float(row_power[1:].max())}


def _infer_debug(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[np.ndarray, str, dict[str, Any]]:
    """@luna-comment
    Input: fixed RGB LQ plus frozen V5 config/checkpoint. Output: RGB prediction, checkpoint hash, and passive debug summaries.
    Key Steps: capture all PReLU pre/post summaries and V5 return_debug tensors without changing inference values. Key Params:
    phase carries are inverse-repacked by the summary helper. State/Dependencies: temporary forward hooks removed before return.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    prelu: dict[str, Any] = {}
    hooks = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.PReLU):
            def hook(_module, inputs, output, key=name):
                prelu[key] = {"input": _tensor_summary(inputs[0]), "output": _tensor_summary(output)}
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            packed = rgb_to_y4uv(lq).to(device)
            debug = network(packed, return_debug=True)
            output = _yuv444_to_rgb(y4uv_to_yuv444(debug["output"]))
    finally:
        for hook in hooks:
            hook.remove()
    keys = {"input": False, "local_residual": False, "context_residual": False, "stem_carry": False,
            "phase_carry1": True, "phase_carry2": True, "decoded1": False, "decoded0": False,
            "decoder_residual2": False, "decoder_residual1": False, "output": False}
    return output, checkpoint_sha256, {"tensors": {key: _tensor_summary(debug[key], packed) for key, packed in keys.items()}, "prelu": prelu}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: parsed bounded control experiment arguments. Output: provenance-rich report and append-only PNG/JSON artifacts.
    Key Steps: validate paths, replay four LQ controls, compare historical LQ, infer C0/R1, and persist diagnostics.
    Key Params: output root must not already exist to prevent overwrite. State/Dependencies: local/Remote58 files named by args.
    """
    paths = [args.manifest, args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_path = Path(record["gt"])
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    gt_rgb = _read_rgb(gt_path)
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    historical = _read_rgb(args.historical_lq)
    results: dict[str, Any] = {}
    for control in VARIANTS:
        lq, trace = _make_lq(gt, manifest["degradation"], item, device, control)
        lq_rgb = lq[0].detach().cpu().permute(1, 2, 0).numpy()
        directory = root / control
        _write_rgb(directory / "gt.png", gt_rgb)
        _write_rgb(directory / "lq.png", lq_rgb)
        for name, tensor in trace["captured"].items():
            _write_rgb(directory / f"{name}.png", tensor[0].cpu().permute(1, 2, 0).numpy())
        c0, c0_sha = _infer(lq, args.c0_config, args.c0_checkpoint, device)
        r1, r1_sha = _infer(lq, args.r1_config, args.r1_checkpoint, device)
        _write_rgb(directory / "c0.png", c0)
        _write_rgb(directory / "r1.png", r1)
        replay = np.abs(lq_rgb - historical)
        payload = {
            "control": control,
            "source_id": str(args.source_id),
            "variant": int(args.variant),
            "replay_against_historical_lq": {
                "pixel_equal_rate": float(np.mean(np.all(replay == 0.0, axis=2))),
                "max_abs": float(replay.max()), "mae": float(replay.mean()),
            },
            "c0_checkpoint_sha256": c0_sha, "r1_checkpoint_sha256": r1_sha,
            "metrics": _metrics(c0, r1, gt_rgb), "path_hit": trace["metadata"],
        }
        (directory / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results[control] = payload
    contact_rows = []
    for control in VARIANTS:
        directory = root / control
        contact_rows.append(np.concatenate([
            _read_rgb(directory / "lq.png"),
            _read_rgb(directory / "c0.png"),
            _read_rgb(directory / "r1.png"),
        ], axis=1))
    _write_rgb(root / "contact_sheet_lq_c0_r1.png", np.concatenate(contact_rows, axis=0))
    report = {"status": "COMPLETE", "source": {"record": record["source_id"], "variant": item["variant"]},
              "manifest_sha256": _sha256(args.manifest), "historical_lq_sha256": _sha256(args.historical_lq),
              "controls": results}
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_revised(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: standard control arguments with revised mode. Output: 14 C0/R1 outputs plus matched-input and passive debug reports.
    Key Steps: match Gaussian/zoom only to motion LQ high-frequency energy, then run four main and three alpha controls.
    Key Params: fixed candidate grids and alpha=.25/.5/.75. State/Dependencies: append-only output directory.
    """
    for path in (args.manifest, args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    motion_lq, _ = _make_lq(gt, manifest["degradation"], item, device, "motion_original")
    target_energy = _high_frequency_energy(motion_lq[0].cpu().permute(1, 2, 0).numpy())
    candidates: dict[str, tuple[str, float]] = {}
    for name, values in {"gaussian": (0.5, 1, 1.5, 2, 2.5, 3, 4), "zoom": (.35, .45, .55, .65, .75, .85, .95)}.items():
        scored = []
        for value in values:
            lq, _ = _make_lq(gt, manifest["degradation"], item, device, f"{name}:{value}")
            energy = _high_frequency_energy(lq[0].cpu().permute(1, 2, 0).numpy())
            scored.append((abs(energy - target_energy), value, energy))
        _, chosen, energy = min(scored)
        candidates[name] = (f"{name}:{chosen}", energy)
    controls = ["motion_off", "motion_original", candidates["gaussian"][0], candidates["zoom"][0],
                "alpha:0.25", "alpha:0.5", "alpha:0.75"]
    rows = {}
    for control in controls:
        lq, trace = _make_lq(gt, manifest["degradation"], item, device, control)
        lq_rgb = lq[0].cpu().permute(1, 2, 0).numpy()
        directory = root / control.replace(":", "_")
        _write_rgb(directory / "lq.png", lq_rgb)
        c0, c0_sha, c0_debug = _infer_debug(lq, args.c0_config, args.c0_checkpoint, device)
        r1, r1_sha, r1_debug = _infer_debug(lq, args.r1_config, args.r1_checkpoint, device)
        _write_rgb(directory / "c0.png", c0); _write_rgb(directory / "r1.png", r1)
        row = {"control": control, "lq_high_frequency_energy": _high_frequency_energy(lq_rgb),
               "metrics": _metrics(c0, r1, gt_rgb), "path_hit": trace["metadata"],
               "c0_checkpoint_sha256": c0_sha, "r1_checkpoint_sha256": r1_sha,
               "c0_debug": c0_debug, "r1_debug": r1_debug}
        (directory / "probe.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows[control] = row
    report = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
              "motion_lq_high_frequency_energy": target_energy,
              "matched_controls": {key: {"control": value[0], "energy": value[1], "absolute_error": abs(value[1] - target_energy)} for key, value in candidates.items()},
              "controls": rows}
    (root / "revised_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_localization(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: one fixed-GT replay configuration and exactly one C1 control. Output: append-only float/PNG stage exports,
    RGB-Y4UV-RGB round-trip planes, C0/R1 outputs, and stage-first-appearance diagnostics. Key Steps: preserve the
    native degradation RNG path, export only stages actually traversed, and report profiles without selecting a visual winner.
    Key Params: localize-control must be a supported C1 replacement. State/Dependencies: frozen checkpoints and Remote58 evaluation storage.
    """
    for path in (args.manifest, args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    lq, trace = _make_lq(gt, manifest["degradation"], item, device, args.localize_control)
    stages: dict[str, torch.Tensor] = {"gt": gt.detach().clone(), **trace["captured"], "final_lq": lq.detach().clone()}
    profiles: dict[str, Any] = {}
    for name, tensor in stages.items():
        value = tensor[0].detach().float().cpu().permute(1, 2, 0).numpy()
        _write_raw_stage(root / "stages" / f"{name}.npy", tensor)
        _write_rgb(root / "stages" / f"{name}.png", value)
        profiles[name] = _localization_profile(value, gt_rgb)

    packed = rgb_to_y4uv(lq).to(device)
    y, u, v = y4uv_to_yuv420(packed)
    roundtrip_yuv = y4uv_to_yuv444(packed)
    roundtrip_rgb = _yuv444_to_rgb(roundtrip_yuv)
    roundtrip_rgb_tensor = torch.from_numpy(np.ascontiguousarray(roundtrip_rgb)).permute(2, 0, 1).unsqueeze(0).to(device)
    planes = {"roundtrip_rgb": roundtrip_rgb_tensor, "roundtrip_y": y.repeat(1, 3, 1, 1),
              "roundtrip_u": u.repeat(1, 3, 1, 1), "roundtrip_v": v.repeat(1, 3, 1, 1)}
    for name, tensor in planes.items():
        _write_raw_stage(root / "roundtrip" / f"{name}.npy", tensor)
        _write_rgb(root / "roundtrip" / f"{name}.png", tensor[0].detach().float().cpu().permute(1, 2, 0).numpy())
    c0, c0_sha = _infer(lq, args.c0_config, args.c0_checkpoint, device)
    r1, r1_sha = _infer(lq, args.r1_config, args.r1_checkpoint, device)
    _write_rgb(root / "outputs" / "c0.png", c0)
    _write_rgb(root / "outputs" / "r1.png", r1)
    np.save(root / "outputs" / "c0.npy", c0.astype(np.float32)); np.save(root / "outputs" / "r1.npy", r1.astype(np.float32))
    profiles["roundtrip_rgb"] = _localization_profile(roundtrip_rgb, gt_rgb)
    profiles["c0"] = _localization_profile(c0, gt_rgb)
    profiles["r1"] = _localization_profile(r1, gt_rgb)
    lq_rgb = lq[0].detach().float().cpu().permute(1, 2, 0).numpy()
    report = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
              "control": args.localize_control, "actual_stages": list(stages), "path_hit": trace["metadata"],
              "c0_checkpoint_sha256": c0_sha, "r1_checkpoint_sha256": r1_sha,
              "roundtrip": {"lq_rgb_max_abs": float(np.abs(roundtrip_rgb - lq_rgb).max()),
                            "lq_rgb_mae": float(np.abs(roundtrip_rgb - lq_rgb).mean())},
              "profiles": profiles, "metrics": _metrics(c0, r1, gt_rgb)}
    (root / "localization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_c2_controls(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: one fixed C1 control and frozen C0/R1 checkpoints. Output: append-only original, bypass, vertical-phase,
    and scale C2 branches. Key Steps: always execute native C2 first to preserve RNG, then replace only its output;
    retain C7/C8 and inference unchanged. Key Params: phase controls set actual phase_xy, while scale controls pin zero
    phase to avoid conflating lattice spacing with offset. State/Dependencies: diagnostic-only Remote58 inference.
    """
    for path in (args.manifest, args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    controls = ("original", "bypass", "phase_y:-0.75", "phase_y:0.0", "phase_y:0.75", "scale:0.40", "scale:0.60")
    results: dict[str, Any] = {}
    for c2_control in controls:
        lq, trace = _make_lq(gt, manifest["degradation"], item, device, args.localize_control, c2_control)
        lq_rgb = lq[0].detach().float().cpu().permute(1, 2, 0).numpy()
        label = c2_control.replace(":", "_").replace("-", "neg").replace(".", "p")
        directory = root / label
        _write_rgb(directory / "lq.png", lq_rgb)
        _write_raw_stage(directory / "lq.npy", lq)
        c0, c0_sha = _infer(lq, args.c0_config, args.c0_checkpoint, device)
        r1, r1_sha = _infer(lq, args.r1_config, args.r1_checkpoint, device)
        _write_rgb(directory / "c0.png", c0); _write_rgb(directory / "r1.png", r1)
        np.save(directory / "c0.npy", c0.astype(np.float32)); np.save(directory / "r1.npy", r1.astype(np.float32))
        results[c2_control] = {"c2_control": c2_control, "path_hit": trace["metadata"],
                               "profiles": {"lq": _localization_profile(lq_rgb, gt_rgb),
                                            "c0": _localization_profile(c0, gt_rgb), "r1": _localization_profile(r1, gt_rgb)},
                               "metrics": _metrics(c0, r1, gt_rgb), "c0_checkpoint_sha256": c0_sha,
                               "r1_checkpoint_sha256": r1_sha}
    report = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
              "c1_control": args.localize_control, "controls": results}
    (root / "c2_control_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _float_gaussian(rgb: torch.Tensor, sigma: float) -> torch.Tensor:
    """@luna-comment
    Input: normalized RGB BCHW tensor and a positive Gaussian sigma. Output: float RGB Gaussian blur without
    quantization, resize, JPEG, or degradation-pipeline stages. Key Steps: convert to linear light, blur, and encode
    back to sRGB. Key Params: sigma=2.5 is the prior motion-energy-nearest Gaussian diagnostic. State/Dependencies: Torch atom math only.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return torch_atoms.srgb(torch_atoms.gaussian(torch_atoms.linear(rgb), sigma)).clamp(0.0, 1.0)


def run_r1_necessity(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: fixed GT plus frozen C0/R1 checkpoints. Output: clean-GT and one-float-Gaussian C0/R1 outputs and factual
    residual profiles. Key Steps: bypass every Lenovo degradation stage and retain only the shared RGB-to-Y4UV inference
    contract. Key Params: Gaussian sigma=2.5, no quantization/resizing/JPEG. State/Dependencies: frozen inference only.
    """
    for path in (args.manifest, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    inputs = {"clean_gt": gt, "float_gaussian_sigma_2p5": _float_gaussian(gt, 2.5)}
    controls: dict[str, Any] = {}
    for name, lq in inputs.items():
        lq_rgb = lq[0].detach().float().cpu().permute(1, 2, 0).numpy()
        directory = root / name
        _write_rgb(directory / "input.png", lq_rgb); _write_raw_stage(directory / "input.npy", lq)
        c0, c0_sha = _infer(lq, args.c0_config, args.c0_checkpoint, device)
        r1, r1_sha = _infer(lq, args.r1_config, args.r1_checkpoint, device)
        _write_rgb(directory / "c0.png", c0); _write_rgb(directory / "r1.png", r1)
        np.save(directory / "c0.npy", c0.astype(np.float32)); np.save(directory / "r1.npy", r1.astype(np.float32))
        controls[name] = {"input": {"pipeline": "none" if name == "clean_gt" else "float_linear_light_gaussian_only",
                                     "gaussian_sigma": None if name == "clean_gt" else 2.5},
                          "profiles": {"input": _localization_profile(lq_rgb, gt_rgb), "c0": _localization_profile(c0, gt_rgb),
                                       "r1": _localization_profile(r1, gt_rgb)}, "metrics": _metrics(c0, r1, gt_rgb),
                          "c0_checkpoint_sha256": c0_sha, "r1_checkpoint_sha256": r1_sha}
    report = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
              "controls": controls, "excluded_operations": ["always", "C1", "C2", "C7", "C8", "JPEG", "resize", "quantization"]}
    (root / "r1_necessity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _infer_branch_dump(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[dict[str, np.ndarray], str, float]:
    """@luna-comment
    Input: RGB LQ BCHW plus one frozen V5 config/checkpoint. Output: decoded input-plus-local, input-plus-context,
    and full RGB branches, checkpoint hash, and Y4UV additive-closure error. Key Steps: invoke existing return_debug
    without changing forward semantics, compose each residual with the native input, and decode each Y4UV result. Key
    Params: local is the shallow skip branch; context is the enc-dec branch. State/Dependencies: frozen inference only.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        debug = network(packed, return_debug=True)
        local_only = debug["input"] + debug["local_residual"]
        context_only = debug["input"] + debug["context_residual"]
        full = debug["output"]
        closure = float((full - (debug["input"] + debug["local_residual"] + debug["context_residual"])).abs().max())
        decoded = {"input": _yuv444_to_rgb(y4uv_to_yuv444(debug["input"])),
                   "local_only": _yuv444_to_rgb(y4uv_to_yuv444(local_only)),
                   "context_only": _yuv444_to_rgb(y4uv_to_yuv444(context_only)),
                   "full": _yuv444_to_rgb(y4uv_to_yuv444(full))}
        raw = {"local_residual": debug["local_residual"], "context_residual": debug["context_residual"],
               "input_y4uv": debug["input"], "full_y4uv": full}
    decoded["_raw"] = raw  # type: ignore[assignment]
    return decoded, checkpoint_sha256, closure


def run_branch_dump(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: historical v2 LQ, fixed GT, and frozen C0/R1 checkpoints. Output: append-only raw Y4UV residuals plus
    local-skip/context-only/full RGB branch dumps for the exact historical input, clean GT, and float-Gaussian control.
    Key Steps: preserve the existing V5 forward contract through return_debug and verify additive branch closure.
    Key Params: historical input is read directly rather than replayed; Gaussian sigma is 2.5.
    State/Dependencies: Remote58 frozen inference; no checkpoint or default forward mutation.
    """
    for path in (args.manifest, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    historical_rgb = _read_rgb(args.historical_lq)
    historical = torch.from_numpy(historical_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if historical.shape != gt.shape:
        raise ValueError(f"historical LQ shape {tuple(historical.shape)} does not match GT shape {tuple(gt.shape)}")
    inputs = {
        "historical_original_v2": historical,
        "clean_gt": gt,
        "float_gaussian_sigma_2p5": _float_gaussian(gt, 2.5),
    }
    checkpoints = {"c0": (args.c0_config, args.c0_checkpoint), "r1": (args.r1_config, args.r1_checkpoint)}
    report: dict[str, Any] = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
                              "historical_lq_sha256": _sha256(args.historical_lq),
                              "branch_contract": {"local_only": "input + local_residual (shallow skip branch)",
                                                  "context_only": "input + context_residual (enc-dec branch)",
                                                  "full": "input + local_residual + context_residual"}, "inputs": {}}
    for input_name, lq in inputs.items():
        input_row: dict[str, Any] = {}
        for model_name, (config, checkpoint) in checkpoints.items():
            decoded, digest, closure = _infer_branch_dump(lq, config, checkpoint, device)
            directory = root / input_name / model_name
            for name in ("input", "local_only", "context_only", "full"):
                _write_rgb(directory / f"{name}.png", decoded[name])
                np.save(directory / f"{name}.npy", decoded[name].astype(np.float32))
            for name, tensor in decoded["_raw"].items():
                np.save(directory / f"{name}.npy", tensor[0].detach().float().cpu().numpy())
            input_row[model_name] = {"checkpoint_sha256": digest, "additive_closure_max_abs": closure,
                                     "profiles": {name: _localization_profile(decoded[name], gt_rgb) for name in ("local_only", "context_only", "full")}}
        report["inputs"][input_name] = input_row
    (root / "branch_dump_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _infer_context_ablation(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, float]:
    """@luna-comment
    Input: one historical RGB LQ, a frozen V5 checkpoint, and the configured device. Output: context-only RGB and
    Y4UV residuals for four depth interventions, checkpoint hash, and V3 reproduction error. Key Steps: retain the
    original terminal/inverse-Haar for every variant while progressively adding context encoder-decoder depth.
    Key Params: V0=stem, V1=shallow carry, V2=stage1 plus second decoder level without stage2, V3=original context.
    State/Dependencies: frozen inference only; PReLU decoder blocks are recomputed per intervention and never linearly split.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        debug = network(packed, return_debug=True)
        stem = debug["stem_carry"]
        phase1 = debug["phase_carry1"]
        stage1 = network._run_blocks(phase1, network.stage1)
        phase2 = network.phase_pack2(stage1)
        residual2 = network.dec2(network.up2(phase2))
        shallow_decoded0 = stem + network.dec1(network.up1(phase1))
        second_level_decoded0 = stem + network.dec1(network.up1(phase1 + residual2))
        decoded0_variants = {
            "v0_stem_only": stem,
            "v1_shallow_carry": shallow_decoded0,
            "v2_second_level_no_stage2": second_level_decoded0,
            "v3_original_context": debug["decoded0"],
        }
        residuals: dict[str, np.ndarray] = {}
        outputs: dict[str, np.ndarray] = {}
        v3_error = 0.0
        for name, decoded0 in decoded0_variants.items():
            context_feature_residual = network.context_terminal(decoded0)
            if not network._deployment:
                alpha = V5_CONTEXT_SCALE_LIMIT * torch.tanh(network.context_scale)
                context_feature_residual = alpha * context_feature_residual
            context_residual = network._decode_luma(context_feature_residual)
            if name == "v3_original_context":
                v3_error = float((context_residual - debug["context_residual"]).abs().max())
            residuals[name] = context_residual[0].detach().float().cpu().numpy()
            outputs[name] = _yuv444_to_rgb(y4uv_to_yuv444(debug["input"] + context_residual))
    return outputs, residuals, checkpoint_sha256, v3_error


def run_context_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: direct historical v2 LQ and frozen C0/R1 checkpoints. Output: append-only context-only depth intervention
    images, raw residuals, and exact V3 reproduction telemetry. Key Steps: reject display composites, run the four
    fixed variants through the production terminal/decode path, and retain only the user-targeted historical input.
    Key Params: V0/V1/V2/V3 are cumulative depth probes; local residual is excluded from every output.
    State/Dependencies: Remote58 frozen diagnostic inference; no model, checkpoint, or default-forward mutation.
    """
    for path in (args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    lq_rgb = _read_rgb(args.historical_lq)
    lq = torch.from_numpy(lq_rgb).permute(2, 0, 1).unsqueeze(0)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoints = {"c0": (args.c0_config, args.c0_checkpoint), "r1": (args.r1_config, args.r1_checkpoint)}
    report: dict[str, Any] = {
        "status": "COMPLETE",
        "historical_lq": str(args.historical_lq),
        "historical_lq_sha256": _sha256(args.historical_lq),
        "variant_contract": {
            "v0_stem_only": "context_terminal(stem)",
            "v1_shallow_carry": "context_terminal(stem + dec1(up1(phase_pack1(stem))))",
            "v2_second_level_no_stage2": "context_terminal(stem + dec1(up1(phase1 + dec2(up2(phase_pack2(stage1(phase1)))))))",
            "v3_original_context": "context_terminal(original decoded0)",
        },
        "outputs": {},
    }
    for model_name, (config, checkpoint) in checkpoints.items():
        outputs, residuals, digest, v3_error = _infer_context_ablation(lq, config, checkpoint, device)
        directory = root / model_name
        directory.mkdir(parents=True)
        for name, image in outputs.items():
            _write_rgb(directory / f"{name}.png", image)
            np.save(directory / f"{name}_context_residual_y4uv.npy", residuals[name].astype(np.float32))
        report["outputs"][model_name] = {
            "checkpoint_sha256": digest,
            "v3_context_residual_reproduction_max_abs": v3_error,
        }
    (root / "context_ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _infer_stage2_ablation(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, float]:
    """@luna-comment
    Input: one historical RGB LQ, a frozen V5 checkpoint, and the configured device. Output: context-only RGB and
    Y4UV residuals for cumulative zero/one/two-block stage2 paths, checkpoint hash, and full-stage2 reproduction error.
    Key Steps: run stage2 blocks sequentially, then preserve the original up2/dec2/dec1/terminal/decode path per variant.
    Key Params: the two-block variant is the production stage2 path and must exactly reproduce original context residual.
    State/Dependencies: frozen inference only; each PReLU block remains in its original sequential order.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        debug = network(packed, return_debug=True)
        if len(network.stage2) != 2:
            raise RuntimeError(f"stage2 ablation expects two blocks, got {len(network.stage2)}")
        stem = debug["stem_carry"]
        phase1 = debug["phase_carry1"]
        stage1 = network._run_blocks(phase1, network.stage1)
        stage2_value = network.phase_pack2(stage1)
        stage2_values = [stage2_value]
        for block in network.stage2:
            stage2_values.append(block(stage2_values[-1]))
        residuals: dict[str, np.ndarray] = {}
        outputs: dict[str, np.ndarray] = {}
        full_error = 0.0
        for count, stage2_output in enumerate(stage2_values):
            residual2 = network.dec2(network.up2(stage2_output))
            decoded0 = stem + network.dec1(network.up1(phase1 + residual2))
            context_feature_residual = network.context_terminal(decoded0)
            if not network._deployment:
                alpha = V5_CONTEXT_SCALE_LIMIT * torch.tanh(network.context_scale)
                context_feature_residual = alpha * context_feature_residual
            context_residual = network._decode_luma(context_feature_residual)
            name = f"stage2_{count}_blocks"
            if count == len(network.stage2):
                full_error = float((context_residual - debug["context_residual"]).abs().max())
            residuals[name] = context_residual[0].detach().float().cpu().numpy()
            outputs[name] = _yuv444_to_rgb(y4uv_to_yuv444(debug["input"] + context_residual))
    return outputs, residuals, checkpoint_sha256, full_error


def run_stage2_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: direct historical v2 LQ and frozen C0/R1 checkpoints. Output: append-only cumulative stage2 context-only
    images, raw residuals, and full-stage2 reproduction telemetry. Key Steps: vary only the number of sequential stage2
    blocks and keep every downstream decoder and terminal operation unchanged. Key Params: exactly zero, one, and two blocks.
    State/Dependencies: Remote58 frozen diagnostic inference; no model, checkpoint, or default-forward mutation.
    """
    for path in (args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    lq_rgb = _read_rgb(args.historical_lq)
    lq = torch.from_numpy(lq_rgb).permute(2, 0, 1).unsqueeze(0)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoints = {"c0": (args.c0_config, args.c0_checkpoint), "r1": (args.r1_config, args.r1_checkpoint)}
    report: dict[str, Any] = {
        "status": "COMPLETE",
        "historical_lq": str(args.historical_lq),
        "historical_lq_sha256": _sha256(args.historical_lq),
        "variant_contract": "stage2_0_blocks, stage2_1_block, and stage2_2_blocks each continue through original up2/dec2/dec1/context_terminal/inverse-Haar",
        "outputs": {},
    }
    for model_name, (config, checkpoint) in checkpoints.items():
        outputs, residuals, digest, full_error = _infer_stage2_ablation(lq, config, checkpoint, device)
        directory = root / model_name
        directory.mkdir(parents=True)
        for name, image in outputs.items():
            _write_rgb(directory / f"{name}.png", image)
            np.save(directory / f"{name}_context_residual_y4uv.npy", residuals[name].astype(np.float32))
        report["outputs"][model_name] = {
            "checkpoint_sha256": digest,
            "stage2_2_blocks_context_residual_reproduction_max_abs": full_error,
        }
    (root / "stage2_ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _context_from_stage2_feature(network: Any, debug: dict[str, Any], stage2_feature: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """@luna-comment Decode one stage2 feature through the unchanged production context tail."""
    residual2 = network.dec2(network.up2(stage2_feature))
    decoded0 = debug["stem_carry"] + network.dec1(network.up1(debug["phase_carry1"] + residual2))
    feature_residual = network.context_terminal(decoded0)
    if not network._deployment:
        feature_residual = (V5_CONTEXT_SCALE_LIMIT * torch.tanh(network.context_scale)) * feature_residual
    residual = network._decode_luma(feature_residual)
    return residual, _yuv444_to_rgb(y4uv_to_yuv444(debug["input"] + residual))


def _roi_detail_metrics(image: np.ndarray) -> dict[str, float]:
    """@luna-comment Measure fixed target ROI contrast and gradient sharpness without using stripe strength."""
    y = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    roi = y[12:78, 156:228]
    return {"contrast_std": float(roi.std()), "gradient_mean": float((np.abs(np.diff(roi, axis=0)).mean() + np.abs(np.diff(roi, axis=1)).mean()) / 2.0)}


def run_stage2_probe(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: direct historical-v2 LQ and frozen C0/R1 checkpoints. Output: real-tail outputs for stage2 block0, midpoint,
    block1, and contrast/sharpness-matched isotropic sharpening of block0. Key Steps: all probes rerun the unchanged
    downstream decoder; select sharpening lambda only from fixed ROI contrast and gradient metrics, never stripe score.
    Key Params: midpoint=.5; lambda candidates are fixed before inference. State/Dependencies: diagnostic-only frozen inference.
    """
    for path in (args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists(): raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    lq = torch.from_numpy(_read_rgb(args.historical_lq)).permute(2, 0, 1).unsqueeze(0)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {"status": "COMPLETE", "historical_lq_sha256": _sha256(args.historical_lq), "lambda_candidates": [0.25, 0.5, 0.75, 1.0, 1.5], "outputs": {}}
    for model_name, config, checkpoint in (("c0", args.c0_config, args.c0_checkpoint), ("r1", args.r1_config, args.r1_checkpoint)):
        network, digest = build_eval_network(checkpoint, device, config)
        with torch.inference_mode():
            debug = network(rgb_to_y4uv(lq).to(device), return_debug=True)
            phase2 = network.phase_pack2(network._run_blocks(debug["phase_carry1"], network.stage1))
            z0 = network.stage2[0](phase2); z1 = network.stage2[1](z0)
            residuals, images = {}, {}
            for name, value in (("t0_block0", z0), ("tmid_half_increment", z0 + 0.5 * (z1-z0)), ("t1_block1", z1)):
                residuals[name], images[name] = _context_from_stage2_feature(network, debug, value)
            target = _roi_detail_metrics(images["t1_block1"])
            blur = F.avg_pool2d(F.pad(z0, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            candidates = []
            for lam in report["lambda_candidates"]:
                residual, image = _context_from_stage2_feature(network, debug, z0 + float(lam) * (z0 - blur))
                metric = _roi_detail_metrics(image)
                score = abs(metric["contrast_std"]-target["contrast_std"])/max(target["contrast_std"],1e-8) + abs(metric["gradient_mean"]-target["gradient_mean"])/max(target["gradient_mean"],1e-8)
                candidates.append((score, float(lam), residual, image, metric))
            score, lam, residuals["tsharp_matched"], images["tsharp_matched"], sharp_metric = min(candidates, key=lambda row: row[0])
            directory = root / model_name; directory.mkdir()
            for name, image in images.items():
                _write_rgb(directory / f"{name}.png", image)
                np.save(directory / f"{name}_context_residual_y4uv.npy", residuals[name][0].detach().float().cpu().numpy())
            report["outputs"][model_name] = {"checkpoint_sha256": digest, "t1_reproduction_max_abs": float((residuals["t1_block1"]-debug["context_residual"]).abs().max()), "t1_metrics": target, "tsharp_lambda": lam, "tsharp_metrics": sharp_metric, "tsharp_match_score": score}
    (root / "stage2_probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def run_stage2_frequency_probe(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment Test the block1 increment's 12-row Nyquist component through the unchanged context tail."""
    for path in (args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    root=args.output_root.resolve()
    if root.exists(): raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True); lq=torch.from_numpy(_read_rgb(args.historical_lq)).permute(2,0,1).unsqueeze(0)
    device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu')
    report={'status':'COMPLETE','historical_lq_sha256':_sha256(args.historical_lq),'outputs':{}}
    for model_name, config, checkpoint in (("c0",args.c0_config,args.c0_checkpoint),("r1",args.r1_config,args.r1_checkpoint)):
        network,digest=build_eval_network(checkpoint,device,config)
        with torch.inference_mode():
            debug=network(rgb_to_y4uv(lq).to(device),return_debug=True)
            p2=network.phase_pack2(network._run_blocks(debug['phase_carry1'],network.stage1)); z0=network.stage2[0](p2); z1=network.stage2[1](z0); delta=z1-z0
            sign=torch.where(torch.arange(delta.shape[-2],device=device)%2==0,1.0,-1.0).view(1,1,-1,1)
            nyquist=(delta*sign).mean(dim=-2,keepdim=True)*sign
            variants={'full_block1':z1,'minus_block1_nyquist':z1-nyquist,'block0':z0}; directory=root/model_name; directory.mkdir()
            output_info={}
            for name,value in variants.items():
                residual,image=_context_from_stage2_feature(network,debug,value); _write_rgb(directory/f'{name}.png',image); np.save(directory/f'{name}_context_residual_y4uv.npy',residual[0].detach().float().cpu().numpy()); output_info[name]=_roi_detail_metrics(image)
            np.save(directory/'z0.npy',z0[0].detach().float().cpu().numpy()); np.save(directory/'z1.npy',z1[0].detach().float().cpu().numpy()); np.save(directory/'block1_increment.npy',delta[0].detach().float().cpu().numpy()); np.save(directory/'block1_increment_nyquist.npy',nyquist[0].detach().float().cpu().numpy())
            power=np.mean(np.abs(np.fft.rfft(delta[0].detach().float().cpu().numpy(),axis=1))**2,axis=(0,2)); report['outputs'][model_name]={'checkpoint_sha256':digest,'full_reproduction_max_abs':float(((_context_from_stage2_feature(network,debug,z1)[0])-debug['context_residual']).abs().max()),'increment_vertical_power':[float(x) for x in power],'roi_metrics':output_info}
    (root/'stage2_frequency_probe_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); return report


def run_cross_context_probe(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment Execute frozen C0/R1 block1 and decoder cross-tests on one historical LQ."""
    for path in (args.historical_lq,args.c0_config,args.c0_checkpoint,args.r1_config,args.r1_checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    root=args.output_root.resolve()
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu'); lq=torch.from_numpy(_read_rgb(args.historical_lq)).permute(2,0,1).unsqueeze(0)
    c0,c0hash=build_eval_network(args.c0_checkpoint,device,args.c0_config); r1,r1hash=build_eval_network(args.r1_checkpoint,device,args.r1_config)
    with torch.inference_mode():
        p=rgb_to_y4uv(lq).to(device); dc=c0(p,return_debug=True); dr=r1(p,return_debug=True)
        p2=c0.phase_pack2(c0._run_blocks(dc['phase_carry1'],c0.stage1)); z0=c0.stage2[0](p2); zc=c0.stage2[1](z0); zr=r1.stage2[1](z0); dzc=zc-z0; dzr=zr-z0
        rows={'h2_c0block1_c0tail':(c0,dc,zc),'h2_r1block1_c0tail':(c0,dc,zr),'h3_c0delta_c0tail':(c0,dc,z0+dzc),'h3_c0delta_r1tail':(r1,dc,z0+dzc),'h3_r1delta_c0tail':(c0,dc,z0+dzr),'h3_r1delta_r1tail':(r1,dc,z0+dzr)}
        report={'status':'COMPLETE','hashes':{'c0':c0hash,'r1':r1hash},'outputs':{}}
        for name,(net,debug,z) in rows.items():
            residual,img=_context_from_stage2_feature(net,debug,z); _write_rgb(root/f'{name}.png',img); np.save(root/f'{name}_context_residual_y4uv.npy',residual[0].detach().cpu().numpy()); report['outputs'][name]=_roi_detail_metrics(img)
        r1p2=r1.phase_pack2(r1._run_blocks(dr['phase_carry1'],r1.stage1)); r1z0=r1.stage2[0](r1p2)
        for name,value in {'c0_p2':p2,'c0_z0':z0,'c0_delta':dzc,'r1_delta_on_c0_z0':dzr,'native_r1_p2':r1p2,'native_r1_z0':r1z0}.items(): np.save(root/f'{name}.npy',value[0].detach().cpu().numpy())
    (root/'cross_context_probe_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); return report


def _stage2_branch_components(block: Any, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """@luna-comment Return exact unfused V5RepECB pre-PReLU branch tensors for a frozen diagnostic only."""
    if block.deploy:
        raise RuntimeError("stage2 branch probe requires unfused V5RepECB checkpoints")
    assert block.conv3x3 is not None
    components = {
        "b3": block.conv3x3(value),
        "b13": block.conv1x1_3x3(block.conv1x1(value)),
        "bx": block._edge_forward(value, block.edge_x_pw, block.edge_x_scale, block.edge_x_bias, block.edge_x_mask),
        "by": block._edge_forward(value, block.edge_y_pw, block.edge_y_scale, block.edge_y_bias, block.edge_y_mask),
        "bl": block._edge_forward(value, block.edge_l_pw, block.edge_l_scale, block.edge_l_bias, block.edge_l_mask),
    }
    if block.with_idt:
        components["bid"] = value
    return components


def _branch_sum(components: dict[str, torch.Tensor]) -> torch.Tensor:
    """@luna-comment Sum real ECB preactivation branches without incorrectly linearizing the subsequent PReLU."""
    return sum(components.values())


def _roi_delta_metrics(image: np.ndarray, baseline: np.ndarray, target_delta: np.ndarray) -> dict[str, float]:
    """@luna-comment Compare one decoded intervention's target ROI delta with the full R1 block1 delta."""
    y = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    base_y = baseline[..., 0] * 0.299 + baseline[..., 1] * 0.587 + baseline[..., 2] * 0.114
    delta = (y - base_y)[12:78, 156:228]
    numerator = float((delta * target_delta).sum())
    denominator = float(np.linalg.norm(delta) * np.linalg.norm(target_delta))
    return {
        **_roi_detail_metrics(image),
        "target_delta_roi_correlation": numerator / denominator if denominator else 0.0,
        "target_delta_roi_energy_ratio": float(np.linalg.norm(delta) / max(np.linalg.norm(target_delta), 1e-12)),
    }


def run_stage2_branch_probe(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Localize the frozen R1 stage2[1] source with shared C0 Z0, decoder, carries, and terminal. Leave-one-out and
    C0-branch-swap variants retain the true R1 PReLU after branch summation; branch tensors are exported only as
    bottleneck diagnostics, never decoded as independent images. No model or checkpoint is changed.
    """
    for path in (args.historical_lq, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    lq = torch.from_numpy(_read_rgb(args.historical_lq)).permute(2, 0, 1).unsqueeze(0)
    c0, c0hash = build_eval_network(args.c0_checkpoint, device, args.c0_config)
    r1, r1hash = build_eval_network(args.r1_checkpoint, device, args.r1_config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        c0_debug = c0(packed, return_debug=True)
        p2 = c0.phase_pack2(c0._run_blocks(c0_debug["phase_carry1"], c0.stage1))
        z0 = c0.stage2[0](p2)
        c0_parts = _stage2_branch_components(c0.stage2[1], z0)
        r1_parts = _stage2_branch_components(r1.stage2[1], z0)
        c0_q, r1_q = _branch_sum(c0_parts), _branch_sum(r1_parts)
        assert c0.stage2[1].act is not None and r1.stage2[1].act is not None
        variants: dict[str, torch.Tensor] = {
            "baseline_c0_full": c0.stage2[1].act(c0_q),
            "r1_full": r1.stage2[1].act(r1_q),
            "prelu_r1q_c0": c0.stage2[1].act(r1_q),
            "prelu_c0q_r1": r1.stage2[1].act(c0_q),
        }
        for name in ("b3", "b13", "bx", "by", "bl"):
            variants[f"r1_minus_{name}"] = r1.stage2[1].act(r1_q - r1_parts[name])
            variants[f"r1_swap_{name}_from_c0"] = r1.stage2[1].act(r1_q - r1_parts[name] + c0_parts[name])
        variants["r1_minus_b3_b13"] = r1.stage2[1].act(r1_q - r1_parts["b3"] - r1_parts["b13"])
        variants["r1_swap_b3_b13_from_c0"] = r1.stage2[1].act(
            r1_q - r1_parts["b3"] - r1_parts["b13"] + c0_parts["b3"] + c0_parts["b13"]
        )
        r1_b13_with_c0_pw = r1.stage2[1].conv1x1_3x3(c0.stage2[1].conv1x1(z0))
        r1_b13_with_c0_spatial = c0.stage2[1].conv1x1_3x3(r1.stage2[1].conv1x1(z0))
        variants["r1_b13_pw_from_c0"] = r1.stage2[1].act(r1_q - r1_parts["b13"] + r1_b13_with_c0_pw)
        variants["r1_b13_spatial3_from_c0"] = r1.stage2[1].act(r1_q - r1_parts["b13"] + r1_b13_with_c0_spatial)
        assert c0.stage2[1].conv3x3 is not None and r1.stage2[1].conv3x3 is not None
        c0_b3, r1_b3 = c0.stage2[1].conv3x3, r1.stage2[1].conv3x3
        assert c0_b3.bias is not None and r1_b3.bias is not None
        weight_delta, bias_delta = r1_b3.weight - c0_b3.weight, r1_b3.bias - c0_b3.bias
        tap_masks = {
            "center": torch.tensor(((0, 0, 0), (0, 1, 0), (0, 0, 0)), device=device, dtype=z0.dtype),
            "axial": torch.tensor(((0, 1, 0), (1, 0, 1), (0, 1, 0)), device=device, dtype=z0.dtype),
            "vertical": torch.tensor(((0, 1, 0), (0, 0, 0), (0, 1, 0)), device=device, dtype=z0.dtype),
            "horizontal": torch.tensor(((0, 0, 0), (1, 0, 1), (0, 0, 0)), device=device, dtype=z0.dtype),
            "corners": torch.tensor(((1, 0, 1), (0, 0, 0), (1, 0, 1)), device=device, dtype=z0.dtype),
        }
        for name, mask in tap_masks.items():
            restored_b3 = c0_parts["b3"] + F.conv2d(z0, weight_delta * mask.view(1, 1, 3, 3), padding=1)
            variants[f"r1_b3_c0_plus_{name}_delta"] = r1.stage2[1].act(r1_q - r1_parts["b3"] + restored_b3)
        restored_b3_bias = c0_parts["b3"] + bias_delta.view(1, -1, 1, 1)
        variants["r1_b3_c0_plus_bias_delta"] = r1.stage2[1].act(r1_q - r1_parts["b3"] + restored_b3_bias)
        decoded: dict[str, np.ndarray] = {}
        raw: dict[str, torch.Tensor] = {"c0_z0": z0, "c0_q": c0_q, "r1_q": r1_q, "c0_z": variants["baseline_c0_full"], "r1_z": variants["r1_full"]}
        raw.update({f"c0_{name}": value for name, value in c0_parts.items()})
        raw.update({f"r1_{name}": value for name, value in r1_parts.items()})
        for name, feature in variants.items():
            residual, image = _context_from_stage2_feature(c0, c0_debug, feature)
            decoded[name] = image
            _write_rgb(root / f"{name}.png", image)
            np.save(root / f"{name}_context_residual_y4uv.npy", residual[0].detach().float().cpu().numpy())
        baseline = decoded["baseline_c0_full"]
        full_y = decoded["r1_full"][..., 0] * 0.299 + decoded["r1_full"][..., 1] * 0.587 + decoded["r1_full"][..., 2] * 0.114
        base_y = baseline[..., 0] * 0.299 + baseline[..., 1] * 0.587 + baseline[..., 2] * 0.114
        target_delta = (full_y - base_y)[12:78, 156:228]
        report = {
            "status": "COMPLETE",
            "contract": "shared C0 P2/Z0, C0 decoder/carries/terminal; only stage2[1] internals vary; PReLU remains after full branch sum",
            "hashes": {"c0": c0hash, "r1": r1hash},
            "outputs": {name: _roi_delta_metrics(image, baseline, target_delta) for name, image in decoded.items()},
            "raw_nodes": {},
        }
        for name, value in raw.items():
            array = value[0].detach().float().cpu().numpy()
            np.save(root / f"{name}.npy", array)
            report["raw_nodes"][name] = {"rms": float(np.sqrt(np.mean(np.square(array)))), "shape": list(array.shape)}
    (root / "stage2_branch_probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _infer_subband_dump(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[dict[str, np.ndarray], str, float]:
    """@luna-comment
    Input: RGB LQ BCHW and a frozen V5 checkpoint. Output: signed full-resolution luma contributions reconstructed
    independently from local-terminal LL/LH/HL/HH bands, checkpoint hash, and exact local-residual closure error.
    Key Steps: request existing debug local_bands, zero all but one band, apply the model's own fixed embed/inverse-Haar,
    and unpack luma. Key Params: band order is LL/LH/HL/HH. State/Dependencies: frozen V5 diagnostic inference only.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        debug = network(packed, return_debug=True)
        bands = debug["local_bands"]
        contributions: dict[str, np.ndarray] = {}
        reconstructed = torch.zeros_like(debug["local_residual"])
        for index, name in enumerate(("ll", "lh", "hl", "hh")):
            one_band = torch.zeros_like(bands)
            one_band[:, index:index + 1] = bands[:, index:index + 1]
            decoded = network._decode_luma(network.local_embed(one_band))
            reconstructed = reconstructed + decoded
            y, _, _ = y4uv_to_yuv420(decoded)
            contributions[name] = y[0, 0].detach().float().cpu().numpy()
        closure = float((reconstructed - debug["local_residual"]).abs().max())
    return contributions, checkpoint_sha256, closure


def _raw_band_background_mask(gt_rgb: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
    """Return a native half-resolution flat-background selector derived only from the fixed GT.

    The selector uses 2x2 GT luma cells, requires low within-cell range, and
    requires closeness to the border-median luma. It is a reproducible
    background proxy for row spectra, not a replacement text annotation.
    """
    if gt_rgb.ndim != 4 or gt_rgb.shape[0] != 1 or gt_rgb.shape[1] != 3:
        raise ValueError("gt_rgb must be one [1,3,H,W] tensor")
    luma = 0.299 * gt_rgb[:, :1] + 0.587 * gt_rgb[:, 1:2] + 0.114 * gt_rgb[:, 2:3]
    mean = torch.nn.functional.avg_pool2d(luma, kernel_size=2, stride=2)
    maximum = torch.nn.functional.max_pool2d(luma, kernel_size=2, stride=2)
    minimum = -torch.nn.functional.max_pool2d(-luma, kernel_size=2, stride=2)
    height, width = mean.shape[-2:]
    border = torch.cat((mean[:, :, 0, :], mean[:, :, height - 1, :], mean[:, :, :, 0], mean[:, :, :, width - 1]), dim=-1)
    border_median = border.median()
    flat = (maximum - minimum) <= 0.02
    background = flat & ((mean - border_median).abs() <= 0.08)
    if int(background.sum()) == 0:
        background = flat
    if int(background.sum()) == 0:
        background = torch.ones_like(flat, dtype=torch.bool)
    mask = background[0, 0].detach().cpu().numpy().astype(bool)
    return mask, {
        "native_height": float(height),
        "native_width": float(width),
        "border_median_luma": float(border_median.detach().cpu()),
        "selected_fraction": float(mask.mean()),
    }


def _raw_band_spectrum(value: np.ndarray, background: np.ndarray) -> dict[str, Any]:
    """Summarize one native coefficient map with a background-selected vertical row spectrum.

    No resize, inverse Haar, PixelShuffle, or per-row normalization is applied.
    Rows without selected cells fall back to their whole-row mean so the
    exported profile remains defined for every native row.
    """
    if value.ndim != 2 or background.shape != value.shape:
        raise ValueError("raw band and background selector must have matching HW shape")
    counts = background.sum(axis=1)
    sums = (value * background).sum(axis=1)
    full_row = value.mean(axis=1)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), full_row).astype(np.float64)
    centered = profile - profile.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    candidates = np.argsort(power[1:])[::-1][:5] + 1
    return {
        "row_profile": profile.astype(np.float32).tolist(),
        "top_vertical_bins": [
            {"bin": int(index), "period_native_rows": float(value.shape[0] / index), "power": float(power[index])}
            for index in candidates
        ],
        "mean": float(value.mean()),
        "rms": float(np.sqrt(np.mean(np.square(value)))),
    }


def _infer_raw_subband_dump(
    lq: torch.Tensor, gt_rgb: torch.Tensor, config: Path, checkpoint: Path, device: torch.device
) -> tuple[dict[str, dict[str, np.ndarray]], str, dict[str, Any]]:
    """Export pre-inverse-Haar coefficient maps at their native local-grid resolution.

    ``input`` is the fixed Haar transform supplied to the local branch and
    ``predicted`` is the post-local_terminal coefficient tensor. The function
    deliberately does not call local_embed, inverse Haar, Y4 phase unpacking,
    resize, or image encoding on either exported array.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        debug = network(packed, return_debug=True)
        if not bool(debug["local_haar_switch"]):
            raise RuntimeError("raw Haar-band diagnostic requires local_haar_switch=true")
        raw_input = debug["local_input"]
        predicted = debug["local_bands"]
        if raw_input.shape != predicted.shape or raw_input.shape[1] != 4:
            raise RuntimeError("local raw Haar tensors must have matching four-band shapes")
        background, selector = _raw_band_background_mask(gt_rgb)
        maps: dict[str, dict[str, np.ndarray]] = {"input": {}, "predicted": {}}
        spectra: dict[str, dict[str, Any]] = {"input": {}, "predicted": {}}
        for tensor_name, tensor in (("input", raw_input), ("predicted", predicted)):
            for index, band_name in enumerate(("ll", "lh", "hl", "hh")):
                value = tensor[0, index].detach().float().cpu().numpy()
                maps[tensor_name][band_name] = value
                spectra[tensor_name][band_name] = _raw_band_spectrum(value, background)
        metadata = {
            "native_shape": [int(raw_input.shape[-2]), int(raw_input.shape[-1])],
            "no_decode_or_resize": True,
            "background_selector": selector,
            "spectra": spectra,
        }
    return maps, checkpoint_sha256, metadata


def run_raw_subband_dump(args: argparse.Namespace) -> dict[str, Any]:
    """Run append-only C0/R1 native Haar-coefficient localization on original, clean, and Gaussian inputs.

    Output arrays remain [48,240] for the configured 96x480 canvas. The
    decoded 96x480 contribution diagnostic is intentionally not produced by
    this mode, preventing inverse-Haar basis texture from being misread as a
    raw coefficient pattern.
    """
    for path in (args.manifest, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb_np = _read_rgb(Path(record["gt"]))
    gt_rgb = torch.from_numpy(gt_rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)
    original_lq, original_trace = _make_lq(gt_rgb, manifest["degradation"], item, device, "motion_original")
    inputs = {
        "original_v2_replay": original_lq,
        "clean_gt": gt_rgb,
        "float_gaussian_sigma_2p5": _float_gaussian(gt_rgb, 2.5),
    }
    checkpoints = {"c0": (args.c0_config, args.c0_checkpoint), "r1": (args.r1_config, args.r1_checkpoint)}
    report: dict[str, Any] = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
                              "band_order": ["ll", "lh", "hl", "hh"],
                              "original_replay_path_hit": original_trace["metadata"], "inputs": {}}
    for input_name, lq in inputs.items():
        input_report: dict[str, Any] = {}
        for model_name, (config, checkpoint) in checkpoints.items():
            maps, digest, metadata = _infer_raw_subband_dump(lq, gt_rgb, config, checkpoint, device)
            directory = root / input_name / model_name
            directory.mkdir(parents=True, exist_ok=True)
            for tensor_name, band_maps in maps.items():
                for band_name, value in band_maps.items():
                    np.save(directory / f"{tensor_name}_{band_name}_raw.npy", value.astype(np.float32))
            np.save(directory / "background_selector_raw.npy", _raw_band_background_mask(gt_rgb)[0].astype(np.uint8))
            input_report[model_name] = {"checkpoint_sha256": digest, **metadata}
        report["inputs"][input_name] = input_report
    (root / "raw_subband_dump_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_subband_dump(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: fixed GT and frozen C0/R1 checkpoints. Output: append-only LL/LH/HL/HH local residual contributions for
    clean and float-Gaussian inputs. Key Steps: use V5 debug-only local bands and exact inverse-Haar reconstruction;
    retain a common source input set for C0/R1 comparison. Key Params: no degradation chain, Gaussian sigma=2.5.
    State/Dependencies: diagnostic-only Remote58 inference.
    """
    for path in (args.manifest, args.c0_config, args.c0_checkpoint, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    inputs = {"clean_gt": gt, "float_gaussian_sigma_2p5": _float_gaussian(gt, 2.5)}
    checkpoints = {"c0": (args.c0_config, args.c0_checkpoint), "r1": (args.r1_config, args.r1_checkpoint)}
    report: dict[str, Any] = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
                              "band_order": ["ll", "lh", "hl", "hh"], "inputs": {}}
    for input_name, lq in inputs.items():
        row: dict[str, Any] = {}
        for model_name, (config, checkpoint) in checkpoints.items():
            maps, digest, closure = _infer_subband_dump(lq, config, checkpoint, device)
            directory = root / input_name / model_name
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in maps.items():
                np.save(directory / f"{name}_y_contribution.npy", value.astype(np.float32))
            row[model_name] = {"checkpoint_sha256": digest, "local_residual_closure_max_abs": closure,
                               "luma_rms": {name: float(np.sqrt(np.mean(np.square(value)))) for name, value in maps.items()}}
        report["inputs"][input_name] = row
    (root / "subband_dump_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _infer_subband_ablation(lq: torch.Tensor, config: Path, checkpoint: Path, device: torch.device) -> tuple[dict[str, np.ndarray], str, dict[str, float]]:
    """@luna-comment
    Input: RGB LQ BCHW and one frozen R1 V5 checkpoint. Output: full RGB plus diagnostic RGB outputs with exactly
    one local HL/LH/HH Haar contribution subtracted. Key Steps: reconstruct each contribution via existing debug bands,
    subtract only that decoded Y4UV term from full output, and record closure. Key Params: no inference parameter or
    checkpoint mutation; LL is excluded because it is not a directional high-frequency control. State/Dependencies: frozen diagnostic inference.
    """
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    with torch.inference_mode():
        packed = rgb_to_y4uv(lq).to(device)
        debug = network(packed, return_debug=True)
        bands = debug["local_bands"]
        contributions: dict[str, torch.Tensor] = {}
        reconstructed = torch.zeros_like(debug["local_residual"])
        for index, name in enumerate(("ll", "lh", "hl", "hh")):
            one_band = torch.zeros_like(bands); one_band[:, index:index + 1] = bands[:, index:index + 1]
            contributions[name] = network._decode_luma(network.local_embed(one_band))
            reconstructed = reconstructed + contributions[name]
        results = {"full": _yuv444_to_rgb(y4uv_to_yuv444(debug["output"]))}
        for name in ("hl", "lh", "hh"):
            results[f"minus_{name}"] = _yuv444_to_rgb(y4uv_to_yuv444(debug["output"] - contributions[name]))
        checks = {"local_residual_closure_max_abs": float((reconstructed - debug["local_residual"]).abs().max())}
    return results, checkpoint_sha256, checks


def run_hl_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """@luna-comment
    Input: fixed GT and frozen R1 checkpoint. Output: clean/Gaussian full, minus-HL, minus-LH, and minus-HH diagnostic
    images. Key Steps: omit one decoded local Haar contribution only after the original full forward, preserving context,
    input, UV, and all other local bands. Key Params: Gaussian sigma=2.5; controls are HL target and LH/HH direction checks.
    State/Dependencies: no training or checkpoint/model mutation.
    """
    for path in (args.manifest, args.r1_config, args.r1_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    root.mkdir(parents=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record, item = _find_variant(manifest, args.source_id, args.variant)
    gt_rgb = _read_rgb(Path(record["gt"]))
    gt = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    inputs = {"clean_gt": gt, "float_gaussian_sigma_2p5": _float_gaussian(gt, 2.5)}
    report: dict[str, Any] = {"status": "COMPLETE", "source_id": str(args.source_id), "variant": int(args.variant),
                              "diagnostic": "each minus image subtracts exactly one decoded local Haar contribution from full R1 output", "inputs": {}}
    for name, lq in inputs.items():
        outputs, digest, checks = _infer_subband_ablation(lq, args.r1_config, args.r1_checkpoint, device)
        directory = root / name
        for output_name, image in outputs.items():
            _write_rgb(directory / f"{output_name}.png", image); np.save(directory / f"{output_name}.npy", image.astype(np.float32))
        report["inputs"][name] = {"checkpoint_sha256": digest, **checks}
    (root / "hl_ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    arguments = _args()
    if sum((arguments.revised, arguments.localize, arguments.c2_controls, arguments.r1_necessity, arguments.branch_dump, arguments.context_ablation, arguments.stage2_ablation, arguments.stage2_probe, arguments.stage2_frequency_probe, arguments.cross_context_probe, arguments.stage2_branch_probe, arguments.subband_dump, arguments.raw_subband_dump, arguments.hl_ablation)) > 1:
        raise ValueError("diagnostic mode flags are mutually exclusive")
    outcome = run_stage2_branch_probe(arguments) if arguments.stage2_branch_probe else (run_cross_context_probe(arguments) if arguments.cross_context_probe else (run_hl_ablation(arguments) if arguments.hl_ablation else (run_raw_subband_dump(arguments) if arguments.raw_subband_dump else (run_subband_dump(arguments) if arguments.subband_dump else (run_stage2_frequency_probe(arguments) if arguments.stage2_frequency_probe else (run_stage2_probe(arguments) if arguments.stage2_probe else (run_stage2_ablation(arguments) if arguments.stage2_ablation else (run_context_ablation(arguments) if arguments.context_ablation else (run_branch_dump(arguments) if arguments.branch_dump else (run_r1_necessity(arguments) if arguments.r1_necessity else (run_c2_controls(arguments) if arguments.c2_controls else (run_localization(arguments) if arguments.localize else (run_revised(arguments) if arguments.revised else run(arguments))))))))))))))
    print(json.dumps({"status": outcome["status"]}, ensure_ascii=False))
