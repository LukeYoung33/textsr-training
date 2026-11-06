"""Restore images using a fixed full-frame checkpoint, without OCR/detection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bootstrap_inference_imports():
    """@luna-comment
    Input: fresh standalone CLI process with local repository path.
    Output: minimal namespace packages for explicit inference imports.
    Key Steps: skip legacy top-level registry scans that initialize OCR globally.
    Key Params: only selected package initializers are bypassed; existing modules remain.
    State/Dependencies: process-local sys.modules; never changes legacy source files.
    """
    for name in ("basicsr", "realesrgan", "realesrgan.data", "realesrgan.archs"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            package.__package__ = name
            sys.modules[name] = package


if __name__ == "__main__":
    bootstrap_inference_imports()

from realesrgan.fullframe_text_priority_inference import restore_fullframe_rgb, tile_geometry


def load_fixed_network(config, checkpoint, device, param_key="auto"):
    """@luna-comment
    Input: parsed YAML, existing checkpoint path, device, parameter key.
    Output: strictly loaded eval network and selected key.
    Key Steps: reject unsupported preprocessing, construct registry arch, load weights only.
    Key Params: auto prefers params_ema then params; raw state dictionaries accepted.
    State/Dependencies: BasicSR architecture registry; no downloads or model mutation.
    """
    from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import BasicTextLineR4StudentY4UVV5

    for settings in [config.get("lq_preprocess"), config.get("inference", {}).get("preprocess")]:
        if settings not in (None, False):
            if not isinstance(settings, dict) or settings.get("enabled", False):
                raise ValueError("full-frame inference supports disabled preprocessing only")
    if config.get("scale", 1) != 1:
        raise ValueError("full-frame pipeline requires scale=1")
    architecture = config.get("network_g")
    if not isinstance(architecture, dict):
        raise ValueError("config requires network_g")
    options = dict(architecture)
    if options.pop("type", None) != "BasicTextLineR4StudentY4UVV5":
        raise ValueError("full-frame inference requires BasicTextLineR4StudentY4UVV5")
    network = BasicTextLineR4StudentY4UVV5(**options).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    selected = param_key
    if selected == "auto":
        selected = "params_ema" if "params_ema" in saved else ("params" if "params" in saved else "raw")
    state = saved if selected == "raw" else saved.get(selected)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no {selected} parameter dictionary")
    network.load_state_dict(state, strict=True)
    return network.eval(), selected


def sha256_file(path):
    """@luna-comment
    Input: readable file path.
    Output: SHA256 hex digest.
    Key Steps: stream bounded chunks through SHA256.
    Key Params: chunk size bounds temporary memory.
    State/Dependencies: filesystem read only; hashlib.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    """@luna-comment
    Input: CLI config/checkpoint/input/output paths and device/tile options.
    Output: restored PNG files plus provenance.json; returns report.
    Key Steps: validate configuration, load fixed weights, restore each RGB image.
    Key Params: inference YAML supplies canvas/halo/stride; CLI overrides batching/device.
    State/Dependencies: filesystem outputs, OpenCV codec, torch, architecture registry.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Single image or directory (nonrecursive)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--param-key", default="auto", choices=("auto", "params", "params_ema", "raw"))
    args = parser.parse_args(argv)
    config_path, checkpoint = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    source, destination = Path(args.input).resolve(), Path(args.output).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    inference = config.get("inference", {})
    canvas = inference.get("canvas_hw", config.get("network_g", {}).get("canvas_hw", (256, 256)))
    halo, stride = inference.get("halo", 32), inference.get("stride")
    tile_geometry(canvas, halo, stride)
    batch_size = args.batch_size if args.batch_size is not None else inference.get("batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not source.exists():
        raise FileNotFoundError(source)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    inputs = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not inputs:
        raise ValueError("input directory contains no supported images")
    outputs = [destination / (p.name + ".restored.png") for p in inputs]
    if any(path.exists() for path in outputs) or (destination / "provenance.json").exists():
        raise FileExistsError("output exists; choose a fresh output directory")
    network, key = load_fixed_network(config, checkpoint, torch.device(args.device), args.param_key)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path, output_path in zip(inputs, outputs):
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"input must be three-channel uint8/uint16 RGB: {image_path}")
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1).float()
        tensor = tensor / np.iinfo(image.dtype).max
        restored, geometry = restore_fullframe_rgb(network, tensor, canvas_hw=canvas, halo=halo,
            stride=stride, batch_size=batch_size, device=args.device, return_report=True)
        encoded_image = restored.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()[:, :, ::-1]
        ok, encoded = cv2.imencode(".png", encoded_image)
        if not ok:
            raise RuntimeError(f"cannot encode output: {output_path}")
        encoded.tofile(str(output_path))
        records.append({"input": str(image_path), "input_sha256": sha256_file(image_path),
                        "input_dtype": str(image.dtype), "output_dtype": "uint8",
                        "output": str(output_path), "output_sha256": sha256_file(output_path), **geometry})
    report = {"config": str(config_path), "config_sha256": sha256_file(config_path),
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
              "parameter_key": key, "device": args.device, "preprocessing": "disabled",
              "color_contract": "RGB -> Y4UV420 -> YUV444 -> RGB; unclamped network output",
              "file_output_clamp": True, "images": records}
    (destination / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "images": len(records), "checkpoint_sha256": report["checkpoint_sha256"]}))
    return report


if __name__ == "__main__":
    main()
