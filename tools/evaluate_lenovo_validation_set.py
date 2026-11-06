"""Evaluate V4 checkpoints on a fixed Lenovo synthetic validation manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB


def _read(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load(path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = next((state[key] for key in ("params_ema", "params", "state_dict") if isinstance(state.get(key), dict)), state)
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    has_probability_head = any(key.startswith("stroke_prob_head.") for key in state)
    model = BasicTextLineR4StudentY4UVV4(haar_switch=True, stroke_prob_enabled=has_probability_head).to(device).eval()
    model.load_state_dict(state, strict=True)
    return model


def _groups(record: dict) -> list[str]:
    metadata = record["stages"]
    result = ["all", f"capture:{metadata['capture_scenario']}"]
    result.extend(f"stage:{stage['branch']}" for stage in metadata["stages"])
    return result


def _summarize(values: list[tuple[float, float]]) -> dict:
    l1 = np.array([item[0] for item in values], dtype=np.float64)
    mse = np.array([item[1] for item in values], dtype=np.float64)
    return {"samples": int(len(values)), "rgb_l1": float(l1.mean()), "y_psnr": float((-10.0 * np.log10(np.maximum(mse, 1e-12))).mean())}


def evaluate(manifest_path: Path, checkpoint_paths: list[Path], output: Path, batch_size: int, device: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = [(record["gt"], variant) for record in manifest["records"] for variant in record["variants"]]
    target = torch.device(device)
    report = {"manifest": str(manifest_path), "pairs": len(pairs), "models": {}}
    for checkpoint in checkpoint_paths:
        model = _load(checkpoint, target)
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        with torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset:offset + batch_size]
                lq = torch.stack([_read(item[1]["lq"]) for item in batch]).to(target)
                gt = torch.stack([_read(item[0]) for item in batch]).to(target)
                native = rgb_to_y4uv(lq)
                prediction = YUV2RGB(y4uv_to_yuv444(model(native)), format="RGB", is_train=True, uv_normalized=True).clamp(0, 1)
                l1 = (prediction - gt).abs().mean(dim=(1, 2, 3)).cpu().numpy()
                y_error = (0.299 * (prediction[:, 0] - gt[:, 0]) + 0.587 * (prediction[:, 1] - gt[:, 1]) + 0.114 * (prediction[:, 2] - gt[:, 2])).square().mean(dim=(1, 2)).cpu().numpy()
                for item, l1_value, mse_value in zip(batch, l1, y_error):
                    for group in _groups(item[1]):
                        grouped[group].append((float(l1_value), float(mse_value)))
        report["models"][checkpoint.parent.parent.name] = {key: _summarize(value) for key, value in sorted(grouped.items())}
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(evaluate(args.manifest, args.checkpoint, args.output, args.batch_size, args.device))


if __name__ == "__main__":
    main()
