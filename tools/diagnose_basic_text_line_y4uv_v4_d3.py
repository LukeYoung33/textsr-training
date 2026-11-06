"""D3 fixed-sample capacity probe for the production BASIC V4 Y4UV+Haar graph.

Runs two short, independent optimizations with the production model, checkpoint,
Adam settings, native loss, and CP weight: GT->GT and LQ->GT.  It writes only a
JSON learning curve and never changes the source checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, yuv420_mask_to_y4uv, y4uv_to_yuv444
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.utils import YUV2RGB


def read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def load(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    model.load_state_dict({str(k).removeprefix("module."): v for k, v in state.items()}, strict=True)


def text_mask(gt: torch.Tensor) -> torch.Tensor:
    gray = cv2.cvtColor((gt[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return torch.from_numpy((gray <= threshold).astype(np.float32)).view(1, 1, 96, 480).to(gt.device)


def rgb_from_y4uv(value: torch.Tensor) -> torch.Tensor:
    return YUV2RGB(y4uv_to_yuv444(value), format="RGB", is_train=True, uv_normalized=True)


def run_case(name, input_rgb, gt_rgb, checkpoint, native, cp, text, iterations, device):
    model = BasicTextLineR4StudentY4UVV4().to(device).train()
    load(model, checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    target = rgb_to_y4uv(gt_rgb)
    valid = torch.ones_like(target)
    text_y4uv = yuv420_mask_to_y4uv(text)
    metadata = {"valid_hw": [[96, 480]], "orientation": ["HORIZONTAL"], "cp_eligible": [True]}
    snapshots = []
    report_at = {0, 10, 50, 100, iterations}
    for step in range(iterations + 1):
        output = model(rgb_to_y4uv(input_rgb))
        l_native = native(output, target, valid_mask=valid, text_mask=text_y4uv)
        l_cp = cp(rgb_from_y4uv(output), gt_rgb, metadata)
        total = l_native + 5e-4 * l_cp
        if step in report_at:
            snapshots.append({"iter": step, "native_l1": float(l_native.detach()), "cp_raw": float(l_cp.detach()), "total": float(total.detach())})
        if step == iterations:
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
    return {"case": name, "iterations": iterations, "curve": snapshots}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--recognizer", required=True, type=Path)
    parser.add_argument("--alphabet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    lq, gt = read_rgb(args.sample_dir / "lq_source_rgb.png", device), read_rgb(args.sample_dir / "gt_source_rgb.png", device)
    if tuple(lq.shape[-2:]) != (96, 480) or tuple(gt.shape[-2:]) != (96, 480):
        raise ValueError("fixture must be 96x480")
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet), layer_weights=(1.4, 1.4, 1.4, 0.4, 0.4), micro_batch_size=1).to(device)
    mask = text_mask(gt)
    result = {
        "model": "BasicTextLineR4StudentY4UVV4", "checkpoint": str(args.checkpoint),
        "optimizer": "Adam(lr=1e-4, betas=[0.9,0.99])", "loss": "native_l1 + 5e-4 * CP",
        "cases": [
            run_case("gt_to_gt", gt, gt, args.checkpoint, native, cp, mask, args.iterations, device),
            run_case("lq_to_gt", lq, gt, args.checkpoint, native, cp, mask, args.iterations, device),
        ],
    }
    (args.out / "d3_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
