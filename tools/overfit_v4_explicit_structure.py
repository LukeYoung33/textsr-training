"""Train32 V4 experiment: explicit occupancy head and Text-Gestalt-style attention.

This is intentionally an experimental wrapper.  The pretrained V4 backbone,
Y4UV contract, Haar transform and native output head stay unchanged; only a
small structure branch is added after the V4 decoded feature tensor.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnose_v4_d4_structure_hf_sensitivity import rgb_from_y4uv
from diagnose_v4_real_structure_masks import load_model
from overfit_v4_real_structure_hf import build_batch, evaluate
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.losses.structure_tversky_loss import tversky_structure_loss


class ExplicitStructureV4(nn.Module):
    """V4 plus a one-channel HR stroke map that controls an added residual path."""

    def __init__(self, base: nn.Module, mode: str) -> None:
        super().__init__()
        if mode not in {"concat", "attention"}:
            raise ValueError(mode)
        self.base = base
        self.mode = mode
        # V4 decoded feature is six channels at 48 x 240; pixel shuffle makes
        # one native 96 x 480 occupancy logit without an interpolation target.
        self.structure_head = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.PReLU(16), nn.Conv2d(16, 4, 3, padding=1), nn.PixelShuffle(2),
        )
        self.guided_residual = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1), nn.PReLU(16), nn.Conv2d(16, 6, 3, padding=1),
        )
        nn.init.zeros_(self.guided_residual[-1].weight)
        nn.init.zeros_(self.guided_residual[-1].bias)

    def forward(self, input_y4uv: torch.Tensor, oracle_mask: torch.Tensor | None = None):
        base = self.base
        features = base._prepare_features(input_y4uv)
        decoded = base._decode_residual(base.backbone(features))
        baseline = input_y4uv + base.refine_head(decoded)
        logits = self.structure_head(decoded)
        prob = logits.sigmoid() if oracle_mask is None else oracle_mask
        half_prob = F.avg_pool2d(prob, 2, 2)
        # concat: retain signed confidence. attention: Text-Gestalt-inspired
        # spatial attention gates only the correction, protecting background.
        if self.mode == "attention":
            correction = self.guided_residual(torch.cat((decoded * half_prob, half_prob), dim=1)) * half_prob
        else:
            correction = self.guided_residual(torch.cat((decoded, half_prob), dim=1))
        return baseline + correction, logits


def run_arm(name, base_state, batch, args, native, cp, device):
    base = load_model(args.checkpoint, device)
    base.load_state_dict(base_state, strict=True)
    mode = "attention" if name == "textgestalt_attention" else "concat"
    model = ExplicitStructureV4(base, mode).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    full_text = torch.from_numpy(np.stack(batch["strokes"]).astype("float32")).to(device).unsqueeze(1)
    full_valid = torch.zeros_like(full_text)
    for i, width in enumerate(batch["widths"]):
        full_valid[i, :, :, :width] = 1.0
    metadata = {"valid_hw": [[96, w] for w in batch["widths"]], "orientation": ["HORIZONTAL"] * len(args.ids), "cp_eligible": [True] * len(args.ids)}
    history = []
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        # The oracle arm is a diagnostic: GT geometry is supplied only to the
        # residual decoder, while the learned structure head is still measured.
        oracle = full_text if name == "tversky_oracle" else None
        output, logits = model(batch["lq"], oracle)
        l_native = native(output, batch["gt"], valid_mask=batch["valid"], text_mask=batch["text"])
        l_cp = cp(rgb_from_y4uv(output), batch["gt_rgb"], metadata)
        l_structure = tversky_structure_loss(logits, full_text, full_valid, alpha=args.alpha, beta=args.beta)
        total = l_native + args.cp_weight * l_cp + args.structure_weight * l_structure
        total.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.iterations:
            item = {"iter": step, "native": float(l_native.detach()), "cp_weighted": float((args.cp_weight * l_cp).detach()), "tversky_weighted": float((args.structure_weight * l_structure).detach()), "total": float(total.detach())}
            history.append(item)
            print(json.dumps({name: item}), flush=True)
    arm_out = args.out / name
    arm_out.mkdir(parents=True, exist_ok=True)
    torch.save({"params": model.state_dict()}, arm_out / "net_g_final.pth")
    # evaluate accepts a module returning one tensor; adapter avoids changing it.
    class ImageOnly(nn.Module):
        def __init__(self, wrapped): super().__init__(); self.wrapped = wrapped
        def forward(self, x): return self.wrapped(x)[0]
    return {"history": history, "final_regions": evaluate(ImageOnly(model), batch, args.ids, arm_out, "final")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train"); parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--cp-weight", type=float, default=5e-4)
    parser.add_argument("--structure-weight", type=float, default=1.0); parser.add_argument("--alpha", type=float, default=.3); parser.add_argument("--beta", type=float, default=.7)
    parser.add_argument("--log-every", type=int, default=250); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    torch.manual_seed(20260831); np.random.seed(20260831)
    batch = build_batch(args.root, args.split, args.ids, device)
    seed = load_model(args.checkpoint, device); base_state = copy.deepcopy(seed.state_dict()); del seed
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet), layer_weights=(1.4, 1.4, 1.4, .4, .4), micro_batch_size=8).to(device)
    report = {"setup": {"architecture": "BasicTextLineR4StudentY4UVV4 + experimental structure branch", "ids": args.ids, "iterations": args.iterations, "alpha": args.alpha, "beta": args.beta, "cp_weight": args.cp_weight, "structure_weight": args.structure_weight}}
    for arm in ("tversky_concat", "tversky_oracle", "textgestalt_attention"):
        report[arm] = run_arm(arm, base_state, batch, args, native, cp, device)
    (args.out / "overfit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
