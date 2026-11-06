"""Train32 V4 control: native + CRNN CP + VGG perceptual + RaGAN.

This is an independent generative-loss comparison arm.  It deliberately uses
no structure mask in its objective; masks remain diagnostics only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from basicsr.losses.basic_loss import PerceptualLoss
from diagnose_v4_d4_structure_hf_sensitivity import rgb_from_y4uv
from diagnose_v4_real_structure_masks import load_model
from overfit_v4_real_structure_hf import build_batch, evaluate
from realesrgan.archs.unet_discriminator_SN import UNetDiscriminatorSN
from realesrgan.losses.basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.losses.relativeGAN_loss import RelativisticSoftplusGANLoss


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--alphabet", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train"); parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--d-lr", type=float, default=5e-5)
    parser.add_argument("--cp-weight", type=float, default=5e-4); parser.add_argument("--vgg-weight", type=float, default=.10)
    parser.add_argument("--gan-weight", type=float, default=.005); parser.add_argument("--d-warmup", type=int, default=1000)
    parser.add_argument("--d-feat", type=int, default=32); parser.add_argument("--log-every", type=int, default=250); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    torch.manual_seed(20260901)
    batch = build_batch(args.root, args.split, args.ids, device)
    generator = load_model(args.checkpoint, device).train()
    discriminator = UNetDiscriminatorSN(3, num_feat=args.d_feat, skip_connection=True).to(device).train()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(.9, .99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(.9, .99))
    native = Y4UVNativeMaskedL1LossV3().to(device)
    cp = CRNNTextLineTileContentLoss(recognizer_path=str(args.recognizer), alphabet_path=str(args.alphabet), layer_weights=(1.4, 1.4, 1.4, .4, .4), micro_batch_size=8).to(device)
    vgg = PerceptualLoss(layer_weights={"conv1_2": .1, "conv2_2": .1, "conv3_4": 1., "conv4_4": 1., "conv5_4": 1.}, vgg_type="vgg19", use_input_norm=True, perceptual_weight=1., style_weight=0., range_norm=False, criterion="l1").to(device).eval()
    set_requires_grad(vgg, False)
    ragan = RelativisticSoftplusGANLoss(loss_weight=1., average=True).to(device)
    metadata = {"valid_hw": [[96, width] for width in batch["widths"]], "orientation": ["HORIZONTAL"] * len(args.ids), "cp_eligible": [True] * len(args.ids)}
    history = []
    for step in range(1, args.iterations + 1):
        optimizer_g.zero_grad(set_to_none=True); set_requires_grad(discriminator, False)
        output = generator(batch["lq"]); pred = rgb_from_y4uv(output).clamp(0, 1)
        l_native = native(output, batch["gt"], valid_mask=batch["valid"], text_mask=batch["text"])
        l_cp = cp(pred, batch["gt_rgb"], metadata)
        l_vgg, _ = vgg(pred, batch["gt_rgb"])
        l_gan = pred.new_zeros(())
        if step > args.d_warmup:
            real_g = discriminator(batch["gt_rgb"]).detach().clone()
            fake_g = discriminator(pred)
            l_gan = ragan(real_g, fake_g, is_disc=False)
        total_g = l_native + args.cp_weight * l_cp + args.vgg_weight * l_vgg + args.gan_weight * l_gan
        total_g.backward(); optimizer_g.step()
        l_d = pred.new_zeros(())
        if step > args.d_warmup:
            set_requires_grad(discriminator, True); optimizer_d.zero_grad(set_to_none=True)
            real_d = discriminator(batch["gt_rgb"]); fake_d = discriminator(pred.detach())
            l_d = ragan(real_d, fake_d, is_disc=True); l_d.backward(); optimizer_d.step()
        if step == 1 or step % args.log_every == 0 or step == args.iterations:
            item = {"iter": step, "native": float(l_native.detach()), "cp_weighted": float((args.cp_weight * l_cp).detach()), "vgg_raw": float(l_vgg.detach()), "vgg_weighted": float((args.vgg_weight * l_vgg).detach()), "gan_weighted": float((args.gan_weight * l_gan).detach()), "d_loss": float(l_d.detach()), "total_g": float(total_g.detach())}
            history.append(item); print(json.dumps(item), flush=True)
    torch.save({"params": generator.state_dict()}, args.out / "net_g_final.pth")
    torch.save({"params": discriminator.state_dict()}, args.out / "net_d_final.pth")
    report = {"setup": {"ids": args.ids, "iterations": args.iterations, "objective": "native + CP + VGG19 perceptual + relativistic-average GAN", "cp_weight": args.cp_weight, "vgg_weight": args.vgg_weight, "gan_weight": args.gan_weight, "d_warmup": args.d_warmup, "d_feat": args.d_feat}, "history": history, "final_regions": evaluate(generator, batch, args.ids, args.out, "final")}
    (args.out / "overfit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
