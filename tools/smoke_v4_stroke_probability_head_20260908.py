"""CPU functional smoke for the V4 training-only stroke probability branch."""
import torch

from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import BasicTextLineR4StudentY4UVV4
from realesrgan.losses.stroke_geometry_losses import StrokeProbabilityBCELoss, SoftCLDiceLoss, StrokeWidthLoss


def main():
    torch.manual_seed(7)
    net = BasicTextLineR4StudentY4UVV4(stroke_prob_enabled=True).train()
    source = torch.rand(1, 6, 48, 240, requires_grad=True)
    plain = net(source)
    restored, logits = net(source, return_stroke_logits=True)
    assert plain.shape == source.shape and logits.shape == (1, 1, 96, 480)
    assert torch.allclose(plain, restored, atol=0.0, rtol=0.0)
    phase = torch.tensor([[[[1.]], [[2.]], [[3.]], [[4.]]]])
    assert torch.equal(net._unpack_y4_logits(phase), torch.tensor([[[[1., 2.], [3., 4.]]]]))
    text = torch.zeros(1, 1, 96, 480); text[:, :, 28:68, 100:140] = 1
    valid = torch.ones_like(text); valid[:, :, :, 220:] = 0
    bce, details = StrokeProbabilityBCELoss()(logits, text, valid, return_details=True)
    # Use an unambiguous dark-stroke / light-background target.  This makes the
    # foreground gate genuinely active, so a nonzero geometry value proves that
    # clDice/width consume the probability branch instead of silently no-oping.
    target = torch.ones(1, 3, 96, 480)
    target[:, :, 28:68, 100:140] = 0
    rgb = target.clone().requires_grad_(True)
    width = StrokeWidthLoss(max_skeleton_iterations=64)(rgb, target, text, valid, pred_probability=logits.sigmoid())
    cldice = SoftCLDiceLoss(max_skeleton_iterations=64)(rgb, target, text, valid, pred_probability=logits.sigmoid())
    total = bce + width + cldice
    total.backward()
    assert torch.isfinite(total) and width > 0 and cldice > 0 and logits.grad is None and net.stroke_prob_head.weight.grad is not None
    before = net(source.detach())
    net.drop_stroke_head()
    after = net(source.detach())
    assert torch.allclose(before, after, atol=0.0, rtol=0.0)
    print("status=pass", {"bce": float(bce), "width": float(width), "cldice": float(cldice), "iou": float(details["iou"].mean())})


if __name__ == "__main__":
    main()
