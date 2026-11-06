import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _check_reduction(reduction):
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError(f'Unsupported reduction mode: {reduction}')


def _reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


@LOSS_REGISTRY.register()
class FFTAmplitudeLoss(nn.Module):
    """L1 loss on FFT amplitude spectrum.

    Penalises missing high-frequency content (sharp edges, fine strokes)
    without adversarial instability. Intended as a lightweight complement to
    GAN loss — keeps G from sacrificing HF structure while D is winning.

    Args:
        loss_weight (float): Overall loss weight. Suggested range: 1e-4 ~ 5e-3.
        reduction (str): 'mean' | 'sum' | 'none'.
        high_freq_alpha (float): Linear slope for frequency-dependent weighting.
            0.0 = uniform weight across all frequencies.
            1.0 = weight rises from 1.0 (DC) to 2.0 (Nyquist) — mild HF emphasis.
            2.0 = weight rises from 1.0 to 3.0 — strong HF emphasis.
        norm (str): FFT normalisation passed to torch.fft.rfft2 ('ortho' recommended).
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
        high_freq_alpha: float = 0.0,
        norm: str = 'ortho',
    ):
        super().__init__()
        _check_reduction(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.high_freq_alpha = high_freq_alpha
        self.norm = norm

    def forward(self, pred, target, **kwargs):
        C = pred.shape[1]
        target = target[:, :C]

        pred_amp   = torch.abs(torch.fft.rfft2(pred,   norm=self.norm))
        target_amp = torch.abs(torch.fft.rfft2(target, norm=self.norm))

        loss = torch.abs(pred_amp - target_amp)

        if self.high_freq_alpha > 0.0:
            H, Wh = pred_amp.shape[-2:]
            fh = torch.arange(H,  device=pred.device, dtype=pred.dtype).div(H)
            fw = torch.arange(Wh, device=pred.device, dtype=pred.dtype).div(Wh)
            # freq_w shape: [H, Wh] → broadcast over [B, C, H, Wh]
            freq_w = (fh[:, None] + fw[None, :]).mul(self.high_freq_alpha).add_(1.0)
            loss = loss * freq_w.unsqueeze(0).unsqueeze(0)

        return self.loss_weight * _reduce_loss(loss, self.reduction)
