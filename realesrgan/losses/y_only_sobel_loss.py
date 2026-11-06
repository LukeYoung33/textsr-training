import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class YOnlySobelLoss(nn.Module):
    """Measure masked Sobel error on the luminance channel only."""

    def __init__(
            self,
            loss_weight=1.0,
            reduction='mean',
            text_weight=1.0,
            bg_weight=0.0,
            eps=1e-6):
        """Initialize fixed Sobel kernels and region weighting parameters.

        @shanlan-comment
        Input: scalar loss/reduction settings and foreground/background mask weights.
        Output: a stateless loss module with device-movable Sobel buffers.
        Key Steps: validate reduction, normalize numeric options, and register horizontal/vertical kernels.
        Key Params: ``text_weight``/``bg_weight`` select masked supervision strength; ``eps`` stabilizes magnitude.
        State/Dependencies: stores immutable Sobel buffers and configuration used by ``forward``.
        """
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. '
                "Supported ones are: ('none', 'mean', 'sum')")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.text_weight = float(text_weight)
        self.bg_weight = float(bg_weight)
        self.eps = float(eps)
        self.register_buffer(
            'sobel_x',
            torch.tensor(
                [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer(
            'sobel_y',
            torch.tensor(
                [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                dtype=torch.float32).view(1, 1, 3, 3))

    def _gradient(self, luminance):
        """Compute differentiable Sobel magnitude for a BCHW luminance tensor.

        @shanlan-comment
        Input: BCHW tensor containing exactly one normalized Y channel.
        Output: BCHW non-negative Sobel magnitude with the same spatial shape.
        Key Steps: cast registered kernels to the input device/dtype, convolve, and stabilize magnitude.
        Key Params: ``eps`` prevents zero-gradient square-root instability.
        State/Dependencies: uses module Sobel buffers and preserves gradients to the input tensor.
        """
        kernel_x = self.sobel_x.to(device=luminance.device, dtype=luminance.dtype)
        kernel_y = self.sobel_y.to(device=luminance.device, dtype=luminance.dtype)
        grad_x = F.conv2d(luminance, kernel_x, padding=1)
        grad_y = F.conv2d(luminance, kernel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def forward(self, pred, target, mask=None, weight=None, **kwargs):
        """Return foreground-weighted Sobel error using only channel zero.

        @shanlan-comment
        Input: BCHW prediction/target tensors in YUV-like channel order and optional BCHW/ BHW mask.
        Output: scalar or unreduced luminance-edge L1 loss multiplied by ``loss_weight``.
        Key Steps: validate channels, slice Y only, compute Sobel magnitudes, build text/background weights,
            optionally apply external weights, and reduce according to ``reduction``.
        Key Params: ``text_weight`` and ``bg_weight`` control mask regions; U/V channels are never read.
        State/Dependencies: mask is resized with nearest interpolation; no mutable external state is changed.
        """
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError('YOnlySobelLoss expects BCHW prediction and target tensors.')
        if pred.shape[1] < 1 or target.shape[1] < 1:
            raise ValueError('YOnlySobelLoss requires at least one channel in prediction and target.')

        pred_y = pred[:, 0:1]
        target_y = target[:, 0:1]
        grad_loss = torch.abs(self._gradient(pred_y) - self._gradient(target_y))

        if mask is None:
            region_weight = torch.ones_like(grad_loss) * self.text_weight
        else:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.dim() != 4:
                raise ValueError('YOnlySobelLoss mask must be BHW or BCHW.')
            if mask.shape[1] < 1:
                raise ValueError('YOnlySobelLoss mask must contain at least one channel.')
            mask = F.interpolate(
                mask[:, :1].to(device=grad_loss.device, dtype=grad_loss.dtype),
                size=grad_loss.shape[-2:],
                mode='nearest')
            region_weight = (
                (mask > 0).to(grad_loss.dtype) * self.text_weight
                + (mask <= 0).to(grad_loss.dtype) * self.bg_weight)

        loss = grad_loss * region_weight
        if weight is not None:
            if weight.dim() == 3:
                weight = weight.unsqueeze(1)
            weight = F.interpolate(
                weight.to(device=loss.device, dtype=loss.dtype),
                size=loss.shape[-2:],
                mode='nearest')
            loss = loss * weight[:, :1]

        if self.reduction == 'none':
            return loss * self.loss_weight
        if self.reduction == 'sum':
            return loss.sum() * self.loss_weight
        return loss.mean() * self.loss_weight
