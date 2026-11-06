import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import sys
sys.path.append('.')
from basicsr.utils.registry import LOSS_REGISTRY
import realesrgan.metrics.crnn as crnn
import numpy as np


@LOSS_REGISTRY.register()
class SobelGradientLoss(nn.Module):
    def __init__(self, loss_weight=1e-2, reduction='mean'):
        super(SobelGradientLoss, self).__init__()
        self.cri_func = nn.L1Loss(reduce=False)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.soble_filter_x, self.soble_filter_y = self._build_sobel_filters()
    
    def _build_sobel_filters(self):
        sobel_filter_x = torch.tensor([[-1., 0., 1.],
                                                                    [-2., 0., 2.],
                                                                    [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_filter_y = torch.tensor([[-1., -2., -1.],
                                                                    [0., 0., 0.],
                                                                    [1., 2., 1.]]).view(1, 1, 3, 3)
        return sobel_filter_x, sobel_filter_y
    
    def _compute_gradient(self, img:torch.Tensor):
        # Expand kernels to match input tensor
        channels = img.shape[1]
        sobel_filter_x = self.soble_filter_x.repeat(channels, 1, 1, 1).to(img.device)
        sobel_filter_y = self.soble_filter_y.repeat(channels, 1, 1, 1).to(img.device)
        
        # Apply filters
        grad_x = F.conv2d(img, sobel_filter_x, padding=1, groups=channels)
        grad_y = F.conv2d(img, sobel_filter_y, padding=1, groups=channels)
        
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return gradient
    
    def forward(self, out_images, target_images, mask=None):
        grad_out = self._compute_gradient(out_images)
        grad_target = self._compute_gradient(target_images)
        
        if mask is None:
            mask = torch.ones_like(grad_out[:, :1, :, :])
        weight_mask = (mask > 0).float() * 1.0 + (mask == 0).float() * 0.0
        
        g_loss = self.cri_func(grad_out, grad_target) * weight_mask
        return self.loss_weight * g_loss.mean()
        