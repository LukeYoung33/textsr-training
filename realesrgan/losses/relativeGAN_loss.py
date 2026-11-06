import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class RelativisticSoftplusGANLoss(nn.Module):
    def __init__(self, loss_weight=1.0, average=False):
        super(RelativisticSoftplusGANLoss, self).__init__()
        self.loss_weight = loss_weight
        self.average = average
        
    def forward(self, real_pred, fake_pred, is_disc=False):
        if is_disc:
            if self.average:
                relativisticLogits = real_pred - fake_pred.mean(dim=0, keepdim=True)
            else:
                relativisticLogits = real_pred - fake_pred
            loss = F.softplus(-relativisticLogits).mean()
        else:
            if self.average:
                relativisticLogits = fake_pred - real_pred.mean(dim=0, keepdim=True)
            else:
                relativisticLogits = fake_pred - real_pred
            loss = self.loss_weight * F.softplus(-relativisticLogits).mean()
        return loss