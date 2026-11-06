import torch
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import numpy as np

def get_local_weights(residual, ksize):

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):

    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
    residual_SR = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = torch.var(residual_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
    pixel_level_weight = get_local_weights(residual_SR.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_SR < residual_ema] = 0

    return overall_weight

def compute_gradient_penalty(D, real_samples, fake_samples):

    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty