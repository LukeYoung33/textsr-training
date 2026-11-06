"""Accepted C3 multiscale fields and luma edge protection, shared by previews and training."""
import math
import torch
from torch.nn import functional as F
from realesrgan import lenovo_atomic_torch as ops

VERSION = "c3_multiscale_edge_protected_v1"
SCHEMA = "c3_multiscale_edge_v1"


@torch.no_grad()
def apply(image, generator, params=None):
    """Execute the accepted complete chain without changing its operator order."""
    params = params or {}
    amplitude = float(params.get('strength', 6.))
    sigmas = tuple(params.get('sigmas', (1.5, 4., 9.)))
    scale = float(params.get('scale', .5))
    rho = float(params.get('rho', -.75))
    if not 0 <= amplitude <= 255 or len(sigmas) != 3 or any(s <= 0 for s in sigmas):
        raise ValueError('Invalid C3 amplitude or multiscale sigmas')
    if not -1 < rho < 1 or not 0 < scale <= 1 or min(image.shape[-2:])*scale < 4:
        raise ValueError('Invalid C3 rho or sampling scale')
    _, base, residual, metadata = generate(image, sigmas=sigmas, amplitude=amplitude,
                                         rho=rho, scale=scale, seed=None, generator=generator)
    result, protected, confidence = protect(image, base, residual)
    metadata.update({'c3_version': VERSION, 'c3_schema': SCHEMA, 'edge_protection': True,
                     'guide': 'input_luminance', 'final_background_floor': 0.,
                     'range_sigma': .035, 'relative_threshold_255': 5.,
                     'protected_uv_rms_255': float(protected.square().mean().sqrt()*255),
                     'zero_confidence_fraction': float((confidence == 0).float().mean()),
                     'order': ['linear_blur', 'area_downsample', 'multiscale_uv',
                               'enlarge', 'luma_guided_residual', 'continuous_edge_attenuation']})
    return result, metadata

def standardized(x):
    x = x-x.mean(dim=(-2, -1), keepdim=True)
    return x/x.std(dim=(-2, -1), unbiased=False, keepdim=True).clamp_min(1e-6)


def guided_smooth(delta, y, spatial_sigma=1.2, range_sigma=.08):
    """Luma-guided joint bilateral on chroma residual, without masking text."""
    radius = math.ceil(2*spatial_sigma)
    h,w = y.shape[-2:]
    yp = F.pad(y, (radius,)*4, mode='reflect')
    dp = F.pad(delta, (radius,)*4, mode='reflect')
    numerator, denominator = torch.zeros_like(delta), torch.zeros_like(y)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            yn = yp[:, :, radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            dn = dp[:, :, radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            weight = torch.exp(-(yn-y).square()/(2*range_sigma**2)-(dx*dx+dy*dy)/(2*spatial_sigma**2))
            numerator += weight*dn
            denominator += weight
    return numerator/denominator.clamp_min(1e-8)


def generate(x, sigmas=(1.5, 4., 9.), amplitude=6., rho=-.75, scale=.5, guided=False, seed=17, generator=None):
    """Low-resolution normalized fields; independent amplitude, scale, correlation."""
    h,w = x.shape[-2:]
    blurred = ops.gaussian(ops.linear(x), .8)
    low = ops.srgb(ops.area(blurred, round(h*scale), round(w*scale)))
    ycc = ops.ycc(low)
    generator = generator if generator is not None else torch.Generator(device=x.device).manual_seed(seed)
    fields = torch.zeros_like(ycc[:, 1:])
    for sigma, coefficient in zip(sigmas, (.55, .8, .45)):
        noise = torch.randn(fields.shape, device=x.device, dtype=x.dtype, generator=generator)
        fields += coefficient*standardized(ops.gaussian(noise, sigma))
    fields = standardized(fields)
    # Orthogonalize this finite realization before applying the requested rho.
    u = fields[:, :1]
    v = fields[:, 1:] - (fields[:, 1:]*u).mean(dim=(-2,-1),keepdim=True)*u
    v = standardized(v)
    correlated = torch.cat((u, rho*u+math.sqrt(1-rho*rho)*v), 1)
    y = ycc[:, :1]
    local = ops.gaussian(y, 3.2)
    darkness = ((.76-y)/.55).clamp(0, 1)
    local_dark = ((local-y)/.16).clamp(0, 1)
    # The positive floor leaves background variation; no OCR/text mask exists.
    modulation = .18 + .82*(.65*darkness+.35*local_dark)
    residual = correlated*(amplitude/255)*modulation
    before_guidance = residual.square().mean().sqrt()
    if guided:
        residual = guided_smooth(residual, y)
    disturbed = ycc.clone()
    disturbed[:, 1:] += residual
    assert torch.equal(disturbed[:, :1], ycc[:, :1])
    # Upsample all channels identically, then convert to RGB only at the end.
    enlarged = F.interpolate(disturbed, size=(h,w), mode='bicubic', align_corners=False)
    base_ycc = F.interpolate(ycc, size=(h,w), mode='bicubic', align_corners=False)
    unclipped = ops.rgb(enlarged)
    result = unclipped.clamp(0, 1)
    base = ops.rgb(base_ycc).clamp(0,1)
    delta = enlarged[:, 1:]-base_ycc[:, 1:]
    metadata = {'seed':seed, 'amplitude_255':amplitude, 'rho':rho, 'source_scale':scale,
                'lr_sigmas':list(sigmas), 'nominal_output_sigmas':[s/scale for s in sigmas],
                'coefficients':[.55,.8,.45], 'background_floor':.18, 'guided':guided,
                'modulation':'continuous absolute and relative luma; no text mask',
                'pre_guidance_uv_rms_255':float(before_guidance*255),
                'post_guidance_uv_rms_255':float(residual.square().mean().sqrt()*255),
                'rgb_clipped_fraction':float(((unclipped<0)|(unclipped>1)).float().mean()),
                'final_y_rms_delta_255':float((ops.ycc(result)[:,:1]-ops.ycc(base)[:,:1]).square().mean().sqrt()*255)}
    assert torch.isfinite(result).all()
    return result, base, delta, metadata


def protect(clean, base, residual, threshold_255=5.):
    """Continuous luma confidence, no text/OCR mask and no background floor.

    Joint filtering limits cross-edge averaging; an output-resolution confidence
    also removes chroma spread introduced by enlargement at the last stage.
    Both gates use luminance only, not the selected UV colour or its amplitude.
    """
    y = ops.ycc(clean)[:, :1]
    local = ops.gaussian(y, 3.2)
    relative = ((local-y-threshold_255/255)/(.10)).clamp(0, 1)
    darkness = ((.76-y)/.35).clamp(0, 1)
    confidence = relative*darkness
    # Smoothstep: continuously approaches zero instead of a hard binary edge.
    confidence = confidence.square()*(3-2*confidence)
    filtered = guided_smooth(residual, y, spatial_sigma=1.2, range_sigma=.035)
    protected = filtered*confidence
    ycc = ops.ycc(base)
    ycc[:, 1:] += protected
    result = ops.rgb(ycc).clamp(0, 1)
    background = (local-y).abs() < 2/255
    assert torch.count_nonzero(protected.expand(-1,2,-1,-1)[background.expand(-1,2,-1,-1)]) == 0
    return result, protected, confidence


