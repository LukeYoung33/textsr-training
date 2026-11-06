"""Training-only affine luminance matching after immutable patch extraction."""
import numpy as np


def match_patch_contrast(gt, lq, valid):
    """@luna-comment Input:finite CHW RGB float32 pairs in0..1 and1HW valid mask. Output:new target and scalar diagnostics. Key Steps:valid luma5/95;positive affine GT luminance;preserve chroma then gamut clip. Key Params:min256pixels/span.02,gain.25..4. State/Dependencies:pure NumPy;inputs never mutated,flat patches unchanged."""
    assert gt.shape == lq.shape and gt.ndim == 3 and gt.shape[0] == 3
    assert valid.shape == (1, *gt.shape[1:])
    assert np.isfinite(gt).all() and np.isfinite(lq).all()
    weights = np.array([.299, .587, .114], np.float32)[:, None, None]
    gy = (gt * weights).sum(0); ly = (lq * weights).sum(0)
    selected = valid[0] > .5
    stats = dict(gain=1., offset=0., applied=0., clipping_fraction=0., level_error_before=0., level_error_after=0., gain_limited=0.)
    if selected.sum() < 256:
        return gt.copy(), stats
    g = np.quantile(gy[selected], [.05, .95]); l = np.quantile(ly[selected], [.05, .95])
    stats['level_error_before'] = float(np.abs(g-l).mean())
    if min(g[1]-g[0], l[1]-l[0]) < .02:
        stats['level_error_after'] = stats['level_error_before']; return gt.copy(), stats
    raw_gain = float((l[1]-l[0])/(g[1]-g[0])); gain = float(np.clip(raw_gain, .25, 4.))
    offset = float(l[1]-gain*g[1])
    delta = (gain-1.)*gy+offset
    raw = gt+delta[None]
    result = np.clip(raw, 0., 1.).astype(np.float32)
    result[:, ~selected] = gt[:, ~selected]
    y = (result*weights).sum(0)
    stats.update(gain=gain, offset=offset, applied=1., gain_limited=float(gain != raw_gain), clipping_fraction=float(((raw<0)|(raw>1))[:, selected].mean()), level_error_after=float(np.abs(np.quantile(y[selected], [.05,.95])-l).mean()))
    return result, stats
