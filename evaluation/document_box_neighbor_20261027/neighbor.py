"""640-only distribution confidence and robust neighboring lower-rail repair."""
from __future__ import annotations

import cv2
import numpy as np
from evaluation.document_box_pre640_revision2_20261027 import revision2

CONFIG = dict(version='neighbor_distribution_trial1', quantiles=[.01,.05,.10,.25,.5,.75,.9],
              windows=[65,33], gap_allowance=12., gap_scale=24., radius=4,
              spatial_scale=24., tail_scale=12., median_scale=24., distribution_scale=48.,
              confidence_min=.5, support_min=.05, support_full=.25,
              repair_strengths=[.5,1.], tail_slope=.1, original_high_unchanged=True)


def descriptors(canvas, nodes):
    """@luna-comment
    Input: Bound640x640uint8 luminance and boundary-inclusive node positions.
    Output: Seven quantile fields and small-window low-tail field, float32.
    Key Steps: Histogram quantiles on65and33windows, clipped to the analysis raster.
    Key Params: Q1/Q5/Q10/Q25/Q50/Q75/Q90; no pixel labels or native statistics.
    State/Dependencies: NumPy only; cached source-bound analysis remains immutable.
    """
    if canvas.shape != (640,640) or canvas.dtype != np.uint8:
        raise ValueError('Expected bound640uint8 analysis')
    fields=np.empty((len(nodes),len(nodes),7),np.float32)
    small=np.empty(fields.shape[:2],np.float32)
    for iy,cy in enumerate(nodes):
        for ix,cx in enumerate(nodes):
            for radius in [32,16]:
                patch=canvas[max(0,cy-radius):min(640,cy+radius+1),max(0,cx-radius):min(640,cx+radius+1)]
                cumulative=np.cumsum(np.bincount(patch.ravel(),minlength=256))
                probabilities=CONFIG['quantiles'] if radius==32 else [.01]
                quantiles=np.searchsorted(cumulative,np.ceil(np.array(probabilities)*patch.size))
                if radius==32:
                    fields[iy,ix]=quantiles
                else:
                    small[iy,ix]=quantiles[0]
    return fields,small


def repair(low, high, quantiles, small_tail, nodes):
    """@luna-comment
    Input: Original node rails, seven640quantiles, small-windowQ1and node geometry.
    Output: Repaired lower rail and confidence/support/amount/target diagnostics.
    Key Steps: Score unsupported rail gaps; match distributions; weighted-median trusted neighbors.
    Key Params: Radius4nodes; soft evidence strengths; unsupported nodes retain original rail.
    State/Dependencies: No native data or text masks; original upper rail is not modified.
    """
    tail=quantiles[:,:,0]
    gap=np.maximum(low-tail-12.,0.)
    scale_gap=np.maximum(np.abs(small_tail-tail)-12.,0.)
    confidence=np.exp(-(gap/24.)**2-(scale_gap/24.)**2)
    support=np.zeros_like(low,np.float32)
    target=low.astype(np.float32).copy()
    amount=np.zeros_like(low,np.float32)
    for iy,cy in enumerate(nodes):
        for ix,cx in enumerate(nodes):
            ys=slice(max(0,iy-4),min(len(nodes),iy+5))
            xs=slice(max(0,ix-4),min(len(nodes),ix+5))
            q=quantiles[ys,xs]
            dy=nodes[ys,None]-cy
            dx=nodes[None,xs]-cx
            differences=q-quantiles[iy,ix]
            weight=np.exp(-(dx*dx+dy*dy)/(24.**2)
                          -(differences[:,:,0]/12.)**2-(differences[:,:,4]/24.)**2
                          -np.mean((differences/48.)**2,axis=2))
            trusted=np.where(confidence[ys,xs]>=.5,confidence[ys,xs],0.)
            evidence=weight*trusted
            fraction=float(evidence.sum()/max(weight.sum(),1e-12))
            support[iy,ix]=fraction
            if fraction<.05:
                continue
            estimates=(low[ys,xs]+tail[iy,ix]-tail[ys,xs]).ravel()
            order=np.argsort(estimates,kind='stable')
            cumulative=np.cumsum(evidence.ravel()[order])
            median=estimates[order[np.searchsorted(cumulative,cumulative[-1]/2)]]
            target[iy,ix]=np.clip(median,0.,min(float(low[iy,ix]),float(high[iy,ix])))
            amount[iy,ix]=(1-confidence[iy,ix])*min(1.,fraction/.25)
    revised=low+amount*(target-low)
    return revised,dict(confidence=confidence,support=support,amount=amount,target=target,
                        corrected_nodes=int(np.count_nonzero(np.abs(revised-low)>.01)),
                        unresolved_nodes=int(np.count_nonzero((confidence<.5)&(support<.05))))


def tables(low, high):
    """@luna-comment
    Input: Finite ordered node rails for one repair strength.
    Output: Float32nodeLUTs using unchanged revision2STANDARD soft-tail formula.
    Key Steps: Evaluate256Ycodes per node and verify finite monotone response.
    Key Params: Slope.1,tau2,support24; only lower rail differs from baseline.
    State/Dependencies: Frozen revision2curve; no image reads or external writes.
    """
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):
        raise ValueError('Invalid rails')
    result=np.empty((*low.shape,256),np.float32)
    for iy,ix in np.ndindex(low.shape):
        result[iy,ix]=revision2.curve(np.arange(256),float(low[iy,ix]),float(high[iy,ix]),.1)
    if not np.isfinite(result).all() or np.min(np.diff(result,axis=2)) < -1e-5:
        raise AssertionError('Invalid response')
    return result
