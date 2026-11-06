"""Foreground-only coefficient boost with unchanged background normalization."""
import torch
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from .basic_text_line_y4uv_v3_losses import Y4UVNativeMaskedL1LossV3, _mask_bchw, _weights
from .uv_structure_supervision_loss import UVStrongSupervisionLoss, _uv_mask


@LOSS_REGISTRY.register()
class TextBoostNativeMaskedL1Loss(Y4UVNativeMaskedL1LossV3):
    def __init__(self,text_multiplier=1.5,**kwargs):
        """@luna-comment
        Input: original mean-reduced native L1 options and multiplier>=1. Output: configured loss.
        Key Steps: preserve original normalization/options; validate additive foreground coefficient.
        Key Params: text_multiplier1.5 means text numerator+50%. State/Dependencies: unchanged native L1.
        """
        super().__init__(**kwargs)
        if self.reduction!='mean' or text_multiplier<1: raise ValueError('Requires mean and multiplier>=1')
        self.text_multiplier=float(text_multiplier)

    def forward(self,pred,target,valid_mask=None,text_mask=None,mask=None,weight=None,**kwargs):
        """@luna-comment
        Input: packed prediction/target and matching valid/text masks. Output: original loss plus foreground increment.
        Key Steps: reuse baseline value; form exact original denominator and text-only numerator.
        Key Params: fg/bg/channel weights unchanged; multiplier1 is exact baseline.
        State/Dependencies: parent helpers; background gradients are unchanged including soft-mask mixtures.
        """
        base=super().forward(pred,target,valid_mask,text_mask,mask,weight,**kwargs)
        valid=_mask_bchw(valid_mask if valid_mask is not None else mask,pred,'valid_mask')
        text=_mask_bchw(text_mask,pred,'text_mask')*valid
        channel=_weights(self.channel_weight,pred)
        full=valid*(text*self.fg_weight+(1-text)*self.bg_weight)*channel
        foreground=valid*text*self.fg_weight*channel
        if weight is not None:
            extra=weight.to(pred.device,pred.dtype)
            while extra.ndim<4:extra=extra.unsqueeze(1)
            full=full*extra;foreground=foreground*extra
        increment=((pred-target).abs()*foreground).sum()/(full.sum()+self.eps)*self.loss_weight
        return base+(self.text_multiplier-1)*increment


@LOSS_REGISTRY.register()
class TextBoostUVStrongSupervisionLoss(UVStrongSupervisionLoss):
    def __init__(self,text_multiplier=1.5,**kwargs):
        """@luna-comment
        Input: original UV options and multiplier>=1. Output: text-only UV boost module.
        Key Steps: retain original regional/multiscale settings and validate multiplier.
        Key Params: original ring/far coefficients retained. State/Dependencies: original UV module.
        """
        super().__init__(**kwargs)
        if text_multiplier<1:raise ValueError('text_multiplier must be>=1')
        self.text_multiplier=float(text_multiplier)

    def forward(self,pred,target,valid_mask=None,text_mask=None,return_details=False,**kwargs):
        """@luna-comment
        Input: packed Y4UV prediction/target and valid/text masks. Output: boosted UV loss and optional diagnostics.
        Key Steps: reuse baseline; add text share of regional UV and first multiscale region only.
        Key Params: original regional denominator and ring/far terms unchanged; multiplier1 is baseline.
        State/Dependencies: original masked blur/reduction; gradients reach prediction only.
        """
        base,details=super().forward(pred,target,valid_mask,text_mask,return_details=True,**kwargs)
        p,t=pred[:,4:6],target[:,4:6]
        valid,text=_uv_mask(valid_mask,p),_uv_mask(text_mask,p)
        dilated=F.max_pool2d(text,2*self.ring_radius+1,1,self.ring_radius)
        regions=(text*valid,(dilated-text).clamp_min(0)*valid,(1-dilated).clamp_min(0)*valid)
        weights=valid*sum(w*m for w,m in zip(self.region_weights,regions))
        residual=p-t;error=torch.sqrt(residual.square()+1e-6)-1e-3
        fgweights=valid*self.region_weights[0]*regions[0]
        uv=((error*fgweights).sum((1,2,3))/(weights.sum((1,2,3)).clamp_min(self.eps)*2)).mean()
        blurs=[self._blur(residual,regions[0],s) for s in self.sigmas]
        bands=(blurs[0]-blurs[1],blurs[1]-blurs[2],blurs[2])
        ms=.4*sum(w*self._mean(torch.sqrt(x.square()+1e-6)-1e-3,regions[0]) for w,x in zip((.35,.35,.30),bands))
        increment=(self.text_multiplier-1)*(self.uv_weight*uv+self.multiscale_weight*ms)
        total=base+increment;details['text_boost_increment']=increment.detach()
        return (total,details) if return_details else total

