"""V5Lite: benchmarked local12/context4->16 shallow V5 with trainable bounded gate."""
import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY
from .basic_text_line_r4_student_y4uv_v5_arch import (
    V5RepECB,FixedPhasePack,V5_CONTEXT_SCALE_LIMIT,V5_CONTEXT_GRAD_WARMUP,
)
from .ecbsr_Unet_arch_server import Y4ToHaarLikeWavelet,HaarLikeToY4


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV5Lite(nn.Module):
    def __init__(self,canvas_hw=(256,256),in_channel=6,out_channel=6,
                 local_channels=12,stage_channels=(4,16),haar_switch=True,
                 local_haar_switch=True,learnable_haar=False,grouped_phase_pack=True):
        """@luna-comment
        Input: canvas divisible by4 and fixed V5Lite channel/Haar settings. Output: native residual model.
        Key Steps: validate; construct exact benchmark branches; initialize local terminal/gate at zero.
        Key Params: local12, context4/16, one phase stage; V5 context bound/warmup retained.
        State/Dependencies: original V5 ECB/phase/Haar classes and deterministic torch initialization.
        """
        super().__init__()
        self.canvas_hw=tuple(int(v) for v in canvas_hw)
        if len(self.canvas_hw)!=2 or any(v<=0 or v%4 for v in self.canvas_hw):raise ValueError('canvas must be positive multiples of4')
        if in_channel!=6 or out_channel!=6 or local_channels!=12 or tuple(stage_channels)!=(4,16):raise ValueError('V5Lite requires6-channel IO, local12 and context4/16')
        if haar_switch is not True or local_haar_switch is not True or learnable_haar or grouped_phase_pack is not True:raise ValueError('V5Lite requires fixed Haar and grouped phase carry')
        self.haar_switch=True;self.local_haar_switch=True
        self.local_channels=12;self.stage_channels=(4,16)
        self.haar=Y4ToHaarLikeWavelet();self.inverse=HaarLikeToY4()
        self.local_pre=V5RepECB(4,12,act_type='prelu')
        self.local_terminal=V5RepECB(12,4,act_type='linear',zero_init=True)
        self.stem=V5RepECB(6,4,act_type='prelu')
        self.phase=FixedPhasePack(4,grouped=True)
        self.stage=V5RepECB(16,16,act_type='prelu',with_idt=True)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.decoder=V5RepECB(16,4,act_type='prelu')
        self.context_terminal=V5RepECB(4,6,act_type='linear')
        self.context_scale=nn.Parameter(torch.tensor(0.))
        self.register_buffer('_deployed_context_scale',torch.tensor(0.),persistent=False)
        self._deployment=False

    @property
    def expected_input_hw(self):
        """@luna-comment Input: configured RGB canvas. Output: packed half-resolution dimensions.
        Key Steps: divide by2. Key Params: Y4UV420 packing. State/Dependencies: constructor metadata only.
        """
        return tuple(v//2 for v in self.canvas_hw)

    def forward(self,x,return_branches=False):
        """@luna-comment
        Input: BCHW six-channel packed data with even spatial dimensions. Output: same-size prediction/branches.
        Key Steps: fixed Haar, local luma residual, one-phase context with stem skip, inverse decode and sum.
        Key Params: bounded0.5*tanh context; training-only zero-value0.1 straight-through warmup.
        State/Dependencies: learned modules and deployment flag; evaluation matches benchmark algebra.
        """
        if x.ndim!=4 or x.shape[1]!=6 or any(v%2 for v in x.shape[-2:]):raise ValueError('Input must be BCHW six-channel packed data with even H/W')
        features=torch.cat((self.haar(x[:,:4]),x[:,4:]),1)
        local_bands=self.local_terminal(self.local_pre(features[:,:4]))
        local=torch.cat((self.inverse(local_bands),torch.zeros_like(x[:,4:])),1)
        stem=self.stem(features);mixed=self.stage(self.phase(stem))
        raw=self.context_terminal(stem+self.decoder(self.up(mixed)))
        if self._deployment:bands=raw
        else:
            alpha=V5_CONTEXT_SCALE_LIMIT*torch.tanh(self.context_scale)
            if self.training:
                # The difference is exactly zero in value, preserving identity
                # initialization while allowing terminal gradients at alpha=0.
                bands=alpha*raw+V5_CONTEXT_GRAD_WARMUP*(raw-raw.detach())
            else:bands=alpha*raw
        context=torch.cat((self.inverse(bands[:,:4]),bands[:,4:]),1)
        output=x+local+context
        return dict(input=x,local_residual=local,context_residual=context,full=output) if return_branches else output

    def switch_to_deploy(self):
        """@luna-comment
        Input: initialized/trained model. Output: self with fused ECBs and baked bounded context amplitude.
        Key Steps: fuse six ECB modules; scale context terminal weights/bias; freeze gate and mark deployment.
        Key Params: idempotent; no gradient warmup in deployed forward. State/Dependencies: model state mutates.
        """
        if self._deployment:return self
        for block in (self.local_pre,self.local_terminal,self.stem,self.stage,self.decoder,self.context_terminal):block.switch_to_deploy()
        alpha=V5_CONTEXT_SCALE_LIMIT*torch.tanh(self.context_scale.detach())
        with torch.no_grad():
            self.context_terminal.reparam.weight.mul_(alpha);self.context_terminal.reparam.bias.mul_(alpha)
            self._deployed_context_scale.copy_(alpha);self.context_scale.requires_grad_(False)
        self._deployment=True
        return self

