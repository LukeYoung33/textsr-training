"""Phase-preserving X2 SR with Compact4x low-resolution local/context features."""
import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY
from .basic_text_line_r4_student_y4uv_v5_compact_arch import BasicTextLineR4StudentY4UVV5Compact


def fixed_projection(matrix):
    """@luna-comment
    Input: Output-by-input coefficient matrix. Output: Frozen dense1x1 convolution.
    Key Steps: Copy exact coefficients and disable gradients.
    Key Params: Dense zeros count toward deployed MACs.
    State/Dependencies: PyTorch module-owned fixed parameters; no global state.
    """
    layer=nn.Conv2d(matrix.shape[1],matrix.shape[0],1,bias=False)
    with torch.no_grad():layer.weight.copy_(matrix[:,:,None,None])
    return layer.requires_grad_(False)


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV5CompactSR(nn.Module):
    """Input RGB256 packed6x128x128; output RGB512 packed6x256x256."""

    def __init__(self,canvas_hw=(256,256)):
        """@luna-comment
        Input: LR RGB canvas, positive multiples of8. Output: Independently initialized X2 SR model.
        Key Steps: Retain Compact hidden features; replace restoration terminals with direct native-phase SR heads.
        Key Params: Context3/12/48, local12, two bottleneck ECBs; fixed0.5 context gain, no trainable zero gate.
        State/Dependencies: Existing Compact feature modules; original architectures remain unchanged.
        """
        super().__init__();base=BasicTextLineR4StudentY4UVV5Compact(canvas_hw=canvas_hw)
        self.canvas_hw=base.canvas_hw;self.expected_input_hw=base.expected_input_hw;self._deployment=False
        for name in ('local_pre','stem','phase_pack1','stage1','phase_pack2','stage2','up2','dec2','up1','dec1'):
            setattr(self,name,getattr(base,name))
        haar=torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=torch.float32)
        matrix=torch.eye(6);matrix[:4,:4]=haar/4
        self.input_transform=fixed_projection(matrix)
        self.local_select=fixed_projection(torch.eye(6)[:4])
        self.local_embed=fixed_projection(torch.eye(6)[:,:4])
        self.local_head=nn.ConvTranspose2d(12,4,2,stride=2)
        self.context_head=nn.ConvTranspose2d(3,6,2,stride=2)
        for head in (self.local_head,self.context_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.skip=nn.ConvTranspose2d(6,6,2,stride=2,bias=False)
        with torch.no_grad():
            self.skip.weight.zero_()
            for phase in range(4):self.skip.weight[phase,:4,phase//2,phase%2]=1
            for channel in (4,5):self.skip.weight[channel,channel,:,:]=1
        self.skip.requires_grad_(False)

    def forward(self,x):
        """@luna-comment
        Input: Native TL/TR/BL/BR/U/V[B,6,H,W]. Output: Native[B,6,2H,2W], without clipping.
        Key Steps: Fixed Haar, independent local/context features, direct SR residuals, phase-correct nearest skip.
        Key Params: Local head changes Y only; context gain0.5 is a fixed multiplier, not an output bound.
        State/Dependencies: Module-owned parameters; no RGB reconstruction, phase averaging or external skip.
        """
        if not torch.jit.is_tracing():
            if x.ndim!=4 or x.shape[1]!=6 or tuple(x.shape[-2:])!=self.expected_input_hw:
                raise ValueError('SR requires configured native[B,6,H,W] input')
        features=self.input_transform(x)
        local=self.local_embed(self.local_head(self.local_pre(self.local_select(features))))
        stem=self.stem(features);phase1=self.phase_pack1(stem);mixed=phase1
        for block in self.stage1:mixed=block(mixed)
        context=self.phase_pack2(mixed)
        for block in self.stage2:context=block(context)
        decoded1=phase1+self.dec2(self.up2(context))
        decoded0=stem+self.dec1(self.up1(decoded1))
        return self.skip(x)+local+0.5*self.context_head(decoded0)

    def switch_to_deploy(self):
        """@luna-comment
        Input: Training-form model. Output: Idempotently fused native-batch deployment form.
        Key Steps: Fuse feature ECBs; expand frozen grouped phase selectors to exact dense convolutions.
        Key Params: SR transpose heads remain intended k2/s2 operators; original3x3 kernels are unchanged.
        State/Dependencies: In-place model transformation; callers copy when retaining training form.
        """
        if self._deployment:return self
        for block in (self.local_pre,self.stem,*self.stage1,*self.stage2,self.dec2,self.dec1):block.switch_to_deploy()
        for name in ('phase_pack1','phase_pack2'):
            source=getattr(self,name);dense=nn.Conv2d(source.in_channels,source.out_channels,2,stride=2,bias=False)
            dense=dense.to(device=source.weight.device,dtype=source.weight.dtype)
            with torch.no_grad():
                dense.weight.zero_()
                for channel in range(source.out_channels):dense.weight[channel,channel//4]=source.weight[channel,0]
            setattr(self,name,dense.requires_grad_(False))
        self._deployment=True;return self

    def deployment_audit(self):
        """@luna-comment
        Input: Current module graph. Output: Fusion and intended SR operator audit.
        Key Steps: Reject unfused ECBs, extra transpose heads, shuffles and nonnative phase groups.
        Key Params: Exactly local/context/skip k2s2 transpose convolutions are allowed.
        State/Dependencies: Static module inspection; device placement requires separate evidence.
        """
        unfused=[name for name,module in self.named_modules() if hasattr(module,'deploy') and not module.deploy]
        heads=[name for name,module in self.named_modules() if isinstance(module,nn.ConvTranspose2d)]
        forbidden=[name for name,module in self.named_modules() if isinstance(module,(nn.PixelShuffle,nn.PixelUnshuffle))]
        invalid_heads=[name for name in heads if not (
            getattr(self,name).kernel_size==(2,2) and getattr(self,name).stride==(2,2)
            and getattr(self,name).padding==(0,0) and getattr(self,name).output_padding==(0,0)
            and getattr(self,name).dilation==(1,1) and getattr(self,name).groups==1)]
        invalid_phase=[name for name in ('phase_pack1','phase_pack2') if not (
            isinstance(getattr(self,name),nn.Conv2d) and getattr(self,name).groups==1
            and getattr(self,name).kernel_size==(2,2) and getattr(self,name).stride==(2,2))]
        return dict(passed=self._deployment and not unfused and not forbidden and not invalid_heads and not invalid_phase
                    and sorted(heads)==['context_head','local_head','skip'],unfused=unfused,
                    sr_transpose_heads=heads,forbidden=forbidden,invalid_heads=invalid_heads,invalid_phase=invalid_phase)
