"""V3M: unchanged V3 depth and boundary, wider 4/8/16/32 branch."""
from basicsr.utils.registry import ARCH_REGISTRY
from .basic_text_line_r4_student_y4uv_v3_arch import BasicTextLineR4StudentY4UVV3, R4_Y4UV_EXPAND_CHANNEL
from .ecbsr_Unet_arch_server import _ECBSRUnetBranch


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV3M(BasicTextLineR4StudentY4UVV3):
    """Named trainable width variant; original V3 definitions remain unchanged."""

    def __init__(self, expand_channel=(4/6,8/6,16/6,32/6), **kwargs):
        """@luna-comment
        Input: original V3 settings and fixed V3M expansion tuple. Output: native six-channel V3M.
        Key Steps: validate through V3; preserve Haar/residual modules; replace only the U-Net branch.
        Key Params: widths4/8/16/32, three levels, depth1; inherited zero residual initialization.
        State/Dependencies: unchanged V3 and existing reparameterizable branch; no checkpoint adaptation.
        """
        expected=(4/6,8/6,16/6,32/6)
        if tuple(float(v) for v in expand_channel)!=expected:
            raise ValueError('V3M requires widths4/8/16/32')
        super().__init__(expand_channel=R4_Y4UV_EXPAND_CHANNEL,**kwargs)
        self.feature_widths=(4,8,16,32)
        self.backbone=_ECBSRUnetBranch(in_channel=6,out_channel=6,layer_num=3,
            expand_channel=list(expected),depth_factor=1,scale_factor=2,
            with_idt=kwargs.get('with_idt',False),act_type=kwargs.get('act_type','prelu'),
            scale=kwargs.get('scale',2),with_proj=kwargs.get('with_proj',True),
            use_upsampler=False,input_conv=False)
        if self.backbone.conv0.out_planes!=4 or self.backbone.encdec.out_planes!=32:
            raise RuntimeError('Unexpected V3M channel rounding')

