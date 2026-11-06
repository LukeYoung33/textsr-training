"""Width-only V3M teachers; existing V3/V3M inference contracts are untouched."""
from basicsr.utils.registry import ARCH_REGISTRY

from .basic_text_line_r4_student_y4uv_v3_arch import BasicTextLineR4StudentY4UVV3
from .ecbsr_Unet_arch_server import _ECBSRUnetBranch


# Calibrated against full deployed learnable-convolution MACs, not width squared.
TEACHER_WIDTHS = {
    16: (20, 40, 80, 160),
    32: (28, 56, 112, 224),
    64: (40, 80, 160, 320),
    128: (57, 114, 228, 456),
}
FEATURE_TAPS = ('conv0', 'enc1', 'enc2', 'enc3', 'encdec', 'dec1', 'dec2', 'dec3')


@ARCH_REGISTRY.register()
class BasicTextLineR4TeacherY4UVV3L(BasicTextLineR4StudentY4UVV3):
    """Same V3M topology, depth1 and expansion2; only stage channels increase."""

    def __init__(self, compute_ratio=16, **kwargs):
        """@luna-comment
        Input: one of four MAC target labels and unchanged V3 canvas/boundary settings.
        Output: six-channel teacher with the original residual/Haar forward contract.
        Key Steps: validate target; initialize V3 boundaries; replace only its backbone.
        Key Params: fixed three levels, depth1, expansion2 and target-specific widths.
        State/Dependencies: existing V3 and ECB implementations; random parameter initialization.
        """
        if isinstance(compute_ratio, bool) or compute_ratio not in TEACHER_WIDTHS:
            raise ValueError('compute_ratio must be 16, 32, 64 or 128')
        forbidden = {'expand_channel', 'depth_factor', 'scale_factor', 'layer_num'} & kwargs.keys()
        if forbidden:
            raise ValueError(f'Width-only teacher owns these fixed settings: {sorted(forbidden)}')
        super().__init__(**kwargs)
        self.compute_ratio = int(compute_ratio)
        self.feature_widths = TEACHER_WIDTHS[self.compute_ratio]
        self.backbone = _ECBSRUnetBranch(
            in_channel=6, out_channel=6, layer_num=3,
            expand_channel=[width / 6 for width in self.feature_widths],
            depth_factor=1, scale_factor=2,
            with_idt=kwargs.get('with_idt', False), act_type=kwargs.get('act_type', 'prelu'),
            scale=kwargs.get('scale', 2), with_proj=kwargs.get('with_proj', True),
            use_upsampler=False, input_conv=False,
        )
        actual = (self.backbone.conv0.out_planes,) + tuple(
            getattr(self.backbone, f'enc{i}').module_list[-1].out_planes for i in (1, 2, 3)
        )
        if actual != self.feature_widths:
            raise RuntimeError(f'Channel rounding changed teacher widths: {actual}')
