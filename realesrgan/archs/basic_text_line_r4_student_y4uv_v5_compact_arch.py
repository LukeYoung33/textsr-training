"""Budgeted V5 local/context architecture rooted at the original V3 stem width.

At256x256 RGB, the dense-phase/fixed-Haar deployment graph costs98,697,216
MACs:4.063x the paired V3 baseline. Original V3/V5 classes are unchanged.
"""
from basicsr.utils.registry import ARCH_REGISTRY
from .basic_text_line_r4_student_y4uv_v5_arch import BasicTextLineR4StudentY4UVV5


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV5Compact(BasicTextLineR4StudentY4UVV5):
    """Preserve V5 phase-volume/local/context design within the four-V3 budget."""

    def __init__(self, canvas_hw=(256,256), **kwargs):
        """@luna-comment
        Input: RGB canvas divisible by8 and inherited six-channel/auxiliary options.
        Output: Trainable compact V5 with3/12/48stages,12local channels and1/2stage repetitions.
        Key Steps: Bind budgeted widths and fixed Haar; delegate unchanged residual initialization to V5.
        Key Params: Budget-defining overrides must match; stroke supervision is training-only and optional.
        State/Dependencies: Existing V5 ECB/phase/fusion implementation; independently initialized parameters.
        """
        fixed=dict(stage_channels=(3,12,48),local_channels=12,stage_repeat=1,bottleneck_repeat=2,
                   haar_switch=True,local_haar_switch=True,learnable_haar=False,grouped_phase_pack=True)
        for key,expected in fixed.items():
            actual=kwargs.pop(key,expected)
            if key=='stage_channels':actual=tuple(actual)
            if actual!=expected:raise ValueError(f'Compact4x requires {key}={expected!r}; received {actual!r}')
        super().__init__(canvas_hw=canvas_hw,**fixed,**kwargs)

    def estimate_macs(self):
        """@luna-comment
        Input: Configured canvas. Output: Native and actual deployment MACs with scaled V3 reference.
        Key Steps: Reuse native ECB cost; add dense phase-selector overhead and all fixed1x1 boundaries.
        Key Params: Counts dense zero weights as work; deployment_tflite is the4x-budget metric.
        State/Dependencies: Static inherited geometry; independently verified by executed convolution hooks.
        """
        result=super().estimate_macs();height,width=self.expected_input_hw
        c0,c1,c2=self.stage_channels
        dense_extra=(height//2)*(width//2)*c1*(c0-1)*4
        dense_extra+=(height//4)*(width//4)*c2*(c1-1)*4
        fixed_boundaries=height*width*(6*6+6*4+6*6+4*6)
        result['deployment_tflite']=result['deployment_fused']+dense_extra+fixed_boundaries
        result['v3_reference']=24293376*height*width//(128*128)
        result['target_four_v3']=4*result['v3_reference']
        return result
