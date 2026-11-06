"""Low-resolution Compact SR residuals added directly to original-resolution LQ."""
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from .basic_text_line_r4_student_y4uv_v5_compact_sr_arch import BasicTextLineR4StudentY4UVV5CompactSR


@ARCH_REGISTRY.register()
class BasicTextLineR4StudentY4UVV5CompactSRHRSkip(BasicTextLineR4StudentY4UVV5CompactSR):
    supports_external_skip_input = True

    def __init__(self, canvas_hw=(128, 128)):
        """@luna-comment
        Input: LR RGB canvas divisible by8. Output: true2x residual generator with required original-HR skip.
        Key Steps: retain Compact features and zero SR heads; remove inherited fixed nearest skip entirely.
        Key Params: no interpolation of the skip, no HR backbone; native original HR input is mandatory.
        State/Dependencies:isolated subclass; original SR architecture and prior exports remain unchanged.
        """
        super().__init__(canvas_hw=canvas_hw)
        del self.skip

    def forward(self, x, *, skip_input_y4uv):
        """@luna-comment
        Input: LR[B,6,H,W] features and original LQ[B,6,2H,2W] native skip.
        Output: original-HR LQ plus learned HR phase residual, unclipped.
        Key Steps: LR Haar/local/context features -> two SR heads -> original HR addition.
        Key Params:fixed0.5 context gain; zero heads give exact HR identity; neither GT nor resized LR is skip.
        State/Dependencies:module-owned weights and two caller inputs; no interpolation of skip input.
        """
        if not torch.jit.is_tracing():
            if x.ndim != 4 or x.shape[1] != 6 or tuple(x.shape[-2:]) != self.expected_input_hw:
                raise ValueError('HR-skip SR requires configured native LR input')
            if skip_input_y4uv.shape != (x.shape[0], 6, 2*x.shape[2], 2*x.shape[3]):
                raise ValueError('Original HR skip must match output geometry')
        features = self.input_transform(x)
        local = self.local_embed(self.local_head(self.local_pre(self.local_select(features))))
        stem = self.stem(features)
        phase1 = self.phase_pack1(stem)
        mixed = phase1
        for block in self.stage1:
            mixed = block(mixed)
        context = self.phase_pack2(mixed)
        for block in self.stage2:
            context = block(context)
        decoded1 = phase1 + self.dec2(self.up2(context))
        decoded0 = stem + self.dec1(self.up1(decoded1))
        return skip_input_y4uv + local + 0.5*self.context_head(decoded0)

    def deployment_audit(self):
        """@luna-comment
        Input: current/fused graph. Output: audit requiring exactly two residual SR heads and no nearest skip.
        Key Steps: reuse phase/kernel/fusion checks and replace the parent's three-head expectation.
        Key Params:local/context k2s2 only; external HR input always required.
        State/Dependencies:static module inspection; does not assert device delegate support for two inputs.
        """
        result = super().deployment_audit()
        result['passed'] = self._deployment and not any(result[k] for k in ('unfused','forbidden','invalid_heads','invalid_phase')) and sorted(result['sr_transpose_heads']) == ['context_head','local_head'] and not hasattr(self, 'skip')
        result['skip_source'] = 'required_original_HR_LQ'
        return result
