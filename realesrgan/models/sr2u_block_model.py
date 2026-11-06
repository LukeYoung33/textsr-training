"""Dedicated LR/HR feed contract with unchanged output-resolution block losses."""
import torch
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, yuv420_mask_to_y4uv
from .block256_region_model import Block256RegionModel


@MODEL_REGISTRY.register()
class SR2uBlockModel(Block256RegionModel):
    @torch.no_grad()
    def feed_data(self, data):
        """@luna-comment
        Input: RGB128 real LQ features, original RGB256 LQ, RGB256 GT and HR masks/CP polygons.
        Output: native LR6x64, HR original-LQ/target/masks6x128 and unchanged HR loss metadata.
        Key Steps: reject incompatible routes, independently pack inputs/targets and retain HR supervision.
        Key Params: fixed2x, paired-only; keep/stroke-logit losses are unsupported and fail explicitly.
        State/Dependencies: current device/batch; bypasses legacy same-size feed, reuses inherited optimizer.
        """
        if any(k in data for k in ('kernel1', 'kernel2', 'sinc_kernel')):
            raise ValueError('SR2u does not allow synthetic degradation kernels')
        train = self.opt.get('train', {})
        if self.opt.get('lq_preprocess', {}).get('enabled') or train.get('keep_weight', 0) or train.get('stroke_prob_enabled', False):
            raise ValueError('Unsupported same-size preprocessing/keep/stroke route')
        self.lq_rgb = data['lq_rgb'].to(self.device)
        self.gt_rgb = data['gt_rgb'].to(self.device)
        batch = self.lq_rgb.shape[0]
        if tuple(self.lq_rgb.shape) != (batch, 3, 128, 128) or tuple(self.gt_rgb.shape) != (batch, 3, 256, 256):
            raise ValueError('SR2u needs RGB128 LQ and matching RGB256 GT batches')
        self.lq, self.gt = rgb_to_y4uv(self.lq_rgb), rgb_to_y4uv(self.gt_rgb)
        original_rgb = data['original_lq_rgb'].to(self.device)
        if original_rgb.shape != self.gt_rgb.shape:
            raise ValueError('Original-resolution LQ skip must be RGB256')
        self.original_lq = rgb_to_y4uv(original_rgb)
        self.valid_mask_rgb = data['valid_mask_rgb'].to(self.device)
        self.text_mask_rgb = data['text_mask_rgb'].to(self.device)
        if self.valid_mask_rgb.shape != (batch, 1, 256, 256) or self.text_mask_rgb.shape != self.valid_mask_rgb.shape:
            raise ValueError('SR2u requires HR256 masks')
        self.valid_mask = yuv420_mask_to_y4uv(self.valid_mask_rgb)
        self.text_mask = yuv420_mask_to_y4uv(self.text_mask_rgb)
        self.structure_available = list(data.get('structure_available', []))
        self.segment_metadata = data
        self.last_paired_selection = torch.ones(batch, dtype=torch.bool)

    def _forward_v5(self, input_y4uv, *, network=None, **kwargs):
        """@luna-comment
        Input: native LR feature tensor, optional network, stored native original-HR LQ skip.
        Output: HR prediction formed from original-HR LQ plus learned residual, never resized LR skip.
        Key Steps: require external-skip architecture and pass the two inputs explicitly.
        Key Params: tonal preprocessing remains disabled; GT is never passed to the generator.
        State/Dependencies:current batch/net_g or EMA; no resizing or image-state mutation.
        """
        generator = self.net_g if network is None else network
        if not getattr(generator, 'supports_external_skip_input', False) or self.original_lq is None:
            raise ValueError('SR2u requires the original-HR external-skip architecture')
        return generator(input_y4uv, skip_input_y4uv=self.original_lq, **kwargs)
