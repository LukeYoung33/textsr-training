"""Native image reconstruction and instance-budgeted text supervision.

The paired boundary and one-sided terms deliberately differ from the 0928
soft-profile objectives: the measured GT is a stationary target for every term.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.losses import build_loss
from basicsr.utils.registry import LOSS_REGISTRY
from realesrgan.data.basic_text_line_y4uv_v3 import (
    yuv420_mask_to_y4uv, y4uv_to_yuv420, y4uv_to_yuv444,
)
from realesrgan.utils import YUV2RGB
from realesrgan.losses.background_low_frequency_loss import BackgroundLowFrequencyGuardLoss


@LOSS_REGISTRY.register()
class FullFrameTextPriorityLoss(nn.Module):
    """Per-tile image means and independently gated, per-instance text means."""

    def __init__(self, image_channel_weight=(1, 1, 1, 1, .25, .25),
                 text_weight=1., pixel_weight=.1, missing_weight=0., false_weight=0.,
                 sdf_weight=0., cp_weight=0., bg_weight=0., uv_weight=0.,
                 area_floor=4., area_mode='hard', ring_radius=3, sdf_radius=3,
                 roi_weight_min=.25, roi_weight_max=4., cp_opt=None, bg_opt=None):
        """@luna-comment
        Input: explicit nonnegative objective coefficients and RGB-pixel geometry.
        Output: configured objective; optional local frozen CP is loaded only on demand.
        Key Steps: validate budgets and mutually exclusive area policy; register weights.
        Key Params: hard area gates or soft denominators, never both; CP defaults off.
        State/Dependencies: BasicSR registry and existing local recognizer assets only.
        """
        super().__init__()
        names = ('pixel', 'missing', 'false', 'sdf', 'cp', 'uv')
        values = (pixel_weight, missing_weight, false_weight, sdf_weight, cp_weight, uv_weight)
        self.coefficients = dict(zip(names, map(float, values)))
        if any(not math.isfinite(v) or v < 0 for v in (*values, text_weight, bg_weight)):
            raise ValueError('loss coefficients must be finite and nonnegative')
        weights = torch.tensor(image_channel_weight, dtype=torch.float32)
        if weights.numel() != 6 or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('image_channel_weight requires six finite nonnegative weights with positive sum')
        if area_mode not in ('hard', 'soft') or not math.isfinite(area_floor) or area_floor <= 0:
            raise ValueError('area_mode must be hard/soft and area_floor positive')
        if int(ring_radius) != ring_radius or int(sdf_radius) != sdf_radius or ring_radius < 1 or sdf_radius < 1:
            raise ValueError('ring_radius and sdf_radius must be positive integers in RGB pixels')
        if not (0 < roi_weight_min <= roi_weight_max) or not math.isfinite(roi_weight_max):
            raise ValueError('invalid ROI weight bounds')
        self.register_buffer('image_channel_weight', weights.view(1, 6, 1, 1))
        self.text_weight = float(text_weight)
        self.bg_weight = float(bg_weight)
        self.background_guard = None
        if self.bg_weight:
            options = dict(bg_opt or {'radius': 7, 'sigma': 4., 'channel': 'luma'})
            configured_type = options.pop('type', 'BackgroundLowFrequencyGuardLoss')
            if configured_type != 'BackgroundLowFrequencyGuardLoss':
                raise ValueError('bg_opt must use BackgroundLowFrequencyGuardLoss')
            self.background_guard = BackgroundLowFrequencyGuardLoss(**options)
        elif bg_opt:
            raise ValueError('bg_opt provided while bg_weight is zero')
        self.area_floor, self.area_mode = float(area_floor), area_mode
        self.ring_radius, self.sdf_radius = int(ring_radius), int(sdf_radius)
        self.roi_weight_min, self.roi_weight_max = float(roi_weight_min), float(roi_weight_max)
        self.cp = None
        if cp_weight > 0:
            if not cp_opt or cp_opt.get('type') != 'CRNNTextLineContentLoss':
                raise ValueError('CP requires explicit CRNNTextLineContentLoss cp_opt')
            if not Path(cp_opt.get('recognizer_path', '')).is_file():
                raise ValueError('CP requires an explicit existing local recognizer_path')
            # Library construction must register the opt-in objective as well
            # as the isolated runner; import does not construct a recognizer.
            from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineContentLoss
            self.cp = build_loss(dict(cp_opt))
        elif cp_opt:
            raise ValueError('cp_opt provided while cp_weight is zero')
        self.last_stats = {}
        self.last_roi_stats = []

    def _mean(self, error, support, term, stats):
        """@luna-comment
        Input: scalar-plane error and soft support with RGB-pixel mass units.
        Output: graph-connected mean, including zero for inadequate hard support.
        Key Steps: measure mass; select exactly one area safeguard; normalize.
        Key Params: area_floor attenuates tiny soft supports or gates hard supports.
        State/Dependencies: increments the caller's per-term skip telemetry.
        """
        mass = support.sum()
        value = float(mass.detach())
        if value <= 0 or (self.area_mode == 'hard' and value < self.area_floor):
            stats['skip_' + term + '_area'] += 1
            return error.sum() * 0
        denominator = mass.clamp_min(self.area_floor if self.area_mode == 'soft' else 1e-12)
        return (error * support).sum() / denominator

    def _geometry(self, mask, context):
        """@luna-comment
        Input: full-canvas instance stroke mask and real-context validity BCHW.
        Output: full-context ring, signed distance and reliable boundary support.
        Key Steps: distance-transform before any ROI crop; exclude uncertain halo edges.
        Key Params: ring_radius/sdf_radius are native RGB pixels, independent of glyph size.
        State/Dependencies: detached OpenCV geometry; no prediction is used for masks.
        """
        binary = (mask[0, 0].detach().cpu().numpy() > .5).astype(np.uint8)
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf = torch.as_tensor(inside - outside, device=mask.device, dtype=mask.dtype)[None, None]
        ring = ((sdf < 0) & (sdf >= -self.ring_radius)).to(mask.dtype)
        radius = max(self.ring_radius, self.sdf_radius)
        # Zero-padding the validity plane prevents cropped glyphs from acquiring
        # fabricated endpoint supervision near unknown exterior context.
        padded = F.pad(context, (radius,) * 4, value=0)
        safe = -F.max_pool2d(-padded, 2 * radius + 1, stride=1)
        boundary = (sdf.abs() <= self.sdf_radius).to(mask.dtype) * safe
        return ring, sdf, boundary

    def forward(self, prediction, target, batch, text_scale=1.):
        """@luna-comment
        Input: matching native B6hw predictions/GT and F009 full-resolution annotations.
        Output: total differentiable scalar and image/text/component scalar dictionary.
        Key Steps: image tile means; full-context geometry; unique ROI means; tile average.
        Key Params: confidence multiplies numerators only; bounded instance weights normalize.
        State/Dependencies: updates skip, size-group, support and per-ROI diagnostic telemetry.
        """
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 6:
            raise ValueError('prediction and target must be matching B6hw Y4UV tensors')
        valid = batch['valid_mask_rgb'].to(prediction)
        context = batch['context_valid_mask_rgb'].to(prediction)
        text = batch['text_mask_rgb'].to(prediction)
        instances = batch['instance_map'].to(prediction.device)
        expected = (prediction.shape[0], 1, prediction.shape[2] * 2, prediction.shape[3] * 2)
        if any(tuple(m.shape) != expected for m in (valid, context, text, instances)):
            raise ValueError('F009 masks must match B1HW full-resolution canvas')
        if not all(torch.isfinite(m).all() for m in (valid, context, text)):
            raise ValueError('annotation masks must be finite')
        if any((m < 0).any() or (m > 1).any() for m in (valid, context, text)) or (valid > context + 1e-6).any():
            raise ValueError('masks must be [0,1], valid must be contained in context')
        if instances.is_floating_point() or (instances < 0).any():
            raise ValueError('instance_map must contain nonnegative integers')
        rois = batch['text_rois']
        if len(rois) != prediction.shape[0]:
            raise ValueError('text_rois must have one list per tile')
        packed = yuv420_mask_to_y4uv(valid)
        weighted = packed * self.image_channel_weight
        image = (((prediction - target).abs() * weighted).flatten(1).sum(1) /
                 weighted.flatten(1).sum(1).clamp_min(1e-12)).mean()
        py, pu, pv = y4uv_to_yuv420(prediction)
        gy, gu, gv = y4uv_to_yuv420(target.detach())
        uv_error = ((pu - gu).abs() + (pv - gv).abs()) / 2
        zero = prediction.sum() * 0
        image_native = image
        image_bg = zero
        bg_skipped, bg_active = 0, 0
        if self.background_guard is not None:
            statuses = batch.get('annotation_status', ['unknown'] * prediction.shape[0])
            available = batch.get('structure_available', [False] * prediction.shape[0])
            if len(statuses) != prediction.shape[0] or len(available) != prediction.shape[0]:
                raise ValueError('background annotation status must have one entry per tile')
            decoded_prediction = YUV2RGB(y4uv_to_yuv444(prediction), is_train=True)
            decoded_target = YUV2RGB(y4uv_to_yuv444(target.detach()), is_train=True)
            bg_tiles = []
            for i, status in enumerate(statuses):
                if status not in ('unknown', 'no_text', 'annotated'):
                    raise ValueError('invalid annotation_status for background supervision')
                trusted = status == 'no_text' or (status == 'annotated' and bool(available[i]))
                if not trusted:
                    bg_skipped += 1
                    bg_tiles.append(zero)
                else:
                    if status == 'no_text' and bool((text[i:i+1] > 0).any()):
                        raise ValueError('no_text contradicts text_mask_rgb')
                    bg_active += 1
                    bg_tiles.append(self.background_guard(decoded_prediction[i:i+1], decoded_target[i:i+1],
                                                           (text[i:i+1] > 0).to(text), valid[i:i+1]))
            image_bg = torch.stack(bg_tiles).mean()
            image = image_native + self.bg_weight * image_bg
        totals = {k: [] for k in self.coefficients}
        stats = Counter({key: 0 for key in (
            'roi_valid', 'skip_duplicate', 'skip_no_support', 'skip_cp_geometry', 'skip_cp_validity',
            'skip_missing_contrast', 'skip_false_contrast',
            *('skip_' + k + '_area' for k in self.coefficients),
            'skip_pixel_uv_area',
            *('skip_' + k + '_confidence' for k in self.coefficients),
            *('effective_' + k for k in self.coefficients),
            *('roi_' + g for g in ('small', 'medium', 'large')),
            *('stroke_mass_' + g for g in ('small', 'medium', 'large')),
        )})
        stats['background_skip_unknown'] = bg_skipped
        stats['background_active_tiles'] = bg_active
        roi_stats = []
        rgb_prediction = rgb_target = None
        for i, tile_rois in enumerate(rois):
            components = {k: zero for k in self.coefficients}
            denominator = 0.
            seen = set()
            geom = []
            for roi in tile_rois:
                instance_id = int(roi['instance_id'])
                if instance_id <= 0:
                    raise ValueError('ROI instance_id must be positive')
                if instance_id in seen:
                    stats['skip_duplicate'] += 1
                    continue
                seen.add(instance_id)
                x0, y0, x1, y1 = map(int, roi['bbox'])
                if not (0 <= x0 < x1 <= expected[3] and 0 <= y0 < y1 <= expected[2]):
                    raise ValueError('ROI bbox must be nonempty and inside the canvas')
                crop_mask = torch.zeros_like(valid[i:i+1])
                crop_mask[..., y0:y1, x0:x1] = 1
                own = text[i:i+1] * (instances[i:i+1] == instance_id)
                stroke = own * valid[i:i+1] * crop_mask
                if float(stroke.sum()) <= 0:
                    stats['skip_no_support'] += 1
                    continue
                ring, sdf, boundary = self._geometry(own, context[i:i+1])
                ring = ring * (text[i:i+1] <= 0).to(ring) * crop_mask
                geom.append((roi, stroke, ring, sdf, boundary * crop_mask, own))
            # Assign each background pixel to its nearest instance exactly once.
            if geom:
                distances = torch.cat([g[3].abs() for g in geom], dim=1)
                candidates = torch.cat([(g[2] > 0) | ((g[4] > 0) & (text[i:i+1] <= 0))
                                        for g in geom], dim=1)
                owner = distances.masked_fill(~candidates, float('inf')).argmin(1, keepdim=True)
            for index, (roi, stroke, ring, sdf, boundary, own) in enumerate(geom):
                confidence = float(roi.get('confidence', 1.))
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError('ROI confidence must be finite in [0,1]')
                weight = float(roi.get('weight', 1.))
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError('ROI weight must be finite and positive')
                weight = min(self.roi_weight_max, max(self.roi_weight_min, weight))
                denominator += weight
                stats['roi_valid'] += 1
                ring = ring * (owner == index).to(ring)
                ring_supervised = ring * valid[i:i+1]
                # Reference rings use true context; supervised rings use center validity.
                reference_ring = ring * context[i:i+1]
                gt_stroke = (gy[i:i+1] * own * context[i:i+1]).sum() / (own * context[i:i+1]).sum().clamp_min(1e-12)
                gt_bg = (gy[i:i+1] * reference_ring).sum() / reference_ring.sum().clamp_min(1e-12)
                contrast = gt_bg - gt_stroke
                direction = contrast.sign().detach()
                reliable = float(reference_ring.sum()) >= self.area_floor and abs(float(contrast)) >= .02
                diff = py[i:i+1] - gy[i:i+1]
                errors = {'pixel': diff.abs(), 'missing': F.relu(diff * direction),
                          'false': F.relu(-diff * direction), 'sdf': diff.abs()}
                supports = {'pixel': stroke, 'missing': stroke, 'false': ring_supervised,
                            'sdf': boundary * valid[i:i+1] * ((own > 0) | ((owner == index) & (text[i:i+1] <= 0))).to(valid),
                            }
                values = {}
                gates = roi.get('loss_confidence', {})
                if any(k not in self.coefficients or not math.isfinite(float(v)) or not 0 <= float(v) <= 1
                       for k, v in gates.items()):
                    raise ValueError('loss_confidence requires known terms with finite values in [0,1]')
                for term in ('pixel', 'missing', 'false', 'sdf'):
                    if not self.coefficients[term]:
                        values[term] = zero
                        continue
                    q = confidence * float(gates.get(term, 1.))
                    if not math.isfinite(q) or not 0 <= q <= 1:
                        raise ValueError('loss_confidence values must be in [0,1]')
                    if term in ('missing', 'false') and not reliable:
                        stats['skip_' + term + '_contrast'] += 1
                        q = 0.
                    if q == 0:
                        stats['skip_' + term + '_confidence'] += 1
                    elif float(supports[term].sum()) >= (self.area_floor if self.area_mode == 'hard' else 1e-12):
                        stats['effective_' + term] += 1
                    values[term] = q * self._mean(errors[term], supports[term], term, stats)
                if self.coefficients['pixel']:
                    # A local native UV anchor accompanies luma reconstruction;
                    # soft chroma coverage preserves the original 420 phase.
                    support_uv = F.avg_pool2d(stroke, 2) * 4
                    values['pixel'] = values['pixel'] + .25 * confidence * float(gates.get('pixel', 1.)) * self._mean(
                        uv_error[i:i+1], support_uv, 'pixel_uv', stats)
                if self.coefficients['uv']:
                    # Convert native chroma coverage to RGB area units before applying A_min.
                    support_uv = F.avg_pool2d(stroke, 2) * 4
                    values['uv'] = confidence * float(gates.get('uv', 1.)) * self._mean(uv_error[i:i+1], support_uv, 'uv', stats)
                else:
                    values['uv'] = zero
                values['cp'] = zero
                if self.cp is not None:
                    if roi.get('cp_eligible', False) and roi.get('complete', False) and roi.get('orientation', 'horizontal') == 'horizontal':
                        x0, y0, x1, y1 = map(int, roi.get('tight_bbox', roi['bbox']))
                        if not (0 <= x0 < x1 <= expected[3] and 0 <= y0 < y1 <= expected[2]):
                            raise ValueError('CP tight_bbox must be nonempty and inside the canvas')
                        cp_valid = bool((valid[i:i+1, :, y0:y1, x0:x1] >= 1 - 1e-6).all())
                        cp_area = float(stroke.sum())
                        cp_gate = (float(cp_area >= self.area_floor) if self.area_mode == 'hard'
                                   else cp_area / max(cp_area, self.area_floor))
                        if not cp_valid:
                            stats['skip_cp_validity'] += 1
                            cp_gate = 0.
                        if cp_area < self.area_floor and self.area_mode == 'hard':
                            stats['skip_cp_area'] += 1
                        cp_confidence = confidence * float(gates.get('cp', 1.))
                        if cp_confidence == 0:
                            stats['skip_cp_confidence'] += 1
                        if rgb_prediction is None:
                            rgb_prediction = YUV2RGB(y4uv_to_yuv444(prediction), is_train=True)
                            rgb_target = YUV2RGB(y4uv_to_yuv444(target.detach()), is_train=True)
                        polygon = [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]]
                        line_record = [coordinate for point in polygon for coordinate in point] + ['']
                        if cp_gate > 0 and cp_confidence > 0:
                            values['cp'] = cp_confidence * cp_gate * self.cp(rgb_prediction[i:i+1], rgb_target[i:i+1], [[line_record]])
                            if self.cp.last_stats.get('accepted_lines', 0) > 0:
                                stats['effective_cp'] += 1
                            else:
                                stats['skip_cp_geometry'] += 1
                    else:
                        stats['skip_cp_geometry'] += 1
                source_bbox = roi.get('original_bbox', roi.get('tight_bbox', roi['bbox']))
                height = int(roi.get('char_height', source_bbox[3] - source_bbox[1]))
                group = 'small' if height <= 24 else ('medium' if height <= 64 else 'large')
                stats['roi_' + group] += 1
                stats['stroke_mass_' + group] += float(stroke.sum())
                record = {'tile': i, 'instance_id': roi['instance_id'], 'group': group,
                          'stroke_mass': float(stroke.sum()), 'ring_mass': float(ring_supervised.sum()),
                          'confidence': confidence, 'weight': weight, 'contrast': float(contrast)}
                record['instance_height'] = height
                record['scheduled_exposure'] = roi.get('scheduled_exposure')
                record['support_mass'] = {term: float(support.sum()) for term, support in supports.items()}
                record['contrast_reliable'] = reliable
                record['loss_confidence'] = dict(gates)
                for term, value in values.items():
                    components[term] = components[term] + weight * value
                    record[term] = float(value.detach())
                roi_stats.append(record)
            for term in totals:
                totals[term].append(components[term] / max(denominator, 1e-12))
        terms = {k: torch.stack(v).mean() for k, v in totals.items()}
        text_loss = sum((self.coefficients[k] * v for k, v in terms.items()), zero)
        total = image + self.text_weight * float(text_scale) * text_loss
        self.last_stats, self.last_roi_stats = dict(stats), roi_stats
        return total, {'l_image': image, 'l_image_native': image_native, 'l_image_bg_guard': image_bg,
                       'l_text': text_loss, 'l_total': total,
                       'l_text_sdf_band_anchor': terms['sdf'],
                       **{'l_text_' + k: v for k, v in terms.items()}}

    def audit_gt_gradient(self, target, batch):
        """@luna-comment
        Input: native GT and annotations to audit at the measured correct prediction.
        Output: objective scalar and maximum absolute GT gradient; no parameter changes.
        Key Steps: detach GT, enable prediction gradients, run objective and autograd.
        Key Params: uses currently enabled terms without synthetic transition profiles.
        State/Dependencies: updates normal diagnostic telemetry; does not call the generator.
        """
        with torch.enable_grad():
            prediction = target.detach().clone().requires_grad_(True)
            total, _ = self(prediction, target.detach(), batch)
            gradient, = torch.autograd.grad(total, prediction)
        return {'loss': float(total.detach()), 'gradient_max': float(gradient.abs().max())}
