"""Bounded native-frame camera contrast-repair pilot; never edits the 280-row gallery."""
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.doc_warp_composition import inverse_residual
from tools.light_text_postprocess_v2_repair import RepairConfig, enhance_bgr_repair
from tools.run_camera_light_post_v2 import (MANIFEST, OUT, digest, documents_for,
                                             mapped_path, read_bgr, write_png)


PILOT = OUT.parent/'pilot_repair'
IDS = ('006', '099', '017')
SCALES = (('lambda_0p5', .5), ('lambda_1p0', 1.))
ROIS = {'006': (1600, 2020, 1850, 2140),
        '099': (1280, 2620, 2570, 2780),
        '017': (1450, 2370, 1940, 2480)}
KEYS = ('BL', 'V1', 'V2std', 'repair_0p5', 'repair_1p0')


def candidate_native(baseline, documents, scale):
    """@luna-comment Input: exact V5 final, bound OCR geometry and contrast lambda. Output: native BGR8 candidate plus per-instance telemetry. Key Steps: warp same final, isolated candidate, inverse only scalar residual with same ownership, final common gamut-limited integer delta. Key Params: existing 8px document feather. State/Dependencies: frozen geometry helpers; no writes."""
    canvas = np.zeros(baseline.shape[:2], np.float32)
    owned = np.zeros(baseline.shape[:2], bool)
    stages = []
    for item, ocr, valid in documents:
        h, w = ocr['warp_hw']
        matrix = np.asarray(item['warp']['image_to_output'], np.float64)
        warped = cv2.warpPerspective(baseline, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        enhanced, telemetry = enhance_bgr_repair(warped, ocr['boxes'], RepairConfig(scale))
        difference = enhanced.astype(np.int16)-warped.astype(np.int16)
        if not (np.array_equal(difference[:, :, 0], difference[:, :, 1])
                and np.array_equal(difference[:, :, 1], difference[:, :, 2])):
            raise AssertionError('Candidate warp BGR delta mismatch')
        mapped, _, take, geometry = inverse_residual(baseline, warped, enhanced, matrix, item['quad'], valid, owned)
        canvas += mapped[:, :, 0]
        owned |= take
        stages.append({'instance_id': item['instance_id'], 'ocr_sha256': ocr['hash'],
                       'telemetry': telemetry, 'geometry': geometry})
        del warped, enhanced, difference, mapped
    lower = -baseline.min(2).astype(np.float32)
    upper = 255-baseline.max(2).astype(np.float32)
    integer = np.rint(np.minimum(np.maximum(canvas, lower), upper)).astype(np.int16)
    final = (baseline.astype(np.int16)+integer[:, :, None]).astype(np.uint8)
    common = (np.array_equal(final[:, :, 0].astype(np.int16)-baseline[:, :, 0].astype(np.int16), integer)
              and np.array_equal(final[:, :, 1].astype(np.int16)-baseline[:, :, 1].astype(np.int16), integer)
              and np.array_equal(final[:, :, 2].astype(np.int16)-baseline[:, :, 2].astype(np.int16), integer))
    outside = bool(np.array_equal(final[~owned], baseline[~owned]))
    if not (common and outside):
        raise AssertionError('Native support/common-channel invariant failed')
    return final, {'stages': stages, 'changed_pixels': int(np.count_nonzero(integer)),
                   'max_abs_delta': int(np.abs(integer).max()), 'owned_pixels': int(owned.sum()),
                   'outside_owned_unchanged': outside, 'common_channel_delta': common}


def roi_metrics(identity, variants):
    """@luna-comment Input: exact five native variants and predefined source crop. Output: source-derived signed contrast and stroke coverage checks. Key Steps: same-pixel Y, 006 G neighbors, 099 baseline-background dark mask and scanline footprint, 017 dark/ring proxy. Key Params: ROI maps are diagnostic, not ground truth. State/Dependencies: no writes."""
    x0, y0, x1, y1 = ROIS[identity]
    y = {key: cv2.cvtColor(value[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.int16)
         for key, value in variants.items()}
    source = y['BL'].astype(np.float32)
    smooth = cv2.GaussianBlur(source, (0, 0), 9.)
    dark = source < smooth-10
    bright = source > smooth+10
    report = {'source_crop_xyxy': ROIS[identity], 'source_dark_mask_pixels': int(dark.sum()),
              'source_bright_mask_pixels': int(bright.sum()), 'variants': {}}
    for key in KEYS:
        value = y[key]
        delta = value-y['BL']
        row = {'mean_Y': float(value.mean()),
               'dark_delta_median': float(np.median(delta[dark])) if dark.any() else None,
               'dark_delta_p05': float(np.percentile(delta[dark], 5)) if dark.any() else None,
               'bright_delta_median': float(np.median(delta[bright])) if bright.any() else None,
               'bright_delta_p95': float(np.percentile(delta[bright], 95)) if bright.any() else None,
               'dark_pixels_below_source_bg_minus_10': int(np.count_nonzero(value<smooth-10)),
               'dark_scanline_width_p50': float(np.median(np.count_nonzero(value<smooth-10, axis=1)))}
        if identity == '006':
            row['G_outer_blue_neighbor_Y'] = {f'{x},{yy}': int(value[yy-y0, x-x0])
                                               for x, yy in ((1746, 2030), (1746, 2031))}
            bgr = variants['BL'][y0:y1, x0:x1].astype(np.int16)
            blue = (bgr[:, :, 0]>bgr[:, :, 2]+20)&(y['BL']<100)
            white = y['BL']>200
            distance = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            row['blue_background_distance_bands'] = {}
            for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32)):
                pick = blue&(distance>=lo)&(distance<hi)
                values = delta[pick]
                row['blue_background_distance_bands'][f'{lo}-{hi}px'] = {
                    'pixels': int(values.size), 'mean_delta': float(values.mean()) if values.size else None,
                    'p05_delta': float(np.percentile(values, 5)) if values.size else None,
                    'darkened_ge_3': int(np.count_nonzero(values<=-3))}
        if identity == '099':
            row['reported_dark_stroke_Y_1707_2725'] = int(value[2725-y0, 1707-x0])
        report['variants'][key] = row
    return report


def write_pilot_html(rows):
    """@luna-comment Input: three completed pilot row descriptors. Output: standalone Y HTML for visual review. Key Steps: ID selector, exact native-coordinate crop per column, source PNG links and explicit pending banner. Key Params: CSS pixelated zoom only; no gallery state. State/Dependencies: writes approved pilot directory."""
    data = json.dumps(rows, ensure_ascii=False).replace('</', '<\\/')
    labels = {'BL': 'V5 原图', 'V1': '冻结 V1 standard', 'V2std': '冻结 V2 standard · 已验收失败',
              'repair_0p5': '修复试验 λ=0.5', 'repair_1p0': '修复试验 λ=1.0'}
    cards = ''.join(f'<article><h3>{html.escape(labels[key])}</h3><a id="link_{key}" target="_blank">打开完整原生图片</a><img id="image_{key}" alt="{html.escape(labels[key])} 原生坐标裁块"></article>' for key in KEYS)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>相机 V5 对比度修复试验</title>
<style>body{{font:16px system-ui;background:#111;color:#eee;margin:1.4rem}}select{{font:inherit}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:12px;overflow:auto}}article{{background:#222;padding:8px}}article img{{display:block;width:100%;image-rendering:pixelated;margin-top:8px}}a{{color:#9cf}}.warn{{background:#513;padding:10px}}</style>
<h1>相机 V5 对比度修复试验</h1><p class="warn">仅供三例视觉评审。旧 V2 已验收失败；新试验的质量尚待您判断。本页没有更新 280 张正式对照。</p>
<p>新方案只把 V2 的对比度增量换成冻结 V1 standard 的原公式，再分别乘以 λ=0.5 或 1.0；V2 的去 halo、AA、锐化、支持域和最终限幅保持原样。因此它不等同完整 V1，也不能据此宣称已经消除 halo。以下五列使用相同原图与原生坐标裁块。</p>
<label>样本 <select id="sample"></select></label><p id="coords"></p><div class="grid">{cards}</div>
<script>const rows={data};const keys={json.dumps(KEYS)};const selector=document.getElementById('sample');
for(const row of rows){{const opt=document.createElement('option');opt.value=row.id;opt.textContent=row.id;selector.appendChild(opt)}}
function show(){{const row=rows.find(item=>item.id===selector.value);document.getElementById('coords').textContent='原生图坐标裁块 [x0,y0,x1,y1]：'+row.roi.join(', ');for(const key of keys){{document.getElementById('image_'+key).src=row.crops[key];document.getElementById('link_'+key).href=row.full[key]}}}}
selector.addEventListener('change',show);show();</script></html>'''
    destination = PILOT/'index.html'
    destination.write_text(page, encoding='utf-8')
    return destination


def run():
    """@luna-comment Input: exact three manifest rows and frozen assets. Output: six new pilot PNG/receipts, ROI JSON and standalone HTML under approved Y directory. Key Steps: hash-bind sources, process lambda .5 before 1, verify all 5 choices, emit provenance and numeric checks. Key Params: IDs 006/099/017 only. State/Dependencies: never edits E gallery or historical Y outputs."""
    cv2.setNumThreads(2)
    PILOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    core = {'v1': digest(ROOT/'tools/light_text_postprocess.py'),
            'v2': digest(ROOT/'tools/light_text_postprocess_v2.py'),
            'repair': digest(ROOT/'tools/light_text_postprocess_v2_repair.py'),
            'runner': digest(Path(__file__))}
    summary = {'status': 'objective pilot complete; visual judgment pending', 'ids': list(IDS),
               'contrast_scales': [.5, 1.], 'core_sha256': core, 'rows': []}
    for identity in IDS:
        row = next(v for v in manifest['rows'] if v['id'] == identity)
        source = mapped_path(row['images']['vosr_v5_block'])
        receipt_path = source.parent.parent/'receipts'/(identity+'.json')
        source_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if digest(source) != source_receipt['final_sha256']:
            raise AssertionError('Exact viewer source receipt mismatch '+identity)
        baseline = read_bgr(source)
        if baseline.shape[:2] != (row['height'], row['width']):
            raise AssertionError('Native dimension mismatch '+identity)
        documents = documents_for(identity, source_receipt)
        variants = {'BL': baseline}
        full = {'BL': source.as_uri()}
        historical_receipts = {}
        for label, key, hash_key in [('V1', 'light_v1_standard', 'v1'), ('V2std', 'light_v2_standard', 'v2')]:
            path = mapped_path(row['images'][key])
            historical = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if (historical['source_sha256'] != source_receipt['final_sha256']
                    or historical['core_sha256'] != core[hash_key]
                    or digest(path) != historical['final_sha256']):
                raise AssertionError('Frozen comparator receipt mismatch '+str(path))
            variants[label] = read_bgr(path)
            historical_receipts[label] = digest(path.with_suffix('.json'))
            full[label] = path.as_uri()
        pilot_receipts = {}
        for label, scale in SCALES:
            candidate, info = candidate_native(baseline, documents, scale)
            if candidate.shape != baseline.shape:
                raise AssertionError('Candidate native shape mismatch')
            key = 'repair_0p5' if scale == .5 else 'repair_1p0'
            target = PILOT/label/(identity+'.png')
            receipt = target.with_suffix('.json')
            if target.exists() or receipt.exists():
                raise FileExistsError('Refusing to overwrite pilot result '+str(target))
            write_png(target, candidate)
            report = {'id': identity, 'variant': key, 'contrast_scale': scale,
                      'source_url': row['images']['vosr_v5_block'],
                      'source_sha256': source_receipt['final_sha256'],
                      'source_receipt_sha256': digest(receipt_path),
                      'document_record_sha256': source_receipt['document_record_sha256'],
                      'core_sha256': core, 'output_sha256': digest(target),
                      'native_hw': list(candidate.shape[:2]), 'integration': 'exact_existing_final_to_existing_ocr_warp_inverse_scalar_residual',
                      **info}
            receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            variants[key] = candidate
            pilot_receipts[key] = str(receipt)
            full[key] = target.as_uri()
            print(json.dumps({'id': identity, 'variant': key, 'changed_pixels': info['changed_pixels']}), flush=True)
        x0, y0, x1, y1 = ROIS[identity]
        crops = {}
        for key in KEYS:
            crop = PILOT/'crops'/identity/(key+'.png')
            if crop.exists():
                raise FileExistsError('Refusing to overwrite pilot crop '+str(crop))
            write_png(crop, variants[key][y0:y1, x0:x1])
            crops[key] = crop.as_uri()
        summary['rows'].append({'id': identity, 'roi': ROIS[identity], 'source_sha256': source_receipt['final_sha256'],
                                'source_receipt_sha256': digest(receipt_path),
                                'historical_receipt_sha256': historical_receipts,
                                'candidate_receipts': pilot_receipts, 'full': full, 'crops': crops,
                                'measurements': roi_metrics(identity, variants)})
        del baseline, variants
    summary['html'] = str(write_pilot_html(summary['rows']))
    destination = PILOT/'pilot_measurements.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    run()
