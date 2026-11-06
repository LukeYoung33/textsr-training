"""Apply both classical postprocess versions to the exact existing 280 V5 block finals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html
from tools.doc_warp_composition import inverse_residual
from tools.light_text_postprocess import enhance_bgr
from tools.light_text_postprocess_v2 import enhance_bgr_v2

MANIFEST = ROOT/'output/camera_2x2_comparison_20261022/manifest.json'
BASE = Path('Y:/guang.yang/TextSR/data/vosr_final_evaluation_20261022')
FIX = Path('Y:/guang.yang/TextSR/data/camera_robust_pre_native_v22_017_fix_20261022')
OUT = Path('Y:/guang.yang/TextSR/data/camera_post_v2_20261023/full')
LOCAL = ROOT/'output/camera_2x2_comparison_20261022'
KEYS = ('light_v1_standard', 'light_v2_low', 'light_v2_standard')
INTEGRATION_VERSION = 'existing-final-to-ocr-warp-inverse-scalar-residual-v2-017-refresh'
LEGACY_INTEGRATION_VERSION = 'existing-final-to-ocr-warp-inverse-scalar-residual-v1'


def digest(path):
    """@luna-comment Input: existing or new file. Output: SHA256 hex. Key Steps: stream 4MiB chunks. Key Params: no whole-image hash buffer. State/Dependencies: read-only file access."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mapped_path(value):
    """@luna-comment Input: approved Y file URL or /AI_DNS path. Output: mapped Y path. Key Steps: validate exact allowed data prefix. Key Params: rejects other mounts. State/Dependencies: read-only source resolution."""
    if value.startswith('file:///Y:/'):
        value = unquote(value[len('file:///Y:/'):])
    elif value.startswith('/AI_DNS/'):
        value = value[len('/AI_DNS/'):]
    else:
        raise ValueError('Unapproved source path '+value)
    if not value.startswith('guang.yang/TextSR/data/'):
        raise ValueError('Unapproved data root')
    return Path('Y:/'+value)


def read_bgr(path):
    """@luna-comment Input: approved image path. Output: BGR8 with checked decode. Key Steps: byte decode to support Windows Unicode paths. Key Params: no resampling. State/Dependencies: OpenCV."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode '+str(path))
    return image


def write_png(path, image):
    """@luna-comment Input: new Y output path/BGR8. Output: lossless PNG. Key Steps: encode then exclusive create. Key Params: compression3. State/Dependencies: never overwrites existing results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError('PNG encoding failed')
    with path.open('xb') as stream:
        stream.write(data.tobytes())


def apply_one(baseline, documents, key):
    """@luna-comment Input: exact V5 final, bound document warp/OCR/valid records and variant. Output: native BGR8 plus per-stage telemetry. Key Steps: warp final, process OCR support, inverse only residual with confidence ownership, common final integer/gamut cap. Key Params: existing 8px document feather; no model rerun. State/Dependencies: v1/v2 operators and inverse_residual."""
    canvas = np.zeros(baseline.shape[:2], np.float32)
    owned = np.zeros(baseline.shape[:2], bool)
    stages = []
    for item, ocr, valid in documents:
        h, w = ocr['warp_hw']
        matrix = np.asarray(item['warp']['image_to_output'], np.float64)
        warped = cv2.warpPerspective(baseline, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if key == 'light_v1_standard':
            enhanced, report = enhance_bgr(warped, 'standard', ocr['boxes'])
        else:
            enhanced, report = enhance_bgr_v2(warped, 'low' if key == 'light_v2_low' else 'standard', ocr['boxes'])
        difference = enhanced.astype(np.int16)-warped.astype(np.int16)
        if not np.array_equal(difference[:, :, 0], difference[:, :, 1]) or not np.array_equal(difference[:, :, 1], difference[:, :, 2]):
            raise AssertionError('Warp BGR delta mismatch')
        mapped, _, take, geometry = inverse_residual(baseline, warped, enhanced, matrix, item['quad'], valid, owned)
        canvas += mapped[:, :, 0]
        owned |= take
        stages.append({'instance_id': item['instance_id'], 'ocr_sha256': ocr['hash'], 'telemetry': report, 'geometry': geometry})
        del warped, enhanced, difference, mapped
    lower = -baseline.min(2).astype(np.float32)
    upper = 255-baseline.max(2).astype(np.float32)
    integer = np.rint(np.minimum(np.maximum(canvas, lower), upper)).astype(np.int16)
    final = (baseline.astype(np.int16)+integer[:, :, None]).astype(np.uint8)
    if not np.array_equal(final[~owned], baseline[~owned]):
        raise AssertionError('Changed outside owned document')
    return final, {'stages': stages, 'changed_pixels': int(np.count_nonzero(integer)), 'max_abs_delta': int(np.abs(integer).max()), 'owned_pixels': int(owned.sum()), 'outside_owned_unchanged': True, 'common_channel_delta': True}


def documents_for(identity, receipt):
    """@luna-comment Input: image ID and verified exact V5 receipt. Output: confidence-sorted bound geometry/OCR/valid triplets. Key Steps: use 017 accepted refresh cache or regular bound document path; verify record/OCR hashes and source linkage. Key Params: existing frozen records only. State/Dependencies: read-only Y originals."""
    record_path = (FIX/'geometry_cache/documents/017.jpg/result.json') if identity == '017' else mapped_path(receipt['document_record'])
    if digest(record_path) != receipt['document_record_sha256']:
        raise AssertionError('Document record hash mismatch')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    doc_root = (FIX/'geometry_cache') if identity == '017' else record_path.parents[2]
    stage_hashes = {v['instance_id']: v for v in receipt['stages']}
    result = []
    for item in sorted(record['instances'], key=lambda v: (-v['confidence'], v['instance_id'])):
        iid = item['instance_id']
        ocr_file = doc_root/'ocr'/identity/(f'{iid:03d}.json')
        ocr_hash = digest(ocr_file)
        if ocr_hash != stage_hashes[iid]['ocr_sha256']:
            raise AssertionError('OCR hash mismatch')
        ocr = json.loads(ocr_file.read_text(encoding='utf-8'))
        if ocr['source_sha256'] != receipt['source_sha256']:
            raise AssertionError('OCR source mismatch')
        valid_path = doc_root/ocr['valid_file']
        if digest(valid_path) != ocr['valid_sha256']:
            raise AssertionError('Valid mask hash mismatch')
        valid = cv2.imdecode(np.frombuffer(valid_path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        ocr['hash'] = ocr_hash
        result.append((item, ocr, valid))
    return result


def run(limit=0, start=1, worker=False, end=280, verify_only=False):
    """@luna-comment Input: production manifest, optional pilot limit/disjoint ID interval or verify-only flag. Output: three exclusive Y columns, E viewer after full verification, Y audit. Key Steps: hash bind exact finals, apply variants or only verify outputs, preserve prior choices. Key Params: OpenCV2 threads; worker ID interval is inclusive. State/Dependencies: approved Y output directory only; no old-result writes."""
    cv2.setNumThreads(2)
    original = json.loads(MANIFEST.read_text(encoding='utf-8'))
    core_hashes = {'v1': digest(ROOT/'tools/light_text_postprocess.py'),
                   'v2': digest(ROOT/'tools/light_text_postprocess_v2.py')}
    rows = original['rows'][start-1:end]
    if limit:
        rows = rows[:limit]
    if len(original['rows']) != 280:
        raise AssertionError('Expected exact 280-row camera manifest')
    audit = []
    for index, row in enumerate(rows, 1):
        identity = row['id']
        baseline_path = mapped_path(row['images']['vosr_v5_block'])
        receipt_path = baseline_path.parent.parent/'receipts'/(identity+'.json')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if digest(baseline_path) != receipt['final_sha256']:
            raise AssertionError('Exact V5 baseline hash/path mismatch '+identity)
        if identity == '017' and receipt['document_record_sha256'] != json.loads((FIX/'receipts/017.json').read_text(encoding='utf-8'))['corrected_document_record_sha256']:
            raise AssertionError('017 accepted geometry binding mismatch')
        baseline = read_bgr(baseline_path)
        if baseline.shape[:2] != (row['height'], row['width']):
            raise AssertionError('Native dimensions mismatch')
        documents = documents_for(identity, receipt)
        for key in KEYS:
            target = OUT/key/(identity+'.png')
            report_path = OUT/key/(identity+'.json')
            if target.exists() and report_path.exists():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                if (report['source_sha256'] != receipt['final_sha256'] or digest(target) != report['final_sha256']
                        or report.get('core_sha256') != core_hashes['v1' if key == 'light_v1_standard' else 'v2']
                        or report.get('integration_version') not in ((INTEGRATION_VERSION,) if identity == '017' else (INTEGRATION_VERSION, LEGACY_INTEGRATION_VERSION))):
                    raise AssertionError('Existing output not verifiable '+str(target))
            else:
                if verify_only:
                    raise AssertionError('Missing output in verify-only mode '+str(target))
                if target.exists() or report_path.exists():
                    raise AssertionError('Incomplete prior result '+str(target))
                final, info = apply_one(baseline, documents, key)
                write_png(target, final)
                report = {'id': identity, 'key': key, 'source_sha256': receipt['final_sha256'], 'final_sha256': digest(target),
                          'source_receipt_sha256': digest(receipt_path),
                          'core_sha256': core_hashes['v1' if key == 'light_v1_standard' else 'v2'],
                          'integration_version': INTEGRATION_VERSION, **info}
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                del final
            row['images'][key] = 'file:///Y:/guang.yang/TextSR/data/camera_post_v2_20261023/full/'+key+'/'+identity+'.png'
            audit.append({'id': identity, 'key': key, 'report': str(report_path), 'final_sha256': report['final_sha256'], 'changed_pixels': report['changed_pixels']})
        print(json.dumps({'done': index, 'total': len(rows), 'id': identity}), flush=True)
    if worker:
        return
    if not limit and len(rows) != 280:
        raise AssertionError('Publishing requires all 280 rows')
    new_models = [
        {'key': 'light_v1_standard', 'label': 'V5 block256 + V1 standard', 'note': 'Same original V5; historical light postprocess comparator.'},
        {'key': 'light_v2_low', 'label': 'V5 block256 + V2 low · 视觉验收未通过', 'note': '对比度不足、可见亮边；仅用于失败结果诊断，不推荐使用。'},
        {'key': 'light_v2_standard', 'label': 'V5 block256 + V2 standard · 视觉验收未通过', 'note': '对比度不足、可见亮边；仅用于失败结果诊断，不推荐使用。'},
    ]
    existing_models = {v['key'] for v in original['models']}
    original['models'].extend(v for v in new_models if v['key'] not in existing_models)
    if len(original['models']) != 16:
        raise AssertionError('Expected 16 unique viewer models')
    original['rows'] = rows
    original['defaults'] = 'vosr_v5_block,light_v1_standard,light_v2_low,light_v2_standard'
    original['title'] = '相机 280 图：V5 轻后处理失败结果诊断对照'
    original['subtitle'] = '同一 V5 block256 原图 · V1 历史对照 · V2 low / standard 均未通过视觉验收'
    original['notes'] = 'V2 low 与 standard 视觉验收未通过：文字对比度不足，仍可见白色亮边。两列仅供诊断，不推荐使用；V5 原图、V1 和既有模型结果继续保留。'
    original['development'] = bool(limit)
    local_manifest = LOCAL/('light_post_v2_pilot_manifest.json' if limit else 'light_post_v2_manifest.json')
    local_html = LOCAL/('light_post_v2_pilot.html' if limit else 'index.html')
    if not limit:
        backup = OUT/'evaluation/original_13_choice_manifest.json'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            if len(json.loads(MANIFEST.read_text(encoding='utf-8'))['models']) != 13:
                raise AssertionError('Original manifest backup must precede the 16-choice update')
            shutil.copyfile(MANIFEST, backup)
    local_manifest.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding='utf-8')
    local_html.write_text(render_html(original), encoding='utf-8')
    if not limit:
        temporary = MANIFEST.with_suffix('.light_post_v2.tmp')
        temporary.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(MANIFEST)
    (OUT/('PILOT_AUDIT.json' if limit else 'BATCH_AUDIT.json')).write_text(json.dumps({'rows': len(rows), 'variants': KEYS, 'records': audit, 'viewer': str(local_html)}, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'variants': len(KEYS), 'html': str(local_html)}), flush=True)


def main():
    """@luna-comment Input: optional pilot row count or disjoint worker start. Output: batch run. Key Steps: parse arguments and dispatch. Key Params: default all280; worker start skips viewer write. State/Dependencies: Y writes limited to dedicated output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=280)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= 280:
        raise ValueError('ID interval must fit 1..280')
    run(args.limit, args.start, args.worker, args.end, args.verify_only)


if __name__ == '__main__':
    main()
