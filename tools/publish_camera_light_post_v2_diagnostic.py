"""Publish the completed 280-row V2 failure as an explicitly diagnostic viewer."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html
from tools.run_camera_light_post_v2 import (INTEGRATION_VERSION, KEYS, LEGACY_INTEGRATION_VERSION,
                                            LOCAL, MANIFEST, OUT, digest, mapped_path)


def publish():
    """@luna-comment Input: original 13-choice viewer, completed source preflight, 840 output receipts/files. Output: backed-up original and 16-choice diagnostic viewer. Key Steps: bind every source/receipt/core/version and preserve old models/row fields before atomic main-manifest replacement. Key Params: structural publication only; full PNG hashing is separately audited. State/Dependencies: E viewer and approved Y backup/audit writes; no image mutation."""
    original = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if len(original['models']) != 13 or len(original['rows']) != 280:
        raise AssertionError('Expected untouched 13-choice, 280-row original manifest')
    preflight = json.loads((OUT/'evaluation/source_preflight.json').read_text(encoding='utf-8'))
    source_records = {v['id']: v for v in preflight['records']}
    if len(source_records) != 280 or preflight['rows'] != 280 or preflight['exceptions'] != ['017']:
        raise AssertionError('Source preflight does not bind all 280 with sole 017 refresh exception')
    core_hashes = {'light_v1_standard': digest(ROOT/'tools/light_text_postprocess.py'),
                   'light_v2_low': digest(ROOT/'tools/light_text_postprocess_v2.py'),
                   'light_v2_standard': digest(ROOT/'tools/light_text_postprocess_v2.py')}
    audit = []
    for row in original['rows']:
        identity = row['id']
        source = source_records[identity]
        if mapped_path(row['images']['vosr_v5_block']) != Path(source['baseline']):
            raise AssertionError('Viewer/source preflight path mismatch '+identity)
        for key in KEYS:
            target = OUT/key/(identity+'.png')
            report_path = target.with_suffix('.json')
            if not target.is_file() or target.stat().st_size == 0 or not report_path.is_file():
                raise AssertionError('Missing trial image/receipt '+identity+' '+key)
            report = json.loads(report_path.read_text(encoding='utf-8'))
            versions = (INTEGRATION_VERSION,) if identity == '017' else (INTEGRATION_VERSION, LEGACY_INTEGRATION_VERSION)
            if (report['id'] != identity or report['key'] != key
                    or report['source_sha256'] != source['baseline_sha256']
                    or report['core_sha256'] != core_hashes[key]
                    or report['integration_version'] not in versions):
                raise AssertionError('Unbound trial receipt '+identity+' '+key)
            row['images'][key] = 'file:///Y:/guang.yang/TextSR/data/camera_post_v2_20261023/full/'+key+'/'+identity+'.png'
            audit.append({'id': identity, 'key': key, 'receipt': str(report_path), 'png_sha256_claim': report['final_sha256']})
    original['models'].extend([
        {'key': 'light_v1_standard', 'label': 'V5 block256 + V1 standard', 'note': '同一 V5 原图的历史轻后处理对照。'},
        {'key': 'light_v2_low', 'label': 'V5 block256 + V2 low · 视觉验收未通过', 'note': '对比度不足、可见亮边；仅用于失败结果诊断，不推荐使用。'},
        {'key': 'light_v2_standard', 'label': 'V5 block256 + V2 standard · 视觉验收未通过', 'note': '对比度不足、可见亮边；仅用于失败结果诊断，不推荐使用。'},
    ])
    original['defaults'] = 'vosr_v5_block,light_v1_standard,light_v2_low,light_v2_standard'
    original['title'] = '相机 280 图：V5 轻后处理失败结果诊断对照'
    original['subtitle'] = '同一 V5 block256 原图 · V1 历史对照 · V2 low / standard 均未通过视觉验收'
    original['notes'] = 'V2 low 与 standard 视觉验收未通过：文字对比度不足，仍可见白色亮边。两列仅供诊断，不推荐使用；V5 原图、V1 和既有模型结果继续保留。'
    original['development'] = False
    backup = OUT/'evaluation/original_13_choice_manifest.json'
    backup_html = OUT/'evaluation/original_13_choice_index.html'
    if backup.exists() or backup_html.exists():
        raise AssertionError('Original backup already exists; inspect before publication')
    shutil.copyfile(MANIFEST, backup)
    shutil.copyfile(LOCAL/'index.html', backup_html)
    (LOCAL/'light_post_v2_manifest.json').write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding='utf-8')
    (LOCAL/'index.html').write_text(render_html(original), encoding='utf-8')
    temporary = MANIFEST.with_suffix('.light_post_v2.tmp')
    temporary.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(MANIFEST)
    (OUT/'evaluation/DIAGNOSTIC_PUBLICATION.json').write_text(json.dumps({'rows': 280, 'models': 16, 'variants': KEYS,
                                                                           'status': 'visual_quality_failed_diagnostic_only',
                                                                           'records': audit}, indent=2), encoding='utf-8')
    print(json.dumps({'rows': 280, 'models': 16, 'status': 'visual_quality_failed_diagnostic_only'}))


if __name__ == '__main__':
    publish()
