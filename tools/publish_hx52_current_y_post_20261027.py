"""Verify and publish current-Y outputs for the two requested HX52 model arms."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html

BASE = ROOT/'output/hx52_finetuned_v3_v3m_0921_20261027'
OUT = BASE/'post_current_y_20261027'
ARMS = ('full_v3', 'text15_v3m')


def publish():
    """@luna-comment
    Input:completed 104 current-Y PNGs/receipts and frozen HX52 manifest. Output:verified nine-choice inline viewer.
    Key Steps:hash and decode every source/output, verify source-clipped common delta and native shape, preserve
    original entries, back up original files before adding two post choices. Key Params:52 IDs/two arms/current cap24.
    State/Dependencies:frozen deployed-formula core; shared offline renderer; never overwrite source model PNGs.
    """
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    manifest_path = BASE/'manifest.json'
    original_html_bytes = (BASE/'index.html').read_bytes()
    manifest_backup = OUT/'manifest_before_post.json'
    html_backup = OUT/'index_before_post.html'
    manifest_bytes = manifest_backup.read_bytes() if manifest_backup.exists() else manifest_path.read_bytes()
    source = json.loads(manifest_bytes)
    assert len(source['rows']) == 52 and len(source['models']) == 7
    records = summary['records']
    assert len(records) == 104
    by_pair = {(r['model'], r['id']): r for r in records}
    assert len(by_pair) == 104
    expected = {(arm, row['id']) for arm in ARMS for row in source['rows']}
    assert set(by_pair) == expected
    assert summary['bindings_sha256']['core'] == hashlib.sha256((ROOT/'tools/native_y_curve_four_images.py').read_bytes()).hexdigest()
    for arm in ARMS:
        assert {p.stem for p in (OUT/arm).glob('*.png')} == {row['id'] for row in source['rows']}
    audit = []
    for row in source['rows']:
        for arm in ARMS:
            record = by_pair[(arm, row['id'])]
            assert record['bindings_sha256'] == summary['bindings_sha256']
            assert record['parameters']['gain'] == .75 and record['parameters']['softcap'] == 24
            assert record['parameters']['deadband'] == 2 and record['zero_candidate_equals_current']
            url = row['images'][arm]
            source_path = Path(unquote(urlparse(url).path).lstrip('/'))
            source_bytes = source_path.read_bytes()
            assert record['source_url'] == url
            assert hashlib.sha256(source_bytes).hexdigest() == record['source_sha256']
            output_path = OUT/arm/(row['id']+'.png')
            output_bytes = output_path.read_bytes()
            assert hashlib.sha256(output_bytes).hexdigest() == record['output_sha256']
            original = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
            result = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_COLOR)
            assert original.shape == result.shape == (row['height'], row['width'], 3)
            delta = result.astype(np.int16)-original.astype(np.int16)
            assert np.max(np.abs(delta)) <= 24
            # At least one unsaturated channel exposes the common integer delta;
            # choose its largest absolute observed difference then reconstruct all channels.
            picked = np.take_along_axis(delta, np.argmax(np.abs(delta), axis=2)[:, :, None], axis=2)
            reconstructed = np.clip(original.astype(np.int16)+picked, 0, 255).astype(np.uint8)
            assert np.array_equal(reconstructed, result)
            audit.append({'id': row['id'], 'model': arm, 'source_sha256': record['source_sha256'],
                          'output_sha256': record['output_sha256'], 'native_hw': list(result.shape[:2]),
                          'changed_pixels': int(np.count_nonzero(np.any(delta != 0, axis=2))),
                          'max_abs_channel_delta': int(np.max(np.abs(delta)))})
            print('verified '+arm+' '+row['id'], flush=True)
    data = copy.deepcopy(source)
    labels = {'full_v3': 'V3 · Full', 'text15_v3m': 'V3M · Text ×1.5'}
    for arm in ARMS:
        data['models'].append({'key': arm+'_post', 'label': labels[arm]+' + 当前后处理',
                               'note': '部署公式离线复现：PNG 派生 BT.601 Y，16×16背景，增益0.75，软限幅24；非原生YUV逐字节验证。'})
        for row in data['rows']:
            row['images'][arm+'_post'] = (OUT/arm/(row['id']+'.png')).as_uri()
    data['defaults'] = 'full_v3,full_v3_post,text15_v3m,text15_v3m_post'
    data['notes'] = source.get('notes', '').replace('No tonal processing.', 'The original seven choices have no tonal processing.')
    data['notes'] += '\n新增 V3-Full 与 V3M-Text×1.5 的当前后处理，共104张；默认左右显示各模型处理前/后。\n仅复用部署公式：阈值2、增益0.75、软限幅24，不使用新曲线或额外锐化。\n输入为PNG派生Y，输出为RGB裁剪预览；不代表手机原生YUV输出或140ms实测。'
    assert hashlib.sha256(manifest_bytes).hexdigest() == summary['source_manifest_sha256']
    assert json.loads(manifest_path.read_bytes()) in (source, data), 'Gallery manifest changed outside this publication'
    assert (BASE/'index.html').read_bytes() == original_html_bytes, 'Gallery HTML changed during verification'
    if not manifest_backup.exists():
        manifest_backup.write_bytes(manifest_bytes)
        html_backup.write_bytes(original_html_bytes)
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    (BASE/'index.html').write_text(render_html(data), encoding='utf-8')
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps({'outputs_checked': 104, 'original_choices_preserved': True,
        'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'records': audit}, indent=2), encoding='utf-8')
    print('published 52 rows, 9 choices, 104 audited outputs', flush=True)


if __name__ == '__main__':
    publish()
