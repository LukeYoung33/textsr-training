"""Append only newly audited five-control outputs to the existing fixed32 page."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html, file_url, digest
from tools.publish_mojito32_all_models import check_image

OUT = ROOT/'evaluation/mojito32_all_models_20261104'
SOURCE = Path('Y:/guang.yang/TextSR/data/v3m_five_controls_mojito32_20261104')


def main():
    """@luna-comment Input:existing18options/frozennew320outputs. Output:28optionpage withonlynewcontentaudited. Key Steps:completion/hashboundmetadata,32ID/inputalignment,10newoptions,imageheadercheck,atomicrender. Key Params:old18options/imagesneverrechecked;originalIDs retained. State/Dependencies:acceptedrenderer,currentlocalpage,NASnewresults."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    audit = json.loads((SOURCE/'AUDIT.json').read_text())
    assert audit['passed'] and audit['outputs'] == 320 and not audit['old_results_rechecked']
    data = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    existing = {r['id']: r for r in data['rows']}
    selection = json.loads((SOURCE/'selection.json').read_text())['samples']
    original = {s['id']: s for s in json.loads((ROOT/'evaluation/mojito280_subset32_20261104/selection.json').read_text())['samples']}
    assert {s['id'] for s in selection} == set(existing) == set(original)
    for s in selection:
        assert s['input_sha256'] == original[s['id']]['input_sha256'] and s['source_sha256'] == original[s['id']]['source_sha256']
    bindings = json.loads((SOURCE/'MODEL_BINDINGS.json').read_text())
    assert len(bindings) == 5
    new_keys = {arm+suffix for arm in bindings for suffix in ['', '_raw']}
    data['models'] = [m for m in data['models'] if m['key'] not in new_keys]
    assert len(data['models']) == 18
    for arm, b in bindings.items():
        for suffix, label in [('_raw', '模型原始输出'), ('', '固化后处理')]:
            data['models'].append(dict(key=arm+suffix, label=b['label']+' | 500K | '+label,
                 note='256/224/batch8/crop8/smooth16'+('；未加后处理' if suffix else '；Y后处理 .75 / 2 / 24')))
    jobs = []
    outputs = json.loads((SOURCE/'OUTPUTS.json').read_text())['samples']
    assert len(outputs) == 160
    for record in outputs:
        row = existing[record['id']]
        for suffix, field in [('_raw', 'raw'), ('', 'post')]:
            key = record['arm']+suffix
            url = file_url(record[field])
            row['images'][key] = url
            jobs.append((row['id'], key, url, row['width'], row['height']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check_image, jobs))
    assert len(checked) == 320 and len(data['models']) == 28
    data['title'] = 'Mojito32 · 历史模型＋五组loss对照＋原始GT笔画+2px'
    data['subtitle'] = '固定32张 · 28个选项 · 新增五组最终500K对照'
    data['defaults'] = ','.join(arm+'_raw' for arm in bindings)+',v3m_l1_raw'
    note = '\n新增五组：Full无guard、L1+Missing/False+UV、该组合50%GT退化混合、L1+Missing/False、L1+UV；均最终500K，共享旧六模型的训练池，本子集24张训练/8张验证。默认展示五组原始输出和旧L1-only基线；其他模型与后处理可从下拉选择。'
    if note not in data['notes']:
        data['notes'] += note
    text = render_html(data)
    assert text.count('for(let i=0;i<4;i++)') == 1
    text = text.replace('for(let i=0;i<4;i++)', 'for(let i=0;i<6;i++)').replace(
        'grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;', 'grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    for filename in ['manifest.json', 'index.html']:
        backup = OUT/(filename+'.before_five_controls')
        if not backup.exists():
            shutil.copy2(OUT/filename, backup)
    for filename, content in [('manifest.json', json.dumps(data, ensure_ascii=False, indent=2)), ('index.html', text)]:
        tmp = OUT/(filename+'.five_controls.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(OUT/filename)
    (OUT/'FIVE_CONTROLS_AUDIT.json').write_text(json.dumps(dict(passed=True, new_options=10, total_options=28,
        checked_new_paths=320, old_images_rechecked=False, source_audit_sha256=digest(SOURCE/'AUDIT.json'), records=checked), indent=2))
    print(json.dumps(dict(passed=True, rows=32, total_options=28, checked_new_paths=320, page=str(OUT/'index.html'))))


if __name__ == '__main__':
    main()
