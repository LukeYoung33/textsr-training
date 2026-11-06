"""Append four frozen GT1/R1 checkpoint results to the existing fixed32 page."""
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
SOURCE = Path('Y:/guang.yang/TextSR/data/gt1_r1_four_latest_mojito32_20261105')


def main():
    """@luna-comment Input:existing options and256 frozen new outputs. Output:eight additional options with actual iterations. Key Steps:completion audit,32 input identities,new image headers,atomic render. Key Params:existing images never rechecked;original IDs retained. State/Dependencies:accepted renderer,current local page,NAS new results."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    audit = json.loads((SOURCE/'AUDIT.json').read_text())
    assert audit['passed'] and audit['outputs'] == 256 and not audit['old_results_rechecked']
    data = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    existing = {r['id']: r for r in data['rows']}
    selection = json.loads((SOURCE/'selection.json').read_text())['samples']
    original = {s['id']: s for s in json.loads((ROOT/'evaluation/mojito280_subset32_20261104/selection.json').read_text())['samples']}
    assert {s['id'] for s in selection} == set(existing) == set(original)
    for s in selection:
        assert s['input_sha256'] == original[s['id']]['input_sha256'] and s['source_sha256'] == original[s['id']]['source_sha256']
    bindings = json.loads((SOURCE/'MODEL_BINDINGS.json').read_text())
    assert len(bindings) == 4
    new_keys = {arm+suffix for arm in bindings for suffix in ['', '_raw']}
    data['models'] = [m for m in data['models'] if m['key'] not in new_keys]
    old_options = len(data['models'])
    for arm, b in bindings.items():
        for suffix, label in [('_raw', '模型原始输出'), ('', '固化后处理')]:
            data['models'].append(dict(key=arm+suffix, label=b['label']+' | '+str(b['iteration']//1000)+'K | '+label,
                 note='256/224/batch8/crop8/smooth16'+('；未加后处理' if suffix else '；Y后处理 .75 / 2 / 24')))
    jobs = []
    outputs = json.loads((SOURCE/'OUTPUTS.json').read_text())['samples']
    assert len(outputs) == 128
    for record in outputs:
        row = existing[record['id']]
        for suffix, field in [('_raw', 'raw'), ('', 'post')]:
            key = record['arm']+suffix
            url = file_url(record[field])
            row['images'][key] = url
            jobs.append((row['id'], key, url, row['width'], row['height']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check_image, jobs))
    assert len(checked) == 256 and len(data['models']) == old_options+8
    data['title'] = 'Mojito32 · 新四组 GT+1px / VOSR R1 对照'
    data['subtitle'] = '固定32张 · 新四组最新权重（各自迭代数见名称）'
    data['defaults'] = ','.join(arm+'_raw' for arm in bindings)+',v3_input,v3m_l1_raw'
    note = '\n新增四组均50%GT退化、关闭对比度对齐，新训练不含Mojito。各组使用本次冻结的最新完整权重，迭代数不同；默认四组原始输出、V3输入及旧L1，后处理版本可从下拉选择。'
    if note not in data['notes']:
        data['notes'] += note
    text = render_html(data)
    assert text.count('for(let i=0;i<4;i++)') == 1
    text = text.replace('for(let i=0;i<4;i++)', 'for(let i=0;i<6;i++)').replace(
        'grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;', 'grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    for filename in ['manifest.json', 'index.html']:
        backup = OUT/(filename+'.before_gt1_r1')
        if not backup.exists():
            shutil.copy2(OUT/filename, backup)
    for filename, content in [('manifest.json', json.dumps(data, ensure_ascii=False, indent=2)), ('index.html', text)]:
        tmp = OUT/(filename+'.gt1_r1.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(OUT/filename)
    (OUT/'GT1_R1_AUDIT.json').write_text(json.dumps(dict(passed=True, new_options=8, total_options=len(data['models']),
        checked_new_paths=256, old_images_rechecked=False, source_audit_sha256=digest(SOURCE/'AUDIT.json'), records=checked), indent=2))
    print(json.dumps(dict(passed=True, rows=32, total_options=len(data['models']), checked_new_paths=256, page=str(OUT/'index.html'))))


if __name__ == '__main__':
    main()
