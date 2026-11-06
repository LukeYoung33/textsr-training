"""Add unthickened original-aligned-GT V3 Full250K to the fixed32 comparison."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html, file_url, digest
from tools.publish_mojito32_all_models import check_image
OUT = ROOT/'evaluation/mojito32_all_models_20261104'
SOURCE = Path('Y:/guang.yang/TextSR/data/v3_original_aligned_gt_250k_mojito32_20261104')
ARM = 'v3_original_aligned_gt_full_250k'
BASE = 'v3m_original_gt_dilate2_l1'


def main():
    """@luna-comment Input:completed64newV3outputs/current30optionpage. Output:32optionpage/defaultoriginalGTreferencepanels. Key Steps:newcompletion/inputIDaudit,64newheaders,appenditeration-labelledraw/post,atomicrender. Key Params:oldresultsnotrechecked;V3Full250KversusV3ML1500Kexplicit. State/Dependencies:acceptedrenderer,frozennewreceipts/currentmanifest."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    assert json.loads((SOURCE/'AUDIT.json').read_text())['outputs'] == 64
    data = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    rows = {r['id']: r for r in data['rows']}
    old = {s['id']: s for s in json.loads((ROOT/'evaluation/mojito280_subset32_20261104/selection.json').read_text())['samples']}
    selection = json.loads((SOURCE/'selection.json').read_text())['samples']
    assert set(rows) == {s['id'] for s in selection} == set(old) and len(rows) == 32
    for s in selection:
        assert s['input_sha256'] == old[s['id']]['input_sha256'] and s['source_sha256'] == old[s['id']]['source_sha256']
    binding = json.loads((SOURCE/'MODEL_BINDINGS.json').read_text())[ARM]
    run = json.loads((SOURCE/'RUN_BINDING.json').read_text())
    assert binding['iteration'] == 250000 and run['architecture'] == 'V3'
    data['models'] = [m for m in data['models'] if m['key'] not in [ARM, ARM+'_raw']]
    assert len(data['models']) == 30
    for suffix, label in [('_raw', '模型原始输出'), ('', '固化后处理')]:
        data['models'].append(dict(key=ARM+suffix, label=binding['label']+' | '+label,
            note='原始GT+2px / 同步新mask / L1+粗细R1；370K中间权重；256/224/crop8/smooth16'+('；未加后处理' if suffix else '；Y后处理 .75/2/24')))
    jobs = []
    records = json.loads((SOURCE/'OUTPUTS.json').read_text())['samples']
    assert len(records) == 32
    for record in records:
        row = rows[record['id']]
        for suffix, field in [('_raw', 'raw'), ('', 'post')]:
            key = ARM+suffix
            url = file_url(record[field])
            row['images'][key] = url
            jobs.append((row['id'], key, url, row['width'], row['height']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check_image, jobs))
    data['title'] = 'Mojito32 · 全部对照＋原始GT粗细R1 370K'
    data['subtitle'] = '32张 · 30个选项 · GT+2px L1-only500K 对照 R1-width370K'
    data['defaults'] = ','.join(['v3_input', BASE+'_raw', ARM+'_raw', 'lq', BASE, ARM])
    note = '\n???????GT????V3 Full??250K?????V3M???????V3??????crop8/smooth16??????????????????GT????????loss?????????????????????????????V3??/??GT-L1??/???GT-V3?????LQ/??GT-L1???/???GT-V3????'
    if note not in data['notes']:
        data['notes'] += note
    text = render_html(data)
    assert text.count('for(let i=0;i<4;i++)') == 1
    text = text.replace('for(let i=0;i<4;i++)', 'for(let i=0;i<6;i++)').replace(
        'grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;', 'grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    for filename, content in [('manifest.json', json.dumps(data, ensure_ascii=False, indent=2)), ('index.html', text)]:
        tmp = OUT/(filename+'.r1.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(OUT/filename)
    (OUT/'ORIGINAL_V3_AUDIT.json').write_text(json.dumps(dict(passed=True, new_options=2, total_options=32,
        checkpoint_iteration=250000, checked_new_paths=64, old_images_rechecked=False,
        source_audit_sha256=digest(SOURCE/'AUDIT.json'), records=checked), indent=2))
    print(json.dumps(dict(passed=True, rows=32, total_options=32, checked_new_paths=64)))


if __name__ == '__main__':
    main()
