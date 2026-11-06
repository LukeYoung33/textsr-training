"""Add frozen R1-width370K results next to the dilated-GT L1-only500K reference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html, file_url, digest
from tools.publish_mojito32_all_models import check_image
OUT = ROOT/'evaluation/mojito32_all_models_20261104'
SOURCE = Path('Y:/guang.yang/TextSR/data/v3m_r1_width_370k_mojito32_20261104')
ARM = 'v3m_dilate2_r1_width_370k'
BASE = 'v3m_original_gt_dilate2_l1'


def main():
    """@luna-comment Input:completed64newR1outputs/current28optionpage. Output:30optionpage/defaultmatchedreferencepanels. Key Steps:newcompletion/inputIDaudit,64newheaders,appenditeration-labelledraw/post,atomicrender. Key Params:oldresultsnotrechecked;370Kversus500Kexplicit. State/Dependencies:acceptedrenderer,frozennewreceipts/currentmanifest."""
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
    assert binding['iteration'] == 370000 and run['architecture_differences'] == {}
    data['models'] = [m for m in data['models'] if m['key'] not in [ARM, ARM+'_raw']]
    assert len(data['models']) == 28
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
    note = '\n新增粗细R1模型使用最新已完整保存的370K权重（训练仍继续），主参考为原始GT笔画+2px L1-only500K。两者网络配置一致，但同步新mask、loss及训练轮数均有差异，不能归因于单一因素。默认上排V3输入/L1原始/R1原始，下排LQ/L1后处理/R1后处理。'
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
    (OUT/'R1_WIDTH_AUDIT.json').write_text(json.dumps(dict(passed=True, new_options=2, total_options=30,
        checkpoint_iteration=370000, checked_new_paths=64, old_images_rechecked=False,
        source_audit_sha256=digest(SOURCE/'AUDIT.json'), records=checked), indent=2))
    print(json.dumps(dict(passed=True, rows=32, total_options=30, checked_new_paths=64)))


if __name__ == '__main__':
    main()
