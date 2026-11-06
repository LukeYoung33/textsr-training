"""Consolidate existing Mojito model variants and original-GT width2 on fixed32."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html, file_url, digest

OUT = ROOT/'evaluation/mojito32_all_models_20261104'
OLD = ROOT/'evaluation/latest_six_500k_mojito280_20261103'
SUBSET = ROOT/'evaluation/mojito280_subset32_20261104'
NEW = Path('Y:/guang.yang/TextSR/data/v3m_original_dilate2_l1_mojito280_20261104')
KEY = 'v3m_original_gt_dilate2_l1'


def check_image(job):
    """@luna-comment Input:model file URL andexpectednative dimensions. Output:verified path/size receipt. Key Steps:decode mappedY URL,open image header,checkgeometry. Key Params:no resizing or imagequality scoring. State/Dependencies:read-only audited images;PIL."""
    from PIL import Image
    identity, key, url, width, height = job
    assert url.startswith('file:///'), url
    path = Path(unquote(url[len('file:///'):]))
    assert path.is_file(), str(path)
    with Image.open(path) as image:
        stored_size = image.size
        orientation = image.getexif().get(274, 1)
        display_size = stored_size[::-1] if orientation in (5, 6, 7, 8) else stored_size
        assert display_size == (width, height), (identity, key, stored_size, orientation)
    return dict(id=identity, key=key, path=str(path), bytes=path.stat().st_size)


def main():
    """@luna-comment Input:confirmed16oldoptions,fixed32IDs,auditednewraw/post. Output:18option sixpanel local viewer andalignment audit. Key Steps:check completion/inputidentity,merge sameIDpaths,verifyall576imageheaders,render. Key Params:old280 page immutable;newmodelMojitoexcluded. State/Dependencies:NAS sourceimages andaccepted renderer;no platform writes."""
    old = json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    subset = json.loads((SUBSET/'selection.json').read_text(encoding='utf-8'))
    assert json.loads((NEW/'COMPLETE.json').read_text())['passed']
    assert json.loads((NEW/'AUDIT.json').read_text())['passed']
    original = {s['id']: s for s in json.loads((OLD/'selection.json').read_text())['samples']}
    current = {s['id']: s for s in json.loads((NEW/'selection.json').read_text())['samples']}
    outputs = {s['id']: s for s in json.loads((NEW/'OUTPUTS.json').read_text())['samples']}
    rows = {r['id']: r for r in old['rows']}
    ids = [s['id'] for s in subset['samples']]
    assert len(ids) == len(set(ids)) == 32
    models = list(old['models'])
    assert len(models) == 16
    for suffix, label in [('_raw', '模型原始输出'), ('', '固化后处理')]:
        models.append(dict(key=KEY+suffix, label='V3M · 原始GT笔画+2px · L1-only | 500K | '+label,
                           note='训练集不含全部Mojito；256/224/batch8/crop8/smooth16'+('；未加后处理' if suffix else '；Y后处理 .75 / 2 / 24')))
    selected = []
    jobs = []
    for identity in ids:
        for field in ['input_sha256', 'source_sha256', 'stored_shape', 'orientation', 'width', 'height']:
            assert original[identity][field] == current[identity][field], (identity, field)
        row = dict(rows[identity])
        row['images'] = dict(row['images'])
        for suffix, field in [('_raw', 'raw'), ('', 'post')]:
            row['images'][KEY+suffix] = file_url(outputs[identity][field])
        selected.append(row)
        for model in models:
            key = model['key']
            url = row['lq'] if key == 'lq' else row['images'][key]
            jobs.append((identity, key, url, row['width'], row['height']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(check_image, jobs))
    manifest = dict(title='Mojito32 · 历史模型与原始GT笔画+2px 500K',
                    subtitle='固定32张 · 18个选项 · 同步缩放与平移', models=models, rows=selected,
                    defaults='v3m_guard_raw,v3m_mix50_raw,previous250_nocp_raw,v3m_l1_raw,v5_l1_raw,'+KEY+'_raw',
                    notes='本页只收录指定32张，保留原始001–272编号。\n新增：V3M 原始对齐GT笔画+2px，L1-only，从头500K；原始输出和固化后处理均可选择。\n新增模型训练集不含Mojito280；旧六模型中本子集有24张训练图、8张验证图，图名旁沿用旧划分标记。不同训练数据/目标风格的比较不作为单因素消融。\n默认六窗口均为模型原始输出；下拉菜单保留原始LQ、V3输入及所有后处理版本。\n输入缓存及切块羽化一致，无重复OCR或预处理。效果由用户看图评审。')
    text = render_html(manifest)
    assert text.count('for(let i=0;i<4;i++)') == 1
    assert text.count('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;') == 1
    text = text.replace('for(let i=0;i<4;i++)', 'for(let i=0;i<6;i++)').replace(
        'grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;',
        'grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'index.html').write_text(text, encoding='utf-8')
    audit = dict(passed=True, selected=32, options=18, verified_image_paths=len(verified), ids=ids,
                 parent_manifest_sha256=digest(OLD/'manifest.json'), subset_selection_sha256=digest(SUBSET/'selection.json'),
                 new_model_audit_sha256=digest(NEW/'AUDIT.json'), records=verified)
    (OUT/'AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=True, images=32, options=18, checked_paths=len(verified), page=str(OUT/'index.html'))))


if __name__ == '__main__':
    main()
