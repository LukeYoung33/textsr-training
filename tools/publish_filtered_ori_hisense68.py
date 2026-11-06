"""Publish new R1_ori outputs alongside accepted historical Hisense68 results."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def publish():
    """@luna-comment Input:complete136 new results and old accepted68 inventory.
    Output:four-panel native comparison. Key Steps:exact-ID join, preserve old paths, raw/post options.
    Key Params:no historical raster revalidation. State/Dependencies:NAS and accepted renderer.
    """
    from tools.build_camera_2x2_viewer import render_html, file_url
    base = Path('Y:/guang.yang/TextSR/data')
    new = base / 'hisense68_filtered_ori_430k_20261106'
    old = base / 'hisense_l772_two_models_20261105'
    assert json.loads((new / 'COMPLETE.json').read_text())['passed']
    samples = json.loads((new / 'selection.json').read_text())['samples']
    previous = {r['id']: r for r in json.loads((old / 'RESULTS.json').read_text())['samples']}
    models = [dict(key='lq', label='LQ · 原始输入', note='原始相机图像'),
              dict(key='v3_input', label='V3 预处理输入', note='复用昨天的缓存，无新预处理/OCR')]
    for key, label in [('new_ori', '新版筛选GT · V3M L1-R1_ori · 430K'),
                       ('r1_width', 'V3M · 原始GT+2px · L1+粗细R1 · 500K'),
                       ('mf_uv_mix50', 'V3M · L1+Missing/False+UV+50%GT退化 · 500K')]:
        for suffix, stage in [('_raw', '原始输出'), ('', '固化后处理')]:
            models.append(dict(key=key+suffix, label=label+' | '+stage,
                 note='256/224/batch8/crop8/smooth16；'+stage))
    rows = []
    for index, entry in enumerate(samples, 1):
        prior = previous[entry['id']]
        assert prior['source'] == entry['source']
        images = {'v3_input': file_url(entry['input'])}
        for key in ['r1_width', 'mf_uv_mix50']:
            images[key+'_raw'] = file_url(prior['stages'][key+'_model']['path'])
            images[key] = file_url(prior['stages'][key+'_post']['path'])
        for folder, suffix in [('model', '_raw'), ('post', '')]:
            remote = '/AI_DNS/guang.yang/TextSR/data/'+new.name+'/'+folder+'/filtered_ori_430k/'+entry['id']+'.png'
            images['new_ori'+suffix] = file_url(remote)
        rows.append(dict(id=f'{index:03d}', name=entry['stem'], width=entry['width'], height=entry['height'],
                         lq=file_url(entry['source']), images=images))
    assert len(rows) == 68
    data = dict(title='Hisense L772 · 68张 · 三模型对比', subtitle='新版筛选GT R1_ori 430K + 昨天两组500K',
       defaults='new_ori,r1_width,mf_uv_mix50,v3_input', models=models, rows=rows,
       notes='默认展示固化后处理，可在各窗口切换模型原始输出。输入使用同一份V3缓存；无新OCR或预处理。新版430K，历史两组500K，训练数据和loss不同，本对比不作单因素归因。原始输入可选；效果请人工评审。')
    out = ROOT / 'evaluation/hisense68_filtered_ori_430k_20261106'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'index.html').write_text(render_html(data), encoding='utf-8')
    print(out / 'index.html')


if __name__ == '__main__':
    publish()
