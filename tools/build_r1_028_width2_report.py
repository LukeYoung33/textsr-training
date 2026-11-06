"""Build a fixed-mask, target-only gradient comparison against the prior audit."""
from pathlib import Path
import json,re,copy,tarfile
import numpy as np
import cv2
from tools.build_r1_028_report import roi_stats,table
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'evaluation/r1_gradient_028_250k_20261103'
OUT=ROOT/'evaluation/r1_gradient_028_width2_250k_20261103'


def main():
    """@luna-comment Input:verified frozen width2 maps and previous audit. Output:Chinese comparison and linked viewer. Key Steps:validate identical output/input, preserve common scale, compare ROI gradients. Key Params:fixed original masks;ROI160/850/640/330. State/Dependencies:owned local report artifacts only."""
    with tarfile.open(OUT/'delivery.tar') as archive:
        for item in archive.getmembers():
            assert (OUT/item.name).resolve().is_relative_to(OUT.resolve())
        archive.extractall(OUT,filter='data')
    r=json.loads((OUT/'REPORT.json').read_text());old=json.loads((OLD/'REPORT.json').read_text())
    assert r['checkpoint_sha256']==old['checkpoint_sha256'] and r['shared_gradient_limit']==old['shared_gradient_limit']
    for v in r['views']:
        for key in ['lq','output']:
            assert np.array_equal(cv2.imread(str(OUT/v['images'][key])),cv2.imread(str(OLD/v['images'][key])))
    for path in [OUT/'full_maps.npz',*OUT.glob('patches/*/maps.npz')]:
        with np.load(path) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
    sl=(slice(850,1180),slice(160,800));(OUT/'roi').mkdir(exist_ok=True)
    full=r['views'][0];roi={}
    for key,path in full['images'].items():
        roi[key]='roi/'+key+'.png';assert cv2.imwrite(str(OUT/roi[key]),cv2.imread(str(OUT/path))[sl])
    maps=np.load(OUT/'full_maps.npz');local=roi_stats(maps,sl)
    prior=json.loads((OLD/'ROI_STATS.json').read_text());(OUT/'ROI_STATS.json').write_text(json.dumps(local,indent=2))
    data=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    data['title']='028 · 原始GT笔画+2px · R1梯度对照'
    data['subtitle']='固定模型、输入、mask和loss权重；仅替换GT，再沿用对比度对齐'
    data['notes']='250K冻结模型；50%轻退化混合训练组，诊断真实LQ分支。\n旧目标：S2.2；新目标：原始对齐GT笔画+2px（单侧半径1px），均逐patch对比度对齐。\n保持原TextMask、valid mask及宽度分组不变，属于GT单变量实验，不是mask同步后的训练模拟。加粗笔画可能伸入原false环带。\n所有梯度共用上次绝对色标；超过上限仅显示饱和，原值保留在NPZ。\n方向图橙红推动变暗、青色推动变亮。\n上排默认旧GT / 新GT / 同一模型输出；下排旧missing / 新missing / 新false。'
    original_models=copy.deepcopy(data['models']);data['models']=[]
    for prefix,label in [('old_','旧GT · '),('','+2px GT · ')]:
        for model in original_models:
            model=copy.deepcopy(model);model['key']=prefix+model['key'];model['label']=label+model['label']
            if model['key'].endswith('gt') and not model['key'].endswith('original_gt'):model['label']=label+'对比度对齐后目标';model['note']='原始GT笔画+2px再逐patch对比度对齐' if not prefix else '上轮S2.2目标逐patch对比度对齐'
            model['note']=model['note'].replace('顶部0.5%幅值饱和显示','超过上次色标上限的幅值饱和显示')
            data['models'].append(model)
    data['defaults']='old_gt,gt,output,old_missing_gradient,missing_gradient,false_gradient'
    newviews=[dict(id='ROI',width=640,height=330,images=roi)]+r['views']
    images=0
    for row,v in zip(data['rows'],newviews):
        assert row['id']==v['id']
        original_images=dict(row['images'],lq=row['lq'])
        row['images']={'old_'+k:'../'+OLD.name+'/'+p for k,p in original_images.items()}
        row['images'].update(v['images']);row['lq']=v['images']['lq']
        for path in row['images'].values():
            value=cv2.imread(str(OUT/path));assert value is not None and value.shape[:2]==(row['height'],row['width']);images+=1
    text=(OLD/'viewer.html').read_text(encoding='utf-8')
    text=re.sub(r'const DATA=.*?;</script>',lambda _: 'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',text,count=1,flags=re.S)
    text=re.sub(r'<title>.*?</title>','<title>'+data['title']+'</title>',text,count=1)
    text=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+data['title']+'</h1><small>'+data['subtitle']+'</small>',text,count=1,flags=re.S)
    text=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+data['notes'].replace('\n','<br>')+'</div>',text,count=1,flags=re.S)
    text=text.replace('顶部0.5%幅值饱和显示','超过上次色标上限的幅值饱和显示')
    (OUT/'viewer.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    rows=[];comparison={}
    for kind in ['missing','false']:
        comparison[kind]={}
        for b,label in [('0','细≤2px'),('1','中2–4px'),('2','粗>4px')]:
            a=prior[kind][b];c=local[kind][b];ratio=c['gradient_per_pixel']/a['gradient_per_pixel'];comparison[kind][b]=dict(old=a,new=c,ratio=ratio)
            rows.append(f'<tr><td>{kind} · {label}</td><td>{a["gradient_per_pixel"]:.3e}</td><td>{c["gradient_per_pixel"]:.3e}</td><td>{ratio:.2f}×</td><td>{a["gradient_share"]:.1%} → {c["gradient_share"]:.1%}</td><td>{a["cancelled_gradient_fraction"]:.1%} → {c["cancelled_gradient_fraction"]:.1%}</td></tr>')
    body='''<!doctype html><meta charset="utf-8"><title>原始GT笔画+2px · R1梯度诊断</title><style>body{font:16px/1.8 system-ui;background:#edf2f7;color:#183048;margin:32px auto;max-width:1200px}section{background:white;padding:22px;margin:18px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd}a{color:#1263b0}</style><h1>028 · 原始GT笔画+2px的R1梯度</h1><p><a href="viewer.html#ROI">打开联动梯度对照</a> · <a href="viewer.html#000">整图</a> · <a href="REPORT.json">原始统计</a></p><section><h2>本次只改变监督GT</h2><p>同一V3M Full-no_guard混合组250K、同一真实V3 LQ、同一25个patch和batch分组。使用此前已生成并校验哈希的“原始GT笔画+2px”，再沿用原逐patch对比度对齐；不做VOSR或V3处理。模型输出与上次逐像素完全相同。</p><p><b>原TextMask保持不变。</b>因此粗细分组仍是原始mask的宽度，不是加粗后宽度。加粗新增区域可能落入原false环带；false仍根据新GT误差判断，不会只因它在环带就一律惩罚。</p><p>两轮热图使用完全相同的绝对色标。梯度衡量加权loss对输出亮度的敏感度，并非训练后的效果或实际Adam更新量。</p></section><section><h2>截图附近ROI：梯度变化</h2><table><tr><th>项 / 原mask宽度</th><th>旧平均梯度</th><th>新平均梯度</th><th>新/旧</th><th>梯度份额</th><th>被其余项抵消量占比</th></tr>'''+''.join(rows)+'</table></section>'
    body+='<section><h2>新目标：整图missing</h2>'+table(r['statistics']['missing'])+'<h2>新目标：整图false</h2>'+table(r['statistics']['false'])+'</section>'
    body+=f'<section><h2>验证</h2><p>26组LQ/模型输出逐像素一致，26个NPZ所有数值有限；解析梯度校验最大误差{r["gradient_analytic_max_error"]:.3e}。模型状态未变化，未执行训练更新。25个唯一patch统计排除最后batch的3个补齐样本。</p><p>这是固定mask的目标替换诊断，不能单独区分“移除教师处理”和“增加2px笔画”各自的贡献。</p></section>'
    (OUT/'index.html').write_text(body,encoding='utf-8');(OUT/'COMPARISON.json').write_text(json.dumps(comparison,indent=2))
    (OUT/'DELIVERY_CHECKS.json').write_text(json.dumps(dict(passed=True,views=27,layers=32,images=images,outputs_equal=True,finite_maps=True),indent=2))
    print(json.dumps({k:{b:v['ratio'] for b,v in x.items()} for k,x in comparison.items()}))


if __name__=='__main__':main()
