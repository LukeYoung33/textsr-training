"""Exact fixed-support width-balanced output-gradient counterfactual."""
from pathlib import Path
import json,copy,re,shutil
import numpy as np
import cv2
from tools.build_r1_028_report import roi_stats
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'evaluation/r1_gradient_028_width2_250k_20261103'
OUT=ROOT/'evaluation/r1_gradient_028_width2_normalized_20261103'


def bins(width):
    """@luna-comment Input:fixed original-mask width. Output:three integer groups. Key Steps:threshold at2/4px. Key Params:original diagnostic tolerance. State/Dependencies:pure NumPy."""
    return np.where(width<=2.00001,0,np.where(width<=4.00001,1,2))


def main():
    """@luna-comment Input:frozen width2 patch autograd maps. Output:exact group-normalized output gradients and Chinese viewer. Key Steps:reconstruct batch support counts including padding;rescale R1;replace total;validate finite-difference reference. Key Params:equal nonempty group means,no cap,missing1/false0.25. State/Dependencies:owned output only;no model updates or parameter-gradient claims."""
    OUT.mkdir(exist_ok=True);r=json.loads((OLD/'REPORT.json').read_text());limit=r['shared_gradient_limit']
    patches={p['id']:{k:v.copy() for k,v in np.load(OLD/'patches'/p['id']/'maps.npz').items()} for p in r['patches']}
    factors=[]
    for batch in r['batches']:
        cohort=[patches[f'{i+1:03d}'] for i in batch['positions']];weights={}
        for term,support,denom in [('missing','support_m','text_pixels'),('false','support_f','ring_pixels')]:
            counts=np.array([sum(int(((bins(z['width'])==b)&(z[support]>0)).sum()) for z in cohort) for b in range(3)])
            assert abs(counts.sum()-batch[denom])<.01
            k=int((counts>0).sum());w=np.divide(counts.sum(),max(k,1)*counts,out=np.zeros(3,float),where=counts>0)
            weights[term]=w;factors.append(dict(offset=batch['offset'],term=term,counts=counts.tolist(),multipliers=w.tolist()))
        for pos in batch['positions'][:min(4,25-batch['offset'])]:
            z=patches[f'{pos+1:03d}'];label=bins(z['width']);old_r1=z['r1'].copy()
            for term in ['missing','false']:z[term]=(z[term]*weights[term][label]).astype(np.float32)
            z['r1']=z['missing']+z['false'];z['total']=z['total']-old_r1+z['r1']
    # Independent finite-difference check of equal-group means and gradient rescaling.
    e=np.array([.011,.04,.07,-.02,.08,.015],dtype=np.float64);g=np.array([0,1,1,2,2,2]);count=np.bincount(g);expected=1/(1+np.exp(-(e-.03)/.01))/(3*count[g]);numeric=[]
    for i in range(len(e)):
        a=e.copy();b=e.copy();a[i]+=1e-7;b[i]-=1e-7
        la=sum(np.logaddexp(0,(a[g==j]-.03)/.01).mean()*.01 for j in range(3))/3
        lb=sum(np.logaddexp(0,(b[g==j]-.03)/.01).mean()*.01 for j in range(3))/3
        numeric.append((la-lb)/2e-7)
    error=float(np.max(np.abs(expected-numeric)));assert error<1e-8
    full={k:v.copy() for k,v in np.load(OLD/'full_maps.npz').items()}
    for p in r['patches']:
        z=patches[p['id']];x,y=p['xy'];h,w=p['hw'];folder=OUT/'patches'/p['id'];folder.mkdir(parents=True,exist_ok=True)
        assert all(np.isfinite(v).all() for v in z.values());np.savez_compressed(folder/'maps.npz',**z)
        for key in ['missing','false','r1','total']:full[key][y:y+h,x:x+w]=z[key]
    np.savez_compressed(OUT/'full_maps.npz',**full)
    # Reuse the exact display implementation without importing its remote-only utilities.
    source=(ROOT/'tools/diagnose_r1_028.py').read_text(encoding='utf-8')
    namespace={'np':np,'cv2':cv2};exec(source[source.index('def overlay('):source.index('\n\ndef main():')],namespace);overlay=namespace['overlay']
    for view in r['views']:
        folder=OUT/('full' if view['id']=='000' else 'patches/'+view['id']);folder.mkdir(exist_ok=True)
        z=full if view['id']=='000' else patches[view['id']];base=cv2.imread(str(OLD/view['images']['output']))
        for key,path in view['images'].items():shutil.copy2(OLD/path,folder/(key+'.png'))
        for term in ['missing','false','r1','total']:assert cv2.imwrite(str(folder/(term+'_gradient.png')),overlay(base,z[term],limit))
        assert cv2.imwrite(str(folder/'r1_direction.png'),overlay(base,-z['r1'],limit,signed=True))
        ratio=np.abs(z['r1'])/(np.abs(z['r1'])+np.abs(z['total']-z['r1'])+1e-20)
        assert cv2.imwrite(str(folder/'r1_relative.png'),overlay(base,ratio,1.))
    sl=(slice(850,1180),slice(160,800));(OUT/'roi').mkdir(exist_ok=True)
    for path in (OUT/'full').glob('*.png'):assert cv2.imwrite(str(OUT/'roi'/path.name),cv2.imread(str(path))[sl])
    current=roi_stats(full,sl);prior=json.loads((OLD/'ROI_STATS.json').read_text());data=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    data['title']='028 · R1粗细归一化 · +2px GT';data['subtitle']='同一GT / 输入 / 模型；比较原R1与粗细等权R1'
    data['notes']='在同一+2px GT上，只改变R1的归一化。\nmissing与false分别：按原mask宽度≤2 / 2–4 / >4px分组，各非空组求均值，再等权平均。每batch统计，含补齐样本；展示统计不含重复样本。\n原missing=1、false=0.25保持不变；没有截断权重，也没有把总梯度量强制配平。\n这是冻结模型输出的精确梯度重加权，不是训练效果；不提供新的参数梯度结论。\n原mask保持不变，宽度代表原mask。共同绝对色标沿用上次；超限饱和。方向图红暗青亮。'
    for m in data['models']:
        old=m['key'].startswith('old_');key=m['key'][4:] if old else m['key']
        m['label']=('原R1 · ' if old else '归一化 · ')+next(x['label'].split(' · ',1)[-1] for x in data['models'] if x['key']==key)
        m['note']=('原R1，+2px GT' if old else '粗细分组等权R1，+2px GT')+'；'+m['note']
        if key=='gt':m['note']='两组使用完全相同的原始GT笔画+2px，再逐patch对比度对齐'
    for row in data['rows']:
        root='roi' if row['id']=='ROI' else 'full' if row['id']=='000' else 'patches/'+row['id']
        row['images']={m['key']:('../'+OLD.name+'/' if m['key'].startswith('old_') else '')+root+'/'+(m['key'][4:] if m['key'].startswith('old_') else m['key'])+'.png' for m in data['models']}
        row['lq']=root+'/lq.png'
        for path in row['images'].values():
            im=cv2.imread(str(OUT/path));assert im is not None and im.shape[:2]==(row['height'],row['width'])
    data['defaults']='old_missing_gradient,old_false_gradient,old_total_gradient,missing_gradient,false_gradient,total_gradient'
    text=(OLD/'viewer.html').read_text(encoding='utf-8');text=re.sub(r'const DATA=.*?;</script>',lambda _: 'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',text,count=1,flags=re.S)
    text=re.sub(r'<title>.*?</title>','<title>'+data['title']+'</title>',text,count=1)
    text=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+data['title']+'</h1><small>'+data['subtitle']+'</small>',text,count=1,flags=re.S)
    text=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+data['notes'].replace('\n','<br>')+'</div>',text,count=1,flags=re.S)
    (OUT/'viewer.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    rows=[];comparison={}
    for term in ['missing','false']:
        comparison[term]={}
        for b,label in [('0','细≤2px'),('1','中2–4px'),('2','粗>4px')]:
            a=prior[term][b];c=current[term][b];ratio=c['gradient_per_pixel']/a['gradient_per_pixel'];comparison[term][b]=dict(before=a,after=c,ratio=ratio)
            rows.append(f'<tr><td>{term} / {label}</td><td>{ratio:.2f}倍</td><td>{a["gradient_share"]:.1%} → {c["gradient_share"]:.1%}</td><td>{a["cancelled_gradient_fraction"]:.1%} → {c["cancelled_gradient_fraction"]:.1%}</td></tr>')
    body='''<!doctype html><meta charset="utf-8"><title>R1粗细归一化诊断</title><style>body{font:17px/1.8 system-ui;max-width:1150px;margin:35px auto;background:#eef3f8;color:#183048}section{padding:24px;background:white;border-radius:12px;margin:20px 0}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd}</style><h1>028：R1粗细分组归一化</h1><a href="viewer.html#ROI">打开梯度对照：上排原R1，下排归一化</a><section><h2>定义</h2><p>原R1已经按整个有效区域的像素数归一化。本次改为：每个batch内，将有效区域按原mask宽度分成≤2、2–4、>4px，分别求loss均值，再对非空组等权平均。missing和false独立计算，原有1和0.25权重保留。没有权重上限。</p><p>若原有效像素数为N，某组为Nₖ，非空组数为K，则该组每像素梯度乘以 N/(K·Nₖ)。mask和分组与预测无关，因此可从上轮autograd梯度精确重建，不需要重新跑网络。</p><p>同一个“原始GT笔画+2px”目标、同一个250K输出。宽度分组仍来自原mask；没有扩张mask。等权的是各组平均loss，不保证实际梯度份额各占1/3。</p></section><section><h2>截图附近：归一化前后</h2><table><tr><th>分项 / 原笔画宽度</th><th>每像素梯度变化</th><th>梯度份额</th><th>被其他项抵消量占比</th></tr>'''+''.join(rows)+'''</table></section><section><h2>解释与边界</h2><p>归一化只重新分配同组梯度大小，不改变missing/false各自的推动方向、margin和激活位置；它可以提高细笔画相对权重，但不能修正错位GT、错误mask或GT本身问题。</p><p>总loss梯度只替换R1，其余项保持不变。总梯度大小不强制配平，因而变化包含重分配与整体强度变化。稀少分组可能产生较大权重，正式训练需检查更多图的稳定性。这里不代表训练后的图像效果，也没有重新计算参数梯度或Adam更新。</p></section>'''
    worst=max(factors,key=lambda x:max(x['multipliers']))
    body+=f'<section><h2>本次已观察到的稀少分组放大</h2><p>第{worst["offset"]//4+1}个batch的{worst["term"]}三个组分别有{worst["counts"]}个有效像素，最大单像素倍率达到<b>{max(worst["multipliers"]):.1f}倍</b>。这不是实际总梯度增幅，但说明无上限等权会过度放大小组；训练版应考虑分母下限、倍率上限或较温和的面积权重。</p></section>'
    (OUT/'index.html').write_text(body,encoding='utf-8');(OUT/'COMPARISON.json').write_text(json.dumps(comparison,indent=2));(OUT/'ROI_STATS.json').write_text(json.dumps(current,indent=2))
    receipt=dict(passed=True,finite_difference_error=error,views=27,layers=32,images=864,cohort_factors=factors,method='exact fixed-support rescaling of autograd output gradients; no new parameter gradients')
    (OUT/'CHECKS.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(comparison))


if __name__=='__main__':main()
