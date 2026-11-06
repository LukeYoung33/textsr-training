"""Offline adaptive R1 weight stress test on real width2 gradients and synthetic cohorts."""
from pathlib import Path
import json,re,html
import numpy as np
import cv2
import torch
from tools.normalize_r1_028_diagnostic import bins
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evaluation/r1_gradient_028_width2_250k_20261103'
OUT=ROOT/'evaluation/r1_adaptive_extremes_20261103'


def adapt(counts,ema=None):
    """@luna-comment Input:three nonnegative support counts and prior EMA orNone. Output:bounded weights/newEMA. Key Steps:first-valid init or0.99EMA;inverse sqrt;active-set joint mean-one/cap solve. Key Params:epsilon1e-4,cap4,zero-support skip. State/Dependencies:pure float64 arrays;caller owns separate missing/false states."""
    counts=np.asarray(counts,np.float64);assert counts.shape==(3,) and np.isfinite(counts).all() and (counts>=0).all()
    if counts.sum()==0:return np.zeros(3),None if ema is None else ema.copy()
    p=counts/counts.sum();state=p.copy() if ema is None else .99*ema+.01*p
    score=(state+1e-4)**-.5;free=counts>0;weights=np.zeros(3)
    for _ in range(3):
        scale=(1-float(p@weights))/float(p[free]@score[free]);candidate=score*scale;over=free&(candidate>4.)
        if not over.any():weights[free]=candidate[free];break
        weights[over]=4.;free[over]=False
    assert np.isfinite(weights).all() and (weights>=0).all() and weights.max()<=4.+1e-12
    assert abs(p@weights-1)<1e-12 and (weights[counts==0]==0).all()
    return weights,state


def naive(counts):
    """@luna-comment Input:three support counts. Output:uncapped equal-group baseline weights. Key Steps:divide total bynonemptygroups andgroupmass. Key Params:absentgroupszero. State/Dependencies:pure."""
    c=np.asarray(counts,float);return np.divide(c.sum(),max(int((c>0).sum()),1)*c,out=np.zeros(3),where=c>0)


def autograd_check(counts,weights,dtype):
    """@luna-comment Input:synthetic groupcounts/fixedweights andtorchdtype. Output:derivative/error/finite evidence. Key Steps:softplus loss withFP32 accumulation forlowprecision;compare closed-form weighted derivative. Key Params:0.03margin,0.01temperature,false0.25 separately. State/Dependencies:CPU test tensors only."""
    labels=np.repeat(np.arange(3),counts);n=len(labels)
    value=torch.linspace(-.1,.15,max(n,1),dtype=dtype,requires_grad=True)
    w=torch.tensor(weights[labels] if n else [0.],dtype=torch.float64 if dtype==torch.float64 else torch.float32)
    x=value if dtype==torch.float64 else value.float();temperature=.01
    errors=[]
    for sign,weight in [(1.,1.),(-1.,.25)]:
        loss=(temperature*torch.nn.functional.softplus((sign*x-.03)/temperature)*w).sum()/max(n,1)*weight
        grad=torch.autograd.grad(loss,value,retain_graph=True)[0]
        expected=(torch.sigmoid((sign*x.detach()-.03)/temperature)*w/max(n,1)*weight*sign).to(dtype)
        error=float((grad.double()-expected.double()).abs().max());errors.append(error)
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
        assert torch.allclose(grad,expected,rtol=.003 if dtype==torch.float16 else 2e-6,atol=6e-8 if dtype==torch.float16 else 1e-10)
        if not n:assert float(loss.detach())==0 and float(grad.abs().max())==0
    return dict(dtype=str(dtype),max_derivative_error=max(errors),pixels=n,passed=True)


def measure(maps,term,weights):
    """@luna-comment Input:realcohort rawmaps,term,andfixedpergroupweights. Output:gradientmass shares andR1-conflict statistics. Key Steps:reweight actualautograd values;retainotherlosses;aggregate unique cohort. Key Params:outputY sensitivity only. State/Dependencies:immutable inputs."""
    mass=np.zeros(3);original=np.zeros(3);cancel=np.zeros(3)
    for z in maps:
        label=bins(z['width']);g=z[term]*weights[term][label]
        r1=z['missing']*weights['missing'][label]+z['false']*weights['false'][label]
        total=z['total']-z['r1']+r1;rest=total-g;support=z['support_m' if term=='missing' else 'support_f']>0
        for b in range(3):
            sel=(label==b)&support;mass[b]+=np.abs(g[sel]).sum();original[b]+=np.abs(z[term][sel]).sum()
            cancel[b]+=np.minimum(np.abs(g),np.abs(rest))[sel&(g*rest<0)].sum()
    return dict(mass=mass.tolist(),shares=(mass/max(mass.sum(),1e-30)).tolist(),per_group_ratio=np.divide(mass,original,out=np.zeros(3),where=original>0).tolist(),total_mass_ratio=float(mass.sum()/max(original.sum(),1e-30)),cancelled_fraction=np.divide(cancel,mass,out=np.zeros(3),where=mass>0).tolist())


def main():
    """@luna-comment Input:verified width2 diagnostic. Output:realextreme viewer,synthetic/EMA checks andChinese report. Key Steps:cohort replay,cold/history shift,10Krandom counts,autograd dtypes,sharedscale PNGs. Key Params:no modeltraining,original masks retained. State/Dependencies:ownedlocal artifacts only."""
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(2)
    r=json.loads((SOURCE/'REPORT.json').read_text());states={'missing':None,'false':None};replay=[]
    for batch in r['batches']:
        maps=[dict(np.load(SOURCE/'patches'/f'{i+1:03d}'/'maps.npz')) for i in batch['positions']];record=dict(offset=batch['offset'],positions=batch['positions'],terms={})
        for term,support,denom in [('missing','support_m','text_pixels'),('false','support_f','ring_pixels')]:
            counts=np.array([sum(int(((bins(z['width'])==b)&(z[support]>0)).sum()) for z in maps) for b in range(3)])
            assert counts.sum()==batch[denom];before=None if states[term] is None else states[term].copy();w,states[term]=adapt(counts,states[term]);cold,_=adapt(counts)
            record['terms'][term]=dict(counts=counts.tolist(),ema_before=None if before is None else before.tolist(),ema_after=states[term].tolist(),adaptive=w.tolist(),cold=cold.tolist(),naive=naive(counts).tolist())
        replay.append(record)
    extreme=max(replay,key=lambda b:max(b['terms']['missing']['naive']));assert extreme['offset']==8
    maps=[dict(np.load(SOURCE/'patches'/f'{i+1:03d}'/'maps.npz')) for i in extreme['positions']]
    variants={'raw':{t:np.ones(3) for t in states}}
    for name,key in [('equal','naive'),('adaptive','adaptive'),('batch','cold')]:variants[name]={t:np.array(extreme['terms'][t][key]) for t in states}
    measurements={name:{t:measure(maps,t,w) for t in states} for name,w in variants.items()}
    cases=[];checks=[]
    for name,counts in [('真实29像素冷启动',[29,231,37254]),('仅1个细笔画像素',[1,0,100000]),('仅1个中等笔画像素',[0,1,100000]),('只有粗笔画',[0,0,65536]),('全背景/无有效支持',[0,0,0]),('三组均衡',[1000,1000,1000])]:
        w,state=adapt(counts);cases.append(dict(name=name,counts=counts,weights=w.tolist(),naive=naive(counts).tolist()))
        for dtype in [torch.float64,torch.float32,torch.float16]:checks.append(autograd_check(np.array(counts),w,dtype))
    rng=np.random.default_rng(20261103);state=None;max_error=0.;max_weight=0.;batch_error=0.;batch_max=0.
    for i in range(10000):
        counts=rng.integers(0,1000000,3);counts[rng.random(3)<.3]=0
        previous=None if state is None else state.copy();w,state=adapt(counts,state)
        independent,_=adapt(counts)
        if counts.sum():batch_error=max(batch_error,abs(float(counts@independent/counts.sum())-1));batch_max=max(batch_max,float(independent.max()))
        if counts.sum():max_error=max(max_error,abs(float(counts@w/counts.sum())-1));max_weight=max(max_weight,float(w.max()))
        else:assert (previous is None and state is None) or np.array_equal(previous,state)
    sequences={}
    for name,first,later in [('粗为主→细为主',[1,1,100000],[100000,1,1]),('细为主→粗为主',[100000,1,1],[1,1,100000]),('均衡→仅1个细像素',[10000,10000,10000],[1,0,100000])]:
        _,state=adapt(first);trajectory=[]
        for step in range(1,301):
            w,state=adapt(later,state);trajectory.append(dict(step=step,weights=w.tolist(),ema=state.tolist()))
            restored=np.array(json.loads(json.dumps(state.tolist())));a,sa=adapt(later,state);b,sb=adapt(later,restored);assert np.array_equal(a,b) and np.array_equal(sa,sb)
        sequences[name]=trajectory
    # A constructed native256 mask: one isolated thin pixel and one thick block.
    label=np.full((256,256),-1,int);label[128,30]=0;label[16:240,80:240]=2;counts=np.array([(label==b).sum() for b in range(3)]);w,_=adapt(counts)
    synth=np.full((256,256,3),245,np.uint8);synth[label>=0]=35;cv2.imwrite(str(OUT/'synthetic_gt.png'),synth)
    for name,weight in [('equal',naive(counts)),('adaptive',w)]:
        amount=np.zeros((256,256));sel=label>=0;amount[sel]=weight[label[sel]];color=cv2.applyColorMap(np.uint8(np.clip(amount/4,0,1)*255),cv2.COLORMAP_TURBO);color[~sel]=245;cv2.imwrite(str(OUT/('synthetic_'+name+'.png')),color)
    source=(ROOT/'tools/diagnose_r1_028.py').read_text(encoding='utf-8');ns={'np':np,'cv2':cv2};exec(source[source.index('def overlay('):source.index('\n\ndef main():')],ns);overlay=ns['overlay']
    models=[dict(key=k,label=label,note='共同色标；输出亮度梯度，不是训练后的图像') for k,label in [('lq','V3 LQ'),('gt','原始GT笔画+2px · 对比度对齐后'),('output','同一250K输出')]]
    for name,title in [('raw','原R1'),('equal','直接等权'),('adaptive','EMA+平方根+cap4'),('batch','当前batch+平方根+cap4')]:
        for term,label in [('missing','missing'),('false','false'),('total','总loss')]:models.append(dict(key=name+'_'+term,label=title+' · '+label,note='统一绝对色标，超限显示饱和；GT、原mask与模型均相同'))
    rows=[]
    for pos,z in zip(extreme['positions'],maps):
        key=f'{pos+1:03d}';folder=OUT/key;folder.mkdir(exist_ok=True);base=cv2.imread(str(SOURCE/'patches'/key/'output.png'));images={};label=bins(z['width'])
        for item in ['lq','gt','output']:images[item]='../'+SOURCE.name+'/patches/'+key+'/'+item+'.png'
        for name,w in variants.items():
            gm=z['missing']*w['missing'][label];gf=z['false']*w['false'][label];gt=z['total']-z['r1']+gm+gf
            for term,g in [('missing',gm),('false',gf),('total',gt)]:
                path=folder/(name+'_'+term+'.png');assert cv2.imwrite(str(path),overlay(base,g,r['shared_gradient_limit']));images[name+'_'+term]=key+'/'+path.name
        rows.append(dict(id=key,name='真实极端batch · patch '+key,width=base.shape[1],height=base.shape[0],lq=images['lq'],images=images))
    data=dict(title='R1自适应 · 极端情况验证',subtitle='上排missing：原R1 / EMA / 当前batch；下排false',notes='真实GT加粗样本028，250K冻结输出。batch内细/中/粗missing像素=29/231/37254。\nEMA按这张图前两个batch初始化和更新，不代表训练历史。原mask不变。\nEMA0.99或仅当前batch；平方根反面积、epsilon1e-4、平均权重1、严格cap4。\n共同色标沿用前次，较强梯度显示饱和。可切换直接等权、输入、GT及总loss。',models=models,rows=rows,defaults='raw_missing,adaptive_missing,batch_missing,raw_false,adaptive_false,batch_false')
    page=(SOURCE/'viewer.html').read_text(encoding='utf-8');page=re.sub(r'const DATA=.*?;</script>',lambda _: 'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',page,count=1,flags=re.S)
    page=re.sub(r'<title>.*?</title>','<title>'+data['title']+'</title>',page,count=1);page=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+data['title']+'</h1><small>'+data['subtitle']+'</small>',page,count=1,flags=re.S);page=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+data['notes'].replace('\n','<br>')+'</div>',page,count=1,flags=re.S)
    (OUT/'viewer.html').write_text(page,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in rows:
        for path in row['images'].values():im=cv2.imread(str(OUT/path));assert im is not None and im.shape[:2]==(row['height'],row['width'])
    result=dict(passed=True,protocol=dict(beta=.99,exponent=.5,epsilon=1e-4,cap=4,real_scope='frozen output gradients, fixed original masks, actual width2 GT; original grouping retained',ema_history='sequential seven controlled image cohorts, not replay of training',checkpoint_sha256=r['checkpoint_sha256']),real_extreme=extreme,measurements=measurements,cases=cases,checks=checks,stress=dict(count=10000,max_mean_error=max_error,max_weight=max_weight,current_batch_max_error=batch_error,current_batch_max_weight=batch_max),sequences=sequences,replay=replay)
    (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def fmt(values):
        """@luna-comment Input:three scalar values. Output:compact table text. Key Steps:fixed decimals. Key Params:three decimal places. State/Dependencies:pure formatting."""
        return ' / '.join(f'{x:.3f}' for x in values)
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R1自适应权重：极端验证</title><style>body{font:16px/1.8 system-ui;margin:25px auto;max-width:1250px;background:#eef3f8;color:#183048}section{background:white;padding:24px;border-radius:12px;margin:18px 0}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:256px;image-rendering:pixelated}a{color:#1260b0}.images{display:flex;gap:16px}</style><h1>R1自适应权重：真实极端 + 构造压力验证</h1><p><a href="viewer.html#009">打开真实极端batch梯度对照</a> · <a href="RESULTS.json">完整原始记录</a></p><section><h2>验证口径</h2><p>真实例子来自此前原始GT笔画+2px的028样本：第3个受控batch，patch009–012；missing细/中/粗像素为29/231/37,254。沿用固定原mask与250K输出，不重做OCR或推理，不更新模型。此处是梯度/数值验证，不是训练效果验证。</p><p>missing、false独立统计有效支持区；EMA系数0.99，首个有效batch初始化；平方根反面积使用epsilon=1e-4。联合求解缩放与截断，保证当前batch平均像素权重=1且每组权重≤4。全空batch跳过且不改变EMA。真实序列EMA只经过本图前两个batch，不冒充完整训练历史。</p></section><section><h2>真实极端的权重与梯度</h2><table><tr><th>项</th><th>直接等权：细/中/粗</th><th>自适应：细/中/粗</th><th>自适应冷启动</th></tr>'''
    for term,v in extreme['terms'].items():body+=f'<tr><td>{term}</td><td>{fmt(v["naive"])}</td><td>{fmt(v["adaptive"])}</td><td>{fmt(v["cold"])}</td></tr>'
    body+='</table><table><tr><th>方案 / 项</th><th>细笔画梯度份额</th><th>总梯度量相对原项</th><th>细笔画梯度被其余项抵消</th></tr>'
    for name,values in measurements.items():
        for term,v in values.items():body+=f'<tr><td>{name} / {term}</td><td>{v["shares"][0]:.4%}</td><td>{v["total_mass_ratio"]:.3f}倍</td><td>{v["cancelled_fraction"][0]:.3%}</td></tr>'
    body+='</table><p>平均权重为1不等于实际梯度范数或总量不变。4倍上限也意味着29个像素不会被强行赋予整组1/3的影响力。</p></section><section><h2>构造的稀少、缺组和全空情况</h2><table><tr><th>情况</th><th>细/中/粗像素</th><th>直接等权</th><th>自适应冷启动</th></tr>'
    for c in cases:body+=f'<tr><td>{c["name"]}</td><td>{c["counts"]}</td><td>{fmt(c["naive"])}</td><td>{fmt(c["weights"])}</td></tr>'
    body+='</table><div class="images"><figure><img src="synthetic_gt.png"><figcaption>构造GT：单个细像素+粗区域</figcaption></figure><figure><img src="synthetic_equal.png"><figcaption>直接等权，倍率色标0–4</figcaption></figure><figure><img src="synthetic_adaptive.png"><figcaption>自适应，倍率色标0–4</figcaption></figure></div></section><section><h2>EMA的限制：分布突变后的滞后</h2><p>下面均先用“切换前”的分布建立EMA，再连续输入切换后的分布。安全上限仍成立，但稀少组不保证立刻获得最大增强。</p><table><tr><th>切换情形</th><th>第1步：细/中/粗</th><th>第10步</th><th>第100步</th><th>第300步</th></tr>'
    for name,seq in sequences.items():body+='<tr><td>'+name+'</td>'+''.join('<td>'+fmt(seq[i-1]['weights'])+'</td>' for i in [1,10,100,300])+'</tr>'
    batch_switch,_=adapt([1,1,100000])
    body+=f'</table><p>因此EMA解决波动，不保证每个batch都即时适配。该限制需在训练采样顺序下观察，不能只看冷启动的单个batch。</p></section><section><h2>更简单的并列方案：不用EMA</h2><p>每batch直接使用当前面积占比，仍保留平方根、epsilon和联合均值1/cap4求解。这等于表中的冷启动列，但每batch独立计算，不继承历史。真实极端missing权重为{fmt(variants["batch"]["missing"])}；突然变为[1,1,100000]时权重为{fmt(batch_switch)}，不出现旧EMA导致的0.10倍抑制。</p><p>此方案通过相同构造计数的autograd检查；代价是权重随batch变化更直接，尚未验证长期训练噪声或图像效果。当前证据支持先把“当前batch+平方根+cap4”作为更简单的训练候选，不直接认定EMA一定更好。</p></section><section><h2>校验结果</h2><p>10,000组随机稀疏计数，EMA版本最大倍率{max_weight:.6f}，平均权重误差最大{max_error:.3e}；构造冷启动例触及但未超过4。缺组权重为0，全空不更新状态。JSON保存恢复后后续权重与状态逐值一致。</p><p>6种构造计数 × float64/float32/float16，共18组autograd校验通过；missing及false均比较解析导数，低精度输入用FP32累计loss。零支持得到零loss/零梯度。真实可视化60张图片尺寸与读取检查通过。</p><p>这里验证了权重和输出梯度，不代表CUDA/AMP完整训练吞吐、参数梯度或图像效果已经验证；尚未接入训练。</p></section></html>'
    (OUT/'index.html').write_text(body,encoding='utf-8');print(json.dumps(dict(extreme=extreme,measurements=measurements,stress=result['stress'],first_switch={k:v[0]['weights'] for k,v in sequences.items()}),ensure_ascii=False))


if __name__=='__main__':main()
