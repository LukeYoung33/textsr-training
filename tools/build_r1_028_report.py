"""Chinese diagnostic summary and synchronized native gradient-map viewer."""
from pathlib import Path
import json,re,html,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluation/r1_gradient_028_250k_20261103'


def percent(v):
    """@luna-comment Input:ratio orNone. Output:Chinese-report numeric percent. Key Steps:format finitevalue. Key Params:1decimal. State/Dependencies:pure."""
    return '—' if v is None else f'{v*100:.1f}%'


def table(stats):
    """@luna-comment Input:three measuredwidthbins. Output:area/gradient/conflict table. Key Steps:explicitdenominators andempty-bin formatting. Key Params:weightedactualmissing/false. State/Dependencies:report data."""
    rows=[]
    for b,label in [('0','细：≤2 px'),('1','中：2–4 px'),('2','粗：>4 px')]:
        v=stats[b];mean=v['gradient_per_pixel']
        rows.append('<tr><th>'+label+'</th><td>'+f'{v["pixels"]:,}'+'</td><td>'+percent(v['area_share'])+'</td><td>'+percent(v['gradient_share'])+'</td><td>'+(f'{mean:.3e}' if mean is not None else '—')+'</td><td>'+percent(v['strong_fraction'])+'</td><td>'+percent(v['cancelled_gradient_fraction'])+'</td><td>'+percent(v['equal_error_expected_share'])+'</td></tr>')
    return '<table><thead><tr><th>笔画宽度</th><th>参与像素</th><th>区域面积占比</th><th>梯度总量占比</th><th>每像素平均 |∂L/∂Y|</th><th>较强激活像素占比</th><th>被其余项抵消的梯度量</th><th>等误差对照占比</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def roi_stats(maps,sl):
    """@luna-comment Input:fullrawmaps andROI slice. Output:localarea/gradient/perpixel descriptors. Key Steps:samebins andweights asfullreport. Key Params:spatialROI only,no renormalizingloss. State/Dependencies:read-onlymaps."""
    width=maps['width'][sl];bins=np.where(width<=2.00001,0,np.where(width<=4.00001,1,2));result={}
    for kind,key in [('missing','support_m'),('false','support_f')]:
        support=maps[key][sl]>0;g=maps[kind][sl];rest=maps['total'][sl]-g;mass=float(np.abs(g)[support].sum());count=int(support.sum());values={}
        for b in range(3):
            select=support&(bins==b);n=int(select.sum());m=float(np.abs(g)[select].sum());act=maps['activation_m' if kind=='missing' else 'activation_f'][sl]
            cancel=float(np.minimum(np.abs(g),np.abs(rest))[select&(g*rest<0)].sum())
            values[str(b)]=dict(pixels=n,area_share=n/max(count,1),gradient_share=m/max(mass,1e-30),gradient_per_pixel=m/n if n else None,strong_fraction=float((act[select]>.5).mean()) if n else None,cancelled_gradient_fraction=cancel/max(m,1e-30),equal_error_expected_share=None)
        result[kind]=values
    return result


def main():
    """@luna-comment Input:verifiedactual-runtime report and416PNGs. Output:Chinese report/viewer,ROI andintegrity receipt. Key Steps:commoncolorlegend,three gradient measures,widthtables,model/source caveats,browser-ready embed. Key Params:ROI160/850/640/330;allmapsnative. State/Dependencies:ownedlocal outputs only."""
    r=json.loads((OUT/'REPORT.json').read_text(encoding='utf-8'));assert r['passed'] and r['model_state_unchanged']
    full=r['views'][0];assert full['id']=='000';roi=OUT/'roi';roi.mkdir(exist_ok=True);sl=(slice(850,1180),slice(160,800));roi_paths={}
    for key,relative in full['images'].items():
        value=cv2.imread(str(OUT/relative));assert value.shape[:2]==(1213,1235);path=roi/(key+'.png');assert cv2.imwrite(str(path),value[sl]);roi_paths[key]='roi/'+key+'.png'
    maps=np.load(OUT/'full_maps.npz');local=roi_stats(maps,sl);(OUT/'ROI_STATS.json').write_text(json.dumps(local,indent=2),encoding='utf-8')
    views=[dict(id='ROI',xy=[160,850],width=640,height=330,images=roi_paths)]+r['views']
    limit=r['shared_gradient_limit'];units=f'共同线性色标：0 → {limit:.3e}；单位 ∂加权loss/∂输出Y，Y∈[0,1]。顶部0.5%幅值饱和显示，原始值保留在NPZ。'
    title='028 · R1 梯度诊断 · V3M Full无guard 250K'
    subtitle='上排：真实 V3 LQ / 模型输出 / 实际训练 GT\n下排：missing / false / 总 loss 梯度'
    notes='对应 capture_261011_003171。模型为已完成500K的50%轻退化混合组，取250K权重；本次读取真实LQ分支。\nGT使用训练时S2.2目标，并逐patch对比度对齐。原始对齐GT可在下拉菜单中查看。\nmissing权重1，false权重0.25；所有梯度图采用同一绝对标尺。\n方向图：橙红=梯度下降推动变暗；青色=推动变亮。\n激活图0–1表示soft hinge梯度系数，0.5对应误差达到margin=0.03。\n支持区：绿色missing，蓝色false，红色为soft mask>0但≤0.5、被二值化排除的像素。\n宽度图：蓝=≤2px，绿=2–4px，红=>4px。只改变显示颜色。\n整图按实际训练256网格拼接，仅作定位；单patch保留实际梯度。图像边缘补齐部分未显示。'
    models=[dict(key=k,label=label,note=note) for k,label,note in [
        ('lq','实际输入 · V3 LQ','训练库真实LQ分支；无二次预处理'),('output','模型输出 · 250K','从冻结模型得到的实际256训练patch输出'),('gt','实际训练 GT · 对比度对齐后','S2.2目标，按当前LQ执行原训练对比度匹配'),
        ('missing_gradient','missing 梯度强度 · 权重1',units),('false_gradient','false 梯度强度 · 权重0.25',units),('total_gradient','总 loss 梯度强度',units),
        ('r1_gradient','R1 合计梯度强度',units),('native_gradient','Native L1 梯度强度',units),('other_aux_gradient','其余辅助项梯度强度 · UV+背景',units),
        ('r1_direction','R1 推动方向 · 红变暗 / 青变亮','显示 -∂R1/∂输出Y；色强度采用共同绝对标尺'),('missing_activation','missing 激活系数 · 0至1','显示sigmoid((有向误差−0.03)/0.01)；与梯度强度不同'),('false_activation','false 激活系数 · 0至1','显示sigmoid((反向误差−0.03)/0.01)；仅外围环带'),
        ('r1_relative','R1 相对推动比例 · 0至1','|gR1|/(|gR1|+|g其余|)，亮度输出空间；不等于参数更新占比'),('support','实际支持区与二值化排除区','绿missing / 蓝false / 红被0.5阈值排除的软mask像素'),('width_bins','笔画粗细分组','蓝≤2px / 绿2–4px / 红>4px；局部直径近似'),('original_gt','原始对齐 GT · 仅参照','截图中的原始GT；该模型的实际训练目标见GT面板')]]
    rows=[]
    for v in views:
        name='截图附近：限速六十 / Tracking / 小字' if v['id']=='ROI' else '整图 · 训练网格拼接' if v['id']=='000' else f'训练patch {v["id"]} · x={v["xy"][0]}, y={v["xy"][1]}'
        for path in v['images'].values():
            im=cv2.imread(str(OUT/path));assert im is not None and im.shape[:2]==(v['height'],v['width'])
        rows.append(dict(id=v['id'],name=name,width=v['width'],height=v['height'],lq=v['images']['lq'],images={k:p for k,p in v['images'].items() if k!='lq'}))
    data=dict(title=title,subtitle=subtitle,notes=notes,models=models,defaults='lq,output,gt,missing_gradient,false_gradient,total_gradient',rows=rows)
    text=(ROOT/'evaluation/original_gt_dilation30_20261102/index.html').read_text(encoding='utf-8')
    text=re.sub(r'const DATA=.*?;</script>',lambda _: 'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',text,count=1,flags=re.S)
    text=re.sub(r'<title>.*?</title>','<title>'+title+'</title>',text,count=1)
    text=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+title+'</h1><small>'+subtitle.replace('\n','<br>')+'</small>',text,count=1,flags=re.S)
    text=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+html.escape(notes).replace('\n','<br>')+'</div>',text,count=1,flags=re.S)
    text=text.replace('href="RESULTS.json"','href="index.html"').replace('查看来源与处理对应表','查看诊断结论与统计')
    legend='<div style="font-size:12px;padding:4px 0">'+units+' <span style="display:inline-block;width:180px;height:10px;background:linear-gradient(90deg,#30123b,#4669dd,#1ae0b6,#a4fc3b,#f9ba38,#d02e05,#7a0403)"></span> 弱 → 强</div>'
    text=text.replace('</header>',legend+'</header>',1)
    (OUT/'viewer.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    m=r['statistics']['missing'];f=r['statistics']['false'];ratio=m['0']['gradient_per_pixel']/m['2']['gradient_per_pixel'];fratio=f['0']['gradient_per_pixel']/f['2']['gradient_per_pixel']
    pnorm=[b['r1_vs_rest']['norm_ratio'] for b in r['batches']];cos=[b['r1_vs_rest']['cosine'] for b in r['batches']]
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>
body{{font:16px/1.7 system-ui,"Microsoft YaHei";background:#edf2f7;color:#1a3048;margin:0}}main{{max-width:1280px;margin:auto;padding:28px}}section{{background:white;border-radius:12px;padding:22px;margin:18px 0}}h1{{font-size:26px}}h2{{font-size:20px}}a{{color:#125bc1}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dce5ec;padding:9px;text-align:right}}th:first-child{{text-align:left}}.lead{{font-size:19px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.card{{background:#f2f6fa;padding:16px;border-radius:8px}}small{{color:#51687e}}.bar{{height:16px;display:flex;background:#eee}}.bar span{{display:block}}figure{{margin:12px 0}}img{{max-width:100%}}details{{margin-top:12px}}code{{word-break:break-all}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}section{{overflow:auto}}}}</style><main>
<h1>{title}</h1><p>样本028：lenovo1011 / capture_261011_003171。冻结250K权重，诊断真实LQ分支。</p>
<p><a href="viewer.html#ROI">打开截图附近的六视图梯度对比 →</a>　<a href="viewer.html#000">查看整图 →</a>　<a href="REPORT.json">完整数据与来源</a></p>
<section><h2>这张图支持什么结论</h2><p class="lead">细笔画的 missing 并没有整体失去梯度；它们的单像素推动反而更强。</p><div class="grid"><div class="card">细笔画占文字面积 <b>{percent(m['0']['area_share'])}</b>，获得 <b>{percent(m['0']['gradient_share'])}</b> 的 missing 梯度。每像素平均梯度约为粗笔画的 <b>{ratio:.2f} 倍</b>。</div><div class="card">细笔画对应的 false 环带占环带面积 <b>{percent(f['0']['area_share'])}</b>，获得 <b>{percent(f['0']['gradient_share'])}</b> 的 false 梯度；每像素约为粗笔画环带的 <b>{fratio:.2f} 倍</b>。</div></div>
<p>missing 仍有 {percent(m['2']['gradient_share'])} 的梯度量落在粗笔画上，因为粗笔画占文字面积 {percent(m['2']['area_share'])}。这体现了面积汇总的影响；但本例细笔画的误差激活更强，已经部分补偿这种影响。</p>
<p>在输出亮度空间，细笔画 missing 梯度中约 <b>{percent(m['0']['cancelled_gradient_fraction'])}</b> 被其余项反向抵消。参数空间另有相互作用：7个固定batch的R1/其余项梯度范数比为 <b>{min(pnorm):.3f}–{max(pnorm):.3f}</b>，其中 <b>{sum(v<0 for v in cos)}/7</b> 个batch方向余弦为负。不能把局部热图直接理解成最终 Adam 更新量。</p>
<p><b>当前证据不足以把细笔画修复不明显主要归因于 missing 梯度太小。</b>图中的实际训练GT与原始对齐GT不同；下一步若验证归一化，应分别检查missing与false，且同时看监督目标。这里只是单图、单checkpoint的诊断。</p></section>
<section><h2>如何读图</h2><ol><li>先比较实际输入、模型输出、实际训练GT，确认模型被要求接近的目标。</li><li>看missing/false/总loss的强度图。它们共享色标，可直接比较绝对强弱；背景保留输出图，彩色覆盖越强，梯度越大。</li><li>切换“R1推动方向”：橙红推动变暗，青色推动变亮，表示对输出亮度做梯度下降的方向。</li><li>用“激活系数”区分误差已小与梯度分母稀释：系数0.5对应有向误差达到margin 0.03（约7.7灰度级）。</li><li>切换支持区/粗细分组；没有支持的像素无法获得该项R1监督。原始GT仅供参照。</li></ol><p>{units}</p><p><small>强度图显示 |∂L/∂输出Y|，missing和false已乘训练权重1与0.25。false只作用于原文字mask外的3px方形膨胀环带。两项均为单侧约束。饱和颜色不是梯度上限。</small></p></section>
<section><h2>全图：missing 的粗细分配</h2>{table(m)}<p><small>面积与梯度占比均在missing支持区域内归一化；梯度量为绝对亮度输出梯度之和。</small></p><h2>全图：false 的粗细分配</h2>{table(f)}<p><small>false按最近文字骨架的宽度划分外围环带；面积分母为环带面积，不是文字面积。宽度估计在弯曲和交叉处是近似值。</small></p></section>
<section><h2>截图附近：限速六十 / Tracking / 小字</h2><p>原图坐标 x=160…799、y=850…1179。这里沿用实际batch梯度，仅裁出观察区域。</p><h3>missing</h3>{table(local['missing'])}<h3>false</h3>{table(local['false'])}<p><a href="viewer.html#ROI">打开这一局部的原生像素热图</a></p></section>
<section><h2>等误差对照能说明什么</h2><p>表中最后一列假设所有参与像素都有相同0.06的有向误差，按本次实际batch分母计算其预期梯度分配。在这个对照下，细笔画只获得 {percent(m['0']['equal_error_expected_share'])} 的missing梯度，而实际为 {percent(m['0']['gradient_share'])}。这证明当前归一化确实不是按粗细均衡；也说明这张图实际细笔画误差较大，不能只按面积推断实际梯度很弱。</p><p><small>等误差对照为loss入口亮度域解析量；不包含下游RGB色域裁切的导数。实际梯度图则完整经过原模型的色彩转换链。</small></p></section>
<section><h2>模型参数上的梯度关系</h2><p>范数比=|R1参数梯度|/|其余loss参数梯度|。余弦&gt;0偏向同向，&lt;0存在反向成分，0附近近似正交；不同batch分别报告。</p><table><tr><th>固定batch</th><th>Patch编号</th><th>范数比</th><th>方向余弦</th></tr>'''
    for i,b in enumerate(r['batches']):body+=f'<tr><th>{i+1}</th><td>'+', '.join(f'{p+1:03d}' for p in b['positions'])+f'</td><td>{b["r1_vs_rest"]["norm_ratio"]:.4f}</td><td>{b["r1_vs_rest"]["cosine"]:.4f}</td></tr>'
    body+=f'''</table></section><section><h2>复现与核验</h2><p>25个原始训练baseline patch，固定batch=4。最后一个batch复用前三个patch补齐；像素统计排除补齐重复项。没有增强变换，边缘padding按valid mask处理，页面只显示真实图像范围。</p><p>严格加载250K参数；实际S22Dataset提供contrast-aligned GT；从训练中真实loss计算函数截取计算图，在backward和optimizer.step之前返回。仅用autograd.grad取导数，网络参数及buffer前后哈希一致，优化器状态为空。</p><p>R1解析RGB导数与autograd最大差：<b>{r['gradient_analytic_max_error']:.3e}</b>。分项梯度相加与总梯度核对通过；宽度分组在1/2/3/4/6/8px直笔画测试通过。</p><details><summary>绑定路径与原始数值</summary><p>模型：<code>{html.escape(r['checkpoint'])}</code></p><p>模型SHA256：<code>{r['checkpoint_sha256']}</code></p><p>配置：<code>{html.escape(r['config'])}</code></p><p><a href="full_maps.npz">全图浮点梯度NPZ</a> · <a href="REPORT.json">全部定量统计JSON</a> · 各patch目录包含独立maps.npz。</p></details></section></main></html>'''
    (OUT/'index.html').write_text(body,encoding='utf-8')
    receipt=dict(passed=True,views=len(rows),layers=len(models),images=len(rows)*len(models),checkpoint_sha256=r['checkpoint_sha256'],actual_gradient_analytic_error=r['gradient_analytic_max_error'])
    (OUT/'DELIVERY_CHECKS.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
