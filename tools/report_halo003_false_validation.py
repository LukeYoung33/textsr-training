"""Chinese evidence report for halo003 and separated false-loss validation probes."""
from pathlib import Path
import json
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/halo003_diagnostic_20261104'
EVIDENCE=OUT/'evidence'
NAS='file:///Y:/guang.yang/TextSR/data/halo003_diagnostic_20261104/'


def main():
    """@luna-comment Input:completedimmutableJSONprobes. Output:ChineseHTML/SVG/PNGreport withsource-linkedfinitecrops. Key Steps:assertgates,summarizeactualgradientdirection/mixedvalidationeffects,showmatched350K/GT/mask/tilelimits. Key Params:nosubjectivequalityranking,nofix/training. State/Dependencies:ownedlocalreport;sourceimagesremainNAS."""
    read=lambda name:json.loads((EVIDENCE/name).read_text())
    val=read('VALIDATION_FALSE_PROBE.json');matched=read('MATCHED_350K.json');tile=read('TILE_PROBE.json');target=read('TARGET_WIDTH_PROBE.json');grad=read('R1_GRADIENT_PROBE.json');mojito=read('MOJITO_VAL8.json');impl=read('IMPLEMENTATION_BINDING.json')
    assert all(x['passed'] for x in [val,matched,tile,target,grad,mojito]) and impl['strict_checkpoint_match']
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(8)
    for j,(term,label,color) in enumerate([('native','L1','#4285d4'),('missing','Missing','#21a477'),('false','False（权重0.25）','#e05b43')]):
        y=[r['gradient_effects'][term]['halo_direction_cosine'] for r in val['records']]
        ax.bar(x+(j-1)*.25,y,width=.24,label=label,color=color)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[str(i+1) for i in x]);ax.set_xlabel('验证样本编号（每张4个patch）');ax.set_ylabel('增大外侧反极性误差方向的余弦');ax.set_title('分项普通梯度方向：正值促发外侧误差，负值抑制');ax.legend();fig.tight_layout();fig.savefig(OUT/'false_gradient_direction.png',dpi=150);fig.savefig(OUT/'false_gradient_direction.svg');plt.close(fig)
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文字halo与变粗根因诊断</title><style>body{font:16px/1.75 system-ui;max-width:1350px;margin:30px auto;color:#203047;padding:0 20px}h1,h2{color:#123456}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ccd5df;padding:8px;text-align:left}th{background:#eaf0f7}img{max-width:100%}.note{background:#fff4d6;padding:16px}.good{background:#e6f4eb;padding:16px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}article{border:1px solid #ccd5df;padding:8px}code{word-break:break-all}</style><h1>文字外黑影、白色halo与中小字变粗：根因诊断</h1>'
    body+='<p>主样本003；原验证集8张、32个patch、四组500K干净对照；另复核Mojito原划分验证8张。全部为模型原始输出。仅诊断，不改loss、不执行优化器更新。</p>'
    body+='<div class="good"><b>核心判断：</b>false是文字外反极性halo的直接风险项。它对黑字外侧推动变亮，对白字外侧推动变暗；验证分项梯度8/8同方向支持这一点。Missing在这批验证样本上的参数梯度反而抑制该指标。尚没有只保留Missing或只保留False的训练消融，因此不能把全部已训练伪影唯一归因于False。</div>'
    body+='<h2>1．为什么false会有这个方向？</h2><p>实际实现以GT判断整块文字的明暗极性，外侧环为TextMask膨胀3px后减去前景。定义 e = polarity × (预测Y − GT Y)，false = mean[0.01 × softplus((-e − 0.03)/0.01)]，权重0.25。</p><table><tr><th>文字极性</th><th>false梯度下降的直接方向</th><th>对应风险</th></tr><tr><td>黑字、浅背景</td><td>外侧背景变亮</td><td>白色轮廓</td></tr><tr><td>白字、深背景</td><td>外侧背景变暗</td><td>黑影、暗轮廓</td></tr></table><p>这是单侧约束，主要惩罚外侧“向文字极性靠近”；对反方向过亮/过暗没有对称惩罚。softplus在预测等于GT时仍有小梯度，越过GT后梯度会衰减。因此它既可能清除伪笔画，也可能与L1的还原方向竞争；不能把每次变亮或变暗都视为错误。</p>'
    body+='<h2>2．验证集：分开测Missing与False</h2><p>从原583张验证源中，HXA、HXB、Lenovo0926、Lenovo1011各取2张，每张固定4个baseline patch。四组均最终500K，输入、GT和数据域匹配。正向余弦表示沿该loss的普通梯度下降会增大当前外侧反极性误差；这是局部一阶方向，不是对实际Adam轨迹的复现。</p><img src="false_gradient_direction.png"><table><tr><th>验证样本</th><th>L1方向</th><th>Missing方向</th><th>False方向</th><th>加入MF后的外环超额变化(DN)</th></tr>'
    for r in val['records']:
        e=r['gradient_effects'];change=r['models']['v3m_mf_only']['halo_gt_excess_dn']-r['models']['v3m_l1']['halo_gt_excess_dn']
        body+='<tr><td>'+html.escape(r['sample'])+'</td>'+''.join('<td>'+f'{e[t]["halo_direction_cosine"]:+.3f}'+'</td>' for t in ['native','missing','false'])+'<td>'+f'{change:+.3f}'+'</td></tr>'
    body+='</table><div class="note">False为8/8正方向；Missing与L1均为8/8负方向。但三项合计的普通梯度方向仍抑制该误差，False在这里是抵消部分修正，不是整个优化过程都在制造halo。终态图像：L1→MF有4/8张指标增加；UV→MF+UV为3/8。两对均值变化仅约+0.08/+0.05DN，不能据此宣称全验证集必然恶化。</div>'
    body+='<p>指标是“GT参考下文字外3px环内、超过2DN的反极性误差”，不是完整视觉halo检测器。GT为历史验证目标（旧教师域），包含背景和边缘差异；表格需结合下列图像查看，不能当作视觉胜率。</p>'
    body+='<h2>3．验证原图与结果：请重点看文字外轮廓</h2><p>每张示例展示固定patch0，点击图片可打开原始256像素PNG；其余patch已计入量化。不同颜色/极性的背景均有覆盖。</p>'
    for i,r in enumerate(val['records'],1):
        body+='<h3>'+str(i)+'．'+html.escape(r['sample'])+'</h3><div class="grid">'
        for key,label in [('input','输入'),('target','验证参考GT'),('v3m_l1','L1-only'),('v3m_mf_only','L1＋Missing/False'),('v3m_uv_only','L1＋UV'),('v3m_mf_uv','L1＋Missing/False＋UV')]:
            url=NAS+f'validation_crops/{i}/{key}_p0.png';body+='<article><b>'+label+'</b><a href="'+url+'"><img src="'+url+'"></a></article>'
        body+='</div>'
    body+='<h2>4．Mojito验证8张：与训练patch统计分开</h2><p>001、002、003、004、269、270、271、272均为原划分验证图。这里以V3输入为参考，无对应可信原始GT；外环高频变亮/变暗只能描述轮廓变化。</p><table><tr><th>编号</th><th>L1→MF前景面积比变化</th><th>UV→MF+UV前景面积比变化</th><th>L1→MF外环高频变亮量变化(DN)</th></tr>'
    for r in mojito['records']:
        m=r['models'];body+='<tr><td>'+r['id']+'</td><td>'+f'{m["v3m_mf_only_raw"]["area_ratio"]-m["v3m_l1_raw"]["area_ratio"]:+.3f}'+'</td><td>'+f'{m["v3m_mf_uv_raw"]["area_ratio"]-m["v3m_uv_only_raw"]["area_ratio"]:+.3f}'+'</td><td>'+f'{m["v3m_mf_only_raw"]["exterior_centered_bright_excess_dn"]-m["v3m_l1_raw"]["exterior_centered_bright_excess_dn"]:+.3f}'+'</td></tr>'
    body+='</table>'
    for identity in ['003','269']:
        body+='<h3>Mojito '+identity+'</h3><div class="grid">'
        for key,label in [('v3_input','V3输入'),('v3m_l1_raw','L1-only'),('v3m_mf_only_raw','L1＋Missing/False')]:
            body+='<article>'+label+'<img src="'+NAS+'mojito_validation/'+identity+'/'+key+'.png"></article>'
        body+='</div>'
    body+='<h2>5．R1为何更严重：GT增粗与mask扩张是额外因素</h2><p>普通Missing/False组也出现相似方向，因此不能只归因于GT增粗。增粗目标＋同步mask＋粗细权重进一步放大了R1组的表现。</p><table><tr><th>相同350K，003区域</th><th>L1前景面积比</th><th>R1＋新mask前景面积比</th><th>L1宽度变化(px)</th><th>R1宽度变化(px)</th></tr>'
    for roi in ['poem','small','medium','large']:
        a=matched['records']['l1_350k']['rois'][roi];b=matched['records']['r1_350k']['rois'][roi]
        body+='<tr><td>'+roi+'</td>'+''.join('<td>'+f'{v:.3f}'+'</td>' for v in [a['area_ratio'],b['area_ratio'],a['width_change_mean_px'],b['width_change_mean_px']])+'</tr>'
    body+='</table><p>面积和宽度以同一V3输入半对比阈值、固定输入骨架测量，不是GT字号真值。匹配350K后差异仍在，排除了仅由370K/500K不一致解释的可能。训练样本028的原GT→增粗GT：细笔画平均相对宽度增加约29%，中等18%，粗笔画13%；“名义+2px”并非每个灰度笔画都精确增加2px。</p>'
    old_count=sum(r['old_text_pixels'] for r in grad['batches']);new_count=sum(r['new_text_pixels'] for r in grad['batches'])
    body+='<p>16个实际训练patch中，同步mask前景支持从'+str(old_count)+'增加到'+str(new_count)+'，约增加'+f'{(new_count/old_count-1)*100:.1f}'+'%；稀少细笔画分组权重达到4。R1输出梯度范数为native的1.28–2.18倍，参数梯度有2/4个batch与native方向相反。GT恒等探针的R1梯度非零，但这不足以单独证明合成loss的最优点错误。</p>'
    body+='<h2>6．切块与后处理的排查</h2><p>所见问题在原始输出中已经存在。严格重放003与现有R1 PNG逐像素一致；切块起点移动112px保持Haar相位，文字区域残差相关系数约0.84–0.97，平均差约0.97–1.46DN。支持问题具有较强的模型/图像内容依赖，而非仅由拼接缝充分解释；仍有切块依赖，不能全部排除。</p><p><a href="'+NAS+'images.html">查看003全部10组模型、明暗残差与定位探针素材</a></p>'
    body+='<h2>7．结论边界与下一步</h2><p>已有证据支持：False是外侧反极性轮廓的直接风险项；Missing不应因与False捆绑就一起被认定为同等成因。中小字变粗还有GT固定像素增粗、mask扩张和宽度重加权的叠加。GT/配准/颜色域也会影响局部轮廓。</p><p>尚缺少Missing-only、False-only的同起点同数据训练消融，不能宣布已唯一定位全部已训练伪影。下一步需先由用户确认本轮根因判断，再设计拆分false与missing的对照；本轮没有修改loss、训练权重或启动训练。</p>'
    body+='<details><summary>原始证据与范围</summary><ul>'
    for name in ['VALIDATION_FALSE_PROBE.json','MOJITO_VAL8.json','MATCHED_350K.json','TILE_PROBE.json','R1_GRADIENT_PROBE.json','TARGET_WIDTH_PROBE.json','IMPLEMENTATION_BINDING.json','IMAGE_METRICS.json']:
        body+='<li><a href="evidence/'+name+'">'+name+'</a></li>'
    body+='</ul></details></html>'
    (OUT/'index.html').write_text(body,encoding='utf-8')
    print(json.dumps(dict(passed=True,original_validation_parents=8,validation_patches=32,mojito_heldout=8,optimizer_steps=0,report=str(OUT/'index.html'))))


if __name__=='__main__':main()
