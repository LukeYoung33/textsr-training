"""Chinese evidence-only report for the approved fixed039 grid diagnosis."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import html
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/grid039_diagnostic_20261104'
SOURCE = Path('Y:/guang.yang/TextSR/data/grid039_diagnostic_20261104')


def chart(series, title, xlabel):
    """@luna-comment Input:named float series/title/axislabel. Output:standaloneSVG curve plot. Key Steps:sharednumericaxes/grid/polyline/legend. Key Params:allseriesfinite,commonYrange. State/Dependencies:reportonly;noqualityranking."""
    values = [v for name, row, color in series for v in row]
    lo, hi = min(values), max(values)
    pad = max((hi-lo)*.12, .1)
    lo -= pad
    hi += pad
    width, height = 860, 310
    text = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 310"><rect width="860" height="310" fill="white"/><text x="65" y="24" font-size="18">'+html.escape(title)+'</text>'
    for i in range(5):
        y = 50+i*45
        v = hi-(hi-lo)*i/4
        text += f'<path d="M65 {y}H830" stroke="#ddd"/><text x="8" y="{y+5}" font-size="13">{v:.2f}</text>'
    for index, (name, row, color) in enumerate(series):
        points = ' '.join(f'{65+765*i/max(len(row)-1,1):.2f},{50+180*(hi-v)/(hi-lo):.2f}' for i,v in enumerate(row))
        text += '<polyline points="'+points+'" fill="none" stroke="'+color+'" stroke-width="2"/>'
        text += f'<text x="{65+index*245}" y="270" fill="{color}" font-size="14">'+html.escape(name)+'</text>'
    text += '<text x="65" y="247" font-size="13">0</text><text x="800" y="247" font-size="13">'+str(len(series[0][1])-1)+'</text><text x="300" y="300" font-size="14">'+html.escape(xlabel)+'</text></svg>'
    return text


def main():
    """@luna-comment Input:completedno-update039evidence. Output:localChineseHTML/SVG/crops pluscopiedJSONaudit. Key Steps:requirecomplete,copyonlynewdiagnosticassets,renderrawandboundedclaims. Key Params:nofix/visualwinner/newdatasetregistration. State/Dependencies:NASdiagnosticandlocalreportonly."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = ['METRICS.json','PHASES.json','PATCH_PROBES.json','SUPPLEMENT.json','DETRENDED_PHASES.json','FEATURE_PHASE.json','BINDING.json','COMPLETE.json']
    pictures = [key+suffix for key in ['input','new_original_grid','new_shift112','old_l1_reference'] for suffix in ['_crop.png','_preview.jpg']]
    pictures += [key+'_residual_x8.png' for key in ['new_original_grid','new_shift112','old_l1_reference']]
    pictures += ['new_flat_residual_x8.png','old_l1_flat_residual_x8.png']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name:shutil.copy2(SOURCE/name,OUT/name),metadata+pictures))
    metrics = json.loads((OUT/'METRICS.json').read_text())
    phases = json.loads((OUT/'PHASES.json').read_text())
    supplement = json.loads((OUT/'SUPPLEMENT.json').read_text())
    feature = json.loads((OUT/'FEATURE_PHASE.json').read_text())
    assert metrics['baseline_bit_exact'] and feature['all_feature_phases_preserved_by_native112']
    keys = [('new_original_grid','新模型·原网格','#1768c4'),('new_shift112','新模型·起点平移112','#d76816'),('old_l1_reference','旧V3M L1-only参考','#597d2a')]
    charts = []
    for axis, label in [('x','沿横坐标折叠（对应竖向条纹）'),('y','沿纵坐标折叠（对应横向条纹）')]:
        svg = chart([(name,phases[key][axis],color) for key,name,color in keys],label+'：原始残差，单位灰阶','原图坐标对224取余 / px')
        name = 'phase_'+axis+'.svg'
        (OUT/name).write_text(svg,encoding='utf-8')
        charts.append(name)
    profiles = supplement['probes']
    (OUT/'edge_distance.svg').write_text(chart([
        ('新模型',[x-profiles['new']['flat_center_residual_dn'] for x in profiles['new']['flat_distance_profile']],'#1768c4'),
        ('旧L1-only',[x-profiles['old_l1']['flat_center_residual_dn'] for x in profiles['old_l1']['flat_distance_profile']],'#597d2a')],
        '均匀暗蓝patch：相对中心的浮点亮度残差 / 灰阶','到最近patch边缘的距离 / px'),encoding='utf-8')
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>039 深色背景网格诊断</title><style>body{font:17px/1.7 system-ui;margin:26px;color:#18314b;background:#eef2f6}section{background:white;border-radius:12px;padding:22px;margin:18px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd4df;text-align:left}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}img{max-width:100%}pre{white-space:pre-wrap}small{color:#52657a}</style><h1>039：深色背景网格诊断</h1>'
    body += '<section><h2>可以确认的机制</h2><p><b>模型对patch边缘与内部的背景预测不同，逐块推理把这种误差重复铺成网格；当前crop8＋smooth16仍保留了受影响区域。</b></p><p>新模型原网格逐像素复现通过。切块起点平移112px后，竖向条纹对应的224px相位曲线也移动112px，相关系数0.950；四个分区的最佳位移均为112px。原图内部像素保持不变，且112px保留了实际网络各级采样相位。</p><p>均匀暗蓝输入也会产生边缘到内部的残差变化，说明并不依赖这张照片已有的背景纹理。新模型在距边缘8px处相对中心偏差4.49灰阶，32px处2.19，48px处0.95；该探针没有RGB截断，统计在PNG量化前进行。</p></section>'
    body += '<section><h2>量化结果</h2><p>共同暗背景有效像素 '+f"{metrics['common_background_pixels']:,}"+'；亮度单位均为0–255灰阶。输入定义同一掩膜，排除文字、较强梯度及物理图像边缘512px。</p><table><tr><th>输出</th><th>平均亮度残差</th><th>残差标准差</th><th>横坐标224相位峰峰值</th><th>纵坐标224相位峰峰值</th></tr>'
    for key,name,color in keys:
        v = metrics['variants'][key]
        body += '<tr><td>'+name+'</td>'+''.join('<td>'+f'{v[k]:.3f}'+'</td>' for k in ['mean_dn','std_dn','x_peak_to_peak_dn','y_peak_to_peak_dn'])+'</tr>'
    body += '</table><p>相位峰峰值用于描述周期变化，不等同于整幅图像的全部网格能量。横向条纹的全幅统计尚未证明统一移动112px；右上分区支持该位移（相关0.753），其他区域仍有内容相关变化，不把所有横纹归为同一机制。</p></section>'
    body += '<section><h2>相位曲线</h2><p>曲线为Y(输出)−Y(输入)，按原图坐标对224取余分组，再减去背景残差均值。横坐标曲线的原网格/平移112相关为−0.084，对齐112后为0.950。σ56去趋势敏感性分析后仍为0.951；去趋势不是修复输出。</p>'
    body += ''.join('<img src="'+name+'">' for name in charts)+'</section>'
    body += '<section><h2>单patch边界响应</h2><p>同一均匀RGB输入：[41,59,110]；原始256×256 patch。旧、新模型都存在边界响应，新模型更强。距边缘8px的偏差：新4.49，旧1.67灰阶；均匀区域中心残差：新−1.71，旧＋0.52灰阶。浮点结果无截断。</p><img src="edge_distance.svg"><p>内部2/4/8/16px相位调制仅约0.013／0.035灰阶（横／纵），在这个暗色探针上不足以解释明显的224px大网格。</p><p>边缘误差延伸超过裁掉的8px，且16px羽化只是加权平均，无法保证两块都给出正确背景。权重覆盖检查通过，最小总权重约1，并非未覆盖像素形成的空洞。</p></section>'
    body += '<section><h2>039 同位置局部证据</h2><p>固定原图区域：x=2816…3839，y=768…1791，未做局部锐化或影调调整。点击图片可看原尺寸。</p><div class="grid">'
    for key,name,color in [('input','V3预处理输入','#000')]+keys:
        body += '<article><b>'+name+'</b><a href="'+key+'_crop.png"><img src="'+key+'_crop.png"></a></article>'
    body += '</div><h3>亮度残差放大8倍，128灰表示零变化</h3><div class="grid">'
    for key,name,color in keys:
        body += '<article><b>'+name+'</b><img src="'+key+'_residual_x8.png"></article>'
    body += '</div></section><section><h2>结论边界与下一步</h2><p>本轮确认了切块位置相关的边界响应，以及当前裁边宽度不足以避开该探针中的影响。尚未分离哪一项训练数据、目标风格或采样设置让新权重更敏感；旧、新训练历史不构成严格的单因素对照。仅凭本轮不能认定是笔画膨胀或L1本身造成。</p><p>未修改loss、训练、模型权重或正式推理实现；没有生成修正版。应先由用户确认根因结论，再进入方案验证。</p><p>可审计证据：<a href="METRICS.json">主指标</a> · <a href="SUPPLEMENT.json">分区／去趋势／旧新探针</a> · <a href="FEATURE_PHASE.json">实际特征采样尺寸</a> · <a href="BINDING.json">输入／模型哈希</a></p></section></html>'
    (OUT/'index.html').write_text(body,encoding='utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),baseline_bit_exact=True,metadata_files=len(metadata),evidence_images=len(pictures))))


if __name__ == '__main__':
    main()
