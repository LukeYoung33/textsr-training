"""Audit staged anchored-Y research outputs and build an isolated review page."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html

OUT = ROOT/'output/anchored_y_contrast_pilot_20261028'
LABELS = {'current': '当前部署公式', 'weak': '低强度机制验证', 'mean_g3': '均值锚 · 中灰增强',
          'upper_g3': '小网格上锚 · 中灰增强', 'upper_g4': '小网格上锚 · 强档'}


def digest(path):
    """@luna-comment Input:read-only asset path. Output:SHA256. Key Steps:stream1MiB chunks. Key Params:exact bytes. State/Dependencies:no source modification."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def url_path(url):
    """@luna-comment Input:source/reference file URL. Output:Windows Path. Key Steps:decode file URI, reject network URL. Key Params:explicit approved manifest assets. State/Dependencies:read-only E/Y paths."""
    parsed = urlparse(url)
    if parsed.scheme != 'file' or parsed.netloc:
        raise ValueError('Expected local mapped file URI')
    return Path(unquote(parsed.path).lstrip('/'))


def collect_and_audit():
    """@luna-comment
    Input:completed weak/full/strong stage summaries and receipts. Output:merged nine cases, available arms and audit.
    Key Steps:verify same source/core per stage, all output hashes/native dimensions and common signed RGB delta,
    exact inventories, source immutability and all reference existence. Key Params:only completed stages; no PS loss.
    State/Dependencies:PNG-derived Y preview; failed/unwritten arms are never added as available viewer models.
    """
    stages = []
    for filename in ('weak_summary.json', 'full_summary.json', 'strong_summary.json'):
        path = OUT/filename
        if path.exists():
            stages.append(json.loads(path.read_text(encoding='utf-8')))
    if not stages:
        raise FileNotFoundError('No completed research stage')
    by_key = {}
    arms = []
    for stage in stages:
        assert stage['case_count'] == 9
        assert stage['bindings_sha256']['core'] == digest(ROOT/'tools/anchored_y_contrast.py')
        assert stage['bindings_sha256']['frozen_current'] == digest(ROOT/'tools/native_y_curve_four_images.py')
        assert stage['bindings_sha256']['runner'] == digest(ROOT/'tools/run_anchored_y_contrast_pilot.py')
        assert stage['bindings_sha256'] == stages[0]['bindings_sha256']
        for arm in stage['arms']:
            if arm not in arms:
                arms.append(arm)
        for case in stage['cases']:
            if case['key'] not in by_key:
                by_key[case['key']] = {**case, 'outputs': {}}
            merged = by_key[case['key']]
            for key in ('source_sha256', 'source_url', 'width', 'height', 'lq_url', 'ps_reference_url'):
                assert merged[key] == case[key], 'Cross-stage source/domain drift'
            for arm, output in case['outputs'].items():
                assert arm not in merged['outputs'], 'Duplicate case/arm'
                merged['outputs'][arm] = output
    assert len(by_key) == 9 and 'current' in arms and 'weak' in arms
    expected_keys = {'ps150', 'ps151', 'ps152', 'cam006', 'cam017', 'cam051', 'cam052', 'cam099', 'cam100'}
    assert set(by_key) == expected_keys
    for arm in arms:
        assert {p.stem for p in (OUT/arm).glob('*.png')} == expected_keys
    records = []
    for case in by_key.values():
        source_path = url_path(case['source_url'])
        assert digest(source_path) == case['source_sha256']
        assert case['lq_url'] and url_path(case['lq_url']).is_file()
        if case['ps_reference_url']:
            assert url_path(case['ps_reference_url']).is_file()
        source = cv2.imdecode(np.frombuffer(source_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert source is not None and source.shape == (case['height'], case['width'], 3)
        assert set(case['outputs']) == set(arms)
        for arm in arms:
            output = case['outputs'][arm]
            path = Path(output['path'])
            assert path.resolve() == (OUT/arm/(case['key']+'.png')).resolve()
            assert digest(path) == output['sha256']
            receipt = json.loads(Path(output['receipt_path']).read_text(encoding='utf-8'))
            assert receipt['source_sha256'] == case['source_sha256']
            assert receipt['output_sha256'] == output['sha256']
            assert receipt['bindings_sha256'] == stages[0]['bindings_sha256']
            assert receipt['source_unchanged_after_generation'] is True
            result = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            assert result is not None and result.shape == source.shape
            change = result.astype(np.int16)-source.astype(np.int16)
            picked = np.take_along_axis(change, np.argmax(np.abs(change), axis=2)[:, :, None], axis=2)
            assert np.array_equal(np.clip(source.astype(np.int16)+picked, 0, 255).astype(np.uint8), result)
            if arm == 'current':
                assert int(np.max(np.abs(change))) <= 24
            records.append({'case': case['key'], 'arm': arm, 'source_sha256': case['source_sha256'],
                            'output_sha256': output['sha256'], 'parameters': receipt['arm_parameters'],
                            'max_abs_channel_change': int(np.max(np.abs(change))),
                            'changed_pixels_vs_source': int(np.count_nonzero(np.any(change, axis=2)))})
            print('audited '+case['key']+' '+arm, flush=True)
        assert digest(source_path) == case['source_sha256']
    audit = {'case_count': 9, 'arms': arms, 'outputs_checked': len(records), 'records': records,
             'source_files_unchanged': True, 'ps_references_used_as_pixel_targets': False}
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return list(by_key.values()), arms, audit


def write_method(arms, audit):
    """@luna-comment
    Input:actual available arms and audited parameters. Output:Chinese method HTML with interactive fixed-anchor plot.
    Key Steps:explain mechanism, failed hypotheses, coarse-field limits and measured-stage status; plot curve only.
    Key Params:PS strength reference only; no native140ms or halo-suppression claim. State/Dependencies:standalone HTML.
    """
    parameter_rows = []
    for arm in arms:
        record = next(r for r in audit['records'] if r['arm'] == arm)
        parameter_rows.append('<tr><td>'+html.escape(LABELS[arm])+'</td><td>'+html.escape(json.dumps(record['parameters'], ensure_ascii=False))+'</td></tr>')
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>背景锚定对比度曲线 · 研究原型</title>
<style>body{font:16px/1.7 system-ui,"Microsoft YaHei";max-width:1050px;margin:30px auto;padding:0 20px;color:#203046}h1,h2{line-height:1.35}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4df;padding:8px;text-align:left}pre{background:#edf2f7;padding:16px;overflow:auto}.note{background:#fff4d8;padding:14px}canvas{width:100%;max-width:850px;background:#f6f8fb}a{color:#146c96}label{display:inline-block;margin-right:15px}</style>
<h1>背景锚定对比度曲线：研究原型</h1><p><a href="index.html">打开九图效果对照</a> · <a href="INDEPENDENT_AUDIT.json">完整性核对</a></p>
<p class="note">PS 图仅作为对比度强度参考，不进行逐像素拟合，也不把它作为 GT。下面的检查、曲线图和差异指标不构成主观质量结论。真实输出是 PNG 派生 Y 的离线预览，手机整体后处理 ≤140 ms 尚未验证。</p>
<h2>目标与变化</h2><p>灰色笔画需要明显压黑，同时尽量稳定纸张亮度，并控制过渡带、笔画粗细和已有明暗环。当前部署曲线的总调整软限幅为24级；新方案研究改变灰阶映射形状，保留原暗底亮字分支，不增加 USM 锐化或全分辨率多轮空间滤波。</p>
<h2>背景与曲线</h2><p>M 来自16×16实际计数均值、小网格两轮横纵[1,2,1]/4平滑和半像素双线性插值。上锚候选 U 先在未平滑均值网格上做7×7灰度闭运算，再使用相同平滑和插值。它只能作为较亮平台的候选，不能当作真实纸面或文字分割。</p>
<pre>p = smoothstep(clip((M-80)/96, 0, 1))
A = max(U-T, 0)
Y ≥ A 或 A=0：F(Y)=Y
否则 t=Y/A：F(Y)=A·t^g·[g-(g-1)t]，g∈{2,3,4}
δnew_dark = p·[F(Y)-Y]
δ = δold_bright + (1-α)·δold_dark + α·δnew_dark
Yout = clamp(Y + δ, 0, 255)，最后ties-even取整</pre>
<p>原暗、亮两分支在 M 下互斥，其各自软限幅之和与当前部署数学公式一致。α=0显式返回当前公式结果。新暗侧在黑点和背景锚处连续；背景锚处的一阶导数为1。固定锚时灰阶次序不反转，但空间变化的背景仍会形成过渡副作用。</p>
<h2>实际生成的分支</h2><table><tr><th>分支</th><th>该次输出绑定的参数</th></tr>__PARAMETERS__</table>
<p>低强度分支用于检查机制，不代表目标对比度。均值锚和上锚需在相同强度下比较，避免把锚变化与增益变化混为一谈。未通过门槛的强档不会伪装成可用结果。</p>
<h2>固定锚曲线示意（不是照片处理器）</h2><p><label>均值背景 M <input id="demoM" type="range" min="40" max="240" value="200"></label><label>上锚高于 M <input id="demoU" type="range" min="0" max="48" value="0"></label><label>保护带 T <input id="demoT" type="range" min="4" max="24" value="12"></label><span id="demoValue"></span></p><canvas id="curvePlot" width="850" height="480"></canvas><p>灰线为恒等映射，蓝线为当前公式，绿线为 g3 均值锚，橙线为 g3 上锚。固定背景曲线单调不代表相邻像素的背景相同；提高 U 仍可能暗化原来接近 M 的纸面。</p>
<h2>已发现的限制与验证</h2><p>初始 T=4 的强档在合成暗环和斜边上失败，因此保留失败记录并扩大近背景保护带；每次还要检查中灰暗化保留量，不能靠接近恒等映射来“通过”。测试包括网格相位、不同宽度/占空比、灰纸/暗底、弱明暗环、平台内波瓣、缓慢阴影、实际输出平台的10–90宽度和50%位置，以及黑位合并与RGB裁剪。主观效果仍由评审人判断。</p>
<p>均值与极性可能被宽笔画、阴影或图案污染；闭运算有固定结构尺度和方向偏好。上锚高于均值时，保留的亮支与新暗支可能共同激活并抵消，收据记录该现象。曲线还可能改变抗锯齿灰阶覆盖率，因此不能宣称保持所有线宽或消除锯齿。</p>
<h2>手机计算量与实现路线</h2><p>全分辨率主阶段仍是一次灰阶映射；整数幂可用乘法和安全倒数实现，额外上锚操作只在1/16网格。二维LUT可作为实现候选，但128–256KB的数据相关查表在NEON上存在缓存和gather代价，不默认更快。本轮没有改手机C++，也没有实机耗时；桌面计时包含验证与预览，不能换算到140ms。</p>
<script>
// @luna-comment Input:three illustrative sliders. Output:fixed-anchor transfer curves. Key Steps:evaluate current and g3 maps, draw axes and legends. Key Params:not fitted to PS and not used to generate any photo. State/Dependencies:standalone canvas only.
function drawCurveDemo(){const m=Number(document.getElementById('demoM').value),u=Math.min(255,m+Number(document.getElementById('demoU').value)),T=Number(document.getElementById('demoT').value),p0=Math.max(0,Math.min(1,(m-80)/96)),p=p0*p0*(3-2*p0),c=document.getElementById('curvePlot'),x=c.getContext('2d'),left=55,top=30,w=750,h=390;document.getElementById('demoValue').textContent=`M=${m} U=${u} T=${T}`;x.clearRect(0,0,c.width,c.height);x.font='14px system-ui';x.fillStyle='#203046';x.fillText('输出亮度',5,20);x.fillText('输入亮度',755,465);x.strokeStyle='#ccd4df';for(let k=0;k<=255;k+=51){const xx=left+k/255*w,yy=top+h-k/255*h;x.beginPath();x.moveTo(xx,top);x.lineTo(xx,top+h);x.moveTo(left,yy);x.lineTo(left+w,yy);x.stroke();x.fillText(String(k),xx-10,top+h+20);x.fillText(String(k),20,yy+4);}for(const [color,label,kind] of [['#89939d','恒等',0],['#2369b1','当前',1],['#168348','g3均值',2],['#ce7320','g3上锚',3]]){x.strokeStyle=color;x.lineWidth=2;x.beginPath();for(let y=0;y<256;y++){const r=y-m,qd=-.75*Math.max(-r-2,0)*p,qb=.75*Math.max(r-2,0)*(1-p),dd=qd/(1+Math.abs(qd)/24),db=qb/(1+Math.abs(qb)/24);let out=y;if(kind===1)out=y+dd+db;if(kind>=2){const a=Math.max((kind===3?u:m)-T,0),t=a>0&&y<a?y/a:0,f=a>0&&y<a?a*t*t*t*(3-2*t):y;out=y+db+p*(f-y);}out=Math.max(0,Math.min(255,out));const xx=left+y/255*w,yy=top+h-out/255*h;if(y===0)x.moveTo(xx,yy);else x.lineTo(xx,yy);}x.stroke();x.fillStyle=color;x.fillText(label,70+(kind*160),20);}}
for(const id of ['demoM','demoU','demoT'])document.getElementById(id).oninput=drawCurveDemo;drawCurveDemo();</script></html>'''
    findings = '''<h2>本轮筛选结果</h2><p>实际采用 T=12；它是在 8、12、16、20 扫描中，均值锚 g3 和上锚 g3 同时通过既定合成门槛的最小值。上锚 g4 仍使暗环额外加深约3.26级、斜边极值增加7级，未生成真实图结果。</p><p>固定 M=U=200、输入 Y=100 时，当前公式输出82，低强度输出81，两个 g3 输出55，即相对当前额外压暗27级。这是映射力度控制，不是实图效果结论。T12 上锚 g3 在24级暗环上的额外加深约1.67级，仍有副作用；通过所列控制不代表所有场景无 halo。</p>'''
    findings += '''<p class="note">Astra 复审发现：上锚 g3 的96像素宽结构，仅改变其与16像素网格的相位，中心调整量跨度就达到10.15级；缓慢阴影控制中的最大变化为18级，尽管95分位仅2级。均值锚 g3 对应最大相位跨度约3.02级、阴影最大变化6级。这些项目未设通过门槛，必须作为已知风险检查实图，不能把所列门槛通过解释成背景稳定、总体安全或已经去 halo。</p>'''
    body = body.replace('<h2>手机计算量与实现路线</h2>', findings+'<h2>手机计算量与实现路线</h2>')
    (OUT/'METHOD_zh.html').write_text(body.replace('__PARAMETERS__', ''.join(parameter_rows)), encoding='utf-8')


def publish():
    """@luna-comment
    Input:completed bounded stage summaries. Output:standalone nine-row viewer, Chinese method and audit artifacts.
    Key Steps:full asset audit, expose only generated arms, retain native sources and optional PS reference links.
    Key Params:no existing gallery edit, no metric-based quality verdict. State/Dependencies:shared offline renderer.
    """
    cases, arms, audit = collect_and_audit()
    models = [{'key': 'lq', 'label': 'LQ 原始照片', 'note': '仅作为输入参照'},
              {'key': 'source', 'label': '模型输出 · 无后处理', 'note': '精确原始输出，不复制或修改'}]
    for arm in arms:
        models.append({'key': arm, 'label': LABELS[arm], 'note': '研究候选；PNG派生Y离线预览；待主观评审，非手机耗时结论。'})
    rows = []
    for case in cases:
        rows.append({'id': case['key'], 'name': case['name'], 'width': case['width'], 'height': case['height'],
                     'lq': case['lq_url'], 'ps_reference': case['ps_reference_url'],
                     'images': {'source': case['source_url'], **{arm: Path(case['outputs'][arm]['path']).as_uri() for arm in arms}}})
    if all(a in arms for a in ('mean_g3', 'upper_g3', 'upper_g4')):
        defaults = ['current', 'mean_g3', 'upper_g3', 'upper_g4']
    elif all(a in arms for a in ('mean_g3', 'upper_g3')):
        defaults = ['current', 'weak', 'mean_g3', 'upper_g3']
    elif 'mean_g3' in arms:
        defaults = ['source', 'current', 'weak', 'mean_g3']
    else:
        defaults = ['source', 'current', 'weak', 'lq']
    data = {'title': '背景锚定对比度 · 九图研究原型', 'subtitle': '3张PS参考组输入＋6张相机回归图\n独立选版本 · 同步缩放和平移',
            'notes': 'PS图只作为对比度强度参考，不拟合像素，不计算PS相似度。\n低强度分支用于机制检查；强档仅在对应合成门槛通过后生成。\n原图及历史结果保持不变。输出是PNG派生Y预览，手机整体≤140ms尚未实测。\n是否达到对比度、halo与线宽要求由人工评审决定。',
            'defaults': ','.join(defaults), 'models': models, 'rows': rows}
    document = render_html(data)
    document = document.replace('<div class="toolbar">', '<div class="toolbar"><a id="psReference" target="_blank" rel="noopener" hidden>打开PS强度参考</a>', 1)
    document = document.replace('<div id="notes">', '<div id="notes"><a href="METHOD_zh.html">中文技术说明与曲线示意</a><br>', 1)
    report_path = OUT/'objective_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        document = document.replace('<div id="notes">', '<div id="notes"><a href="'+html.escape(Path(report['path']).as_uri(), quote=True)+'">客观差异与人工评审报告</a><br>', 1)
    extra = '''<script>
// @luna-comment Input:active pilot row. Output:optional external PS reference link. Key Steps:show only on the three PS rows, label strength-only. Key Params:not a fitted target or metric reference. State/Dependencies:existing viewer sample text and row metadata.
function updatePSReference(){const link=document.getElementById('psReference'),url=DATA.rows[current].ps_reference;link.hidden=!url;if(url)link.href=url;else link.removeAttribute('href');}
new MutationObserver(updatePSReference).observe(document.getElementById('sample'),{childList:true,characterData:true,subtree:true});updatePSReference();</script>'''
    assert document.count('</html>') == 1
    document = document.replace('</html>', extra+'</html>')
    (OUT/'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'index.html').write_text(document, encoding='utf-8')
    write_method(arms, audit)
    (OUT/'publication.json').write_text(json.dumps({'cases': 9, 'arms': arms, 'outputs': audit['outputs_checked'],
        'viewer_sha256': digest(OUT/'index.html'), 'method_sha256': digest(OUT/'METHOD_zh.html')}, indent=2), encoding='utf-8')
    print(str(OUT/'index.html'), flush=True)


if __name__ == '__main__':
    publish()
