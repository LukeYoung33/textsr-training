"""Audit and publish the direct, uncalibrated V3-reference postprocess pilot."""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_camera_2x2_viewer import render_html
from tools.publish_anchored_y_contrast_pilot import digest,url_path
OUT=ROOT/'output/v3_reference_post_pilot_20261028'
IDS=['006','017','051','052','089','099','100','197','251']
LABELS={'current':'当前部署公式 · 输出统计对照','mean_g3':'上一轮均值锚 g3 · 输出统计对照','reuse_weak':'V3参考直用 · 弱档35% · 无校准','reuse_full':'V3参考直用 · 全档 · 无校准'}
CODE_PATHS={'candidate_core':'tools/v3_reference_post.py','runner':'tools/run_v3_reference_post_pilot.py',
    'old_post_core':'tools/anchored_y_contrast.py','old_current_core':'tools/native_y_curve_four_images.py',
    'revision2':'evaluation/document_box_pre640_revision2_20261027/revision2.py',
    'neighbor':'evaluation/document_box_neighbor_20261027/neighbor.py',
    'pre640':'evaluation/document_box_pre640_astra_20261027/pre640.py',
    'prepare_v3pre_training':'tools/prepare_v3pre_training.py',
    'cluster_module':'tools/adaptive_scurve_robust_v2.py',
    'lowres_module':'evaluation/adaptive_prepost_frozen_v1_20261020/lowres_curve.py'}


def load_stages():
    """@luna-comment Input:finished weak/full stage summaries. Output:merged nine bound cases and common bindings. Key Steps:enforce exact IDs/arms and immutable cross-stage source/code identities. Key Params:no inferred source pairing. State/Dependencies:read-only summaries; partial full stage never published."""
    stages=[json.loads((OUT/name).read_text(encoding='utf-8')) for name in ['weak_summary.json','full_summary.json']]
    cases={};arms=[];bindings=stages[0]['bindings_sha256']
    for key,relative in CODE_PATHS.items():assert digest(ROOT/relative)==bindings[key],key
    base=ROOT/'output/v3pre_three_mojito280_20261028'
    for key,filename in [('selection_sha256','selection.json'),('manifest_sha256','manifest.json'),('source_manifest_sha256','SOURCE_MANIFEST.json'),('model_bindings_sha256','MODEL_BINDINGS.json')]:
        assert digest(base/filename)==bindings[key],key
    for stage in stages:
        assert stage['bindings_sha256']==bindings
        assert stage['case_count']==9
        for arm in stage['arms']:
            if arm not in arms:arms.append(arm)
        for case in stage['cases']:
            key=case['key']
            if key not in cases:cases[key]={**case,'outputs':{}}
            target=cases[key]
            for field in ['id','source_url','source_sha256','width','height','lq_url','v3_input_url','field_receipt_path']:
                assert target[field]==case[field],field
            for arm,record in case['outputs'].items():
                assert arm not in target['outputs']
                target['outputs'][arm]=record
    assert sorted(case['id'] for case in cases.values())==IDS
    assert set(arms)==set(LABELS)
    for arm in arms:assert {p.stem for p in (OUT/arm).glob('*.png')}==set(cases)
    return sorted(cases.values(),key=lambda c:c['id']),arms,bindings


def audit_case(case,arms,bindings):
    """@luna-comment Input:one source-bound case plus actual arm names. Output:independent SHA/geometry/common RGB-delta audit and source-Y-bin changes. Key Steps:decode immutable original model output, check receipts, compare clipped common deltas and aggregate eight luminance bins. Key Params:bins are not semantic text masks; no metric feeds candidate parameters. State/Dependencies:serial bounded image memory, Unicode-safe decoding."""
    source_path=url_path(case['source_url']);assert digest(source_path)==case['source_sha256']
    assert url_path(case['lq_url']).is_file() and url_path(case['v3_input_url']).is_file()
    assert Path(case['field_receipt_path']).is_file()
    field_path=Path(case['field_receipt_path'])
    field=json.loads(field_path.read_text(encoding='utf-8'))
    assert field['exact_v3_replay_equal'] and field['exact_v3_replay_max_abs']==0
    assert digest(field['field_cache_path'])==field['field_cache_sha256']==case['field_cache_sha256']
    for key,value in field['v3_code_sha256'].items():assert bindings[key]==value
    with np.load(field['field_cache_path'],allow_pickle=False) as arrays:
        assert np.array_equal(arrays['nodes'],np.r_[np.arange(0,640,8),639])
        assert arrays['low3'].shape==arrays['high'].shape==arrays['population'].shape==(81,81)
        assert np.isfinite(arrays['low3']).all() and np.isfinite(arrays['high']).all()
        assert np.all(arrays['low3']<=arrays['high'])
    selection=json.loads((ROOT/'output/v3pre_three_mojito280_20261028/selection.json').read_text(encoding='utf-8'))
    entry=next(e for e in selection['samples'] if e['id']==case['id'])
    for name,expected in [(entry['preprocess']['source_files']['lq'],field['source_sha256']),(entry['input'],field['v3_stored_sha256'])]:
        prefix='/AI_DNS/guang.yang/TextSR/data/'
        assert name.startswith(prefix)
        assert digest(Path('Y:/guang.yang/TextSR/data')/name[len(prefix):])==expected
    assert field['orientation']==entry['orientation'] and field['stored_shape']==entry['stored_shape']
    source=cv2.imdecode(np.frombuffer(source_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    assert source.shape==(case['height'],case['width'],3)
    y=cv2.cvtColor(source,cv2.COLOR_BGR2YCrCb)[:,:,0]
    records=[]
    for arm in arms:
        item=case['outputs'][arm];path=Path(item['path'])
        assert path.resolve()==(OUT/arm/(case['key']+'.png')).resolve()
        assert digest(path)==item['sha256']
        receipt=json.loads(Path(item['receipt_path']).read_text(encoding='utf-8'))
        assert receipt['source_sha256']==case['source_sha256']
        assert receipt['output_sha256']==item['sha256']
        assert receipt['bindings_sha256']==bindings
        assert receipt['field_cache_sha256']==field['field_cache_sha256']
        assert receipt['field_receipt_sha256']==digest(field_path)
        assert receipt['exact_v3_replay_max_abs']==0
        assert digest(receipt['model_receipt_path'])==receipt['model_receipt_sha256']
        assert digest(receipt['v3_receipt_path'])==receipt['v3_receipt_sha256']
        model_record=json.loads(Path(receipt['model_receipt_path']).read_text(encoding='utf-8'))
        assert model_record['final_sha256']==case['source_sha256']
        assert model_record['input_sha256']==field['v3_stored_sha256']
        assert model_record['source_sha256']==field['source_sha256']
        assert model_record['display_orientation']==field['orientation']
        if arm.startswith('reuse_'):
            assert receipt['output_calibration_enabled'] is False
            assert receipt['output_statistics_used_for_candidate_parameters'] is False
        result=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        assert result.shape==source.shape
        changed=bright=dark=0;maxabs=0
        for top in range(0,len(source),128):
            a=source[top:top+128].astype(np.int16);b=result[top:top+128]
            delta=b.astype(np.int16)-a
            picked=np.take_along_axis(delta,np.argmax(np.abs(delta),axis=2)[:,:,None],axis=2)
            assert np.array_equal(np.clip(a+picked,0,255).astype(np.uint8),b)
            changed+=int(np.count_nonzero(np.any(delta,axis=2)))
            maxabs=max(maxabs,int(np.max(np.abs(delta))))
        ry=cv2.cvtColor(result,cv2.COLOR_BGR2YCrCb)[:,:,0]
        dy=ry.astype(np.int16)-y.astype(np.int16)
        bright=int(np.count_nonzero(dy>0));dark=int(np.count_nonzero(dy<0))
        bins=[]
        for lo in range(0,256,32):
            mask=(y>=lo)&(y<lo+32);count=int(mask.sum())
            bins.append({'source_Y_range':[lo,lo+31],'pixels':count,'mean_delta_Y':float(dy[mask].mean()) if count else None})
        if arm=='current':assert maxabs<=24
        records.append({'id':case['id'],'arm':arm,'output_sha256':item['sha256'],'changed_pixels':changed,'brighter_pixels':bright,'darker_pixels':dark,'max_abs_channel_change':maxabs,'source_Y_bins_not_text_labels':bins})
        print('audited '+case['key']+' '+arm,flush=True)
    assert digest(source_path)==case['source_sha256']
    return records


def write_method(audit):
    """@luna-comment Input:complete artifact audit and diagnostic JSONs. Output:Chinese mechanism/complex-background HTML. Key Steps:separate fixed-field assumptions, synthetic observations, real metrics and manual judgments. Key Params:explicit no calibration/native timing proof. State/Dependencies:source-linked local report only."""
    synthetic=OUT/'synthetic_controls.json'
    evidence=synthetic.read_text(encoding='utf-8') if synthetic.exists() else 'See task-linked synthetic evidence files.'
    extended=OUT/'synthetic_controls_extended.json'
    if extended.exists():evidence+='\nExtended controls:\n'+extended.read_text(encoding='utf-8')
    bins=json.dumps(audit['records'],ensure_ascii=False,indent=2)
    measured_rows=[]
    for row in audit['records']:
        if row['arm']!='reuse_full':continue
        receipt=json.loads((OUT/'receipts'/('cam'+row['id']+'_reuse_full.json')).read_text(encoding='utf-8'))
        telemetry=receipt['telemetry'];fields=telemetry['fields'];render=telemetry['render']
        baseline=next(r for r in audit['records'] if r['id']==row['id'] and r['arm']=='current')
        new_bin=row['source_Y_bins_not_text_labels'][3];old_bin=baseline['source_Y_bins_not_text_labels'][3]
        relative=round(new_bin['mean_delta_Y']-old_bin['mean_delta_Y'],3) if new_bin['pixels'] else None
        measured_rows.append('<tr><td>'+row['id']+'</td><td>'+str(fields['dark_nodes'])+' / '+str(fields['bright_nodes'])+'</td><td>'+str(row['darker_pixels'])+' / '+str(row['brighter_pixels'])+'</td><td>'+str(render['rgb_clipped_pixels'])+'</td><td>'+str(relative)+'</td></tr>')
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V3参考直用：复杂背景适应性与未校准试验</title><style>body{font:16px/1.75 system-ui,"Microsoft YaHei";max-width:1100px;margin:30px auto;padding:0 20px;color:#203046}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd4df;padding:10px;text-align:left}pre{white-space:pre-wrap;background:#edf2f7;padding:15px}.note{background:#fff2cf;padding:15px}a{color:#126c96}</style>\n<h1>V3参考直用：复杂背景适应性与未校准试验</h1><p><a href="index.html">九图原生尺寸对照</a> · <a href="INDEPENDENT_AUDIT.json">独立核查与亮度分段数据</a></p>\n<p class="note">按用户要求，本轮完全跳过模型输出的低分辨率采样校准。候选曲线只由预处理阶段的 L3/H/亮类占比决定。输出 Y 只用于查曲线，所有输出统计仅作事后诊断，不反馈参数。本轮直接检验“模型前后参考一致”假设的影响，不能把该假设写成已证实事实。</p>\n<h2>实际输入和数据关系</h2><p>选择经过 V3 预处理的 V3M Full、Pretrain250K 模型，固定九个场景006、017、051、052、089、099、100、197、251。每个场景的原始输入、旧V3输入和模型输出严格配对。旧记录未保存全部节点字段，因此在原始输入的存储方向复算冻结V3；重现旧V3输入逐像素相同后，才允许复用字段。候选不在模型图上重估。需要旋转的输出先无插值逆转到统计方向，处理后再无插值转回展示方向。</p>\n<h2>如何适应两种明暗方向</h2><pre>S(x)=smoothstep(clamp(x,0,1))\nad=S((p-0.5)/0.25)；ab=S((0.5-p)/0.25)\nspan=H-L3；w=span²/(span²+24²)\nA=max(H-12,0)；t=Y/A\nD(Y)=A*t³*(3-2*t)，当0≤Y&lt;A；其他位置D(Y)=Y\nB(Y)=255-D_mirror(255-Y)，镜像上锚为255-L3\nG(Y)=Y+λ*w*[ad*(D(Y)-Y)+ab*(B(Y)-Y)]\nλ=0.35或1；两档仅混合强度不同</pre>\n<p>亮类占多数时优先暗化；暗类占多数时优先亮化；占比接近一半时处理减弱。低跨度减弱，零跨度恒等回退。这是分布驱动的方向假设，不是文字识别。原始暗参考置信度不能代表亮参考或修复后的整体可信度，本轮不把它当通用门控。</p>\n<p>每个节点的固定曲线单调，两方向是恒等与单调曲线的凸组合。原生像素在同一个输出Y上求四个邻接节点的映射值，然后插值；不先插值参考再构造曲线。空间参考或方向变化仍可产生光晕、亮度不均和笔画变化，单调性不构成这些质量保证。</p>\n<h2>复杂背景适用条件与失效方式</h2><table><tr><th>场景</th><th>预期机制</th><th>必须检查的反例</th></tr>\n<tr><td>亮底暗字</td><td>亮类占多数时，压暗中灰笔画并保护接近H的亮度</td><td>粗字、密集文字或裁剪过紧会改变占比，可能改为亮化或处理不足</td></tr>\n<tr><td>暗底亮字</td><td>暗类占多数时，按L3镜像提亮文字</td><td>大块亮图案会改变占比；错误L3可能连暗背景一起抬亮</td></tr>\n<tr><td>彩色背景、彩字</td><td>只根据Y亮度分布处理，保留已有色度的意图</td><td>等亮异色无法区分；饱和颜色的RGB预览会裁剪通道，不能据此声称原生UV等价或颜色完全不变</td></tr>\n<tr><td>渐变、阴影、材质分界</td><td>节点场随局部明暗分布变化</td><td>Otsu可能区分两块纸面，而非纸与字；V3暗尾修复不自动修正占比和H</td></tr>\n<tr><td>模型输出亮度漂移</td><td>本轮直接使用旧参考，无回校准</td><td>偏移或增益变化可让旧背景落入增强区。固定字段的漂移控制用于量化这个风险，不能消除风险</td></tr></table>\n<h2>比较的含义</h2><p>无后处理、当前部署公式、上一轮均值锚，以及本轮V3直用弱/全档都作用于同一张模型输出。前两个统计型对照允许计算输出背景，但这些信息绝不进入本轮候选。若新旧曲线和方向机制同时改变，实图差异属于方案整体差异，不能全部归因于复用V3统计。</p>\n<p>像素变化量、亮度分段和差异图不自动代表质量改善。亮度分段不是文字分割。197亮字、251框角是已有风险场景；本轮无法恢复在前序处理/模型中已丢失的信息。原PS样例不属于本批同源V3管线，不用于拟合或GT评估。</p>\n<h2>计算量边界</h2><p>工程复用时可保留已有节点信息，避免后处理重复做直方图估计。本次为证明来源而复算V3，这部分主机时间不是未来复用开销；也不能直接从总延时中扣除。参考实现采用81×81×256 float32节点LUT，单套约6.4MiB，四节点查表并不自动适合NEON。部署需比较解析曲线、缓存和查表成本。本轮没有改手机C++，整体140ms尚未实测。</p>\n<h2>已观察到的机制反例</h2><p class="note">在65×65亮底220/暗条80的直方图控制中，条宽由8增加到56时，亮类占比从0.877降至0.138，同一个中灰Y110从映射61.19变成114.77，方向反转。该控制未进行空间邻域修复，用于隔离占比机制。固定预处理参考、仅让模拟模型输出乘0.9再减8时，直用全档又把纸面压暗6级。等亮灰与绿的预览则触发不同的RGB裁剪。这些反例已经否定“复杂背景普遍安全”的判断；实图仅供观察该假设的影响。</p><p class="note">扩展控制还人为构造了“源参考字段发生阶跃，但模拟模型输出均匀Y180”的不匹配场景。弱档产生175–196的亮度范围，相邻最大跳变7；全档产生166–226，相邻最大跳变19。这是空间参考不一致的压力测试，并非实图上已观察到的同等缺陷；它说明四节点插值不能保证不产生伪边。</p><h2>合成机制控制</h2><details><summary>展开原始控制结果；失败必须保留</summary><pre>__SYNTHETIC__</pre></details>\n<h2>实图客观变化</h2><details><summary>展开九图各臂亮度变化（不作质量排名）</summary><pre>__BINS__</pre></details></html>'
    measured='<h2>九图实际分支与变化</h2><p>下表是全档收据的占比方向统计与独立核查的像素亮度变化。节点方向不等于文字类别；变暗、变亮和裁剪数量不作质量排名。</p><table><tr><th>场景</th><th>占比指向暗化 / 亮化节点</th><th>实际变暗 / 变亮像素</th><th>RGB裁剪像素</th><th>源Y96–127区间：相对当前公式的平均Y差</th></tr>'+''.join(measured_rows)+'</table>'
    body=body.replace('<h2>实图客观变化</h2>',measured+'<h2>实图客观变化</h2>')
    (OUT/'METHOD_zh.html').write_text(body.replace('__SYNTHETIC__',html.escape(evidence)).replace('__BINS__',html.escape(bins)),encoding='utf-8')


def publish():
    """@luna-comment Input:both completed nine-case stages. Output:audited native-coordinate comparison/method/publication receipts. Key Steps:audit every output, expose no missing models, link optional objective review. Key Params:direct uncalibrated candidates explicitly marked research. State/Dependencies:existing file-open renderer; original galleries untouched."""
    cases,arms,bindings=load_stages();records=[]
    for case in cases:records.extend(audit_case(case,arms,bindings))
    audit={'case_count':9,'arms':arms,'outputs_checked':len(records),'source_files_unchanged':True,'output_calibration_enabled':False,'records':records}
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    models=[{'key':'lq','label':'原始LQ','note':'相机原始输入'}, {'key':'v3_input','label':'V3预处理输入','note':'实际送入模型的同源V3输入'}, {'key':'source','label':'V3M Full · 无后处理','note':'经过V3预处理的模型输出，保持原文件'}]
    for arm in arms:models.append({'key':arm,'label':LABELS[arm],'note':'研究对照；全程PNG派生Y预览。复杂背景质量和手机耗时待评审。'})
    rows=[{'id':c['id'],'name':c['name'],'width':c['width'],'height':c['height'],'lq':c['lq_url'],'images':{'source':c['source_url'],'v3_input':c['v3_input_url'],**{a:Path(c['outputs'][a]['path']).as_uri() for a in arms}}} for c in cases]
    data={'title':'V3参考直用 · 无输出校准','subtitle':'九图 / 同一V3M Full输出\n两种明暗方向与复杂背景检查','notes':'候选直接复用预处理参考；输出亮度不参与参数校准。\n均值锚和当前公式仅作独立对照。\n全档用于观察假设影响；不代表复杂背景质量通过。\n手机整体140ms尚未实测。','defaults':'source,current,reuse_weak,reuse_full','models':models,'rows':rows}
    document=render_html(data)
    document=document.replace('<div id="notes">','<div id="notes"><a href="METHOD_zh.html">中文方法与复杂背景分析</a><br>',1)
    pointer=OUT/'objective_report.json'
    if pointer.exists():
        report=json.loads(pointer.read_text(encoding='utf-8'))
        document=document.replace('<div id="notes">','<div id="notes"><a href="'+html.escape(Path(report['path']).as_uri(),quote=True)+'">客观差异与人工评审报告</a><br>',1)
    (OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.html').write_text(document,encoding='utf-8');write_method(audit)
    publication={'cases':9,'arms':arms,'outputs':len(records),'viewer_sha256':digest(OUT/'index.html'),'method_sha256':digest(OUT/'METHOD_zh.html'),'bindings_sha256':bindings}
    (OUT/'publication.json').write_text(json.dumps(publication,indent=2),encoding='utf-8');print(OUT/'index.html',flush=True)


if __name__=='__main__':publish()
