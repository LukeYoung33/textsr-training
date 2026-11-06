"""Audit and publish nine-case source-V3-evidence brightening attenuation."""
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
from tools.publish_v3_reference_post_pilot import load_stages
OUT=ROOT/'output/v3_reference_bright_guard_20261028'
OLD=ROOT/'output/v3_reference_post_pilot_20261028'


def audit_results(summary,old_cases):
    """@luna-comment Input:completed9-case guard summary and immutable old stage cases. Output:independent18-image audit including order, paper samples and brightening retention. Key Steps:verify hashes/geometry/cache source, compare full<=weak<=old in row chunks, measure retained positive Y on old-brightened pixels. Key Params:diagnostic bins are not semantic labels. State/Dependencies:read-only old assets and serial image memory."""
    assert summary['case_count']==9
    lookup={c['key']:c for c in old_cases};assert set(lookup)=={c['key'] for c in summary['cases']}
    records=[]
    for case in summary['cases']:
        old=lookup[case['key']]
        for key in ['source_sha256','source_url','width','height','field_cache_sha256']:
            assert case[key]==old[key],key
        field=json.loads(Path(old['field_receipt_path']).read_text(encoding='utf-8'))
        assert digest(field['field_cache_path'])==case['field_cache_sha256']
        source_path=url_path(case['source_url']);assert digest(source_path)==case['source_sha256']
        source=cv2.imdecode(np.frombuffer(source_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        old_item=old['outputs']['reuse_full'];assert digest(old_item['path'])==old_item['sha256']
        previous=cv2.imdecode(np.frombuffer(Path(old_item['path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        images={};receipts={}
        for arm in ['guard_weak','guard_full']:
            item=case['outputs'][arm];path=Path(item['path'])
            assert path.resolve()==(OUT/arm/(case['key']+'.png')).resolve()
            assert digest(path)==item['sha256']
            receipts[arm]=json.loads(Path(item['receipt_path']).read_text(encoding='utf-8'))
            assert receipts[arm]['source_sha256']==case['source_sha256']
            assert receipts[arm]['output_sha256']==item['sha256']
            assert receipts[arm]['bindings_sha256']==summary['bindings_sha256']
            assert receipts[arm]['field_cache_sha256']==case['field_cache_sha256']
            assert receipts[arm]['field_receipt_sha256']==digest(old['field_receipt_path'])
            assert receipts[arm]['old_reuse_full_sha256']==old_item['sha256']
            assert receipts[arm]['output_calibration_enabled'] is False
            assert receipts[arm]['output_statistics_used_for_candidate_parameters'] is False
            assert receipts[arm]['telemetry']['old_reuse_full_exact_eta0'] is True
            images[arm]=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            assert images[arm].shape==source.shape==previous.shape
        for top in range(0,len(source),128):
            weak=images['guard_weak'][top:top+128];full=images['guard_full'][top:top+128]
            assert np.all(full<=weak) and np.all(weak<=previous[top:top+128])
            base=source[top:top+128].astype(np.int16)
            for result in [weak,full]:
                delta=result.astype(np.int16)-base
                picked=np.take_along_axis(delta,np.argmax(np.abs(delta),axis=2)[:,:,None],axis=2)
                assert np.array_equal(np.clip(base+picked,0,255).astype(np.uint8),result)
        y=cv2.cvtColor(source,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.int16)
        old_y=cv2.cvtColor(previous,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.int16)
        new_y=cv2.cvtColor(images['guard_full'],cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.int16)
        mask=old_y>y;denom=float((old_y-y)[mask].sum())
        retained=float(np.maximum(new_y-y,0)[mask].sum()/denom) if denom else None
        record={'id':case['id'],'positive_Y_sum_retention_on_old_bright_pixels':retained,
                'old_bright_pixels':int(mask.sum()),'changed_pixels_vs_old_full':int(np.count_nonzero(np.any(images['guard_full']!=previous,axis=2)))}
        if case['id']=='089':
            record['paper_y2750']=[{'x':x,'source':int(y[2750,x]),'old_full':int(old_y[2750,x]),'guard_weak':int(cv2.cvtColor(images['guard_weak'][2749:2751],cv2.COLOR_BGR2YCrCb)[1,x,0]),'guard_full':int(new_y[2750,x])} for x in [520,600,640,720]]
        records.append(record);assert digest(source_path)==case['source_sha256']
        print('audited '+case['key'],flush=True)
    for arm in ['guard_weak','guard_full']:assert {p.stem for p in (OUT/arm).glob('*.png')}==set(lookup)
    return {'cases':9,'outputs':18,'ordered_full_le_weak_le_old':True,'source_files_unchanged':True,'records':records}


def write_method(audit):
    """@luna-comment Input:native audit of18guard images. Output:Chinese hypothesis and limits HTML. Key Steps:explain actual repair movement gate versus blend amount, report089 samples and retained brightening. Key Params:no semantic reliability or mobile timing claim. State/Dependencies:standalone local report."""
    details=html.escape(json.dumps(audit,ensure_ascii=False,indent=2))
    body='<html lang="zh-CN"><meta charset="utf-8"><title>V3矫正证据抑制错误亮化</title><style>body{font:16px/1.8 system-ui;max-width:1000px;margin:30px auto;padding:0 20px}pre{white-space:pre-wrap;background:#eef2f6;padding:15px}</style><h1>V3矫正证据抑制错误亮化</h1><p><a href="index.html">打开九图效果对照</a></p><p>本轮完全跳过输出校准。继续使用原V3的字段、模型输出及曲线，仅抑制部分亮化，不反转方向，也不搬动亮参考H。原图、当前公式和旧版直用结果均保留。</p><pre>a = V3邻域修复的混合系数\nd = max(L0-L3,0)\ne = a * smoothstep(clamp((d-12)/24,0,1))\n新亮化权重 = 原亮化权重 * (1-η*e)\nη=0.35为弱修复，η=1为全修复</pre><p>a大并不保证发生实际修正，所以同时要求L0到L3有明显位移。12和24沿用V3的容差/尺度作为新试验假设；这个门控不是原V3算法，也不是已证实的误分类检测器。暗化分支、原曲线强度、四节点插值、旋转和RGB裁剪预览保持一致。</p><p>η0必须精确回到旧版；无实际位移、无修复混合及纯暗化节点保持旧映射。全部像素满足全修复≤弱修复≤旧全档，不保证修复结果始终高于无后处理图。目标是减少089错误提亮，不是去除原有阴影。暗底亮字可能损失增强，下面报告旧版变亮像素上的正向Y变化总量保留率；它不是文字识别或主观质量评分。手机整体140ms尚未验证，源码注释审阅按此前安排延后。</p><p>下文 source 指无后处理的模型输出，不是相机LQ。089已检查的四个纸面点及指定扫描行恢复，不代表整张纸面修好。含文字的更大ROI里，全修复有1867个像素由相对模型非负变为负，最大负差10；它们尚未区分文字与纸面，不能全部当作暗陷。减少亮化可暴露此前被抵消的暗化，仍需人工查看。</p><h2>独立核查和实际变化</h2><pre>'+details+'</pre></html>'
    (OUT/'METHOD_zh.html').write_text(body,encoding='utf-8')


def publish():
    """@luna-comment Input:completed guard9summary and frozen previous stages. Output:audited7-choice9-row viewer/method/hash receipts. Key Steps:verify old code and new assets, preserve exact source/native coordinates, expose old/current/guard arms. Key Params:old galleries unchanged; optional one-off objective report link. State/Dependencies:shared offline renderer."""
    old_cases,_,old_binding=load_stages();summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    binding=summary['bindings_sha256']
    assert binding['old_bindings']==old_binding
    assert binding['guard_core']==digest(ROOT/'tools/v3_reference_bright_guard.py')
    assert binding['guard_runner']==digest(ROOT/'tools/run_v3_reference_bright_guard.py')
    assert binding['prior_weak_summary']==digest(OLD/'weak_summary.json')
    assert binding['prior_full_summary']==digest(OLD/'full_summary.json')
    audit=audit_results(summary,old_cases)
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    lookup={c['key']:c for c in old_cases}
    models=[{'key':k,'label':label,'note':note} for k,label,note in [
        ('lq','原始LQ','相机输入'),('v3_input','V3预处理输入','同源模型输入'),('source','V3M Full · 无后处理','原模型结果'),('current','当前部署公式','已有输出统计对照'),('old_full','旧版V3参考直用 · 全档','089错误亮带的旧候选'),('guard_weak','V3矫正证据 · 弱修复35%','只抑制不可信亮化，无输出校准'),('guard_full','V3矫正证据 · 修复强度100%','保守试验；暗底亮字收益可能降低')]]
    rows=[]
    for case in summary['cases']:
        old=lookup[case['key']]
        rows.append({'id':case['id'],'name':case['name'],'width':case['width'],'height':case['height'],'lq':old['lq_url'],'images':{'source':old['source_url'],'v3_input':old['v3_input_url'],'current':Path(old['outputs']['current']['path']).as_uri(),'old_full':Path(old['outputs']['reuse_full']['path']).as_uri(),**{a:Path(case['outputs'][a]['path']).as_uri() for a in ['guard_weak','guard_full']}}})
    data={'title':'V3矫正证据 · 亮带修复试验','subtitle':'089先验证 / 同源九图回归\n无输出校准 / 旧结果保留','notes':'只降低有大幅V3暗参考矫正证据的亮化。\n不去除真实阴影；不保证复杂背景质量。\n暗底亮字收益需人工比较；手机140ms未测。','defaults':'source,old_full,guard_weak,guard_full','models':models,'rows':sorted(rows,key=lambda c:c['id'])}
    doc=render_html(data).replace('<div id="notes">','<div id="notes"><a href="METHOD_zh.html">中文方法与实际变化</a><br>',1)
    if (OUT/'objective_report.json').exists():
        report=json.loads((OUT/'objective_report.json').read_text(encoding='utf-8'))
        doc=doc.replace('<div id="notes">','<div id="notes"><a href="'+html.escape(Path(report['path']).as_uri(),quote=True)+'">客观差异与人工评审报告</a><br>',1)
    (OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'index.html').write_text(doc,encoding='utf-8');write_method(audit)
    (OUT/'publication.json').write_text(json.dumps({'cases':9,'outputs':18,'viewer_sha256':digest(OUT/'index.html'),'method_sha256':digest(OUT/'METHOD_zh.html')},indent=2),encoding='utf-8')
    print(OUT/'index.html',flush=True)


if __name__=='__main__':publish()
