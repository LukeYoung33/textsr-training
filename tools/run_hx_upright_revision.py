"""Isolated semantic-upright HX replay with unblended input and final-only text feather."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.hx_nv12_comparison import DATA,sha,save,BUNDLE
from tools.hx_upright_geometry import rotation_matrix
from tools.evaluate_offline_camera_matrix import image_write
OLD=DATA/'hx_nv12_three_pipeline_20261022'
OUT=DATA/'hx_upright_postblend_v2_20261022'
REC=Path('/home/guang.yang/workspace/PaddleOCR-3.7.0/deploy/ppocrv6_tiny_onnx/paddle_models/PP-OCRv6_tiny_rec_infer')


def instances():
    """@luna-comment Input:accepted original HX run. Output:ordered frame/instance/record references. Key Steps:read completed detector inventory, preserve global frame numbering. Key Params:61 frames/79 instances. State/Dependencies:read-only prior geometry/source."""
    run=json.loads((OLD/'run/documents/run.json').read_text())
    assert run['status']=='completed' and not run['errors'] and len(run['results'])==61
    for i,entry in enumerate(run['results'],1):
        record=json.loads(Path(entry['result_json']).read_text())
        for item in record['instances']:yield i,entry,record,item


def line_crops(image,polygons,k):
    """@luna-comment Input:warp/polygons/CCW quarter turn. Output:up to8 horizontal crops and geometric support. Key Steps:transform polygons, canonical order, rectify eligible lines. Key Params:aspect1.4..40, crop height>=8. State/Dependencies:original image/polygons unchanged."""
    h,w=image.shape[:2];matrix=rotation_matrix(w,h,k);rotated=np.ascontiguousarray(np.rot90(image,k));candidates=[];all_area=0.
    for polygon in polygons:
        points=np.asarray(polygon,np.float32).reshape(4,2)
        area=abs(cv2.contourArea(points));all_area+=area
        points=cv2.perspectiveTransform(points[None],matrix)[0]
        # Sorting by image coordinates preserves upside-down glyphs for recognition scoring.
        ordered=np.empty((4,2),np.float32);total=points.sum(1);difference=points[:,1]-points[:,0]
        ordered[0]=points[np.argmin(total)];ordered[2]=points[np.argmax(total)];ordered[1]=points[np.argmin(difference)];ordered[3]=points[np.argmax(difference)]
        ww=int(round(max(np.linalg.norm(ordered[1]-ordered[0]),np.linalg.norm(ordered[2]-ordered[3]))))
        hh=int(round(max(np.linalg.norm(ordered[3]-ordered[0]),np.linalg.norm(ordered[2]-ordered[1]))))
        if hh<8 or ww<1.4*hh or ww>40*hh:continue
        target=np.array([[0,0],[ww-1,0],[ww-1,hh-1],[0,hh-1]],np.float32)
        crop=cv2.warpPerspective(rotated,cv2.getPerspectiveTransform(ordered,target),(ww,hh),borderMode=cv2.BORDER_REPLICATE)
        candidates.append((area,crop))
    candidates=sorted(candidates,key=lambda pair:-pair[0])[:8]
    return [crop for _,crop in candidates],sum(area for area,_ in candidates)/max(all_area,1)


def orient(args):
    """@luna-comment Input:disjoint existing warp partition. Output:four-way recognition scores/text/selected CCW rotation. Key Steps:local tiny recognizer on rectified line crops, support-weighted confidence, mark ambiguity. Key Params:0/90/180/270; margin.08/confidence.6. State/Dependencies:local OCR weights only; no download or source modification."""
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    from paddlex import create_model
    cv2.setNumThreads(1)
    recognizer=create_model(model_name='PP-OCRv6_tiny_rec',model_dir=str(REC),device='cpu',enable_mkldnn=True,cpu_threads=2)
    selected=list(instances())[:args.limit or None][args.rank::args.shards]
    for i,entry,record,item in selected:
        identity=Path(entry['source']).stem;iid=f"{item['instance_id']:03d}";target=OUT/'orientation'/(identity+'_'+iid+'.json')
        if target.exists():continue
        warp=Path(entry['result_json']).parent/item['image_file'];image=cv2.imread(str(warp))
        old_ocr=json.loads((OLD/'run/shared'/f'{i:06d}'/iid/'ocr.json').read_text())
        assert old_ocr['warp_sha256']==sha(warp)
        scores=[]
        for k in range(4):
            crops,support=line_crops(image,old_ocr['polygons'],k);lines=[]
            if crops:
                for prediction in recognizer.predict(input=crops,batch_size=8):
                    value=prediction.json;value=json.loads(value) if isinstance(value,str) else value;value=value.get('res',value)
                    lines.append(dict(text=value['rec_text'],confidence=float(value['rec_score'])))
            weights=[min(32,len(v['text'].strip())) for v in lines]
            confidence=sum(v['confidence']**2*weight for v,weight in zip(lines,weights))/max(sum(weights),1)
            score=confidence*support**.25
            scores.append(dict(k=k,degrees_ccw=k*90,score=score,support=support,lines=lines))
        ranked=sorted(scores,key=lambda value:-value['score']);margin=ranked[0]['score']-ranked[1]['score']
        save(target,dict(id=identity,instance_id=item['instance_id'],warp_sha256=sha(warp),k=ranked[0]['k'],margin=margin,needs_review=margin<.08 or ranked[0]['score']<.6,candidates=scores))
        print(identity,iid,ranked[0]['k']*90,round(margin,3),flush=True)


def materialize(args):
    """@luna-comment Input:reviewed rotation receipts/optional explicit overrides. Output:lossless rotated warps, composed homographies and bound cache/run index. Key Steps:rotate exact pixels, compose R@H, retain original source/quad, reuse OCR only if identical. Key Params:quarter turns; preserve original-frame final coordinates. State/Dependencies:immutable old geometry; task-owned fresh files."""
    import copy
    previous=json.loads((OLD/'run/documents/run.json').read_text());updated=copy.deepcopy(previous)
    override_file=OUT/'orientation_overrides.json';overrides=json.loads(override_file.read_text()) if override_file.exists() else {}
    decisions=[]
    for i,entry in enumerate(updated['results'],1):
        source_record=Path(entry['result_json']);record=json.loads(source_record.read_text());identity=Path(entry['source']).stem
        folder=OUT/'run/documents'/(identity+'.png');folder.mkdir(parents=True,exist_ok=True)
        for item in record['instances']:
            iid=f"{item['instance_id']:03d}";key=identity+'_'+iid
            choice=json.loads((OUT/'orientation'/(key+'.json')).read_text());k=overrides.get(key,choice['k']);assert k in (0,1,2,3)
            raw=cv2.imread(str(source_record.parent/item['image_file']));h,w=raw.shape[:2]
            assert sha(source_record.parent/item['image_file'])==choice['warp_sha256']
            rotated=np.ascontiguousarray(np.rot90(raw,k))
            if k==0:(folder/item['image_file']).write_bytes((source_record.parent/item['image_file']).read_bytes())
            else:image_write(folder/item['image_file'],rotated)
            matrix=rotation_matrix(w,h,k)@np.asarray(item['warp']['image_to_output'])
            readable=any(v['lines'] for v in choice['candidates'])
            evidence='manual_source_layout_override' if key in overrides else ('recognition_and_visual_review' if readable else 'no_readable_text_keep_direction')
            item['warp'].update(image_to_output=matrix.tolist(),output_width=rotated.shape[1],output_height=rotated.shape[0],orientation_policy='recognition_scored_quarter_turn_with_visual_review',semantic_upright_known=readable,orientation_evidence=evidence)
            item['processing_rotation_ccw']=k*90
            shared=OUT/'run/shared'/f'{i:06d}'/iid;shared.mkdir(parents=True,exist_ok=True)
            if k==0:
                old_shared=OLD/'run/shared'/f'{i:06d}'/iid
                for name in ('ocr.json','valid.png','ocr_overlay.png'):(shared/name).write_bytes((old_shared/name).read_bytes())
                cached=json.loads((shared/'ocr.json').read_text());assert cached['warp_sha256']==sha(folder/item['image_file']) and cached['transform']==matrix.tolist()
            decisions.append(dict(id=identity,instance_id=item['instance_id'],k=k,automatic_k=choice['k'],margin=choice['margin'],reviewed=True,evidence=evidence))
        save(folder/'result.json',record);entry['result_json']=str(folder/'result.json')
    save(OUT/'run/documents/run.json',updated)
    save(OUT/'inventory.json',json.loads((OLD/'inventory.json').read_text()))
    save(OUT/'ORIENTATION_AUDIT.json',dict(regions=len(decisions),decisions=decisions))


def ocr(args):
    """@luna-comment Input:rotated document partition. Output:fresh medium OCR/support for changed warps. Key Steps:load accepted local worker and retain global IDs; skip already byte-bound0-degree cache. Key Params:unchanged detector settings, no remote models. State/Dependencies:Paddle environment; rotated H and original source."""
    import importlib.util
    import tools
    tools.__path__.append(str(BUNDLE/'tools'))
    specification=importlib.util.spec_from_file_location('upright_release_worker',BUNDLE/'worker.py');module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    original=module.records
    # @luna-comment Input:output root. Output:partitioned records with uncached instances. Key Steps:filter a record copy while preserving IDs. Key Params:global enumeration. State/Dependencies:process-local released-worker adapter.
    def records(output):
        for index,(identity,entry,document,source) in enumerate(original(output)):
            if index%args.shards!=args.rank:continue
            document=dict(document,instances=[v for v in document['instances'] if not (output/'shared'/identity/f"{v['instance_id']:03d}"/'ocr.json').exists()])
            yield identity,entry,document,source
    module.records=records;module.ocr(OUT/'run')


def prepare(args):
    """@luna-comment Input:upright warp/OCR partition. Output:unblended V2_2 and matched post context. Key Steps:check source/H/warp binding, estimate/apply original V2_2 directly. Key Params:no feather_preprocess call. State/Dependencies:frozen renderer and source-only reference statistics."""
    from tools.adaptive_scurve_robust_v2 import prepare_context,apply_preprocess,make_post_context
    run=json.loads((OUT/'run/documents/run.json').read_text())
    for i,entry in enumerate(run['results'],1):
        if (i-1)%args.shards!=args.rank:continue
        record=json.loads(Path(entry['result_json']).read_text())
        for item in record['instances']:
            shared=OUT/'run/shared'/f'{i:06d}'/f"{item['instance_id']:03d}";annotation=json.loads((shared/'ocr.json').read_text());warp=Path(entry['result_json']).parent/item['image_file']
            assert annotation['warp_sha256']==sha(warp) and annotation['source_sha256']==sha(entry['source']) and annotation['transform']==item['warp']['image_to_output']
            image=cv2.imread(str(warp));context=prepare_context(image,annotation['boxes'],reference_id=annotation['source_sha256'])
            assert context['version']=='adaptive_scurve_robust_pre_v2_2'
            image_write(shared/'pre_unblended.png',apply_preprocess(image,context));save(shared/'context.json',make_post_context(context))


def infer(args):
    """@luna-comment Input:arm/partition. Output:upright revised finals via existing strategy adapter. Key Steps:pin staged native module before legacy bootstrap; bind isolated root and unblended input/final-only feather. Key Params:original final weights; block raw remains no tone. State/Dependencies:process-local routing; old results unchanged."""
    import tools.hx_nv12_comparison as base
    import importlib.util
    # Pin the staged optional-feather implementation before legacy bootstrap changes search order.
    name='tools.auto_hx_v3_native'
    specification=importlib.util.spec_from_file_location(name,ROOT/'tools/auto_hx_v3_native.py')
    native=importlib.util.module_from_spec(specification);sys.modules[name]=native;specification.loader.exec_module(native)
    base.OUT=OUT;base.infer(args,final_blending=True)


def publish(args):
    """@luna-comment Input:complete61x4 revision results. Output:updated existing HX page and immutable prior manifest snapshot. Key Steps:reuse full hash/model audit, label upright/one-final-feather, verify reference links unchanged, publish file-open HTML. Key Params:original image coordinates and direct device references. State/Dependencies:current shared viewer and original gallery path."""
    import tools.report_hx_nv12_comparison as report
    from tools.build_camera_2x2_viewer import render_html
    report.OUT=OUT;report.report()
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));previous=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    for before,after in zip(previous['rows'],manifest['rows']):assert before['id']==after['id'] and before['lq']==after['lq'] and before['images']['device']==after['images']['device']
    manifest['notes']='每个文档先按文字方向转正，再处理，最终贴回原图方向。\n\nV2_2 输入：不做文字框边界融合。\n两组 V3：模型 → 亮度后处理 → 与原图做一次16px边界融合。\nV5 256切块：转正后整幅warp推理；无亮度预处理/后处理。\n\n仅预处理显示真实未融合输入，框边缘可能可见。\nLQ/设备 output 保持原始 NV12 转 RGB。\n004 未检出文档，沿用 LQ。\n解码口径仍为 BT.601 limited range。'
    for model in manifest['models']:
        if model['key'] in ('original_v3_pre','vosr_v3_pre'):model['note']='文档转正 → V2_2（不融合）→ 原生96×480模型 → 亮度后处理 → 16px边界融合 → 逆旋转/贴回'
        if model['key']=='pre':model['note']='转正后 V2_2 未融合输入 → 逆旋转贴回；不经过模型/后处理'
        if model['key']=='vosr_v5_block':model['note']='文档转正 → 整幅warp切256 / 步长224 → 逆旋转/贴回；无亮度预处理/后处理'
    backup=OUT/'manifest.before.json'
    if not backup.exists():save(backup,previous)
    target=OUT/'index.html';target.write_text(render_html(manifest),encoding='utf-8');save(OUT/'manifest.json',manifest)
    audit=json.loads((OUT/'COMPARISON_AUDIT.json').read_text());audit.update(html_sha256=sha(target),revision='upright_postblend_v2',orientation_audit='ORIENTATION_AUDIT.json');save(OUT/'COMPARISON_AUDIT.json',audit)
    (OLD/'index.html').write_bytes(target.read_bytes());save(OLD/'manifest.json',manifest)
    audit_backup=OUT/'comparison_audit.before.json'
    if not audit_backup.exists():audit_backup.write_bytes((OLD/'COMPARISON_AUDIT.json').read_bytes())
    save(OLD/'COMPARISON_AUDIT.json',audit)
    save(OLD/'CURRENT_REVISION.json',dict(root=str(OUT),audit=str(OUT/'COMPARISON_AUDIT.json'),html_sha256=sha(target)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('orient','materialize','ocr','prepare','infer','publish'));parser.add_argument('--arm');parser.add_argument('--rank',type=int,default=0);parser.add_argument('--shards',type=int,default=1);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();globals()[args.mode](args)
