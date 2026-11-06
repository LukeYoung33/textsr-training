"""Extract seven current V5 finals, evaluate fixed classical presets and publish a portable viewer."""
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.hx_nv12_comparison import DATA,sha,save
from tools.evaluate_offline_camera_matrix import image_write
from tools.doc_warp_composition import inverse_residual
from tools.light_text_postprocess import PRESETS,enhance_bgr
from tools.build_camera_2x2_viewer import render_html
SOURCE=DATA/'hx_upright_postblend_v2_20261022'
OUT=DATA/'hx_v5_light_post_v1_20261022'
IDS=('003','014','027','035','049','055','056')


def run():
    """@luna-comment Input:seven hash-bound V5 receipts and upright intermediates. Output:exact baseline/LQ copies, three fixed preset finals, stage telemetry and standalone viewer. Key Steps:verify provenance, enhance only added warp residual, reuse ownership/unwarp, validate each delta, export relative assets. Key Params:CPU2; original61 gallery untouched; no model rerun/GT. State/Dependencies:existing OCR optional prior and frozen geometry, standalone classical operator."""
    cv2.setNumThreads(2)
    inventory={v['id']:v for v in json.loads((SOURCE/'inventory.json').read_text())['pairs']}
    records=json.loads((SOURCE/'run/documents/run.json').read_text())['results']
    entries={Path(v['source']).stem:v for v in records}
    rows=[];evidence=[]
    for identity in IDS:
        receipt_path=SOURCE/'results/vosr_v5_block/receipts'/(identity+'.json');receipt=json.loads(receipt_path.read_text())
        baseline_file=SOURCE/receipt['final'];assert sha(baseline_file)==receipt['final_sha256']
        assert receipt['binding']['preprocessing']=='none' and receipt['binding']['revision']=='upright_postblend_v2'
        baseline=cv2.imread(str(baseline_file));entry=entries[identity];record=json.loads(Path(entry['result_json']).read_text())
        baseline_copy=OUT/'v5_baseline'/(identity+'.png');baseline_copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(baseline_file,baseline_copy);assert sha(baseline_copy)==receipt['final_sha256']
        lq=Path(inventory[identity]['input']['png']);assert sha(lq)==receipt['source_sha256']
        lq_copy=OUT/'lq'/(identity+'.png');lq_copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(lq,lq_copy)
        images=dict(v5='v5_baseline/'+identity+'.png')
        for preset in PRESETS:
            canvas=baseline.astype(np.float32);owned=np.zeros(baseline.shape[:2],bool);stages=[]
            for item in sorted(record['instances'],key=lambda value:(-value['confidence'],value['instance_id'])):
                iid=f"{item['instance_id']:03d}";shared=SOURCE/'run/shared'/f'{int(identity):06d}'/iid
                annotation=json.loads((shared/'ocr.json').read_text());model_file=SOURCE/'results/vosr_v5_block/stages'/identity/iid/'model.png'
                model=cv2.imread(str(model_file));valid=cv2.imread(str(shared/'valid.png'),0)
                assert annotation['transform']==item['warp']['image_to_output']
                enhanced,telemetry=enhance_bgr(model,preset,annotation['boxes'])
                diff=enhanced.astype(np.int16)-model.astype(np.int16)
                assert np.array_equal(diff[:,:,0],diff[:,:,1]) and np.array_equal(diff[:,:,1],diff[:,:,2])
                assert np.abs(diff).max()<=PRESETS[preset]['max_delta']
                delta,alpha,take,geometry=inverse_residual(baseline,model,enhanced,item['warp']['image_to_output'],item['quad'],valid,owned)
                canvas+=delta;owned|=take
                image_write(OUT/'stages'/preset/identity/(iid+'.png'),enhanced)
                stages.append(dict(instance_id=item['instance_id'],model_sha256=sha(model_file),ocr_sha256=sha(shared/'ocr.json'),telemetry=telemetry,geometry=geometry))
            added=canvas[:,:,0]-baseline[:,:,0].astype(np.float32)
            added=np.rint(np.minimum(np.maximum(added,-baseline.min(2).astype(np.float32)),255-baseline.max(2).astype(np.float32))).astype(np.int16)
            final=(baseline.astype(np.int16)+added[:,:,None]).astype(np.uint8)
            assert np.array_equal(final[~owned],baseline[~owned]) and final.shape==baseline.shape
            target=OUT/preset/(identity+'.png');image_write(target,final);images[preset]=preset+'/'+identity+'.png'
            evidence.append(dict(id=identity,preset=preset,baseline_sha256=receipt['final_sha256'],final_sha256=sha(target),outside_document_unchanged=True,algorithm_seconds=sum(v['telemetry']['seconds'] for v in stages),stages=stages))
        rows.append(dict(id=identity,name=inventory[identity]['input']['name'],width=baseline.shape[1],height=baseline.shape[0],lq='lq/'+identity+'.png',images=images))
        print('completed '+identity,flush=True)
    manifest=dict(title='V5 · 7 张轻量后处理',subtitle='同一算法 · 三档固定参数\n同步缩放 / 框选 / 前后比较',notes='基线：本轮已转正推理的 V5 256切块输出。\n轻度 / 标准 / 增强：传统局部对比度＋限幅锐化。\n仅共享亮度增量，复用现有文字区域，最后平滑贴回。\n不新增 OCR 或深度模型。\n\n所有图片仍为原图坐标。耗时与参数见 METHOD.md / EVALUATION.json。',defaults='v5,mild,standard,strong',models=[dict(key='lq',label='LQ · 原始输入',note='原始 NV12 解码 RGB'),dict(key='v5',label='V5 · 原始切块结果',note='从最新61张版本逐字节提取；未加本次传统后处理'),dict(key='mild',label='传统后处理 · 轻度',note='局部对比度0.40 / 锐化0.35 / 增量上限14'),dict(key='standard',label='传统后处理 · 标准',note='局部对比度0.75 / 锐化0.65 / 增量上限24'),dict(key='strong',label='传统后处理 · 增强',note='局部对比度1.10 / 锐化1.00 / 增量上限32')],rows=rows)
    save(OUT/'manifest.json',manifest);save(OUT/'EVALUATION.json',dict(status='passed',ids=IDS,presets=PRESETS,records=evidence,core_sha256=sha(ROOT/'tools/light_text_postprocess.py')))
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    shutil.copyfile(ROOT/'tools/light_text_postprocess.py',OUT/'light_text_postprocess.py');save(OUT/'presets.json',PRESETS)


if __name__=='__main__':run()
