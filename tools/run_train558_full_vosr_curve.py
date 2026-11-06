"""Full delivered OCR/large-line VOSR route with the previous curve study unchanged."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_train558_vosr_curve as core
from tools.train_v3_edge_pilot import save,digest
OUT=core.DATA/'train558_full_vosr_scurve_20261028'
SNAPSHOT=Path(__file__).with_name('frozen_displayfit_snapshot.py')


def audit_masks(route,model_outputs,final):
    """@luna-comment Input:actual routing/model directory/native BGR final. Output:independent small/unused-area invariance telemetry. Key Steps:rebuild9px small masks and21px polygon/distance12 alpha;compare to full-frame baseline. Key Params:exact documented geometry. State/Dependencies:OpenCV/PIL output PNGs;no new blending policy."""
    import cv2
    import numpy as np
    from PIL import Image
    w,h=route['size']
    with Image.open(model_outputs/(str(route['index'])+'_base.png')) as image:baseline=np.array(image.convert('RGB').crop((0,0,w,h)))
    small=np.zeros((h,w),np.uint8);used=np.zeros((h,w),bool)
    for line in route['lines']:
        if not line['large']:cv2.fillPoly(small,[np.int32(line['polygon'])],255)
    small=cv2.dilate(small,np.ones((9,9),np.uint8))>0
    rebuilt=baseline.astype(np.float32)
    for patch in route['patches']:
        x,y,x2,y2=patch['box'];q=np.array(route['lines'][patch['line']]['polygon'])-[x,y]
        mask=np.zeros((y2-y,x2-x),np.uint8);cv2.fillPoly(mask,[q.astype(np.int32)],255)
        mask=cv2.dilate(mask,np.ones((21,21),np.uint8));alpha=np.minimum(1,cv2.distanceTransform(mask,cv2.DIST_L2,3)/12)
        alpha[small[y:y2,x:x2]]=0;used[y:y2,x:x2]|=alpha>0
        with Image.open(model_outputs/patch['name']) as image:
            restored=np.array(image.convert('RGB').crop((0,0,*patch['target'])).resize((x2-x,y2-y),Image.Resampling.LANCZOS))
        rebuilt[y:y2,x:x2]=rebuilt[y:y2,x:x2]*(1-alpha[:,:,None])+restored*alpha[:,:,None]
    rgb=cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
    assert np.array_equal(rgb[~used],baseline[~used]) and np.array_equal(rgb[small],baseline[small])
    assert np.array_equal(rgb,np.uint8(np.clip(np.rint(rebuilt),0,255)))
    return dict(small_protected_pixels=int(small.sum()),used_paste_pixels=int(used.sum()),small_protection_equal=True,outside_used_equal=True,independent_full_composition_equal=True)


def full_infer(image,arm):
    """@luna-comment Input:native BGR image/raw-or-pre arm. Output:full documented composite plus routing receipt. Key Steps:isolated source roots;bind only DATA/RUN/single-image selection in delivered snapshot;actual OCR+teacher+composition;independent mask audit. Key Params:GPU3,exact document defaults,do not force large eligibility. State/Dependencies:unmodified deployment/OCR environment;no original GT overwrite."""
    import cv2
    import numpy as np
    folder=OUT/'teacher'/arm;data=folder/'source';run=folder/'run'
    for split in ['train','val']:(data/split/'LQ').mkdir(parents=True,exist_ok=True)
    source=data/'train/LQ/batch_3_043-128_000558.png';assert cv2.imwrite(str(source),image)
    text=SNAPSHOT.read_text(encoding='utf-8')
    changes={"DATA=Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261014/full_images')":'DATA=Path('+repr(str(data))+')',"RUN=BASE/'runs/displayfit_20261014'":'RUN=Path('+repr(str(run))+')','if i%4==gpu':'if True'}
    for old,new in changes.items():assert text.count(old)==1; text=text.replace(old,new)
    adapter=folder/'run_full_snapshot.py';adapter.write_text(text,encoding='utf-8')
    save(folder/'BINDING_DIFF.json',dict(source_snapshot_sha256=digest(SNAPSHOT),adapter_sha256=digest(adapter),replacements=changes,policy='Only isolated source/output roots and selection of the one owned image;all OCR/inference/composition statements unchanged'))
    command=[str(core.DEPLOY/'orientation_env/bin/python'),'-u',str(adapter),'3'];started=time.time()
    core.status('full_ocr_teacher_pipeline',arm=arm)
    if not (run/'worker3/batch0000/0_done.json').exists():
        with (folder/'pipeline.log').open('wb') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
    batch=run/'worker3/batch0000';routes=json.loads((batch/'routing.json').read_text());assert len(routes)==1
    route=routes[0];done=json.loads((batch/'0_done.json').read_text());assert done['verified'] and done['quality_gate'] is False
    assert len(route['patches'])==sum(line['large'] for line in route['lines'])
    for line in route['lines']:
        assert line['large']==bool(line['score']>=.6 and line['height_estimate']>100 and abs(line['angle'])<15)
    for patch in route['patches']:
        line=route['lines'][patch['line']];assert abs(patch['scale']-max(.25,64/line['height_estimate']))<1e-12
        assert all(size>=512 and size%16==0 for size in patch['model_input'])
    output=cv2.imread(route['output'],cv2.IMREAD_UNCHANGED);assert output is not None and output.shape==image.shape
    # Match the delivery OpenCV version for exact distance-transform rounding.
    audit_code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import cv2;from tools.run_train558_full_vosr_curve import audit_masks;p=Path(sys.argv[2]);r=json.loads((p/'routing.json').read_text())[0];v=audit_masks(r,p/'model_output',cv2.imread(r['output']));v['opencv_version']=cv2.__version__;Path(sys.argv[3]).write_text(json.dumps(v))"
    audit_path=folder/'composition_audit.json'
    subprocess.run([command[0],'-c',audit_code,str(ROOT),str(batch),str(audit_path)],check=True)
    invariants=json.loads(audit_path.read_text())
    path=OUT/('vosr_'+arm+'.png');assert cv2.imwrite(str(path),output)
    baseline=cv2.imread(str(batch/'model_output/0_base.png'))[:image.shape[0],:image.shape[1]]
    assert cv2.imwrite(str(OUT/('baseline_'+arm+'.png')),baseline)
    receipt=dict(command=command,seconds=time.time()-started,native_hw=list(image.shape[:2]),padded_hw=list(reversed(route['baseline_model_input'])),final_sha256=digest(path),u=1,infer_steps=1,seed=42,align_method='wavelet',tile_size=512,tile_overlap=32,vae_tile_size=512,vae_tile_overlap=32,full_documented_pipeline=True,ocr_model='PP-OCRv5_mobile_det',detected_lines=len(route['lines']),large_lines=len(route['patches']),small_lines=sum(not line['large'] for line in route['lines']),routing=route,routing_sha256=digest(batch/'routing.json'),snapshot_sha256=digest(SNAPSHOT),adapter_sha256=digest(adapter),invariants=invariants)
    save(folder/'receipt.json',receipt);print(json.dumps(dict(arm=arm,lines=receipt['detected_lines'],large=receipt['large_lines'],seconds=receipt['seconds'])),flush=True)
    return output,receipt


def main():
    """@luna-comment Input:explicit authorized corrected diagnostic launch. Output:full OCR teacher+same curve study in separate root. Key Steps:process-local inference binding;unchanged source and postcurve core;correct metadata after completion. Key Params:no old output replacement. State/Dependencies:frozen delivered snapshot and native invariants."""
    core.OUT=OUT;core.infer=full_infer
    core.main()
    receipt=json.loads((OUT/'COMPLETE.json').read_text())
    receipt['full_documented_pipeline']=True
    receipt['limitations']=[s for s in receipt['limitations'] if s!='VOSR direct padded whole-image;no historical large-line OCR rerouting']
    receipt['limitations'].append('Actual OCR eligibility determines large-line routing;zero eligible lines is not forced')
    save(OUT/'COMPLETE.json',receipt)


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);save(OUT/'FAILED.json',dict(error=repr(error)));raise
