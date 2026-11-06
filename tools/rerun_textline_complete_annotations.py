"""Rerun historical V3 using complete aligned annotations, without training visibility gates."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OLD=DATA/'textline_val583_train200_20261026'
OUT=DATA/'textline_complete_annotations_20261026'

def annotation_boxes(document,shape):
    """@luna-comment Input: existing aligned annotation JSON/native shape. Output: complete boxes/audit.
    Key Steps: clone annotation entries, remove training-only flags, reuse finite geometry parsing/clipping.
    Key Params: no coordinate remapping or expansion; invalid/outside rectangles excluded geometrically only.
    State/Dependencies: existing parser and numpy; immutable original annotations, training behavior untouched.
    """
    import copy
    from tools.build_offline_augmented_dataset import read_annotations,clipped_boxes
    full=copy.deepcopy(document)
    nested=full.get('ocr',full.get('res',full))
    entries=nested.get('annotations',full.get('annotations',[]))
    old=clipped_boxes(read_annotations(document),shape)
    for entry in entries:
        entry.pop('visible',None);entry.pop('ignore',None)
    boxes=clipped_boxes(read_annotations(full),shape)
    assert set(map(tuple,old)).issubset(set(map(tuple,boxes)))
    return boxes,dict(policy='all_valid_aligned_annotations_no_training_flags',raw_annotations=len(entries),old_eligible_boxes=len(old),complete_boxes=len(boxes),training_excluded_recovered=len(boxes)-len(old))

def worker(suite,rank):
    """@luna-comment Input: fixed suite and disjoint rank0or1. Output: complete-annotation native predictions.
    Key Steps: strict same frozen V3 load, hash source/OCR, complete primary/supplement union, native restore.
    Key Params:96x480/64x448/16px feather; no GT input or training-eligibility gate.
    State/Dependencies: CUDA/OpenCV/immutable selections; owned new outputs and per-image receipts only.
    """
    import cv2,numpy as np,torch,yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    cv2.setNumThreads(1);torch.set_num_threads(2)
    binding=json.loads((OLD/'MODEL_BINDINGS.json').read_text())['textline_v3']
    checkpoint=OLD/'models/textline_v3.pth';config=OLD/'models/textline_v3.yml'
    assert sha(checkpoint)==binding['checkpoint_sha256'] and sha(config)==binding['config_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    entries=json.loads((OLD/suite/'selection.json').read_text())['samples'][rank::2]
    for done,item in enumerate(entries,1):
        source=Path(item['input']);assert sha(source)==item['source_sha256']
        image=cv2.imread(str(source));assert image is not None
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);row=item['row']
        ocr=Path(row['files']['ocr']);assert sha(ocr)==row['output_sha256']['ocr']
        boxes,audit=annotation_boxes(json.loads(ocr.read_text()),rgb.shape)
        ocr_hashes=dict(primary=sha(ocr));primary_count=len(boxes)
        extra=DATA/'adaptive_vosr_preprocessed_augmented_v1_20261021/source_view/OCR_ppocrv6_medium'/row['bucket']/(row['sample_id']+'.json')
        if extra.exists():
            extra_boxes,extra_audit=annotation_boxes(json.loads(extra.read_text()),rgb.shape)
            ocr_hashes['supplemental']=sha(extra);audit['supplemental']=extra_audit
            for box in extra_boxes:
                if box not in boxes:boxes.append(box)
        assert boxes
        result,details=restore(rgb,boxes,model,'cuda')
        mask=np.zeros(rgb.shape[:2],np.uint8)
        for x0,y0,x1,y1 in boxes:mask[y0:y1,x0:x1]=1
        assert np.array_equal(result[mask==0],rgb[mask==0]) and result.shape==rgb.shape
        previous=json.loads((OLD/suite/'textline_v3/receipts'/(item['id']+'.json')).read_text())
        assert previous['binding']==binding and previous['source_sha256']==item['source_sha256']
        final=OUT/suite/'final'/(source.stem+'.png');image_write(final,cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
        save_json(OUT/suite/'receipts'/(item['id']+'.json'),dict(id=item['id'],sample_id=row['sample_id'],source_sha256=item['source_sha256'],source_hw=list(rgb.shape[:2]),binding=binding,ocr_hashes=ocr_hashes,annotation_audit=audit,primary_count=primary_count,boxes=boxes,restoration=details,old_final_sha256=previous['final_sha256'],final=str(final),final_sha256=sha(final),adapter_sha256=sha(__file__)))
        save_json(OUT/suite/f'progress.{rank}.json',dict(done=done,total=len(entries)))

def execute(job):
    """@luna-comment Input: suite/rank/GPU tuple. Output: child exit record.
    Key Steps: dedicated log and CUDA environment, bounded child wait. Key Params: four independent shards.
    State/Dependencies: subprocess, own logs only; existing jobs/files untouched.
    """
    suite,rank,gpu=job;log=OUT/'logs'/f'{suite}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker',suite,str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    return dict(suite=suite,rank=rank,exit_code=result.returncode)

def run():
    """@luna-comment Input: authorized corrective783-image scope. Output: four zero-exit shard records.
    Key Steps: exclusive lock, preserve old snapshot metadata, run one shard per idle GPU.
    Key Params:200+583 only; no other arms recomputed. State/Dependencies: Linux flock and own new output.
    """
    import fcntl
    from tools.evaluate_offline_camera_matrix import save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shutil.copy2(OLD/'MODEL_BINDINGS.json',OUT/'MODEL_BINDINGS.json')
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(execute,[('train200',0,0),('train200',1,1),('validation',0,2),('validation',1,3)]))
        save_json(OUT/'QUEUE.json',results);assert all(r['exit_code']==0 for r in results),results
        save_json(OUT/'COMPLETE.json',dict(passed=True,new_finals=783))

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
    else:run()
