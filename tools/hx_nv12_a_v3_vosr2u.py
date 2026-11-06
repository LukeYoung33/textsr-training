"""A completed VOSR2u V3 on the hash-bound reviewed upright HX NV12 set."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OLD=DATA/'hx_upright_postblend_v2_20261022'
OUT=DATA/'hx_nv12_a_v3_vosr2u_20261026'
MODELS=DATA/'v3_val583_abc_camera280_20261026/camera/models'

def worker(rank):
    """@luna-comment Input: disjoint rank0..3 of accepted61 frames. Output: native A V3 finals/stages/receipts.
    Key Steps: strict snapshot/source verification, upright whole-warp block restore, residual ownership unwarp.
    Key Params:256/224/batch8; no tonal transforms or OCR gate; original source-coordinate output.
    State/Dependencies: CUDA/existing composition helper and immutable geometry; task-owned writes only.
    """
    import cv2,numpy as np,torch,yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    from tools.evaluate_vosr_six_models import restore_block
    from tools.doc_warp_composition import inverse_residual
    cv2.setNumThreads(1);torch.set_num_threads(2)
    config=MODELS/'a_v3.yml';checkpoint=MODELS/'a_v3.pth'
    binding=json.loads((MODELS.parent/'MODEL_BINDINGS.json').read_text())['a_v3']
    assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']=='ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    inventory=json.loads((OLD/'inventory.json').read_text())['pairs'];assert len(inventory)==61
    run=json.loads((OLD/'run/documents/run.json').read_text());assert run['status']=='completed' and not run['errors'] and len(run['results'])==61
    for index,entry in enumerate(run['results']):
        if index%4!=rank:continue
        asset=inventory[index]['input'];identity=inventory[index]['id']
        assert Path(entry['source']).stem==identity and sha(entry['source'])==asset['png_sha256']
        source=cv2.imread(entry['source']);assert list(source.shape[:2])==[asset['height'],asset['width']]
        record=json.loads(Path(entry['result_json']).read_text());canvas=source.astype(np.float32);owned=np.zeros(source.shape[:2],bool);stages=[]
        for item in sorted(record['instances'],key=lambda v:(-v['confidence'],v['instance_id'])):
            iid=f"{item['instance_id']:03d}";warp=Path(entry['result_json']).parent/item['image_file']
            shared=OLD/'run/shared'/f'{index+1:06d}'/iid;ocr=json.loads((shared/'ocr.json').read_text())
            assert ocr['warp_sha256']==sha(warp) and ocr['source_sha256']==asset['png_sha256'] and ocr['transform']==item['warp']['image_to_output']
            original=cv2.imread(str(warp));value,details=restore_block(cv2.cvtColor(original,cv2.COLOR_BGR2RGB),model,'cuda')
            predicted=cv2.cvtColor(value,cv2.COLOR_RGB2BGR);valid=cv2.imread(str(shared/'valid.png'),0)
            delta,alpha,take,geometry=inverse_residual(source,original,predicted,item['warp']['image_to_output'],item['quad'],valid,owned)
            canvas+=delta;owned|=take
            stage=OUT/'stages'/identity/iid;image_write(stage/'model.png',predicted)
            stages.append(dict(instance_id=item['instance_id'],input_warp=str(warp),warp_sha256=sha(warp),valid_sha256=sha(shared/'valid.png'),ocr_sha256=sha(shared/'ocr.json'),model=str(stage/'model.png'),model_sha256=sha(stage/'model.png'),restoration=details,geometry=geometry,processing_rotation_ccw=item.get('processing_rotation_ccw',0)))
        final=np.clip(np.rint(canvas),0,255).astype(np.uint8);assert np.array_equal(final[~owned],source[~owned])
        target=OUT/'final'/(identity+'.png');image_write(target,final)
        save_json(OUT/'receipts'/(identity+'.json'),dict(id=identity,binding=binding,raw_nv12_sha256=asset['sha256'],source_sha256=asset['png_sha256'],source_hw=list(source.shape[:2]),document_record_sha256=sha(entry['result_json']),outside_owned_unchanged=True,no_document=not bool(stages),stages=stages,final=str(target),final_sha256=sha(target),adapter_sha256=sha(__file__)))
        save_json(OUT/f'progress.{rank}.json',dict(done=index//4+1,total=len(run['results'][rank::4]),last=identity))

def execute(rank):
    """@luna-comment Input: GPU/shard rank. Output: exit record. Key Steps: isolated log and CUDA environment.
    Key Params: one worker per GPU0..3. State/Dependencies: own process only, existing workloads untouched.
    """
    log=OUT/'logs'/f'{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker',str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    return dict(rank=rank,exit_code=result.returncode)

def run():
    """@luna-comment Input: approved61-input source audit. Output: complete isolated native inference.
    Key Steps: exclusive lock, source inventory hash audit, four CUDA children, assert zero exits.
    Key Params: no stopped/repeat training weights. State/Dependencies: Linux flock/NAS/existing frozen assets.
    """
    import fcntl
    from tools.evaluate_offline_camera_matrix import save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit=json.loads((OUT/'_incoming/SOURCE_AUDIT.json').read_text());assert audit['passed'] and audit['inputs']==61
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(execute,range(4)))
        save_json(OUT/'QUEUE.json',results);assert all(r['exit_code']==0 for r in results),results
        save_json(OUT/'COMPLETE.json',dict(passed=True,images=61))

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]))
    else:run()
