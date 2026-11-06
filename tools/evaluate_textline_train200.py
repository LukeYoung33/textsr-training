"""Historical V3 val583 and three V3 strategies on actual supplied200 JPEGs."""
import argparse
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
OUT=DATA/'textline_val583_train200_20261026'
OLD=DATA/'v3_val583_abc_camera280_20261026'

def prepare():
    """@luna-comment Input: uploaded200JPEGs and fixed manifests. Output: immutable selections/models.
    Key Steps: exact identity/geometry/hash checks, snapshot final weights. Key Params:200/583 counts.
    State/Dependencies: PIL/yaml/NAS; owned outputs only; never replace source JPEG by PNG.
    """
    from PIL import Image
    from tools.evaluate_offline_camera_matrix import sha,save_json
    history=Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/v3_native_four_20261015')
    refs={'textline_v3':(history/'configs/02_displayfit_paired.yml',history/'runs/02_displayfit_paired/models/net_g_250000.pth')}
    for arm in ('a_v3','l1_v3'):refs[arm]=(OLD/'camera/models'/(arm+'.yml'),OLD/'camera/models'/(arm+'.pth'))
    bindings={}
    for arm,paths in refs.items():
        for source,suffix in zip(paths,('.yml','.pth')):
            target=OUT/'models'/(arm+suffix);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            assert sha(target)==sha(source)
        bindings[arm]=dict(config_sha256=sha(paths[0]),checkpoint_sha256=sha(paths[1]),original_config=str(paths[0]),original_checkpoint=str(paths[1]))
    save_json(OUT/'MODEL_BINDINGS.json',bindings)
    base=DATA/'vosr_2u_block256_v1_20261023/source_view/manifests'
    train=[json.loads(x) for x in (base/'train.jsonl').read_text().splitlines()]
    lookup={r['sample_id']:r for r in train};assert len(lookup)==4873
    uploaded=json.loads((OUT/'_incoming/INPUTS.json').read_text())
    rows=[]
    for index,item in enumerate(uploaded,1):
        path=OUT/'_incoming/source200'/item['name'];row=lookup[path.stem]
        assert sha(path)==item['sha256']
        with Image.open(path) as image:size=image.size
        with Image.open(row['files']['lq']) as image:assert image.size==size,(path.name,size,image.size)
        rows.append(dict(id=f'{index:03d}',input=str(path),source_sha256=item['sha256'],row=row))
    assert len(rows)==200
    save_json(OUT/'train200/selection.json',dict(samples=rows))
    val=json.loads((OLD/'validation/selection.json').read_text())['samples'];assert len(val)==583
    for item in val:item.update(input=item['row']['files']['lq'],source_sha256=item['row']['output_sha256']['lq'])
    save_json(OUT/'validation/selection.json',dict(samples=val))

def worker(suite,arm,rank):
    """@luna-comment Input: suite/model and shard0or1. Output: nativePNG and hash-bound receipts.
    Key Steps: strict frozen load; source hash/geometry, source OCR only; unchanged native helpers.
    Key Params:96x480 OCR vs256/224 full frame, no tone transforms. State/Dependencies: CUDA and owned shards.
    """
    import cv2,torch,yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    from tools.auto_hx_v3_native import restore
    from tools.evaluate_vosr_six_models import restore_block
    from tools.build_offline_augmented_dataset import read_annotations,clipped_boxes
    cv2.setNumThreads(1);torch.set_num_threads(2)
    binding=json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    checkpoint=OUT/'models'/(arm+'.pth');config=OUT/'models'/(arm+'.yml')
    assert sha(checkpoint)==binding['checkpoint_sha256'] and sha(config)==binding['config_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    rows=json.loads((OUT/suite/'selection.json').read_text())['samples'][rank::2]
    for done,item in enumerate(rows,1):
        source=Path(item['input']);assert sha(source)==item['source_sha256']
        image=cv2.imread(str(source));assert image is not None
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ocr_hashes={}
        if arm=='textline_v3':
            row=item['row'];ocr=Path(row['files']['ocr']);ocr_hashes['primary']=sha(ocr)
            assert ocr_hashes['primary']==row['output_sha256']['ocr']
            boxes=clipped_boxes(read_annotations(json.loads(ocr.read_text())),rgb.shape)
            extra=DATA/'adaptive_vosr_preprocessed_augmented_v1_20261021/source_view/OCR_ppocrv6_medium'/row['bucket']/(row['sample_id']+'.json')
            if extra.exists():
                ocr_hashes['supplemental']=sha(extra)
                for box in clipped_boxes(read_annotations(json.loads(extra.read_text())),rgb.shape):
                    if not any(abs((box[1]+box[3])-(b[1]+b[3]))<.5*min(box[3]-box[1],b[3]-b[1]) and min(box[2],b[2])>max(box[0],b[0]) for b in boxes):boxes.append(box)
            result,details=restore(rgb,boxes,model,'cuda')
        else:result,details=restore_block(rgb,model,'cuda')
        assert result.shape==rgb.shape
        final=OUT/suite/arm/'final'/(source.stem+'.png')
        image_write(final,cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
        save_json(OUT/suite/arm/'receipts'/(item['id']+'.json'),dict(id=item['id'],sample_id=item['row']['sample_id'],source_sha256=item['source_sha256'],source_hw=list(rgb.shape[:2]),binding=binding,ocr_hashes=ocr_hashes,final=str(final),final_sha256=sha(final),restoration=details))
        save_json(OUT/suite/arm/f'progress.{rank}.json',dict(done=done,total=len(rows)))

def execute(job):
    """@luna-comment Input: suite/arm/shard/GPU tuple. Output: zero-exit child record.
    Key Steps: isolated log and environment, synchronous child wait. Key Params: at most4 pool workers.
    State/Dependencies: own logs/processes only, existing workloads untouched.
    """
    suite,arm,rank,gpu=job;log=OUT/'logs'/f'{suite}.{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--suite',suite,'--arm',arm,'--rank',str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    return dict(suite=suite,arm=arm,rank=rank,exit_code=result.returncode)

def run():
    """@luna-comment Input: authorized583+600 inference scope. Output: completed queue receipt.
    Key Steps: exclusive controller lock, prepare then four-worker job pool, assert all exits.
    Key Params: immutable old results and actualJPEG inputs. State/Dependencies: Linux flock/CUDA/NAS.
    """
    import fcntl
    from tools.evaluate_offline_camera_matrix import save_json
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
        jobs=[(s,a,r,i%4) for i,(s,a,r) in enumerate((s,a,r) for s,arms in [('train200',('textline_v3','a_v3','l1_v3')),('validation',('textline_v3',))] for a in arms for r in (0,1))]
        save_json(OUT/'PIPELINE.json',dict(phase='running',pid=os.getpid(),jobs=jobs))
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(execute,jobs))
        save_json(OUT/'QUEUE.json',results);assert all(r['exit_code']==0 for r in results),results
        save_json(OUT/'PIPELINE.json',dict(phase='complete',new_finals=1183))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('run','worker'));parser.add_argument('--suite');parser.add_argument('--arm');parser.add_argument('--rank',type=int)
    args=parser.parse_args()
    if args.action=='run':run()
    else:worker(args.suite,args.arm,args.rank)
