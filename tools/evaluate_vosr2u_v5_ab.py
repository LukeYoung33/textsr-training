"""Fast matched Camera280 inference for the completed VOSR2u V5 A/B pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
BASE = DATA / 'vosr_2u_block256_v1_20261023'
OUT = DATA / 'camera_vosr2u_v5_ab_20261023'
OLD = DATA / 'adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020/evaluation'
DOC = OLD / 'camera_docwarp_robust_20261021'
FIX = DATA / 'camera_robust_pre_native_v22_017_fix_20261022/geometry_cache'
INV = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016/inventory.json')


def worker(arm, rank):
    """@luna-comment
    Input: A/B checkpoint snapshot and disjoint rank0/1 over280 exact sources. Output: native PNGs/receipts.
    Key Steps: strict V5 load, verify accepted warp/support (017 fixed), whole-block inference and residual unwarp.
    Key Params:256/224/batch8 via accepted helper; no tonal pre/post or new OCR; two ranks per arm.
    State/Dependencies:CUDA, immutable caches; writes isolated arm outputs, never modifies prior results.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_vosr_six_models import restore_block
    from tools.doc_warp_composition import inverse_residual
    from tools.evaluate_offline_camera_matrix import sha, save_json, image_write
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    config, checkpoint = OUT / 'models' / (arm+'.yml'), OUT / 'models' / (arm+'.pth')
    binding = dict(arm=arm, checkpoint_sha256=sha(checkpoint), config_sha256=sha(config),
        evaluator_sha256=sha(__file__), block_helper_sha256=sha(ROOT/'tools/evaluate_vosr_six_models.py'),
        composition_sha256=sha(ROOT/'tools/doc_warp_composition.py'), inventory_sha256=sha(INV),
        preprocessing='none', postprocessing='none', canvas_hw=[256,256], stride=224)
    expected = json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    assert binding['checkpoint_sha256']==expected['checkpoint_sha256'] and binding['config_sha256']==expected['config_sha256']
    model = build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    rows = json.loads(INV.read_text())['images']
    assert len(rows)==280
    selected = rows[rank::2]
    destination = OUT/'results'/arm
    for index,row in enumerate(selected,1):
        started = time.perf_counter()
        identity = row['id']
        source_file = OLD/'camera_final_20261021/source'/(identity+Path(row['name']).suffix.lower())
        assert sha(source_file)==row['sha256']
        source = cv2.imread(str(source_file))
        assert source is not None
        cache = FIX if identity=='017' else DOC
        record_file = cache/'documents'/source_file.name/'result.json'
        record_hash = sha(record_file)
        record = json.loads(record_file.read_text())
        assert [record['image_height'],record['image_width']]==list(source.shape[:2])
        receipt = destination/'receipts'/(identity+'.json')
        if receipt.exists():
            existing = json.loads(receipt.read_text())
            assert existing['binding']==binding and existing['document_record_sha256']==record_hash
            assert sha(OUT/existing['final'])==existing['final_sha256']
            continue
        canvas,owned = source.astype(np.float32),np.zeros(source.shape[:2],bool)
        stages=[]
        for item in sorted(record['instances'],key=lambda x:(-x['confidence'],x['instance_id'])):
            ocr_file=cache/'ocr'/identity/(f"{item['instance_id']:03d}.json")
            ocr=json.loads(ocr_file.read_text())
            assert ocr['source_sha256']==row['sha256'] and sha(ocr['warp_file'])==ocr['warp_sha256']
            valid_file=cache/ocr['valid_file']
            assert sha(valid_file)==ocr['valid_sha256'] and np.allclose(ocr['transform'],item['warp']['image_to_output'])
            original=cv2.imread(ocr['warp_file'])
            valid=cv2.imread(str(valid_file),0)
            assert original is not None and valid is not None
            prediction,details=restore_block(cv2.cvtColor(original,cv2.COLOR_BGR2RGB),model,'cuda')
            post=cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR)
            delta,alpha,take,geometry=inverse_residual(source,original,post,item['warp']['image_to_output'],item['quad'],valid,owned)
            canvas+=delta
            owned|=take
            stages.append(dict(instance_id=item['instance_id'],ocr_sha256=sha(ocr_file),warp_sha256=ocr['warp_sha256'],restoration=details,geometry=geometry))
        final=np.clip(np.rint(canvas),0,255).astype(np.uint8)
        assert np.array_equal(final[~owned],source[~owned])
        output=destination/'final'/(identity+'.png')
        image_write(output,final)
        save_json(receipt,dict(id=identity,binding=binding,source_sha256=row['sha256'],source_hw=list(source.shape[:2]),
            final=str(output.relative_to(OUT)),final_sha256=sha(output),document_record_sha256=record_hash,
            stages=stages,outside_owned_unchanged=True,geometry017_fixed=(identity=='017'),seconds=time.perf_counter()-started))
        save_json(destination/f'progress.{rank}.json',dict(done=index,total=len(selected),last=identity))
        print(json.dumps(dict(arm=arm,rank=rank,id=identity,done=index,total=len(selected),seconds=round(time.perf_counter()-started,2))),flush=True)
    save_json(destination/f'COMPLETE.{rank}.json',dict(images=len(selected),binding=binding))


def publish():
    """@luna-comment
    Input: complete paired280 finals and original gallery references. Output: checked four-choice2x2 viewer/audit.
    Key Steps: verify every final/source/geometry hash and native size, retain old V5 reference including fixed017.
    Key Params: LQ/old V5/A/B defaults; no metrics or subjective quality labels.
    State/Dependencies:original gallery read-only; writes only isolated manifest/HTML/audit.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import file_url,render_html
    from tools.evaluate_offline_camera_matrix import sha,save_json
    original=json.loads((DATA/'camera_compare_2x2_20261022/manifest.json').read_text(encoding='utf-8'))
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    assert len(original['rows'])==280
    manifest=dict(title='VOSR2u V5 · A/B 280张对比',subtitle='LQ / 上一版 V5 / A组 / B组 · 同步缩放与切图',
        notes='A：原始权重起训；B：上一版V5分块权重起训。两组均为最新VOSR2u数据、250K最终权重。\n无亮度预处理/后处理；文档检测→warp→256分块/224步长→融合→unwarp；017使用修复区域。\n仅提供看图，不计算指标；效果请自行评审。',
        defaults='lq,previous_v5,a_v5,b_v5',models=[
            dict(key='lq',label='LQ · 原始图像',note='原始相机图像'),
            dict(key='previous_v5',label='上一版 V5 · 旧VOSR GT · 256切块',note='B组的预训练模型；原有结果引用'),
            dict(key='a_v5',label='A组 V5 · 新VOSR2u GT · 原始权重起训',note='250K最终权重 · 无预处理 · 256切块'),
            dict(key='b_v5',label='B组 V5 · 新VOSR2u GT · 上一版分块权重起训',note='250K最终权重 · 无预处理 · 256切块')],rows=[])
    audit=[]
    for row in original['rows']:
        item={k:row[k] for k in ('id','name','width','height','lq')}
        item['images']=dict(previous_v5=row['images']['vosr_v5_block'])
        pair=[]
        for arm,key in [('A','a_v5'),('B','b_v5')]:
            receipt=json.loads((OUT/'results'/arm/'receipts'/(row['id']+'.json')).read_text())
            file=OUT/receipt['final']
            assert receipt['binding']['checkpoint_sha256']==bindings[arm]['checkpoint_sha256']
            assert receipt['binding']['config_sha256']==bindings[arm]['config_sha256']
            assert receipt['outside_owned_unchanged'] and sha(file)==receipt['final_sha256']
            with Image.open(file) as image:assert image.size==(row['width'],row['height'])
            item['images'][key]=file_url(file)
            pair.append(receipt)
        assert pair[0]['source_sha256']==pair[1]['source_sha256']
        assert pair[0]['document_record_sha256']==pair[1]['document_record_sha256']
        assert [(x['ocr_sha256'],x['warp_sha256']) for x in pair[0]['stages']]==[(x['ocr_sha256'],x['warp_sha256']) for x in pair[1]['stages']]
        if row['id']=='017':assert all(r['geometry017_fixed'] for r in pair)
        manifest['rows'].append(item)
        audit.append(dict(id=row['id'],source_sha256=pair[0]['source_sha256'],document_record_sha256=pair[0]['document_record_sha256'],final_A=pair[0]['final_sha256'],final_B=pair[1]['final_sha256']))
    for arm in ('A','B'):
        expected={r['id'] for r in manifest['rows']}
        assert {p.stem for p in (OUT/'results'/arm/'final').glob('*.png')}==expected
        assert {p.stem for p in (OUT/'results'/arm/'receipts').glob('*.json')}==expected
    save_json(OUT/'manifest.json',manifest)
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save_json(OUT/'COMPARISON_AUDIT.json',dict(passed=True,images=280,finals=560,models=bindings,records=audit,metrics='not_requested'))


def execute(job):
    """@luna-comment
    Input: arm/rank/GPU job. Output: child exit receipt. Key Steps: exact worker subprocess with isolated log/env.
    Key Params: two processes each on idleGPU0/2; train processes never signaled.
    State/Dependencies: staged immutable source, owned output directories.
    """
    arm,rank,gpu=job
    log=OUT/'logs'/f'{arm}.{rank}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--rank',str(rank)],
            stdout=stream,stderr=subprocess.STDOUT,cwd=str(ROOT),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,rank=rank,gpu=gpu,exit_code=result.returncode)


def run():
    """@luna-comment
    Input: completed A/B250K configs/checkpoints. Output: frozen bindings, four-worker inference and checked viewer.
    Key Steps: snapshot hashes, record ongoing train identities, execute disjoint shards, require all success.
    Key Params: only A/B V5, latest VOSR2u, native280 sources; no metric report.
    State/Dependencies:read-only old training roots and caches; isolated new output, Linux process probes.
    """
    import fcntl
    import yaml
    from tools.evaluate_offline_camera_matrix import sha,save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bindings={}
        (OUT/'models').mkdir(exist_ok=True)
        for arm,folder in [('A','training_comparison_20261023'),('B','training_comparison_prev_pretrain_20261023')]:
            config=BASE/folder/'configs/05_v5_block256_raw.yml'
            options=yaml.safe_load(config.read_text())
            checkpoint=BASE/folder/'experiments'/options['name']/'models/net_g_250000.pth'
            config_hash,weight_hash=sha(config),sha(checkpoint)
            for source,target in [(config,OUT/'models'/(arm+'.yml')),(checkpoint,OUT/'models'/(arm+'.pth'))]:
                if target.exists():assert sha(target)==sha(source)
                else:shutil.copy2(source,target)
            bindings[arm]=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=config_hash,checkpoint_sha256=weight_hash)
        assert bindings['A']['checkpoint_sha256']!=bindings['B']['checkpoint_sha256']
        save_json(OUT/'MODEL_BINDINGS.json',bindings)
        active={}
        for path in [BASE/'training_comparison_20261023/launch/06_v3_block256_raw.json',BASE/'training_comparison_prev_pretrain_20261023/launch/06_v3_block256_raw.json',DATA/'vosr2u_compact_sr_x2_20261023/training_20261023/launch.json']:
            receipt=json.loads(path.read_text())
            command=Path('/proc')/str(receipt['pid'])/'cmdline'
            if command.exists() and command.read_bytes():active[receipt['pid']]=command.read_bytes()
        with ThreadPoolExecutor(max_workers=4) as pool:
            complete=list(pool.map(execute,[('A',0,0),('A',1,2),('B',0,0),('B',1,2)]))
        save_json(OUT/'QUEUE_COMPLETE.json',complete)
        assert all(x['exit_code']==0 for x in complete)
        for pid,command in active.items():
            path=Path('/proc')/str(pid)/'cmdline'
            if path.exists() and path.read_bytes():assert path.read_bytes()==command
        publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker','publish'))
    parser.add_argument('--arm',choices=('A','B'))
    parser.add_argument('--rank',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.mode=='worker':
        if args.arm is None or args.rank is None:parser.error('--arm and --rank required')
        worker(args.arm,args.rank)
    else:
        dict(run=run,publish=publish)[args.mode]()
