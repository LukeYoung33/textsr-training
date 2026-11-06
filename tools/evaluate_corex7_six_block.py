"""Six frozen block models on two exact user-authorized vehicle JPEGs."""
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
OUT=DATA/'corex7_two_photos_six_block_20261026'
ARMS=('original_v3','original_v5','vosr_v3','vosr_v5','vosr2u_v3','vosr2u_v5')

def prepare():
    """@luna-comment Input: two source JPEG inventory and prior six-model audit. Output: frozen strict snapshots.
    Key Steps: source hashes and all model/config hashes match existing280 results, copy immutable model assets.
    Key Params: full-loss250K256 V3/V5 only. State/Dependencies: YAML and existing NAS weights, own writes only.
    """
    import yaml
    from tools.evaluate_offline_camera_matrix import sha,save_json
    refs={}
    for arch in ('v3','v5'):
        refs['original_'+arch]=DATA/'original_gt_vosr2u_l1_six_20261023/training_20261023/configs'/('original_full_'+arch+'.yml')
        arm='06_v3_block256_raw.yml' if arch=='v3' else '05_v5_block256_raw.yml'
        refs['vosr_'+arch]=DATA/'vosr_fullimage_block256_v1_20261021/training_comparison_20261021/configs'/arm
        refs['vosr2u_'+arch]=DATA/'vosr_2u_block256_v1_20261023/training_comparison_20261023/configs'/arm
    expected=json.loads((DATA/'camera280_six_block_comparison_20261026/AUDIT.json').read_text())['records'][0]['groups']
    bindings={}
    for arm in ARMS:
        config=refs[arm];cfg=yaml.safe_load(config.read_text());checkpoint=Path(cfg['path']['experiments_root'])/cfg['name']/'models/net_g_250000.pth'
        assert cfg['network_g']['canvas_hw']==[256,256] and cfg['train']['cp_weight']>0
        assert sha(config)==expected[arm]['binding']['config_sha256'] and sha(checkpoint)==expected[arm]['binding']['checkpoint_sha256']
        for source,suffix in [(config,'.yml'),(checkpoint,'.pth')]:
            target=OUT/'models'/(arm+suffix);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            assert sha(source)==sha(target)
        bindings[arm]=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=sha(config),checkpoint_sha256=sha(checkpoint))
    source=json.loads((OUT/'_incoming/SOURCE.json').read_text(encoding='utf-8'));assert len(source)==2
    for row in source:assert sha(OUT/'_incoming'/row['alias'])==row['sha256']
    save_json(OUT/'MODEL_BINDINGS.json',bindings)

def worker(rank):
    """@luna-comment Input: rank0..3 assigned immutable models. Output: two native predictions per arm.
    Key Steps: strict snapshot load, identical original JPEG decode, unchanged full-frame256/224 helper.
    Key Params: no warp/OCR/tonal transforms; finite and shape assertions in helper. State/Dependencies: CUDA.
    """
    import cv2,torch,yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    cv2.setNumThreads(1);torch.set_num_threads(2)
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());rows=json.loads((OUT/'_incoming/SOURCE.json').read_text(encoding='utf-8'))
    for arm in ARMS[rank::4]:
        binding=bindings[arm];config=OUT/'models'/(arm+'.yml');checkpoint=OUT/'models'/(arm+'.pth')
        assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']
        model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
        for row in rows:
            path=OUT/'_incoming'/row['alias'];assert sha(path)==row['sha256']
            image=cv2.imread(str(path));assert image is not None and list(image.shape[:2])==row['hw']
            value,details=restore_block(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),model,'cuda');assert value.shape==image.shape
            final=OUT/arm/'final'/(row['id']+'.png');image_write(final,cv2.cvtColor(value,cv2.COLOR_RGB2BGR))
            save_json(OUT/arm/'receipts'/(row['id']+'.json'),dict(id=row['id'],source_sha256=row['sha256'],source_hw=list(image.shape[:2]),binding=binding,restoration=details,final=str(final),final_sha256=sha(final),adapter_sha256=sha(__file__)))
        del model;torch.cuda.empty_cache()

def execute(rank):
    """@luna-comment Input: shard/GPU rank. Output: child exit record. Key Steps: own logs/environment/child wait.
    Key Params: one worker per GPU, sequential model loads per shard. State/Dependencies: subprocess only.
    """
    log=OUT/'logs'/f'{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker',str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    return dict(rank=rank,exit_code=result.returncode)

def run():
    """@luna-comment Input: two authorized photos/six prior models. Output: complete twelve-image receipt.
    Key Steps: exclusive lock, immutable preparation, four-GPU join and zero-exit gate.
    Key Params: existing datasets/results unaffected. State/Dependencies: Linux flock/CUDA/NAS.
    """
    import fcntl
    from tools.evaluate_offline_camera_matrix import save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
        with ThreadPoolExecutor(max_workers=4) as pool:queue=list(pool.map(execute,range(4)))
        save_json(OUT/'QUEUE.json',queue);assert all(r['exit_code']==0 for r in queue),queue
        save_json(OUT/'COMPLETE.json',dict(passed=True,images=12))

def deliver():
    """@luna-comment Input: twelve complete native outputs. Output: offline local full/plate seven-choice viewers.
    Key Steps: rehash sources/models/native copies, identical source-selected crop rectangles across all groups.
    Key Params: crop only for extra plate page, no resize; originals/finals unchanged. State/Dependencies: cv2/PIL.
    """
    import cv2
    from PIL import Image
    from tools.build_vosr2u_local_three_viewer import digest
    from tools.build_camera_2x2_viewer import render_html
    remote=Path(OUT.as_posix().replace('/AI_DNS/','Y:/',1));local=ROOT/'evaluation/corex7_two_photos_six_block_20261026';view=ROOT/'output/corex7_two_photos_six_block_20261026'
    assert json.loads((remote/'COMPLETE.json').read_text())['passed']
    queue=json.loads((remote/'QUEUE.json').read_text());assert len(queue)==4 and all(r['exit_code']==0 for r in queue)
    bindings=json.loads((remote/'MODEL_BINDINGS.json').read_text());sources=json.loads((remote/'_incoming/SOURCE.json').read_text(encoding='utf-8'))
    rows=[];crops=[];records=[]
    for source in sources:
        path=local/'source'/source['alias'];assert digest(path)==source['sha256']
        image=cv2.imread(str(path));h,w=image.shape[:2];assert [h,w]==source['hw']
        row=dict(id=source['id'],name=source['name'],width=w,height=h,lq=path.resolve().as_uri(),images={})
        x0,y0,x1,y1=[round(v*extent) for v,extent in zip(source['plate_context_xyxy'],[w,h,w,h])]
        crop=dict(id=row['id'],name=row['name'],width=x1-x0,height=y1-y0,images={})
        choices={'lq':path}
        for arm in ARMS:
            receipt=json.loads((remote/arm/'receipts'/(source['id']+'.json')).read_text());assert receipt['binding']==bindings[arm] and receipt['source_sha256']==source['sha256'] and receipt['source_hw']==[h,w]
            assert receipt['restoration']['canvas_hw']==[256,256] and not receipt['restoration']['ocr_cropping']
            original=Path(receipt['final'].replace('/AI_DNS/','Y:/',1));assert digest(original)==receipt['final_sha256']
            with Image.open(original) as im:assert im.size==(w,h)
            target=local/arm/original.name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(original,target)
            assert digest(target)==receipt['final_sha256'];row['images'][arm]=target.resolve().as_uri();choices[arm]=target;records.append(receipt)
        for key,path in choices.items():
            target=local/'plate'/key/(source['id']+'.png');target.parent.mkdir(parents=True,exist_ok=True)
            pixels=cv2.imread(str(path))[y0:y1,x0:x1];assert cv2.imwrite(str(target),pixels,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if key=='lq':crop['lq']=target.resolve().as_uri()
            else:crop['images'][key]=target.resolve().as_uri()
        crop['source_crop_xyxy']=[x0,y0,x1,y1];rows.append(row);crops.append(crop)
    labels={'original':'原始GT','vosr':'VOSR GT','vosr2u':'VOSR2u GT · A组'}
    models=[dict(key='lq',label='LQ · 原图',note='原始JPEG直接解码，无额外处理')]
    for arm in ARMS:
        family,arch=arm.rsplit('_',1);models.append(dict(key=arm,label=arch.upper()+' · '+labels[family],note='相同250K完整损失256切块权重；整图256/224，无亮度预处理/后处理'))
    manifest=dict(title='COREX7 · 昼夜两张 · 六模型切块对比',subtitle='原尺寸整图 / 车牌局部 · 同步缩放与切图',notes='使用刚才280张对比中的同一六组权重与配置。\n两张为车辆场景，直接整幅原图256×256/步长224重叠推理，不使用文档检测/warp、OCR框或亮度预处理/后处理。\n全图与车牌局部页都使用相同像素结果；车牌区域按LQ固定坐标截取，不缩放。\n所有原图、六组结果和车牌裁剪均在本地，打开页面无需Y盘。默认三组V3，窗口下拉可切换V5。',defaults='lq,original_v3,vosr_v3,vosr2u_v3',models=models,rows=rows)
    view.mkdir(parents=True,exist_ok=True);(view/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(view/'index.html').write_text(render_html(manifest),encoding='utf-8')
    plate=dict(manifest,title=manifest['title']+' · 车牌局部',rows=crops);(view/'plate.html').write_text(render_html(plate),encoding='utf-8');(view/'plate_manifest.json').write_text(json.dumps(plate,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(records)==12
    (local/'DELIVERY.json').write_text(json.dumps(dict(passed=True,images=12,records=records,crops=[r['source_crop_xyxy'] for r in crops]),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,images=12,viewer=str(view/'index.html'))))

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(int(sys.argv[2]))
    elif sys.argv[1]=='deliver':deliver()
    else:run()
