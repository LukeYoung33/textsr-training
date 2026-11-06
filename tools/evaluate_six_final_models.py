"""Six strict final models on accepted Camera280 and separately labeled sampled parents."""
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
TRAIN=DATA/'original_gt_vosr2u_l1_six_20261023/training_20261023'
OUT=DATA/'six_final_camera280_samples_20261024'
ARMS=tuple(group+'_'+arch for group in ('original_full','vosr2u_l1') for arch in ('v3','v5','sr'))
LABELS={a:('原始GT · 完整损失 · ' if a.startswith('original') else 'VOSR2u GT · L1 only · ')+({'v3':'V3','v5':'V5','sr':'V5compact X2SR'}[a.rsplit('_',1)[1]]) for a in ARMS}


def select_validation():
    """@luna-comment
    Input: fixed583 validation LQ parents. Output: frozen four bright/four dark selection with contact sheet.
    Key Steps: classify LQ contrast/neutral-bright polarity, prefer all available families, fill from other families.
    Key Params: dark median<100; bright>175/chroma<35/dark tail<130; percentile contrast>60; unique identities.
    State/Dependencies: OpenCV/NumPy/manifest; no model results or training-parent substitutions.
    """
    import cv2,numpy as np
    from tools.evaluate_offline_camera_matrix import sha,save_json
    source=DATA/'vosr_2u_block256_v1_20261023/source_view/manifests/val.jsonl'
    rows=[json.loads(x) for x in source.read_text().splitlines()];assert len(rows)==583
    candidates={'dark':[],'bright':[]}
    for row in rows:
        image=cv2.imread(row['files']['lq']);assert image is not None
        small=cv2.resize(image,(320,320));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        low,med,high=np.percentile(gray,[5,50,95]).tolist()
        chroma=np.median(small.max(2).astype(float)-small.min(2))
        polarity='dark' if med<100 else ('bright' if med>175 and chroma<35 and low<130 else None)
        if polarity is not None and high-low>=60:
            candidates[polarity].append(dict(row=row,polarity=polarity,input_percentiles=[low,med,high]))
    chosen=[];identities=set()
    for polarity in ('bright','dark'):
        group=[]
        for family in sorted({r['dataset'] for r in rows}):
            available=[r for r in candidates[polarity] if r['row']['dataset']==family and r['row']['source_identity'] not in identities]
            if available:
                group.append(available[0]);identities.add(available[0]['row']['source_identity'])
        for entry in candidates[polarity]:
            if len(group)>=4:break
            if entry['row']['source_identity'] not in identities:
                group.append(entry);identities.add(entry['row']['source_identity'])
        assert len(group)==4,(polarity,len(group))
        chosen.extend(group)
    sheet=np.full((4*260,720,3),245,np.uint8)
    for i,entry in enumerate(chosen):
        entry['id']=f'{i+1:03d}';row=entry['row'];assert sha(row['files']['lq'])==row['output_sha256']['lq']
        im=cv2.imread(row['files']['lq']);h,w=im.shape[:2];scale=min(350/w,225/h)
        im=cv2.resize(im,(round(w*scale),round(h*scale)));y,x=i//2*260,i%2*360
        sheet[y:y+im.shape[0],x:x+im.shape[1]]=im
        cv2.putText(sheet,entry['id']+' '+row['dataset']+' '+entry['polarity'],(x+5,y+248),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    folder=OUT/'validation';folder.mkdir(exist_ok=True)
    assert cv2.imwrite(str(folder/'selection.jpg'),sheet)
    save_json(folder/'selection.json',dict(source_manifest_sha256=sha(source),selection_basis='LQ only before inference; family coverage where polarity available; four per polarity',samples=chosen))


def sr_restore(rgb,network,device):
    """@luna-comment
    Input: RGB8 full image, strict HR-skip SR network/device. Output: RGB8 native frame and adapter telemetry.
    Key Steps: edge-pad256 tiles, area-average RGB to128, pack original HR skip/LR independently, decode/feather.
    Key Params: stride224,batch8,feather16; no GT, interpolation skip or tonal processing.
    State/Dependencies: torch/OpenCV/native packers; immutable input, isolated accumulation and source hash.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tools.auto_hx_v3_native import positions
    from tools.evaluate_offline_camera_matrix import sha
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv,y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    h,w=rgb.shape[:2]
    grid=[(x,y) for y in positions(h,256,224) for x in positions(w,256,224)]
    total=np.zeros((h,w,3),np.float32);weights=np.zeros((h,w),np.float32)
    axis=np.minimum(np.minimum(np.arange(256)+1,256-np.arange(256))/16,1).astype(np.float32)
    window=axis[:,None]*axis[None,:]
    assert getattr(network,'supports_external_skip_input',False)
    for start in range(0,len(grid),8):
        selected=grid[start:start+8];patches=[]
        for x,y in selected:
            patch=rgb[y:y+256,x:x+256]
            patches.append(np.pad(patch,((0,256-patch.shape[0]),(0,256-patch.shape[1]),(0,0)),mode='edge'))
        batch=torch.from_numpy(np.stack(patches).transpose(0,3,1,2).copy()).to(device).float()/255
        with torch.inference_mode():
            hr=rgb_to_y4uv(batch);lr=rgb_to_y4uv(F.avg_pool2d(batch,2,2))
            assert list(lr.shape[1:])==[6,64,64] and list(hr.shape[1:])==[6,128,128]
            prediction=network(lr,skip_input_y4uv=hr)
            decoded=YUV2RGB(y4uv_to_yuv444(prediction),is_train=True).clamp(0,1)
        assert decoded.shape==batch.shape and torch.isfinite(decoded).all()
        for (x,y),value in zip(selected,decoded.permute(0,2,3,1).cpu().numpy()):
            hh,ww=min(256,h-y),min(256,w-x)
            total[y:y+hh,x:x+ww]+=value[:hh,:ww]*window[:hh,:ww,None]
            weights[y:y+hh,x:x+ww]+=window[:hh,:ww]
    assert (weights>0).all()
    return np.clip(np.rint(total/weights[...,None]*255),0,255).astype(np.uint8),dict(canvas_hw=[256,256],stride_hw=[224,224],patches=len(grid),ocr_cropping=False,feature_rgb_hw=[128,128],original_lq_skip_hw=[256,256],downsample='RGB float area2x before packing',adapter_sha256=sha(__file__))


def load(arm):
    """@luna-comment
    Input: arm with frozen final config/weights. Output: strict eval CUDA network and verified binding.
    Key Steps: explicit bootstrap, compare hashes with completed-training identity, strict state loading.
    Key Params:250K final only; no checkpoint fallback. State/Dependencies: process-local registries/CUDA/NAS.
    """
    import torch,yaml
    from tools.train_vosr2u_sr import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_offline_camera_matrix import sha
    config=OUT/'camera/models'/(arm+'.yml');checkpoint=OUT/'camera/models'/(arm+'.pth')
    binding=json.loads((OUT/'camera/MODEL_BINDINGS.json').read_text())[arm]
    assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    return model,binding


def pilot():
    """@luna-comment
    Input: actual final SR and source-selected256 tile. Output: independent area/skip/quantization pilot receipt.
    Key Steps: compare adapter output with direct original-HR and independently summed2x2 LR model call.
    Key Params: RGB256->128->256; <=1 byte quantization difference; finite exact native output.
    State/Dependencies: strict final network and native decoder; no inference target or model mutation.
    """
    import cv2,numpy as np,torch
    model,binding=load('original_full_sr')
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv,y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    from tools.evaluate_offline_camera_matrix import save_json
    row=json.loads((OUT/'training/selection.json').read_text())['samples'][0]['row']
    image=cv2.cvtColor(cv2.imread(row['files']['lq']),cv2.COLOR_BGR2RGB)[:256,:256].copy()
    result,details=sr_restore(image,model,'cuda')
    batch=torch.from_numpy(image.transpose(2,0,1).copy()).float().cuda()[None]/255
    lr=(batch[:,:,0::2,0::2]+batch[:,:,0::2,1::2]+batch[:,:,1::2,0::2]+batch[:,:,1::2,1::2])/4
    with torch.inference_mode():
        reference=YUV2RGB(y4uv_to_yuv444(model(rgb_to_y4uv(lr),skip_input_y4uv=rgb_to_y4uv(batch))),is_train=True).clamp(0,1)
    reference=np.clip(np.rint(reference[0].permute(1,2,0).cpu().numpy()*255),0,255).astype(np.uint8)
    delta=int(np.abs(result.astype(int)-reference.astype(int)).max())
    assert result.shape==image.shape and delta<=1
    save_json(OUT/'SR_PILOT.json',dict(passed=True,max_byte_difference=delta,details=details,binding=binding,reference='independent2x2 RGB sum and direct original-HR skip; single tile'))


def worker(arm,suite,rank):
    """@luna-comment
    Input: arm/suite/disjoint camera rank. Output: native finals and source/model-bound receipts.
    Key Steps: bootstrap strict model; reuse legacy camera composition with process-local SR adapter or aligned samples.
    Key Params: camera rank0/1; no preprocessing/post tone;017 fixed; sample inputs never use GT.
    State/Dependencies: immutable accepted caches, own output; ordinary restore helper never modified on disk.
    """
    import cv2,numpy as np,torch
    cv2.setNumThreads(1);torch.set_num_threads(2)
    model,binding=load(arm)
    if suite=='camera':
        del model
        from tools import evaluate_vosr2u_v5_ab as legacy
        from tools import evaluate_vosr_six_models as block
        legacy.OUT=OUT/'camera'
        if arm.endswith('_sr'):block.restore_block=sr_restore
        legacy.worker(arm,rank)
        return
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    restore=sr_restore if arm.endswith('_sr') else restore_block
    selection=json.loads((OUT/suite/'selection.json').read_text())
    for entry in selection['samples']:
        row=entry['row'];identity=entry['id'];receipt=OUT/suite/arm/'receipts'/(identity+'.json')
        if receipt.exists():
            saved=json.loads(receipt.read_text());assert saved['binding']==binding and sha(OUT/saved['final'])==saved['final_sha256'];continue
        assert {k:sha(v) for k,v in row['files'].items()}==row['output_sha256']
        source=cv2.imread(row['files']['lq']);assert source is not None
        rgb,details=restore(cv2.cvtColor(source,cv2.COLOR_BGR2RGB),model,'cuda')
        final=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);assert final.shape==source.shape
        path=OUT/suite/arm/'final'/(identity+'.png');image_write(path,final)
        save_json(receipt,dict(id=identity,sample_id=row['sample_id'],binding=binding,source_sha256=row['output_sha256']['lq'],source_hashes=row['output_sha256'],source_hw=list(source.shape[:2]),final=str(path.relative_to(OUT)),final_sha256=sha(path),restoration=details,selection_sha256=sha(OUT/suite/'selection.json'),evaluator_sha256=sha(__file__)))
    save_json(OUT/suite/arm/'COMPLETE.json',dict(images=len(selection['samples']),binding=binding))


def publish(suite):
    """@luna-comment
    Input: completed native results and immutable source/target references. Output: verified synchronized2x2 page.
    Key Steps: verify final hash/size and shared source/geometry; retain explicit training/validation target labels.
    Key Params: six model choices plus LQ and aligned GT choices; no pending-camera GT or quality judgement.
    State/Dependencies: existing inline viewer and mapped-Y assets; old galleries unchanged.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import file_url,render_html
    from tools.evaluate_offline_camera_matrix import sha,save_json
    definitions=[dict(key='lq',label='LQ · 原始输入',note='原分辨率输入')]+[dict(key=a,label=LABELS[a],note='250K最终模型 · 256分块/224步长 · 无亮度预处理/后处理'+(' · 特征128，原尺寸LQ256直通' if a.endswith('_sr') else '')) for a in ARMS]
    camera=suite=='camera'
    if camera:
        source_manifest=json.loads((DATA/'camera_compare_2x2_20261022/manifest.json').read_text())
        rows=[{k:r[k] for k in ('id','name','width','height','lq')} for r in source_manifest['rows']]
    else:
        definitions+= [dict(key='gt_original',label='GT · 原始GT',note='已对齐原始GT；仅展示，不参与推理'),dict(key='gt_vosr',label='GT · '+('VOSR2u新版训练目标' if suite=='training' else '旧版GT_VOSR验证目标'),note='训练与验证版本不同；仅展示，不参与推理')]
        selection=json.loads((OUT/suite/'selection.json').read_text())['samples'];rows=[]
        for entry in selection:
            r=entry['row'];image=Path(r['files']['lq']);target=Path(r['files']['gt']);original=Path(r['original_gt_lineage']['path'])
            assert sha(image)==r['output_sha256']['lq'] and sha(target)==r['output_sha256']['gt'] and sha(original)==r['original_gt_lineage']['sha256']
            with Image.open(image) as im:w,h=im.size
            for p in (target,original):
                with Image.open(p) as im:assert im.size==(w,h)
            label='暗底亮字' if entry['polarity']=='dark' else '亮底暗字'
            rows.append(dict(id=entry['id'],name=label+' · '+r['dataset']+' · '+r['sample_id'],width=w,height=h,lq=file_url(image),images=dict(gt_original=file_url(original),gt_vosr=file_url(target))))
    audit=[]
    for row in rows:
        row.setdefault('images',{});receipts=[]
        for arm in ARMS:
            root=OUT/suite/('results' if camera else '')/arm
            receipt=json.loads((root/'receipts'/(row['id']+'.json')).read_text())
            path=(OUT/'camera' if camera else OUT)/receipt['final']
            assert sha(path)==receipt['final_sha256']
            expected=json.loads((OUT/'camera/MODEL_BINDINGS.json').read_text())[arm]
            assert receipt['binding']['checkpoint_sha256']==expected['checkpoint_sha256']
            assert receipt['binding']['config_sha256']==expected['config_sha256']
            with Image.open(path) as image:assert image.size==(row['width'],row['height'])
            if camera:
                assert receipt['outside_owned_unchanged']
                if row['id']=='017':assert receipt['geometry017_fixed']
            row['images'][arm]=file_url(path);receipts.append(receipt)
        assert len({r['source_sha256'] for r in receipts})==1
        if camera:
            assert len({r['document_record_sha256'] for r in receipts})==1
            assert all([(x['ocr_sha256'],x['warp_sha256']) for x in r['stages']]==[(x['ocr_sha256'],x['warp_sha256']) for x in receipts[0]['stages']] for r in receipts)
        audit.append(dict(id=row['id'],source_sha256=receipts[0]['source_sha256'],finals={a:r['final_sha256'] for a,r in zip(ARMS,receipts)}))
    for arm in ARMS:
        root=OUT/suite/('results' if camera else '')/arm
        assert {p.stem for p in (root/'final').glob('*.png')}=={r['id'] for r in rows}
    title={'camera':'280张相机测试','training':'上次训练集抽样 · 8张','validation':'验证集抽样 · 8张'}[suite]
    notes='六组均为250K最终权重；原始GT完整损失 vs 新版VOSR2u GT仅L1。无亮度预处理/后处理。\n'+('复用文档检测→warp→整幅256分块→残差融合→unwarp；017使用修复区域。' if camera else '已对齐整图直接256切块；不做文档warp。GT仅用于展示，未输入模型。\n'+('这是上次实际抽样的训练集，不能代表泛化表现。' if suite=='training' else '从固定583对验证图按输入明暗选取；VOSR验证GT仍为旧版，原始GT另列。'))
    manifest=dict(title=title+' · 六模型',subtitle='六模型可选 · 同步缩放 / 切图 / 列表 / 前后翻页',notes=notes,defaults=('lq,original_full_v5,vosr2u_l1_v5,original_full_sr' if camera else 'lq,gt_vosr,original_full_v5,vosr2u_l1_v5'),models=definitions,rows=rows)
    save_json(OUT/suite/'manifest.json',manifest)
    (OUT/suite/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save_json(OUT/suite/'AUDIT.json',dict(passed=True,rows=len(rows),finals=len(rows)*6,records=audit,metrics='not_requested'))


def execute(job):
    """@luna-comment
    Input: arm/suite/rank/GPU tuple. Output: checked child exit receipt. Key Steps: isolated explicit command/log.
    Key Params: four concurrent GPU workers maximum, threads2. State/Dependencies: own output and immutable stage.
    """
    arm,suite,rank,gpu=job;log=OUT/'logs'/f'{suite}.{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--suite',suite,'--rank',str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,suite=suite,rank=rank,gpu=gpu,exit_code=result.returncode)


def run():
    """@luna-comment
    Input: six completed training configs and immutable inventories. Output: frozen snapshots, pilots, all suites/pages.
    Key Steps: lock, bind final hashes, preserve old training8, select validation8 before inference; bounded GPU queue.
    Key Params: twelve camera140-frame shards plus twelve8-parent suite workers; no existing process signaled.
    State/Dependencies: Linux/CUDA/NAS and approved cache; resumable receipts reject changed provenance.
    """
    import fcntl,yaml
    from tools.evaluate_offline_camera_matrix import sha,save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (OUT/'camera/models').mkdir(parents=True,exist_ok=True);bindings={}
        for arm in ARMS:
            config=TRAIN/'configs'/(arm+'.yml');cfg=yaml.safe_load(config.read_text());checkpoint=TRAIN/'experiments'/cfg['name']/'models/net_g_250000.pth'
            assert 'End of training' in (TRAIN/'logs'/(arm+'.log')).read_text(errors='replace')
            for source,target in [(config,OUT/'camera/models'/(arm+'.yml')),(checkpoint,OUT/'camera/models'/(arm+'.pth'))]:
                if target.exists():assert sha(target)==sha(source)
                else:shutil.copy2(source,target)
            bindings[arm]=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=sha(config),checkpoint_sha256=sha(checkpoint),adapter_sha256=sha(__file__))
        save_json(OUT/'camera/MODEL_BINDINGS.json',bindings)
        (OUT/'training').mkdir(exist_ok=True)
        old=DATA/'training_vosr2u_v5_ab_polarities_20261023/selection.json'
        target=OUT/'training/selection.json'
        if target.exists():assert sha(target)==sha(old)
        else:shutil.copy2(old,target)
        if not (OUT/'validation/selection.json').exists():select_validation()
        save_json(OUT/'PIPELINE.json',dict(phase='sr_pilot',controller_pid=os.getpid()))
        with (OUT/'pilot.log').open('ab') as stream:
            subprocess.run([sys.executable,'-u',__file__,'pilot'],stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        jobs=[(arm,suite,rank,i%4) for i,(arm,suite,rank) in enumerate([(a,s,0) for s in ('training','validation') for a in ARMS]+[(a,'camera',r) for a in ARMS for r in range(2)])]
        save_json(OUT/'PIPELINE.json',dict(phase='inference',controller_pid=os.getpid(),jobs=jobs))
        with ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(execute,jobs))
        save_json(OUT/'QUEUE_COMPLETE.json',completed)
        assert all(r['exit_code']==0 for r in completed),completed
        for suite in ('camera','training','validation'):publish(suite)
        for arm,binding in bindings.items():
            assert sha(binding['checkpoint'])==binding['checkpoint_sha256'] and sha(binding['config'])==binding['config_sha256']
        (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Six final models</title><h1>六模型效果对比</h1><p><a href="camera/index.html">280张相机测试</a></p><p><a href="training/index.html">上次训练集抽样8张</a></p><p><a href="validation/index.html">验证集抽样8张</a></p>',encoding='utf-8')
        save_json(OUT/'PIPELINE.json',dict(phase='complete',camera_finals=1680,training_finals=48,validation_finals=48))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','worker','publish','pilot'))
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--suite',choices=('camera','training','validation'));parser.add_argument('--rank',type=int,default=0)
    args=parser.parse_args()
    if args.action=='worker':worker(args.arm,args.suite,args.rank)
    elif args.action=='publish':publish(args.suite)
    else:{'run':run,'pilot':pilot}[args.action]()
