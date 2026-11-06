"""Five uniformly selected native candidate GT images from each user-named IQ subset."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import digest,save,curve_pair
from tools.prepare_v3pre_training import REFERENCES
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'IQ_data_grouped_560'
OUT=DATA/'iq_seven_subsets_gt_s22_20261029'
DEPLOY=Path('/data/guang.yang/vosr2_u1_tile512')
NAMES=['training_ready_bicubic_sigma5_center','aligned_hybrid_batch_20260730_1x_training','aligned_hybrid_1x_3rd_20260812_fixed_training','aligned_hybrid_2x_1st_20260812_fixed_training','aligned_hybrid_2x_2nd_20260812_fixed_training','aligned_hybrid_batch_20260730_4x_training','aligned_hybrid_4x_2nd_20260812_fixed_training']


def inventory():
    """@luna-comment Input:seven exact lowercase lq directories. Output:population records and35 unique selected paths. Key Steps:reject unexpected nesting;filename sort;five endpoint-inclusive indices. Key Params:round(i*(N-1)/4),no content filter. State/Dependencies:read-only requested IQ roots."""
    populations=[];selected=[]
    for name in NAMES:
        folder=SOURCE/name/'lq';assert folder.is_dir() and not any(p.is_dir() for p in folder.iterdir())
        files=sorted(p for p in folder.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'])
        assert len(files)>=5
        indices=[round(i*(len(files)-1)/4) for i in range(5)];assert len(set(indices))==5
        populations.append(dict(dataset=name,path=str(folder),count=len(files),indices=indices,files=[p.name for p in files]))
        for index in indices:
            selected.append(dict(id=f'{len(selected)+1:03d}',dataset=name,filename=files[index].name,population_index=index,population_count=len(files),source=str(files[index])))
    assert len(selected)==35 and len({r['source'] for r in selected})==35
    return populations,selected


def prepare(row):
    """@luna-comment Input:selected originalRGB path. Output:V3 native input and source/shape/hash receipt. Key Steps:verify source bytes;exact estimate/neighbor full repair/tables/native render;common-channel delta assertion. Key Params:STANDARD strength1,exterior.1,chunk128. State/Dependencies:frozen V3 modules,owned per-image outputs."""
    import cv2,numpy as np
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    cv2.setNumThreads(1);folder=OUT/'images'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    source=Path(row['source']);source_hash=digest(source)
    image=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
    assert image is not None and image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3
    assert cv2.imwrite(str(folder/'lq.png'),image)
    estimate=revision2.estimate(image);q,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
    low,_=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],q,tail,estimate['nodes'])
    packed=dict(nodes=estimate['nodes'],tables=neighbor.tables(low,estimate['fields']['high']),native_shape=list(image.shape[:2]))
    result,telemetry=pre640.render(image,packed,strength=1.,chunk=128)
    delta=result.astype(np.int16)-image.astype(np.int16)
    assert result.shape==image.shape and np.array_equal(delta[...,0],delta[...,1]) and np.array_equal(delta[...,1],delta[...,2])
    assert cv2.imwrite(str(folder/'pre_lq.png'),result)
    assert digest(source)==source_hash
    teacher_name=row['id']+'__'+source.stem+'.png'
    shutil.copyfile(folder/'pre_lq.png',OUT/'source/train/LQ'/teacher_name)
    record=dict(**row,source_sha256=source_hash,lq_sha256=digest(folder/'lq.png'),pre_lq_sha256=digest(folder/'pre_lq.png'),shape_hw=list(image.shape[:2]),teacher_name=teacher_name,preprocess_telemetry=telemetry,preprocess_shared_delta=True)
    save(folder/'input_receipt.json',record);return record


def gpu_inventory():
    """@luna-comment Input:local NVIDIA runtime. Output:index/freeMiB pairs. Key Steps:query CSV and parse integers. Key Params:read-only capacity selection. State/Dependencies:nvidia-smi,no process modification."""
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return [tuple(int(x.strip()) for x in line.split(',')) for line in output.splitlines() if line.strip()]


def main():
    """@luna-comment Input:authorized seven-subset diagnostic. Output:35 stage results,candidateGT aliases and full routing/provenance. Key Steps:four CPU preparations;capacity-bound single teacher worker;independent delivery audit;unchanged old/newS;native/hash checks. Key Params:35,5perroot,u1 seed42 tile512/32,newS2.2. State/Dependencies:isolated output and unmodified deployed teacher,existing training stays active."""
    import cv2
    cv2.setNumThreads(1);started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'COMPLETE.json').exists(),'Refusing completed output overwrite'
    assert {key:digest(ROOT/key) for key in REFERENCES}==REFERENCES
    for split in ['train','val']:(OUT/'source'/split/'LQ').mkdir(parents=True,exist_ok=True)
    populations,selected=inventory();save(OUT/'INVENTORY.json',dict(subsets=populations))
    save(OUT/'STATUS.json',dict(phase='preprocessing',selected=35,completed=0,started_epoch=started))
    inputs=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(prepare,selected):
            inputs.append(row);save(OUT/'STATUS.json',dict(phase='preprocessing',selected=35,completed=len(inputs),seconds=time.time()-started))
    save(OUT/'SAMPLE_SELECTION.json',dict(count=35,per_subset=5,index_rule='round(i*(N-1)/4),i=0..4,sorted filename within each exact lowercase lq root',inventory_sha256=digest(OUT/'INVENTORY.json'),records=inputs))
    while True:
        occupancy=gpu_inventory();gpu,free=max(occupancy,key=lambda v:v[1])
        if free>=9000:break
        save(OUT/'STATUS.json',dict(phase='waiting_gpu',selected=35,preprocessed=35,free_mib=occupancy,required_free_mib=9000,seconds=time.time()-started));time.sleep(15)
    snapshot=Path(__file__).with_name('frozen_displayfit_snapshot.py');text=snapshot.read_text(encoding='utf-8')
    bindings={"DATA=Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261014/full_images')":'DATA=Path('+repr(str(OUT/'source'))+')',"RUN=BASE/'runs/displayfit_20261014'":'RUN=Path('+repr(str(OUT/'run'))+')','if i%4==gpu':'if True'}
    for old,new in bindings.items():assert text.count(old)==1;text=text.replace(old,new)
    adapter=OUT/'run_full_snapshot.py';adapter.write_text(text,encoding='utf-8')
    teacher={str(p):digest(p) for p in [DEPLOY/'src/run_remote58_vosr2.sh',DEPLOY/'src/run_remote58_vosr2.py',DEPLOY/'src/preset/ckpts/VOSR2/args.json',DEPLOY/'src/preset/ckpts/VOSR2/checkpoints/ema_model.safetensors']}
    save(OUT/'BINDING_DIFF.json',dict(snapshot_sha256=digest(snapshot),adapter_sha256=digest(adapter),replacements=bindings,teacher_files=teacher,gpu=gpu,initial_free_mib=occupancy,preprocessing_references=REFERENCES))
    with (OUT/'teacher.log').open('wb') as log:
        process=subprocess.Popen([str(DEPLOY/'orientation_env/bin/python'),'-u',str(adapter),str(gpu)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        while process.poll() is None:
            completed=len(list((OUT/'source/train/GT_VOSR').glob('*.png')))
            save(OUT/'STATUS.json',dict(phase='full_teacher',selected=35,completed=completed,gpu=gpu,free_mib=gpu_inventory(),seconds=time.time()-started));time.sleep(10)
        assert process.returncode==0,('Teacher failed',process.returncode)
    audit_code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import cv2;from tools.run_train558_full_vosr_curve import audit_masks;p=Path(sys.argv[2]);r=json.loads((p/'routing.json').read_text());a=[dict(input=x['input'],**audit_masks(x,p/'model_output',cv2.imread(x['output']))) for x in r];Path(sys.argv[3]).write_text(json.dumps(a))"
    routes={}
    for path in sorted((OUT/'run').glob('worker*/batch*/routing.json')):
        batch=path.parent;audit_path=batch/'composition_audit.json'
        subprocess.run([str(DEPLOY/'orientation_env/bin/python'),'-c',audit_code,str(ROOT),str(batch),str(audit_path)],check=True)
        audits={r['input']:r for r in json.loads(audit_path.read_text())}
        for route in json.loads(path.read_text()):
            done=json.loads((batch/(str(route['index'])+'_done.json')).read_text());assert done['verified'] and not done['quality_gate']
            assert len(route['patches'])==sum(line['large'] for line in route['lines'])
            routes[Path(route['input']).name]=(route,audits[route['input']])
    assert len(routes)==35
    records=[]
    for i,row in enumerate(inputs,1):
        folder=OUT/'images'/row['id'];route,audit=routes[row['teacher_name']]
        image=cv2.imread(route['output']);assert list(image.shape[:2])==row['shape_hw']
        assert cv2.imwrite(str(folder/'vosr.png'),image)
        curve=curve_pair(image,folder)
        gt=OUT/'candidate_gt'/row['dataset']/(Path(row['filename']).stem+'.png');gt.parent.mkdir(parents=True,exist_ok=True)
        assert not gt.exists();os.link(folder/'steep_s.png',gt)
        assert digest(gt)==curve['arms']['steep_s']['sha256']
        record=dict(**row,**curve,teacher_sha256=digest(folder/'vosr.png'),candidate_gt=str(gt),large_lines=len(route['patches']),detected_lines=len(route['lines']),composition_audit=audit,routing=route)
        save(folder/'receipt.json',record);records.append(record)
        save(OUT/'STATUS.json',dict(phase='post_curves',selected=35,completed=i,seconds=time.time()-started));print('COMPLETE_IMAGE',row['id'],row['dataset'],row['filename'],flush=True)
    save(OUT/'COMPLETE.json',dict(passed=True,count=35,subsets=7,per_subset=5,seconds=time.time()-started,selection_sha256=digest(OUT/'SAMPLE_SELECTION.json'),binding_sha256=digest(OUT/'BINDING_DIFF.json'),records=records,parameters=dict(alpha=.75,gamma=2.6,midpoint_derivative=2.2,exterior_slope=.1,usm=False),limitations=['Generative candidate GT requires visual review','Native LQ geometry preserved;dataset1x/2x/4x names are not extra output scaling']))
    save(OUT/'STATUS.json',dict(phase='complete',selected=35,completed=35,seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except Exception as error:save(OUT/'FAILED.json',dict(error=repr(error)));raise
