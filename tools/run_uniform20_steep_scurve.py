"""Uniform20 source-bound complete teacher and stronger interior-S diagnostic."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'uniform20_steep_scurve_20261028'
MANIFEST=DATA/'v3pre_v3m_three_20261027/pretrain/source_view/manifests/train.jsonl'
DEPLOY=Path('/data/guang.yang/vosr2_u1_tile512')
SNAPSHOT=Path(__file__).with_name('frozen_displayfit_snapshot.py')


def digest(path):
    """@luna-comment Input:existing path. Output:SHA256. Key Steps:stream1MiB chunks. Key Params:bytes unchanged. State/Dependencies:hashlib,read-only file."""
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):result.update(chunk)
    return result.hexdigest()


def save(path,value):
    """@luna-comment Input:owned JSON path/serializable value. Output:atomic UTF8 JSON. Key Steps:write same-directory temp then replace. Key Params:indent2. State/Dependencies:owned isolated output only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temporary,path)


def choose(rows,count=20):
    """@luna-comment Input:unique training manifest records,count>=2. Output:sorted population and endpoint-inclusive evenly spaced indices. Key Steps:sort dataset/sample_id;round i(N-1)/(count-1);assert uniqueness. Key Params:20,no image-content filtering. State/Dependencies:no mutation or RNG."""
    ordered=sorted(rows,key=lambda r:(r['dataset'],r['sample_id']))
    assert len(ordered)>=count>=2 and len({(r['dataset'],r['sample_id']) for r in ordered})==len(ordered)
    indices=[round(i*(len(ordered)-1)/(count-1)) for i in range(count)]
    assert len(set(indices))==count
    return ordered,indices


def curve_pair(image,folder):
    """@luna-comment Input:native BGR teacher and owned folder. Output:old/new curve PNG plus shared-rail receipt. Key Steps:once estimate/repairL/H;finite monotone LUTs;unchanged native common-channel render. Key Params:old(.5,1.8),new(.75,2.6),outside.1. State/Dependencies:exact frozen V3 reference modules;GT unused."""
    import cv2,numpy as np
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    estimate=revision2.estimate(image);q,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
    low,repair=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],q,tail,estimate['nodes']);high=estimate['fields']['high']
    np.savez_compressed(folder/'fixed_rails.npz',low=low,high=high,nodes=estimate['nodes'])
    records={}
    for key,alpha,gamma in [('old_s',.5,1.8),('steep_s',.75,2.6)]:
        tables=np.empty((*low.shape,256),np.float32)
        for iy,ix in np.ndindex(low.shape):
            l,h=float(low[iy,ix]),float(high[iy,ix]);values=np.arange(256)
            tables[iy,ix]=revision2.curve(pre640.curve(values,l,(l+h)/2,h,alpha,gamma),l,h,.1)
        assert np.isfinite(tables).all() and np.diff(tables,axis=2).min()>=-1e-5
        result,telemetry=pre640.render(image,dict(nodes=estimate['nodes'],tables=tables,native_shape=list(image.shape[:2])),strength=1.,chunk=128)
        delta=result.astype(np.int16)-image.astype(np.int16)
        assert result.shape==image.shape and np.array_equal(delta[...,0],delta[...,1]) and np.array_equal(delta[...,1],delta[...,2])
        assert cv2.imwrite(str(folder/(key+'.png')),result)
        records[key]=dict(alpha=alpha,gamma=gamma,midpoint_derivative=1+alpha*(gamma-1),exterior_slope=.1,finite_monotone_lut=True,shared_channel_delta=True,changed_pixels=int(np.any(delta!=0,axis=2).sum()),sha256=digest(folder/(key+'.png')),telemetry=telemetry)
    return dict(arms=records,rails_sha256=digest(folder/'fixed_rails.npz'))


def main():
    """@luna-comment Input:authorized20-image full-pipeline diagnostic. Output:sample/worker/audit/native comparison receipts. Key Steps:freeze selection/hashes;bind snapshot paths/partition only;workersGPU2/3;exact delivery audit;native old/newS. Key Params:seed42 u1 full original OCR routing,alpha.75 gamma2.6. State/Dependencies:existing deployments;owned output,no training/source overwrite."""
    import cv2,numpy as np
    from tools.prepare_v3pre_training import REFERENCES
    cv2.setNumThreads(1);started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'COMPLETE.json').exists(),'Refusing completed output overwrite'
    assert {key:digest(ROOT/key) for key in REFERENCES}==REFERENCES
    rows,indices=choose([json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()])
    source=OUT/'source';run=OUT/'run'
    for split in ['train','val']:(source/split/'LQ').mkdir(parents=True,exist_ok=True)
    selection=[]
    for i,index in enumerate(indices,1):
        row=rows[index];key=f'{i:03d}';folder=OUT/'images'/key;folder.mkdir(parents=True,exist_ok=True)
        bindings={'lq':row['preprocess']['source_files']['lq'],'pre_lq':row['files']['lq'],'training_gt':row['files']['gt']}
        expected={'lq':row['preprocess']['source_hashes']['lq'],'pre_lq':row['output_sha256']['lq'],'training_gt':row['output_sha256']['gt']}
        hashes={}
        for name,path in bindings.items():
            assert digest(path)==expected[name];shutil.copyfile(path,folder/(name+'.png'));hashes[name]=expected[name]
        image=cv2.imread(str(folder/'pre_lq.png'));shape=list(image.shape[:2]);assert cv2.imread(str(folder/'training_gt.png')).shape==image.shape
        teacher_name=key+'__'+row['sample_id']+'.png';shutil.copyfile(folder/'pre_lq.png',source/'train/LQ'/teacher_name)
        selection.append(dict(id=key,population_index=index,sample_id=row['sample_id'],dataset=row['dataset'],shape_hw=shape,source_files=bindings,source_hashes=hashes,teacher_name=teacher_name))
    save(OUT/'SAMPLE_SELECTION.json',dict(population=len(rows),count=20,manifest=str(MANIFEST),manifest_sha256=digest(MANIFEST),sort=['dataset','sample_id'],index_rule='round(i*(N-1)/19),i=0..19',indices=indices,records=selection))
    teacher={str(p):digest(p) for p in [DEPLOY/'src/run_remote58_vosr2.sh',DEPLOY/'src/run_remote58_vosr2.py',DEPLOY/'src/preset/ckpts/VOSR2/args.json',DEPLOY/'src/preset/ckpts/VOSR2/checkpoints/ema_model.safetensors']}
    text=SNAPSHOT.read_text(encoding='utf-8');bindings={"DATA=Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261014/full_images')":'DATA=Path('+repr(str(source))+')',"RUN=BASE/'runs/displayfit_20261014'":'RUN=Path('+repr(str(run))+')','if i%4==gpu':'if i%2==gpu-2'}
    for old,new in bindings.items():assert text.count(old)==1;text=text.replace(old,new)
    adapter=OUT/'run_full_snapshot.py';adapter.write_text(text,encoding='utf-8')
    save(OUT/'BINDING_DIFF.json',dict(snapshot_sha256=digest(SNAPSHOT),adapter_sha256=digest(adapter),replacements=bindings,teacher_files=teacher))
    processes=[]
    save(OUT/'STATUS.json',dict(phase='full_teacher',selected=20,curves_completed=0,started_epoch=started))
    for gpu in [2,3]:
        owned=selection[gpu-2::2]
        completed=list((run/f'worker{gpu}').glob('batch*/*_done.json'))
        if len(completed)==len(owned):
            done=[json.loads(p.read_text()) for p in completed]
            assert {Path(r['input']).name for r in done}=={r['teacher_name'] for r in owned}
            assert all(r['verified'] and not r['quality_gate'] and Path(r['output']).is_file() for r in done)
            print('REUSE_VERIFIED_WORKER',gpu,len(done),flush=True)
            continue
        log=(OUT/f'worker{gpu}.log').open('wb')
        process=subprocess.Popen([str(DEPLOY/'orientation_env/bin/python'),'-u',str(adapter),str(gpu)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
        processes.append((process,log))
    for process,log in processes:code=process.wait();log.close();assert code==0,('Teacher worker failed',code)
    routes={}
    audit_code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import cv2;from tools.run_train558_full_vosr_curve import audit_masks;p=Path(sys.argv[2]);r=json.loads((p/'routing.json').read_text());a=[dict(input=x['input'],**audit_masks(x,p/'model_output',cv2.imread(x['output']))) for x in r];Path(sys.argv[3]).write_text(json.dumps(a))"
    for routing in sorted(run.glob('worker*/batch*/routing.json')):
        batch=routing.parent;audited=batch/'composition_audit.json'
        subprocess.run([str(DEPLOY/'orientation_env/bin/python'),'-c',audit_code,str(ROOT),str(batch),str(audited)],check=True)
        audits={x['input']:x for x in json.loads(audited.read_text())}
        for route in json.loads(routing.read_text()):
            done=json.loads((batch/(str(route['index'])+'_done.json')).read_text());assert done['verified'] and not done['quality_gate']
            assert len(route['patches'])==sum(line['large'] for line in route['lines'])
            for line in route['lines']:assert line['large']==bool(line['score']>=.6 and line['height_estimate']>100 and abs(line['angle'])<15)
            routes[Path(route['input']).name]=(route,audits[route['input']])
    assert len(routes)==20
    records=[]
    for i,row in enumerate(selection,1):
        folder=OUT/'images'/row['id'];route,audit=routes[row['teacher_name']]
        image=cv2.imread(route['output']);assert list(image.shape[:2])==row['shape_hw']
        assert cv2.imwrite(str(folder/'vosr.png'),image)
        curve=curve_pair(image,folder)
        record=dict(**row,**curve,teacher_sha256=digest(folder/'vosr.png'),large_lines=len(route['patches']),detected_lines=len(route['lines']),composition_audit=audit,routing=route)
        save(folder/'receipt.json',record);records.append(record)
        save(OUT/'STATUS.json',dict(phase='post_curves',selected=20,curves_completed=i,seconds=time.time()-started))
        print('COMPLETE_IMAGE',row['id'],row['sample_id'],record['large_lines'],flush=True)
    save(OUT/'COMPLETE.json',dict(passed=True,count=20,seconds=time.time()-started,selection_sha256=digest(OUT/'SAMPLE_SELECTION.json'),binding_sha256=digest(OUT/'BINDING_DIFF.json'),records=records,parameters=dict(alpha=.75,gamma=2.6,midpoint_derivative=2.2,exterior_slope=.1),limitations=['Twenty uniformly selected pretraining images,not a holdout','Fresh full documented teacher on cached V3LQ,not trainingGT replay','Quality preference pending user review']))
    save(OUT/'STATUS.json',dict(phase='complete',selected=20,curves_completed=20,seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except Exception as error:save(OUT/'FAILED.json',dict(error=repr(error)));raise
