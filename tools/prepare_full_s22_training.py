"""Incremental full documented teacher/S2.2 refresh and gated training bank."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import digest,save,curve_pair
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'full_s22_training_matrix_20261028'
PARENT=DATA/'v3pre_v3m_three_20261027/pretrain/source_view'
DEPLOY=Path('/data/guang.yang/vosr2_u1_tile512')
SNAPSHOT=Path(__file__).with_name('frozen_displayfit_snapshot.py')


class ExistingWorker:
    def __init__(self,pid,gpu):
        """@luna-comment Input:recorded owned teacherPID/GPU. Output:read-only process observer. Key Steps:bind original log/PID without relaunch. Key Params:never terminate teacher. State/Dependencies:existing launch receipt."""
        self.pid,self.gpu=pid,gpu

    def poll(self):
        """@luna-comment Input:existing worker observer. Output:None while verified live,0 only original complete log,1 otherwise. Key Steps:check completion footer;validate proc command and non-zombie state. Key Params:owned adapter only. State/Dependencies:read-only proc/log."""
        with (OUT/f'teacher{self.gpu}.log').open('rb') as stream:
            stream.seek(max(0,stream.seek(0,2)-4096));tail=stream.read()
        if ('WORKER_COMPLETE '+str(self.gpu)+' ').encode() in tail:return 0
        proc=Path('/proc')/str(self.pid)
        if not (proc/'cmdline').exists():return 1
        if str(OUT/'run_full_snapshot.py').encode() not in (proc/'cmdline').read_bytes():return 1
        return 1 if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z' else None

    def wait(self):
        """@luna-comment Input:owned observer. Output:final status after bounded one-second polling. Key Steps:delegate exact poll. Key Params:original teacher is never relaunched. State/Dependencies:read-only observer."""
        while self.poll() is None:time.sleep(1)
        return self.poll()


def target(job):
    """@luna-comment Input:owned batch route/original cached row/audit. Output:newGT row/curve/hash receipt. Key Steps:verify input;reuse exact native curve_pair;select S2.2 asGT;preserve cached LQ and masks. Key Params:no second V3,no source edits. State/Dependencies:immutable teacherPNG,owned per-parent outputs,resume hash check."""
    import cv2
    cv2.setNumThreads(1)
    row,route,audit=job;folder=OUT/'targets'/row['dataset']/row['sample_id'];receipt=folder/'receipt.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text());assert previous['source_lq_sha256']==row['output_sha256']['lq'] and digest(previous['row']['files']['gt'])==previous['row']['output_sha256']['gt'];return previous
    assert digest(row['files']['lq'])==row['output_sha256']['lq']
    image=cv2.imread(route['output']);assert image is not None
    source=cv2.imread(row['files']['lq']);assert source.shape==image.shape
    folder.mkdir(parents=True,exist_ok=True);curves=curve_pair(image,folder)
    updated=copy.deepcopy(row);updated['files']['gt']=str(folder/'steep_s.png');updated['output_sha256']['gt']=curves['arms']['steep_s']['sha256']
    updated['target_refresh']=dict(version='V3LQ_full_documented_VOSR_S2.2',teacher_output=route['output'],teacher_sha256=digest(route['output']),alpha=.75,gamma=2.6,midpoint_derivative=2.2,exterior_slope=.1,no_second_v3=True)
    result=dict(row=updated,source_lq_sha256=row['output_sha256']['lq'],curves=curves,audit=audit,large_lines=len(route['patches']),shape_hw=list(image.shape[:2]))
    save(receipt,result);return result


def state(phase,started,teacher_done=0,gt_done=0,**extra):
    """@luna-comment Input:actual stage/counts/start. Output:atomic progress/rates. Key Steps:elapsed wall-rate and conservative remaining branches. Key Params:4873,bank separate timing. State/Dependencies:owned STATUS only."""
    elapsed=max(time.time()-started,1)
    save(OUT/'DATA_STATUS.json',dict(phase=phase,total=4873,teacher_done=teacher_done,gt_done=gt_done,started_epoch=started,elapsed_seconds=elapsed,teacher_per_hour=teacher_done/elapsed*3600,gt_per_hour=gt_done/elapsed*3600,updated_epoch=time.time(),**extra))


def main():
    """@luna-comment Input:authorized full4873 refresh. Output:incremental targets/verified full bank/DATA_READY. Key Steps:isolate snapshot;original4GPU split;exact batch audits;bounded CPU pool;final manifests and accepted bank/sidecars. Key Params:8CPU,12bank workers,original VOSR defaults. State/Dependencies:old datasets immutable;new workers owned;no training here."""
    from tools.prepare_v3pre_training import REFERENCES
    assert {p:digest(ROOT/p) for p in REFERENCES}==REFERENCES
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'DATA_READY.json').exists(),'Already ready'
    started=json.loads((OUT/'DATA_STATUS.json').read_text())['started_epoch'] if (OUT/'DATA_STATUS.json').exists() else time.time()
    manifest=PARENT/'manifests/train.jsonl';rows=sorted([json.loads(x) for x in manifest.read_text().splitlines() if x.strip()],key=lambda r:(r['dataset'],r['sample_id']))
    assert len(rows)==4873
    source=OUT/'teacher_source';run=OUT/'teacher_run'
    for split in ['train','val']:(source/split/'LQ').mkdir(parents=True,exist_ok=True)
    lookup={}
    for i,row in enumerate(rows):
        name=f'{i:05d}__'+row['sample_id']+'.png';path=source/'train/LQ'/name
        if not path.exists():path.symlink_to(row['files']['lq'])
        assert path.resolve()==Path(row['files']['lq']).resolve();lookup[name]=row
    save(OUT/'SOURCE_BINDING.json',dict(population=4873,train_manifest=str(manifest),train_manifest_sha256=digest(manifest),references=REFERENCES,teacher_snapshot_sha256=digest(SNAPSHOT)))
    text=SNAPSHOT.read_text();changes={"DATA=Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261014/full_images')":'DATA=Path('+repr(str(source))+')',"RUN=BASE/'runs/displayfit_20261014'":'RUN=Path('+repr(str(run))+')'}
    for old,new in changes.items():assert text.count(old)==1;text=text.replace(old,new)
    adapter=OUT/'run_full_snapshot.py'
    if (OUT/'OCR_REUSE_BINDING.json').exists():
        reused=json.loads((OUT/'OCR_REUSE_BINDING.json').read_text());assert digest(adapter)==reused['adapter_sha256']
        changes=reused['replacements']
    else:adapter.write_text(text)
    save(OUT/'TEACHER_BINDING.json',dict(snapshot_sha256=digest(SNAPSHOT),adapter_sha256=digest(adapter),replacements=changes,teacher={str(p):digest(p) for p in [DEPLOY/'src/run_remote58_vosr2.sh',DEPLOY/'src/run_remote58_vosr2.py',DEPLOY/'src/preset/ckpts/VOSR2/args.json',DEPLOY/'src/preset/ckpts/VOSR2/checkpoints/ema_model.safetensors']}))
    processes=[];streams=[]
    for gpu in range(4):
        if (OUT/f'launch/teacher{gpu}.json').exists():
            previous=json.loads((OUT/f'launch/teacher{gpu}.json').read_text());worker=ExistingWorker(previous['pid'],gpu)
            assert worker.poll() in (None,0),'Existing teacher failed;inspect original log before retry'
            processes.append(worker);continue
        assert not (OUT/f'teacher{gpu}.log').exists(),'Refusing duplicate teacher launch'
        stream=(OUT/f'teacher{gpu}.log').open('xb');streams.append(stream)
        command=[str(DEPLOY/'orientation_env/bin/python'),'-u',str(adapter),str(gpu)]
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));processes.append(p)
        save(OUT/f'launch/teacher{gpu}.json',dict(pid=p.pid,gpu=gpu,command=command,started_epoch=started))
    audited=set();submitted=set();pending={};completed={}
    audit_code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import cv2;from tools.run_train558_full_vosr_curve import audit_masks;p=Path(sys.argv[2]);r=json.loads((p/'routing.json').read_text());a=[dict(input=x['input'],**audit_masks(x,p/'model_output',cv2.imread(x['output']))) for x in r];Path(sys.argv[3]).write_text(json.dumps(a))"
    with ProcessPoolExecutor(max_workers=8) as pool:
        while len(completed)<4873:
            for p in processes:assert p.poll() in (None,0),'Teacher worker failed'
            for routing in sorted(run.glob('worker*/batch*/routing.json')):
                batch=routing.parent
                if str(batch) in audited:continue
                if not any(batch.glob('*_done.json')):continue
                try:routes=json.loads(routing.read_text())
                except json.JSONDecodeError:continue
                if not routes or not all((batch/(str(r['index'])+'_done.json')).exists() for r in routes):continue
                output=batch/'independent_audit.json'
                subprocess.run([str(DEPLOY/'orientation_env/bin/python'),'-c',audit_code,str(ROOT),str(batch),str(output)],check=True,stdout=subprocess.DEVNULL)
                audits={r['input']:r for r in json.loads(output.read_text())}
                for route in routes:
                    name=Path(route['input']).name;assert name not in submitted
                    assert json.loads((batch/(str(route['index'])+'_done.json')).read_text())['verified']
                    pending[pool.submit(target,(lookup[name],route,audits[route['input']]))]=name;submitted.add(name)
                audited.add(str(batch))
            for future in list(pending):
                if future.done():completed[pending.pop(future)]=future.result()
            teacher_done=sum(1 for _ in run.glob('worker*/batch*/*_done.json'))
            state('teacher_and_targets',started,teacher_done,len(completed),audited_batches=len(audited))
            if all(p.poll()==0 for p in processes) and not pending and len(completed)<4873:raise RuntimeError('Missing completed teacher inputs')
            time.sleep(10)
    assert all(p.wait()==0 for p in processes)
    for stream in streams:stream.close()
    view=OUT/'source_view';(view/'manifests').mkdir(parents=True,exist_ok=True)
    records=[completed[name] for name in sorted(completed)]
    (view/'manifests/train.jsonl').write_text(''.join(json.dumps(r['row'])+'\n' for r in records))
    val=PARENT/'manifests/val.jsonl';assert len([x for x in val.read_text().splitlines() if x.strip()])==583;shutil_copy=val.read_bytes();(view/'manifests/val.jsonl').write_bytes(shutil_copy)
    save(OUT/'TARGETS_COMPLETE.json',dict(passed=True,count=4873,seconds=time.time()-started,train_manifest_sha256=digest(view/'manifests/train.jsonl'),val_manifest_sha256=digest(view/'manifests/val.jsonl'),validation_domain='Existing olderGT_VOSR583 V3 cache;not refreshed'))
    bank=OUT/'bank256';state('bank_build',started,4873,4873)
    with (OUT/'bank.log').open('ab') as stream:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(view),'--output',str(bank),'--workers','12','--preprocess-mode','none'],stdout=stream,stderr=subprocess.STDOUT)
        while p.poll() is None:state('bank_build',started,4873,4873);time.sleep(15)
        assert p.returncode==0
    state('line_sidecars',started,4873,4873)
    from tools.run_v3pre_training import sidecars
    sidecars(bank,view,12)
    assert json.loads((bank/'COMPLETE.json').read_text())['complete'] and json.loads((bank/'LINE_REGIONS_COMPLETE.json').read_text())['passed']
    save(OUT/'DATA_READY.json',dict(passed=True,train_parents=4873,val_parents=583,bank_receipt_sha256=digest(bank/'COMPLETE.json'),line_receipt_sha256=digest(bank/'LINE_REGIONS_COMPLETE.json'),targets_receipt_sha256=digest(OUT/'TARGETS_COMPLETE.json'),completed_epoch=time.time()))
    state('complete',started,4873,4873)


if __name__=='__main__':
    try:main()
    except Exception as error:save(OUT/'DATA_FAILED.json',dict(error=repr(error),epoch=time.time()));raise
