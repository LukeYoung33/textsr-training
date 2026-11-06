"""Complete already-authorized validation preparation; leave platform writes pending."""
from pathlib import Path
import csv,json,os,subprocess,time
OUT=Path('/AI_DNS/guang.yang/TextSR/data/trusted_gt_validation_20261106')
REGISTRY=OUT.parent/'registry/vosr_gt_repaired_20261106'
PYTHON='/data/guang.yang/vosr2_u1_tile512/orientation_env/bin/python'


def save(path,value):
    """@luna-comment Input:owned path/JSON. Output:atomic status. Key Steps:sibling temp/replace. Key Params:UTF8. State/Dependencies:isolated preparation directory."""
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def run():
    """@luna-comment Input:owned controller identity/completion receipt. Output:verified validation copies and reviewable two-row registration CSV. Key Steps:observe exact controller;copy after success;record counts/paths. Key Params:no MCP writes or retraining. State/Dependencies:authorized background job/owned NAS outputs only."""
    import fcntl
    lock=(OUT/'completion_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller=json.loads((OUT/'CONTROLLER.json').read_text())
    while not (OUT/'COMPLETE.json').exists():
        proc=Path('/proc')/str(controller['pid'])
        assert (proc/'stat').exists(),'Validation controller exited before completion; inspect controller.log'
        stat=(proc/'stat').read_text().split()
        assert stat[2]!='Z' and stat[21]==controller['start_token'],'Validation controller identity changed/exited'
        progress=json.loads((OUT/'STATUS.json').read_text()) if (OUT/'STATUS.json').exists() else {}
        save(OUT/'COMPLETION_SUPERVISOR.json',dict(phase='waiting_for_validation',controller=controller,progress=progress,updated=time.time(),pid=os.getpid()))
        time.sleep(20)
    assert json.loads((OUT/'COMPLETE.json').read_text())['passed']
    save(OUT/'COMPLETION_SUPERVISOR.json',dict(phase='copying_validation',updated=time.time(),pid=os.getpid()))
    subprocess.run([PYTHON,'-u',str(OUT/'code/tools/prepare_trusted_validation_registration_next.py'),'--materialize','val'],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    plan=[]
    for split,kind,name in [('train','training','VOSR-GT修复版'),('val','testing','VOSR-GT修复版-验证')]:
        receipt=json.loads((REGISTRY/split/'COMPLETE.json').read_text());assert receipt['passed'] and receipt['all_sha256_verified']
        plan.append(dict(version_alias=name,data_type=kind,sample_count=receipt['pairs'],image_count=receipt['images'],image_bytes=receipt['image_bytes'],data_path=str(REGISTRY/split),storage_alias='NAS-ai_dns',project='texthd-base',module='basic-restore',reason='更新数据GT版本',pair_rule='LQ/GT matching relative filenames',lineage_status='pending user confirmation',registration_status='not registered; final write approval required'))
    temporary=REGISTRY/'registration_plan.csv.tmp'
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(plan[0]));writer.writeheader();writer.writerows(plan)
    temporary.replace(REGISTRY/'registration_plan.csv');save(REGISTRY/'registration_plan.json',plan)
    save(OUT/'COMPLETION_SUPERVISOR.json',dict(phase='ready_for_registration_approval',plan=plan,updated=time.time(),pid=os.getpid()))


if __name__=='__main__':
    try:run()
    except Exception as error:
        import traceback
        save(OUT/'COMPLETION_SUPERVISOR_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),updated=time.time()));raise
