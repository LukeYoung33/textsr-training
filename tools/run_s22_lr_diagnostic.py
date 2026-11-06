"""Isolated, optimizer-preserving LR forks of the four completed S22 guard runs."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import html
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA / 's22_guard_lr_diagnostic_20261030'
REPO = '/home/guang.yang/workspace/Real-ESRGAN'
TZ = timezone(timedelta(hours=8))
FACTORS = {'lr1': 1., 'lr3': 1./3., 'lr10': .1}
GROUPS = {'cp': 's22_patch_contrast_guard_20261029', 'nocp': 's22_nocp_text_guard_20261029'}


def save(path, value):
    """@luna-comment Input:path/JSON value. Output:atomic owned metadata. Key Steps:write sibling then replace. Key Params:UTF8. State/Dependencies:filesystem only."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'); temp.replace(path)


def digest(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream bytes. Key Params:1MiB chunks. State/Dependencies:read-only file."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): result.update(part)
    return result.hexdigest()


def equal_state(left, right):
    """@luna-comment Input:nested optimizer/network state. Output:exact equality. Key Steps:recursive tensor/value comparison. Key Params:no tolerance. State/Dependencies:torch CPU or CUDA tensors."""
    import torch
    if torch.is_tensor(left): return torch.equal(left, right.to(left.device))
    if isinstance(left, dict): return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)): return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def prepare():
    """@luna-comment Input:four completed parent snapshots. Output:12 frozen configs/bindings. Key Steps:validate completion and source hashes;retain all losses/data;bind250K Adam/calibration. Key Params:20K constant LR continuation. State/Dependencies:torch/yaml;owned directory only."""
    import torch, yaml
    assert not (OUT/'BINDINGS.json').exists(), 'Refusing duplicate preparation'
    parents = {}; runs = {}
    for group, folder in GROUPS.items():
        source = DATA/folder
        assert json.loads((source/'TRAINING_COMPLETE.json').read_text())['passed']
        records = json.loads((source/'CONFIG_BINDINGS.json').read_text())['runs']
        for arm in ['full_sdf', 'full_nosdf']:
            parent = group+'_'+arm; record = records[arm]
            source_cfg = Path(record['config']); assert digest(source_cfg) == record['sha256']
            cfg = yaml.safe_load(source_cfg.read_text())
            experiment = Path(cfg['path']['experiments_root'])/cfg['name']
            network = experiment/'models/net_g_250000.pth'; state = experiment/'training_states/250000.state'
            saved = torch.load(state, map_location='cpu'); assert saved['iter'] == 250000 and len(saved['optimizers']) == 1
            rates = [p['lr'] for p in saved['optimizers'][0]['param_groups']]; assert rates and len(set(rates)) == 1
            calibration = source/'audits'/arm/'CALIBRATION.json'
            cal = json.loads(calibration.read_text()); assert cal['passed'] and len(cal['records']) == 8
            paths = {'config': source_cfg, 'network': network, 'state': state, 'calibration': calibration,
                     'train_manifest': Path(cfg['datasets']['train']['manifest']), 'val_manifest': Path(cfg['datasets']['val']['manifest'])}
            parents[parent] = dict(paths={k:str(v) for k,v in paths.items()}, hashes={k:digest(v) for k,v in paths.items()}, base_lr=rates[0], guard_weight=cal['weight'])
            for variant, factor in FACTORS.items():
                name = parent+'_'+variant; derived = copy.deepcopy(cfg)
                derived['name'] = 'diagnostic_'+name
                derived['path'].update(pretrain_network_g=str(network), resume_state=None, strict_load_g=True, experiments_root=str(OUT/'experiments'))
                derived['s22_strategy']['audit_root'] = str(OUT/'runs'/name/'audit')
                derived['train']['total_iter'] = 270000
                derived['train']['optim_g']['lr'] = rates[0]*factor
                derived['train']['scheduler'] = dict(type='MultiStepLR', milestones=[], gamma=1.)
                path = OUT/'configs'/(name+'.yml'); path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding='utf-8')
                runs[name] = dict(parent=parent, variant=variant, factor=factor, lr=rates[0]*factor, config=str(path), config_sha256=digest(path))
    save(OUT/'BINDINGS.json', dict(parents=parents, runs=runs, steps=20000, sampling_seed=20261030, created=datetime.now(TZ).isoformat(), adapter_sha256=digest(__file__)))


def fixed_probe(model, dataset, indices, iteration):
    """@luna-comment Input:model/frozen dataset IDs/global step. Output:fixed-batch full losses plus conditional CP. Key Steps:suppress Adam step;run unchanged loss/backward;assert exact state;restore RNG/counters. Key Params:batch4,not training update. State/Dependencies:original S22 model,isolated temporary step binding."""
    import random, numpy as np, torch
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    weights = {k:v.detach().clone() for k,v in model.net_g.state_dict().items()}
    adam = copy.deepcopy(model.optimizer_g.state_dict())
    counters = {k:copy.deepcopy(getattr(model,k)) for k in ['total_draws','synthetic_count','extra_stats','calibration','extra_weight']}
    old_step = model.optimizer_g.step; records = []
    def no_update(*args, **kwargs):
        """@luna-comment Input:Adam call arguments. Output:no mutation. Key Steps:return. Key Params:probe only. State/Dependencies:temporary owned optimizer binding."""
        return None
    try:
        model.optimizer_g.step = no_update
        for start in range(0,len(indices),4):
            batch = dataset.collate_fn([dataset[int(i)] for i in indices[start:start+4]])
            model.feed_data(batch); model.optimize_parameters(iteration)
            records.append({k:float(v) for k,v in model.get_current_log().items()})
        assert equal_state(weights,model.net_g.state_dict()), 'Probe mutated network'
        assert equal_state(adam,model.optimizer_g.state_dict()), 'Probe mutated Adam'
    finally:
        model.optimizer_g.step = old_step; model.optimizer_g.zero_grad(set_to_none=True)
        for k,v in counters.items(): setattr(model,k,v)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
    keys = sorted(set.intersection(*(set(r) for r in records)))
    values = {k:sum(r[k] for r in records)/len(records) for k in keys if k.startswith('l_')}
    active = [r for r in records if r.get('cp_accepted_lines',0)>0]
    return dict(values=values, batches=len(records), samples=len(indices), cp_active_batches=len(active), cp_lines=sum(r.get('cp_accepted_lines',0) for r in records), cp_active_mean=sum(r.get('l_cp',0) for r in active)/len(active) if active else None, state_unchanged=True)


def worker(name, smoke=False):
    """@luna-comment Input:bound parent/LR fork. Output:20K continuation or two-step CUDA gate. Key Steps:strict weights/Adam/calibration restore;common deterministic draws;fixed probes;owned checkpoints. Key Params:constant LR,no historical scheduler replay;seed reset identical within parent. State/Dependencies:CUDA and immutable existing bank."""
    import random, numpy as np, torch
    from tools.train_full_s22_strategy import bootstrap
    bootstrap()
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from basicsr.data import build_dataset
    binding = json.loads((OUT/'BINDINGS.json').read_text()); run = binding['runs'][name]; parent = binding['parents'][run['parent']]
    for key,path in parent['paths'].items(): assert digest(path)==parent['hashes'][key], key
    assert digest(run['config']) == run['config_sha256']
    sys.argv = [sys.argv[0],'-opt',run['config']]; opt,_ = parse_options(REPO,is_train=True)
    opt['dist'] = False; opt['rank'] = 0; opt['world_size'] = 1
    run_dir = OUT/('smoke' if smoke else 'runs')/name; run_dir.mkdir(parents=True,exist_ok=True)
    opt['s22_strategy']['audit_root'] = str(run_dir/'audit')
    for key in ['models','training_states','log','visualization']:
        opt['path'][key] = str(run_dir/key); Path(opt['path'][key]).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    train = build_dataset(opt['datasets']['train']); val = build_dataset(opt['datasets']['val'])
    seed = binding['sampling_seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_model(opt)
    expected = torch.load(parent['paths']['network'],map_location='cpu')['params']
    assert equal_state(expected,model.net_g.state_dict())
    state = torch.load(parent['paths']['state'],map_location='cpu')
    model.optimizer_g.load_state_dict(state['optimizers'][0]); assert equal_state(state['optimizers'][0],model.optimizer_g.state_dict())
    cal = json.loads(Path(parent['paths']['calibration']).read_text())
    model.calibration = cal['records']; model.extra_weight = cal['weight']
    for group in model.optimizer_g.param_groups: group['lr'] = run['lr']; group['initial_lr'] = run['lr']
    generator = np.random.default_rng(seed)
    # A permutation of draw-index entries retains the parent's balanced multiplicities.
    schedule = []
    while len(schedule)<80000: schedule.extend(generator.permutation(len(train)).tolist())
    schedule = np.asarray(schedule[:80000],dtype=np.int64)
    train_ids = np.random.default_rng(seed+1).choice(len(train),size=min(128,len(train)//4*4),replace=False)
    val_ids = np.random.default_rng(seed+2).choice(len(val),size=min(256,len(val)//4*4),replace=False)
    frozen = dict(train_ids=train_ids.tolist(),val_ids=val_ids.tolist(),schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),train_index_sha256=hashlib.sha256(train.index.tobytes()).hexdigest(),val_index_sha256=hashlib.sha256(val.index.tobytes()).hexdigest())
    save(run_dir/'INPUT_BINDING.json',frozen)
    save(run_dir/'RESTORE.json',dict(passed=True,parent=run['parent'],network_exact=True,adam_exact_before_lr_override=True,source_iter=250000,lr=run['lr'],guard_weight=model.extra_weight,calibration_restored=True))
    start=time.time(); records=[]; steps=2 if smoke else 20000
    probe_steps = {0,2} if smoke else {0,2000,5000,10000,15000,20000}
    ids = train_ids[:8] if smoke else train_ids; vids = val_ids[:8] if smoke else val_ids
    def probe(offset):
        """@luna-comment Input:continuation offset. Output:atomic train/val fixed probe. Key Steps:run frozen sample sets and retain checkpoint-relative labels. Key Params:global250K+offset. State/Dependencies:nonupdating diagnostic helper."""
        result=dict(step=offset,global_iter=250000+offset,train=fixed_probe(model,train,ids,250000+offset),validation=fixed_probe(model,val,vids,250000+offset))
        records.append(result);save(run_dir/'PROBES.json',records)
    probe(0)
    subset=torch.utils.data.Subset(train,schedule[:steps*4].tolist())
    loader=torch.utils.data.DataLoader(subset,batch_size=4,shuffle=False,num_workers=2,collate_fn=train.collate_fn,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed))
    history=[]; update_seconds=0.; probe_seconds=0.
    before={k:v.detach().clone() for k,v in model.net_g.state_dict().items()}
    for offset,batch in enumerate(loader,1):
        tic=time.time();model.feed_data(batch);model.optimize_parameters(250000+offset);torch.cuda.synchronize();update_seconds+=time.time()-tic
        assert all(g['lr']==run['lr'] for g in model.optimizer_g.param_groups)
        logs={k:float(v) for k,v in model.get_current_log().items()}
        if offset==1: assert not equal_state(before,model.net_g.state_dict()), 'No parameter update'
        if offset%100==0 or offset<=2:
            history.append(dict(step=offset,values={k:v for k,v in logs.items() if k.startswith('l_') or k=='cp_accepted_lines'}))
            save(run_dir/'HISTORY.json',history)
            save(run_dir/'STATUS.json',dict(state='running',step=offset,lr=run['lr'],elapsed=time.time()-start,update_seconds=update_seconds,probe_seconds=probe_seconds,pid=os.getpid(),updated=datetime.now(TZ).isoformat()))
        if offset in probe_steps:
            tic=time.time();probe(offset);probe_seconds+=time.time()-tic
            if not smoke:
                torch.save(dict(params={k:v.detach().cpu() for k,v in model.net_g.state_dict().items()}),run_dir/'models'/f'net_g_{250000+offset}.pth')
    assert offset==steps
    if not smoke:
        torch.save(dict(epoch=0,iter=270000,optimizers=[model.optimizer_g.state_dict()],calibration=model.calibration,extra_weight=model.extra_weight),run_dir/'training_states/270000.state')
    save(run_dir/'COMPLETE.json',dict(passed=True,steps=steps,lr=run['lr'],parent=run['parent'],seconds=time.time()-start,probe_state_checks=True,source_hashes_preserved=all(digest(p)==parent['hashes'][k] for k,p in parent['paths'].items())))


def publish(processes, queues):
    """@luna-comment Input:owned processes and pending queues. Output:atomic live dashboard and status. Key Steps:read receipts/history/probes;render explicit states and shared loss chart. Key Params:30s refresh/manual button;ETA from elapsed progression. State/Dependencies:read-only trainer artifacts."""
    from tools.offline_loss_charts import render_loss_charts
    binding=json.loads((OUT/'BINDINGS.json').read_text());states={};cards=[];histories={}
    for name,run in binding['runs'].items():
        folder=OUT/'runs'/name;path=folder/'STATUS.json';s=json.loads(path.read_text()) if path.exists() else dict(step=0)
        process=processes.get(name);complete=(folder/'COMPLETE.json').exists()
        state='complete' if complete else 'failed' if process is not None and process.poll() is not None else 'running' if process is not None else 'queued'
        step=20000 if complete else s.get('step',0);s.update(state=state,step=step,lr=run['lr']);states[name]=s
        hp=folder/'HISTORY.json'
        if hp.exists():histories[name]=json.loads(hp.read_text())
        pp=folder/'PROBES.json';probes=json.loads(pp.read_text()) if pp.exists() else []
        detail=''
        for row in probes:
            detail+='<tr><td>'+str(row['step'])+'</td>'+''.join('<td>'+format(row[split]['values'].get(key,0),'.6f')+'</td>' for split,key in [('train','l_native'),('train','l_total'),('validation','l_native')])+'</tr>'
        eta=(s.get('elapsed',0)/step*(20000-step)/60) if step and state=='running' else None
        labels={'complete':'已完成','running':'运行中','queued':'排队','failed':'失败'}
        cards.append('<section><h2>'+html.escape(name)+'</h2><p>'+labels[state]+f' | {step}/20000 | LR={run["lr"]:.6g}</p><p>'+('剩余约 '+str(round(eta))+' 分钟' if eta is not None else '')+'</p><table><tr><th>续训步数</th><th>固定训练L1</th><th>固定训练总损失</th><th>固定验证L1</th></tr>'+detail+'</table></section>')
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>四组学习率诊断</title><style>body{font:15px system-ui;background:#edf2f7;margin:24px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}section{background:white;padding:14px;border-radius:10px}h2{font-size:17px}td,th{padding:5px;text-align:right}button{padding:10px}</style><h1>四组250K起点 · 学习率诊断续训</h1><button onclick="location.reload()">刷新</button><p>每组 LR×1 / ÷3 / ÷10，各20K；原Adam状态、保护系数和损失继承。三分支使用相同新采样序列。固定训练128patch / 验证256patch；验证仍为旧GT_VOSR域。随机batch曲线不替代固定样本评估。</p><p>'+datetime.now(TZ).isoformat()+'</p><main>'+''.join(cards)+'</main>'+render_loss_charts(histories)+'</html>'
    path=OUT/'dashboard.html';tmp=path.with_suffix('.tmp');tmp.write_text(body,encoding='utf-8');tmp.replace(path)
    save(OUT/'MONITOR.json',dict(runs=states,queues=queues,updated=datetime.now(TZ).isoformat()))
    return states


def controller():
    """@luna-comment Input:authorized twelve-fork diagnostic. Output:gated GPU queues/monitor/completion. Key Steps:exclusive lock;prepare;two-update smoke each fork;queue one parent perGPU with >=3GiB free. Key Params:never signal other jobs;stop failed parent's queue. State/Dependencies:owned subprocesses/NVIDIA query."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare();binding=json.loads((OUT/'BINDINGS.json').read_text());parents=list(binding['parents'])
    queues={str(g):[p+'_'+v for v in FACTORS] for g,p in enumerate(parents)};processes={};active={};streams=[]
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=datetime.now(TZ).isoformat(),command=sys.argv))
    while any(queues.values()) or active:
        for gpu,(name,p) in list(active.items()):
            if p.poll() is not None:
                if p.returncode:save(OUT/'FAILED.json',dict(name=name,returncode=p.returncode));queues[gpu]=[]
                del active[gpu]
        free={r.split(',')[0].strip():int(r.split(',')[1]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
        for gpu,queue in queues.items():
            if gpu in active or not queue or free.get(gpu,0)<3072:continue
            name=queue.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            path=OUT/'logs'/(name+'.log');path.parent.mkdir(exist_ok=True);stream=path.open('xb');streams.append(stream)
            p=subprocess.Popen([sys.executable,'-u',__file__,'gated',name],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env,cwd=OUT)
            processes[name]=p;active[gpu]=(name,p);save(OUT/'launch'/(name+'.json'),dict(pid=p.pid,gpu=gpu,config_sha256=binding['runs'][name]['config_sha256']))
        publish(processes,queues);time.sleep(15)
    states=publish(processes,queues)
    assert all(s['state']=='complete' for s in states.values()), 'Diagnostic queue failed'
    # Common-start probe equality is a final triplet acceptance gate.
    for parent in parents:
        probes=[json.loads((OUT/'runs'/(parent+'_'+v)/'PROBES.json').read_text())[0] for v in FACTORS]
        assert probes[0]==probes[1]==probes[2], 'Baseline probes differ within a parent'
    save(OUT/'COMPLETE.json',dict(passed=True,runs=12,steps_each=20000))


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='worker':worker(sys.argv[2])
    elif mode=='smoke':worker(sys.argv[2],True)
    elif mode=='gated':
        name=sys.argv[2]
        for stage in ['smoke','worker']:
            subprocess.run([sys.executable,'-u',__file__,stage,name],check=True)
    elif mode=='run':controller()
