"""Three controlled scratch500K V3M loss recipes with original data and initialization."""
from pathlib import Path
import sys,os,copy,json,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import launch_latest_vosr_500k as monitor
from tools.run_uniform20_steep_scurve import save,digest
BASE=monitor.OUT;OUT=BASE.parent/'v3m_loss_ablation_500k_20261103'
LABELS={'v3m_full_noguard':'V3M · Full无guard · 真实LQ','v3m_mf_uv':'V3M · L1＋Missing/False＋UV · 真实LQ','v3m_mf_uv_mix50':'V3M · L1＋Missing/False＋UV · 50%GT轻退化'}
monitor.OUT=OUT;monitor.LABELS=LABELS
original_publish=monitor.publish


def publish():
    """@luna-comment Input:owned logs/state. Output:correctly titled three-arm ETA andloss page. Key Steps:reuse read-only observer;replace task headings andstartup wording. Key Params:500K,same absoluteLR. State/Dependencies:owned metadata only."""
    original_publish();status=monitor.read(OUT/'STATUS.json')
    for row in status.get('runs',{}).values():
        if row['finish']=='采样中':row['finish']='等待训练日志估时'
        if row['phase']=='等待固定快照patch构建和启动检查':row['phase']='等待启动检查'
    save(OUT/'STATUS.json',status)
    path=OUT/'index.html';text=path.read_text(encoding='utf-8')
    text=text.replace('最新VOSR · 四组500K','V3M · 三组loss对照500K').replace('V3M / V5 · guard / L1-only · 从头训练500K','V3M · 三组loss对照 · 从头训练500K')
    text=text.replace('等待固定快照patch构建和启动检查','等待启动检查').replace('采样中','等待训练日志估时')
    text=text.replace('<div class="grid">','<p><a href="CONTROL_MATRIX.html">查看单变量对照关系与补充建议</a></p><div class="grid">',1)
    tmp=OUT/'index.corrected.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(path)


def prepare():
    """@luna-comment Input:completed V3M500K basebindings. Output:three immutable matchedrecipes. Key Steps:strictold hashes;only disableedgeguard andselectiveBG;share banks/init/schedule. Key Params:MF1/.25,UV1,BG2or0,500K. State/Dependencies:original artifacts read-only."""
    import yaml
    assert json.loads((BASE/'COMPLETE.json').read_text())['passed'] and json.loads((BASE/'DATA_READY.json').read_text())['passed']
    declared=json.loads((BASE/'CONFIG_BINDINGS.json').read_text())['runs']['v3m_guard'];source=Path(declared['config']);assert digest(source)==declared['config_sha256']
    base=yaml.safe_load(source.read_text());assert digest(base['path']['pretrain_network_g'])==declared['initial_sha256'];records={}
    assert base['train']['total_iter']==500000 and base['train']['optim_g']['lr']==1e-4
    for gpu,(arm,label) in enumerate(LABELS.items()):
        cfg=copy.deepcopy(base);mixed=arm.endswith('mix50');selective=arm!='v3m_full_noguard';kind='LatestVOSRMix50' if mixed else 'S22Model'
        cfg['name']=arm+'_500k_20261103';cfg['model_type']=kind;cfg['train']['model_type']=kind;cfg['s22_strategy'].update(name='mixed50' if mixed else 'paired',audit_root=str(OUT/'audits'/arm))
        cfg['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None);cfg['dist_params']['port']=35930+gpu
        if selective:cfg['train'].update(bg_lf_weight=0.,bg_lf_opt=None)
        assert cfg['network_g']==base['network_g'] and cfg['datasets']==base['datasets'] and cfg['manual_seed']==base['manual_seed']
        assert all(cfg['train'][k]==v for k,v in base['train'].items() if k not in ['model_type','bg_lf_opt','bg_lf_weight'])
        assert cfg['train']['recall_aware_weight_missing']==1. and cfg['train']['recall_aware_weight_false']==.25 and cfg['train']['uv_strong_weight']==1.
        assert cfg['train']['cp_weight']==cfg['train']['sdf_boundary_weight']==0
        path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True);text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/arm);sm=OUT/'configs'/(arm+'_smoke.yml');sm.write_text(yaml.safe_dump(smoke,sort_keys=False))
        records[arm]=dict(label=label,name=cfg['name'],config=str(path),config_sha256=digest(path),smoke=str(sm),gpu=gpu,initial=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']),bg=cfg['train']['bg_lf_weight'],mixture=.5 if mixed else 0.)
    for name in ['DATA_READY.json','SOURCE_ELIGIBILITY.json']:
        link=OUT/name
        if not link.exists():link.symlink_to(BASE/name)
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=records,base_config_sha256=digest(source),bank_ready_sha256=digest(BASE/'DATA_READY.json'),initialization='same frozen seeded V3M random state as completed six runs;fresh Adam',schedule=[(0,1e-4),(200000,5e-5),(300000,2.5e-5),(400000,1e-5),(450000,5e-6)]))
    return records


def matrix():
    """@luna-comment Input:authorized/existing experiment set. Output:Chinese causalcomparison plan. Key Steps:pair differing factors;identify missingaux controls andreplicates. Key Params:recommendations do notlaunch jobs. State/Dependencies:owned static HTML."""
    text='''<!doctype html><meta charset="utf-8"><title>V3M 对照矩阵</title><style>body{font:17px/1.8 system-ui;max-width:1100px;margin:35px auto;color:#18314b}td,th{padding:10px;border-bottom:1px solid #ccd6e0;text-align:left}table{width:100%;border-collapse:collapse}</style><h1>V3M · 已有模型＋三组新对照</h1><p>统一7628训练/620验证、同一随机初始化、batch4、500K、相同LR。所有模型保留GT-LQ对比度对齐和有效文字采样；无SDF/CP。</p><table><tr><th>要判断的因素</th><th>单变量比较</th></tr><tr><td>可信边缘guard</td><td>已有Full＋guard真实LQ ↔ 新Full无guard真实LQ</td></tr><tr><td>背景低频loss</td><td>新Full无guard真实LQ ↔ 新L1＋MF＋UV真实LQ；已有Full无guard混合50% ↔ 新L1＋MF＋UV混合50%</td></tr><tr><td>50%GT轻退化</td><td>新Full无guard真实LQ ↔ 已有Full无guard混合50%；新L1＋MF＋UV真实LQ ↔ 新L1＋MF＋UV混合50%</td></tr><tr><td>MF＋UV的联合贡献</td><td>已有L1-only真实LQ ↔ 新L1＋MF＋UV真实LQ</td></tr></table><p>Full＝L1＋Missing(1)＋False(0.25)＋UV(1)＋背景低频(2)。这里的无guard只关闭可信边缘抵消保护；第2/3组另外关闭背景低频loss。</p><h2>还建议补充的对照（仅建议）</h2><ol><li><b>L1＋Missing/False，真实LQ，500K</b>；<b>L1＋UV，真实LQ，500K</b>：与已有L1-only和新L1＋MF＋UV构成2×2矩阵，可拆分MF、UV及其交互。</li><li><b>L1-only＋50%GT轻退化，500K</b>：与已有L1-only真实LQ直接比较，判断混合退化在没有辅助loss时的作用。</li><li>先看上述方向，再对关键差异换随机种子复验；单个种子可支持本轮归因，但不能证明所有初始化都稳定成立。</li></ol><h2>评价口径</h2><p>主要比较同一37张Mojito验证图和583张原验证图；068属于训练图，可做机制观察，不能代替验证结论。先看模型原始输出，再统一后处理；同一iter比较，分别记录笔画内部、边缘、伪影和相对GT误差。不同loss组合的训练总loss不直接排名。</p><p><a href="index.html">返回三组训练监控</a></p>'''
    (OUT/'CONTROL_MATRIX.html').write_text(text,encoding='utf-8')


def main():
    """@luna-comment Input:three explicitly authorized arms. Output:checked resident training andETA. Key Steps:lock,freeze recipes,allCUDAgates,capacitylaunch,monitor500K checkpoints. Key Params:GPU0/1/2,no oldjobcontrol. State/Dependencies:owned processes only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not list((OUT/'launch').glob('*.json'))
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()));records=prepare();matrix();command=[sys.executable,'-u',str(ROOT/'tools/train_v3m_loss_ablation.py')]
    for arm,r in records.items():
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='启动检查',arm=arm));env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        monitor.run_checked(command+['--smoke-config',r['smoke'],'--smoke-output',str(OUT/(arm+'_smoke.json'))],OUT/(arm+'_smoke.log'),env)
        assert monitor.read(OUT/(arm+'_smoke.json'))['passed']
    processes={}
    for arm,r in records.items():
        assert digest(r['config'])==r['config_sha256'] and digest(r['initial'])==r['initial_sha256']
        while True:
            free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
            if free[r['gpu']]>=6000:break
            save(OUT/'CONTROLLER_STATUS.json',dict(phase='等待GPU显存',gpu=r['gpu']));publish();time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None);(OUT/'logs').mkdir(exist_ok=True)
        with (OUT/'logs'/(arm+'.log')).open('xb') as f:p=subprocess.Popen(command+['-opt',r['config']],env=env,cwd=OUT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        processes[arm]=p;save(OUT/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=r['gpu'],config=r['config'],initial_sha256=r['initial_sha256']))
    save(OUT/'CONTROLLER_STATUS.json',dict(phase='三组训练中'))
    while processes:
        for arm,p in list(processes.items()):
            if p.poll() is not None:
                final=OUT/'experiments'/records[arm]['name']/'models/net_g_500000.pth';assert p.returncode==0 and final.exists(),arm+' failed'
                save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=str(final)));del processes[arm]
        publish();time.sleep(30)
    save(OUT/'COMPLETE.json',dict(passed=True,arms=3,total_iter=500000));save(OUT/'CONTROLLER_STATUS.json',dict(phase='全部完成'));publish()


monitor.publish=publish
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'FAILED.json',dict(error=repr(e)));save(OUT/'CONTROLLER_STATUS.json',dict(phase='异常停止',error=repr(e)));publish();raise
