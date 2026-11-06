"""Freeze three complete380K checkpoints and append accepted fixed32 results."""
from pathlib import Path
import sys,json,shutil,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if sys.platform=='win32':
    from types import SimpleNamespace
    core=SimpleNamespace(DATA=Path('/AI_DNS/guang.yang/TextSR/data'))
else:
    from tools import evaluate_five_controls_mojito32 as core
from tools.run_uniform20_steep_scurve import save,digest
SOURCE=core.DATA/'filtered_trusted_no_mojito_three_500k_20261105'
core.OUT=core.DATA/'filtered_trusted_three_380k_mojito32_20261106'
core.ARMS={'filtered_l1_380k':'V3M · 新版可信GT · L1-only','filtered_ori_380k':'V3M · 新版可信GT · L1+R1_ori','filtered_norm_380k':'V3M · 新版可信GT · L1+R1_norm'}
core.__file__=__file__


def prepare():
    """@luna-comment Input:three numeric380K model/state receipts and exact fixed32. Output:immutable evaluation copies/hash bindings. Key Steps:strict source config/model/state verification;freeze before CUDA launch. Key Params:380000 shared iteration,no latest alias. State/Dependencies:read-only training artifacts;owned output only."""
    import yaml
    declared=json.loads((SOURCE/'CONFIG_BINDINGS.json').read_text())['runs'];bindings={};(core.OUT/'models').mkdir(exist_ok=True)
    for arm,kind in zip(core.ARMS,['l1','ori','norm']):
        b=declared[kind];config=Path(b['config']);assert digest(config)==b['config_sha256'];cfg=yaml.safe_load(config.read_text());receipt=SOURCE/'experiments'/cfg['name']/'training_states/380000.verified.json';r=json.loads(receipt.read_text())
        assert r['iter']==380000 and digest(r['model'])==r['model_sha256'] and digest(r['state'])==r['state_sha256']
        model=core.OUT/'models'/(arm+'.pth');conf=core.OUT/'models'/(arm+'.yml');shutil.copy2(r['model'],model);shutil.copy2(config,conf);assert digest(model)==r['model_sha256'] and digest(conf)==b['config_sha256']
        bindings[arm]=dict(config=str(conf),config_sha256=digest(conf),checkpoint=str(model),checkpoint_sha256=digest(model),source_checkpoint=r['model'],iteration=380000,label=core.ARMS[arm],source_state_sha256=r['state_sha256'],stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    shutil.copy2(core.DATA/'mojito280_subset32_20261104/selection.json',core.OUT/'selection.json');assert len(json.loads((core.OUT/'selection.json').read_text())['samples'])==32
    save(core.OUT/'MODEL_BINDINGS.json',bindings);save(core.OUT/'RUN_BINDING.json',dict(selection_sha256=digest(core.OUT/'selection.json'),post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'),outputs=192,shared_iteration=380000,train_count=5284,val_count=583,no_mojito_training=True,gt_mixture_probability=0,contrast_alignment=False))


def publish():
    """@luna-comment Input:audited192 new outputs/current local32 page. Output:six added380K choices and unchanged historical options. Key Steps:bind exact input identities;validate only new native headers;atomic backup/render/manifest. Key Params:no old image revalidation,raw/post both retained. State/Dependencies:accepted six-panel renderer,NAS images."""
    from concurrent.futures import ThreadPoolExecutor
    from tools.build_camera_2x2_viewer import render_html,file_url
    from tools.publish_mojito32_all_models import check_image
    source=Path('Y:/guang.yang/TextSR/data')/core.OUT.name;assert json.loads((source/'COMPLETE.json').read_text())['passed'];audit=json.loads((source/'AUDIT.json').read_text());assert audit['outputs']==192 and not audit['old_results_rechecked']
    out=ROOT/'evaluation/mojito32_all_models_20261104';data=json.loads((out/'manifest.json').read_text(encoding='utf-8'));rows={r['id']:r for r in data['rows']};selection=json.loads((source/'selection.json').read_text())['samples'];original={r['id']:r for r in json.loads((ROOT/'evaluation/mojito280_subset32_20261104/selection.json').read_text())['samples']}
    assert {r['id'] for r in selection}==set(rows)==set(original)
    for r in selection:assert r['input_sha256']==original[r['id']]['input_sha256'] and r['source_sha256']==original[r['id']]['source_sha256']
    bindings=json.loads((source/'MODEL_BINDINGS.json').read_text());assert len(bindings)==3 and all(b['iteration']==380000 for b in bindings.values())
    keys={a+s for a in bindings for s in ['','_raw']};data['models']=[m for m in data['models'] if m['key'] not in keys];old_count=len(data['models'])
    for arm,b in bindings.items():
        for suffix,label in [('_raw','模型原始输出'),('','固化后处理')]:data['models'].append(dict(key=arm+suffix,label=b['label']+' | 380K | '+label,note='新版筛选可信背景GT；5284训练图，无Mojito；无退化/对比度对齐；256/224/batch8/crop8/smooth16'+('；未加后处理' if suffix else '；Y后处理0.75/2/24')))
    outputs=json.loads((source/'OUTPUTS.json').read_text())['samples'];assert len(outputs)==96;jobs=[]
    for r in outputs:
        row=rows[r['id']]
        for suffix,field in [('_raw','raw'),('','post')]:
            key=r['arm']+suffix;url=file_url(r[field]);row['images'][key]=url;jobs.append((row['id'],key,url,row['width'],row['height']))
    with ThreadPoolExecutor(max_workers=4) as pool:checked=list(pool.map(check_image,jobs))
    assert len(checked)==192 and len(data['models'])==old_count+6
    data.update(title='Mojito32 · 新版可信GT 三组380K',subtitle='固定32张 · L1-only / R1_ori / R1_norm · 相同380K',defaults='filtered_l1_380k_raw,filtered_ori_380k_raw,filtered_norm_380k_raw,v3_input,v3m_l1_raw,lq')
    note='\n新增三组为新版筛选可信背景GT训练的380K：5284张训练/583张验证均不含Mojito；不混合GT退化、关闭对比度对齐。三组只改变loss策略。文件名中的训练/验证标记沿用旧Mojito划分，本轮三组均未使用Mojito。旧版本数据/策略/迭代可能不同，保留作历史参照。原始输出和固化后处理均可选。'
    if note not in data['notes']:data['notes']+=note
    text=render_html(data);assert text.count('for(let i=0;i<4;i++)')==1;text=text.replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;')
    for name in ['manifest.json','index.html']:
        backup=out/(name+'.before_filtered380k')
        if not backup.exists():shutil.copy2(out/name,backup)
    for name,payload in [('manifest.json',json.dumps(data,ensure_ascii=False,indent=2)),('index.html',text)]:
        tmp=out/(name+'.filtered380k.tmp');tmp.write_text(payload,encoding='utf-8');tmp.replace(out/name)
    save(out/'FILTERED380K_AUDIT.json',dict(passed=True,rows=32,new_options=6,checked_new_paths=192,old_images_rechecked=False,source_audit_sha256=digest(source/'AUDIT.json'),records=checked));print(json.dumps(dict(passed=True,rows=32,total_options=len(data['models']),page=str(out/'index.html'))))


core.prepare=prepare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker','publish']);p.add_argument('--arm',choices=list(core.ARMS));a=p.parse_args()
    if a.mode=='publish':publish()
    elif a.mode=='worker':core.worker(a.arm)
    else:core.run()
