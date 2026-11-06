"""Four frozen Full models on selected4x/8x photos at user-requested input scales."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time

DATA=Path('/AI_DNS/guang.yang/TextSR/data')
BASE=DATA/'v3pre_three_mojito280_20261028'
PRIOR=DATA/'s22_full_four_mojito280_20261029'
OUT=DATA/'s22_full_mojito280_downsample_20261029'
sys.path.insert(0,str(BASE/'_build'))
ARMS=('paired_full','gt_degrade30_full','identity_full','guard_full')


def digest(path):
    """@luna-comment Input:read-only file. Output:SHA256. Key Steps:stream4MiB. Key Params:no source writes. State/Dependencies:filesystem."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    """@luna-comment Input:owned JSON/value. Output:atomic receipt. Key Steps:sibling replacement. Key Params:UTF8. State/Dependencies:owned directory only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def now():
    """@luna-comment Input:none. Output:CST timestamp. Key Steps:explicit UTC+8. Key Params:client timezone. State/Dependencies:clock."""
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def tag(factor):
    """@luna-comment Input:denominator. Output:stable scale key. Key Steps:decimal formatting. Key Params:1.5/2/3/4 only. State/Dependencies:pure."""
    return 'd'+format(factor,'g').replace('.','p')


def image_write(path,bgr):
    """@luna-comment Input:owned path/BGR8. Output:lossless PNG. Key Steps:accepted atomic writer. Key Params:no JPEG export. State/Dependencies:accepted utility."""
    from tools.evaluate_offline_camera_matrix import image_write as accepted
    accepted(path,bgr)


def prepare_parent(entry):
    """@luna-comment Input:one selected original V3 sample. Output:2or3 reduced inputs and roundtrip controls. Key Steps:verify original,area resize,lossless display rotation,cubic return. Key Params:round(native dimension/factor). State/Dependencies:read-only parent;owned per-case files."""
    import cv2
    from tools.evaluate_v3pre_three_mojito280 import canonical
    cv2.setNumThreads(1);number=int(entry['id']);role=(number-1)%4
    assert role in (2,3);zoom=4 if role==2 else 8;factors=(1.5,2.) if zoom==4 else (2.,3.,4.)
    assert digest(entry['input'])==entry['input_sha256']
    source=cv2.imread(entry['input'],cv2.IMREAD_UNCHANGED);assert list(source.shape)==entry['stored_shape']
    h,w=source.shape[:2];variants=[]
    for factor in factors:
        key=entry['id']+'_'+tag(factor);receipt=OUT/'prepared/receipts'/(key+'.json')
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['input_sha256']==entry['input_sha256'] and r['factor']==factor
            assert digest(r['reduced_input'])==r['reduced_input_sha256'] and digest(r['roundtrip'])==r['roundtrip_sha256'];variants.append(r);continue
        width,height=int(round(w/factor)),int(round(h/factor))
        reduced=cv2.resize(source,(width,height),interpolation=cv2.INTER_AREA)
        path=OUT/'prepared/inputs'/(key+'.png');image_write(path,reduced)
        display=canonical(cv2.cvtColor(reduced,cv2.COLOR_BGR2RGB),entry['orientation'])
        control=cv2.resize(display,(entry['width'],entry['height']),interpolation=cv2.INTER_CUBIC)
        control_path=OUT/'prepared/roundtrip'/(key+'.png');image_write(control_path,cv2.cvtColor(control,cv2.COLOR_RGB2BGR))
        r=dict(task_id=key,id=entry['id'],scene=(number-1)//4+1,camera_zoom=zoom,factor=factor,scale_key=tag(factor),input_sha256=entry['input_sha256'],reduced_input=str(path),reduced_input_sha256=digest(path),reduced_stored_hw=[height,width],original_stored_hw=[h,w],actual_down_factor_hw=[h/height,w/width],orientation=entry['orientation'],display_wh=[entry['width'],entry['height']],roundtrip=str(control_path),roundtrip_sha256=digest(control_path),down_interpolation='INTER_AREA',up_interpolation='INTER_CUBIC')
        save(receipt,r);variants.append(r)
    return variants


def prepare():
    """@luna-comment Input:complete prior280 experiment. Output:350 immutable scale cases and same model bindings. Key Steps:verify prior completion,copy exact bindings,select70scenes lasttwo,parallel CPU preparation. Key Params:4preparation threads,no new preprocessing. State/Dependencies:owned output and immutable previous results."""
    import cv2
    import numpy as np
    assert json.loads((PRIOR/'COMPLETE.json').read_text())['passed']
    bindings=json.loads((PRIOR/'MODEL_BINDINGS.json').read_text());assert tuple(bindings)==ARMS
    for b in bindings.values():
        for field in ('checkpoint','config'):assert digest(b[field])==b[field+'_sha256']
    save(OUT/'MODEL_BINDINGS.json',bindings)
    samples=json.loads((PRIOR/'selection.json').read_text())['samples'];assert {e['id'] for e in samples}=={f'{i:03d}' for i in range(1,281)}
    selected=[e for e in samples if (int(e['id'])-1)%4 in (2,3)];assert len(selected)==140
    probe=np.arange(15*17*3,dtype=np.uint16).reshape(15,17,3).astype(np.uint8)
    assert np.array_equal(cv2.resize(probe,(17,15),interpolation=cv2.INTER_AREA),probe)
    cases=[]
    save(OUT/'PREPARATION.json',dict(phase='preparing',parents=0,total_parents=140,variants=0,total_variants=350))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,result in enumerate(pool.map(prepare_parent,selected),1):
            cases.extend(result);save(OUT/'PREPARATION.json',dict(phase='preparing',parents=n,total_parents=140,variants=len(cases),total_variants=350,updated_at=now()))
    cases.sort(key=lambda r:(int(r['id']),r['factor']))
    assert len(cases)==350 and len({r['task_id'] for r in cases})==350
    assert sum(r['camera_zoom']==4 for r in cases)==140 and sum(r['camera_zoom']==8 for r in cases)==210
    save(OUT/'selection.json',dict(cases=cases,original_selection_sha256=digest(PRIOR/'selection.json'),grouping='User-defined consecutive four IDs:1x,2x,4x,8x',same_prior_checkpoints=True))
    save(OUT/'PREPARATION.json',dict(phase='complete',parents=140,variants=350,updated_at=now()))
    save(OUT/'RUN_BINDING.json',dict(prior_bindings_sha256=digest(PRIOR/'MODEL_BINDINGS.json'),selection_sha256=digest(OUT/'selection.json'),runner_sha256=digest(__file__),new_model_outputs=1400,roundtrip_controls=350,down_interpolation='INTER_AREA',up_interpolation='INTER_CUBIC'))


def worker(arm):
    """@luna-comment Input:one frozen model and350 scale cases. Output:native reduced predictions/original-size display outputs. Key Steps:strict weights,accepted stitcher,cubic display return,hash-bound receipt. Key Params:256/224/batch8/crop8/smooth16. State/Dependencies:selected CUDA,owned outputs,no training changes."""
    import cv2
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    from tools.evaluate_v3pre_three_mojito280 import canonical
    import inspect
    cv2.setNumThreads(1);torch.set_num_threads(2)
    binding=json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    assert digest(inspect.getfile(restore_block))==binding['stitcher_sha256']
    for name in ('checkpoint','config'):assert digest(binding[name])==binding[name+'_sha256']
    model=build_network(yaml.safe_load(Path(binding['config']).read_text())['network_g']).cuda().eval();model.load_state_dict(torch.load(binding['checkpoint'],map_location='cpu')['params'],strict=True)
    selection_hash=digest(OUT/'selection.json');cases=json.loads((OUT/'selection.json').read_text())['cases']
    for n,case in enumerate(cases,1):
        key=case['task_id'];receipt=OUT/'results'/arm/'receipts'/(key+'.json')
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['binding']==binding and r['selection_sha256']==selection_hash and r['case']==case
            assert digest(r['native_output'])==r['native_sha256'] and digest(r['display_output'])==r['display_sha256'];continue
        started=time.time();assert digest(case['reduced_input'])==case['reduced_input_sha256']
        source=cv2.imread(case['reduced_input'],cv2.IMREAD_UNCHANGED);assert list(source.shape[:2])==case['reduced_stored_hw']
        prediction,details=restore_block(cv2.cvtColor(source,cv2.COLOR_BGR2RGB),model,'cuda')
        native=canonical(prediction,case['orientation']);display=cv2.resize(native,tuple(case['display_wh']),interpolation=cv2.INTER_CUBIC)
        native_path=OUT/'results'/arm/'native'/(key+'.png');display_path=OUT/'results'/arm/'final'/(key+'.png')
        image_write(native_path,cv2.cvtColor(native,cv2.COLOR_RGB2BGR));image_write(display_path,cv2.cvtColor(display,cv2.COLOR_RGB2BGR))
        save(receipt,dict(case=case,binding=binding,selection_sha256=selection_hash,native_output=str(native_path),native_sha256=digest(native_path),display_output=str(display_path),display_sha256=digest(display_path),native_display_hw=list(native.shape[:2]),display_hw=list(display.shape[:2]),restoration=details,seconds=time.time()-started))
        print(json.dumps(dict(arm=arm,done=n,total=350,task_id=key,seconds=time.time()-started)),flush=True)


def progress(started,phase,error=None):
    """@luna-comment Input:controller phase. Output:readable preparation/inference progress. Key Steps:receipt counts,observed ETA. Key Params:1400outputs,30s refresh. State/Dependencies:owned metadata."""
    counts={a:len(list((OUT/'results'/a/'receipts').glob('*.json'))) for a in ARMS};done=sum(counts.values());elapsed=time.time()-started;eta=elapsed/done*(1400-done) if done else None
    state=dict(phase=phase,counts=counts,completed=done,total=1400,elapsed_seconds=elapsed,eta_seconds=eta,updated_at=now(),error=error);save(OUT/'PROGRESS.json',state)
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Mojito4x/8x缩放验证</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;background:#f4f7fa}section{padding:18px;background:white;margin:14px}progress{width:100%}</style><h1>Mojito4x/8x下采样对比</h1><p>70场景 · 350缩放组合 · 4组模型 · 复用上一轮固定权重</p>'
    for a in ARMS:body+=f'<section><h2>{a}</h2><progress value="{counts[a]}" max="350"></progress><p>{counts[a]}/350</p></section>'
    body+=f'<p>{html.escape(phase)} · {done}/1400 · ETA '+(f'{eta/60:.1f}min' if eta is not None else 'warming up')+'</p>'
    if phase=='complete':body+='<p><a href="4x.html">4x拍摄对比</a> · <a href="8x.html">8x拍摄对比</a></p>'
    if error:body+='<pre>'+html.escape(error)+'</pre>'
    (OUT/'progress.html').write_text(body+'</html>',encoding='utf-8')


def verify_case(case,bindings):
    """@luna-comment Input:one prepared variant. Output:four audited receipt hashes. Key Steps:input/control/native/display hashes and geometry. Key Params:no synthetic quality score. State/Dependencies:read-only prepared/results files."""
    from PIL import Image
    for field in ('reduced_input','roundtrip'):assert digest(case[field])==case[field+'_sha256']
    audit={}
    for arm in ARMS:
        p=OUT/'results'/arm/'receipts'/(case['task_id']+'.json');r=json.loads(p.read_text());assert r['binding']==bindings[arm] and r['case']==case
        for field,sha in [('native_output','native_sha256'),('display_output','display_sha256')]:
            assert digest(r[field])==r[sha]
            with Image.open(r[field]) as im:assert list(im.size)==(list(reversed(r['native_display_hw'])) if field=='native_output' else case['display_wh'])
        audit[arm]=dict(receipt_sha256=digest(p),native_sha256=r['native_sha256'],display_sha256=r['display_sha256'])
    return dict(task_id=case['task_id'],models=audit)


def publish():
    """@luna-comment Input:1400 complete model outputs/350 controls. Output:two70-row scale viewers plus audit. Key Steps:full ID/hash checks,matched native references,existing synchronized renderer,model presets. Key Params:all displayed at original dimensions. State/Dependencies:owned pages,NAS image references."""
    from tools.build_camera_2x2_viewer import render_html,file_url
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());cases=json.loads((OUT/'selection.json').read_text())['cases'];expected={r['task_id'] for r in cases}
    for arm in ARMS:
        assert {p.stem for p in (OUT/'results'/arm/'receipts').glob('*.json')}==expected
        for field in ('checkpoint','config'):assert digest(bindings[arm][field])==bindings[arm][field+'_sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:audits=list(pool.map(lambda c:verify_case(c,bindings),cases))
    prior=json.loads((PRIOR/'manifest.json').read_text());previous={r['id']:r for r in prior['rows']};lookup={(r['id'],r['scale_key']):r for r in cases}
    native_audit=[]
    for zoom,factors in [(4,(1.5,2.)),(8,(2.,3.,4.))]:
        models=[dict(key='lq',label='Original camera input',note='Original camera source'),dict(key='v3_input',label='V3 preprocessed input',note='Exact original-sized model input cache')]
        for factor in factors:models.append(dict(key='control_'+tag(factor),label=f'Resize only / {factor:g}x',note='AREA downsample + CUBIC upsample;no model'))
        for arm in ARMS:
            label=bindings[arm]['label']+f' | {bindings[arm]["iteration"]//1000}K'
            models.append(dict(key=arm+'_native',label=label+' | original-size inference',note='Reused exact preceding280 output,same checkpoint'))
            for factor in factors:models.append(dict(key=arm+'_'+tag(factor),label=label+f' | input / {factor:g}x',note='AREA reduction -> model -> CUBIC return to original size'))
        ids=sorted({r['id'] for r in cases if r['camera_zoom']==zoom});assert len(ids)==70;rows=[]
        for identity in ids:
            old=previous[identity];images={'v3_input':old['images']['v3_input']}
            for factor in factors:
                case=lookup[(identity,tag(factor))];images['control_'+tag(factor)]=file_url(Path(case['roundtrip']))
            for arm in ARMS:
                receipt=PRIOR/'results'/arm/'receipts'/(identity+'.json');r=json.loads(receipt.read_text());path=PRIOR/'results'/arm/'final'/(identity+'.png')
                assert r['binding']==bindings[arm] and digest(path)==r['final_sha256'];images[arm+'_native']=file_url(path)
                native_audit.append(dict(id=identity,arm=arm,sha256=r['final_sha256']))
                for factor in factors:images[arm+'_'+tag(factor)]=file_url(OUT/'results'/arm/'final'/(identity+'_'+tag(factor)+'.png'))
            rows.append(dict(id=identity,name=f'Scene{(int(identity)-1)//4+1:02d} | {zoom}x capture | '+old['name'],width=old['width'],height=old['height'],lq=old['lq'],images=images))
        defaults=['v3_input',ARMS[0]+'_native',ARMS[0]+'_d1p5',ARMS[0]+'_d2'] if zoom==4 else [ARMS[0]+'_native',ARMS[0]+'_d2',ARMS[0]+'_d3',ARMS[0]+'_d4']
        notes=f'User-defined scene grouping:1x/2x/4x/8x in every four IDs. This page has70 {zoom}x photos. Cached V3 input -> INTER_AREA downsample -> unchanged model -> INTER_CUBIC upsample to original dimensions. Same four frozen checkpoints as preceding280 run. Resize-only controls isolate resampling effects;no second V3,tone or warp. Native reduced outputs are retained in receipts. Quality is user-reviewed.'
        manifest=dict(title=f'Mojito {zoom}x capture | input downsampling comparison',subtitle='70 scenes | fixed checkpoints | original-size display',models=models,rows=rows,defaults=','.join(defaults),notes=notes)
        text=render_html(manifest)
        preset="""<script>// @luna-comment Input:selected strategy. Output:four matching scale panels. Key Steps:set existing model keys,load and update hash. Key Params:original coordinate system. State/Dependencies:existing viewer only.
const scaleArms=ARMS_PLACEHOLDER,scaleZoom=ZOOM_PLACEHOLDER;const presetLabel=document.createElement('label');presetLabel.textContent='Model scale preset ';const preset=document.createElement('select');preset.id='scale-preset';preset.setAttribute('aria-label','Model scale preset');for(const arm of scaleArms){const option=document.createElement('option');option.value=arm;option.textContent=DATA.models.find(m=>m.key===arm+'_native').label;preset.append(option);}presetLabel.append(preset);document.querySelector('.toolbar').append(presetLabel);function applyScalePreset(){const arm=preset.value;const keys=scaleZoom===4?['v3_input',arm+'_native',arm+'_d1p5',arm+'_d2']:[arm+'_native',arm+'_d2',arm+'_d3',arm+'_d4'];panels.forEach((p,i)=>{p.select.value=keys[i];loadPanel(p);});updateHash();}preset.onchange=applyScalePreset;</script>""".replace('ARMS_PLACEHOLDER',json.dumps(ARMS)).replace('ZOOM_PLACEHOLDER',str(zoom))
        text=text.replace('</body>',preset+'</body>',1) if '</body>' in text else text+'\n'+preset
        text=text.replace('<aside>','<aside><p><a href="index.html">Scale test overview</a> · <a href="'+('8x.html' if zoom==4 else '4x.html')+'">Other zoom group</a></p>',1)
        save(OUT/f'manifest_{zoom}x.json',manifest);(OUT/f'{zoom}x.html').write_text(text,encoding='utf-8')
    save(OUT/'AUDIT.json',dict(passed=True,selected_photos=140,scenes=70,scale_cases=350,new_model_outputs=1400,native_reference_outputs=560,roundtrip_controls=350,records=audits,native_references=native_audit,selection_sha256=digest(OUT/'selection.json')))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mojito input-scale comparison</title><style>body{font:18px/1.7 system-ui;max-width:1100px;margin:40px auto}section{padding:24px;background:#eef3f8;margin:20px}</style><h1>Mojito4x/8x input-scale comparison</h1><section><a href="4x.html">4x capture: input /1.5 and /2</a><p>70 scenes;original-size inference and resize-only controls included.</p></section><section><a href="8x.html">8x capture: input /2,/3 and /4</a><p>70 scenes;all results displayed at original dimensions.</p></section><p>Same frozen four models as preceding280 comparison. Images remain on mappedY NAS.</p>',encoding='utf-8')


def run(resume=False):
    """@luna-comment Input:explicit scale experiment. Output:complete1400-result comparison. Key Steps:owned lock,prepare,one worker perGPU,progress,final audit. Key Params:do not touch training. State/Dependencies:owned evaluation children only."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not resume:
        assert not (OUT/'selection.json').exists(),'Existing immutable experiment';prepare()
    else:assert (OUT/'selection.json').exists()
    started=time.time();processes={};streams=[]
    try:
        for gpu,arm in enumerate(ARMS):
            folder=OUT/'logs';folder.mkdir(exist_ok=True);stream=(folder/(arm+'.log')).open('ab');streams.append(stream)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--arm',arm]
            p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'));processes[arm]=p;save(OUT/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=gpu,command=command))
        while any(p.poll() is None for p in processes.values()):
            for arm,p in processes.items():assert p.poll() in (None,0),(arm,p.returncode)
            progress(started,'running');time.sleep(15)
        assert all(p.returncode==0 for p in processes.values());progress(started,'auditing');publish();progress(started,'complete');save(OUT/'COMPLETE.json',dict(passed=True,new_model_outputs=1400,completed_at=now()))
    except Exception as error:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for p in processes.values():
            if p.poll() is None:p.wait(timeout=20)
        progress(started,'failed',repr(error));save(OUT/'FAILED.json',dict(error=repr(error)));raise
    finally:
        for stream in streams:stream.close()


def main():
    """@luna-comment Input:validated run/resume/worker/publish CLI. Output:owned scale evaluation. Key Steps:explicit dispatch. Key Params:existing run never refreshes weights. State/Dependencies:no public model API change."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','resume','worker','publish'));parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    if args.mode=='worker':assert args.arm;worker(args.arm)
    elif args.mode=='publish':publish()
    else:run(args.mode=='resume')


if __name__=='__main__':main()
