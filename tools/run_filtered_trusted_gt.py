"""Resident resumable quality screening and trusted-background GT materialization."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import sys,os,json,time,copy,fcntl,html,traceback
import hashlib
from datetime import datetime,timezone,timedelta
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.screen_mask_gt_quality import SOURCE,OUT,measure,decide
from tools.prepare_r1_width_bank import save,digest
WORKERS=8
REASONS={'spatial_mask_image_misalignment':'局部错位 / 空腔风险','insufficient_lq_text_detail':'失焦 / 有效文字高频不足'}


def read(path):
    """@luna-comment Input:owned atomic JSON. Output:document or empty if absent. Key Steps:UTF8 parse. Key Params:corruption raises. State/Dependencies:filesystem."""
    p=Path(path);return json.loads(p.read_text()) if p.exists() else {}


def configure():
    """@luna-comment Input:completed source and calibration/parity receipts. Output:frozen rules/code/source binding. Key Steps:verify all identities,assert four failures and07693 keep;freeze exact settings. Key Params:no new training or original mutation. State/Dependencies:existing immutable8352 artifacts."""
    thresholds=dict(misaligned_patch_fraction=.18,misaligned_text_fraction=.15,max_good_detail_fraction=.10,minimum_coherent_hf_dn=.60,minimum_text_band_dn=1.5,minimum_hf_ratio=.12)
    calibration=read(OUT/'CALIBRATION.json');assert calibration['passed'] and read(OUT/'PARITY_07693.json')['different_pixels']==0
    byid={r['viewer_id']:r for r in calibration['records']}
    decisions={i:decide(byid[i],thresholds) for i in ['07610','07613','07647','07649','07693']}
    assert all(not decisions[i]['keep'] for i in ['07610','07613','07647','07649']) and decisions['07693']['keep']
    source=SOURCE/'SOURCE_ROWS.json';manifest=SOURCE/'source_view/manifests/train.jsonl';complete=read(SOURCE/'GT_COMPLETE.json')
    assert complete['passed'] and digest(source)==complete['source_rows_sha256'] and digest(manifest)==complete['manifest_sha256']
    files=['tools/screen_mask_gt_quality.py','tools/trusted_small_background_repair.py','tools/preview_07693_trusted_background.py']
    binding=dict(thresholds=thresholds,source_rows_sha256=digest(source),published_gt_manifest_sha256=digest(manifest),code_sha256={f:digest(ROOT/f) for f in files},calibration_sha256=digest(OUT/'CALIBRATION.json'),parity_sha256=digest(OUT/'PARITY_07693.json'),workers=WORKERS,source_count=8352,training_changed=False)
    path=OUT/'BINDING.json'
    if path.exists():assert read(path)==binding,'Frozen source/code/settings changed'
    else:save(path,binding)
    save(OUT/'CALIBRATION_DECISIONS.json',dict(passed=True,thresholds=thresholds,cases=decisions))
    final_rules=dict(version='dual-observation-local-shift-v1',minimum_lq_correlation=.45,maximum_gt_lq_displacement_difference=2.,note='Final misalignment exclusion requires original teacher and LQ to corroborate the same local mask displacement; preliminary screening counts are not final.')
    if (OUT/'FINAL_RULES.json').exists():assert read(OUT/'FINAL_RULES.json')==final_rules
    else:save(OUT/'FINAL_RULES.json',final_rules)
    return binding


def final_decisions(records):
    """@luna-comment Input:completed original measurements and frozen consensus rule. Output:final keep/reject records with corroborated shifts. Key Steps:require both original GT/LQ evidence within2px;retain raw metrics for audit;recheck four negatives/positive. Key Params:reuse all image measurements,no new registration. State/Dependencies:owned final decision metadata."""
    rules=read(OUT/'FINAL_RULES.json');thresholds=read(OUT/'BINDING.json')['thresholds'];result=[]
    for record in records:
        r=copy.deepcopy(record);m=r['measurements'];patches=m['patches']
        confirmed=[p for p in patches if p['gt']['peak']>=.45 and np.hypot(p['gt']['dx'],p['gt']['dy'])>=1.75 and p['gt']['peak']-p['gt']['zero']>=.04 and p['lq']['peak']>=rules['minimum_lq_correlation'] and np.hypot(p['gt']['dx']-p['lq']['dx'],p['gt']['dy']-p['lq']['dy'])<=rules['maximum_gt_lq_displacement_difference']]
        m['gt_only_misaligned_fraction']=m['misaligned_fraction'];m['gt_only_misaligned_text_fraction']=m['misaligned_text_fraction']
        m['misaligned_fraction']=len(confirmed)/max(len(patches),1);m['misaligned_text_fraction']=sum(p['text_pixels'] for p in confirmed)/max(sum(p['text_pixels'] for p in patches),1)
        r['preliminary_decision']=r['decision'];r['decision']=decide(m,thresholds);r['final_rules_sha256']=digest(OUT/'FINAL_RULES.json');result.append(r)
    byid={r['id']:r for r in result}
    assert all(not byid[i]['decision']['keep'] for i in ['07610','07613','07647','07649']) and byid['07693']['decision']['keep']
    counts=dict(retained=sum(r['decision']['keep'] for r in result),rejected=sum(not r['decision']['keep'] for r in result))
    save(OUT/'FINAL_DECISIONS.json',dict(passed=True,source_records_sha256=digest(OUT/'SCREEN_RECORDS.json'),rules_sha256=digest(OUT/'FINAL_RULES.json'),counts=counts,records=result))
    return result


def screen_one(job):
    """@luna-comment Input:original row and frozen binding. Output:hash-bound quality/decision receipt. Key Steps:resume exact bindings;measure only original teacher/LQ/M0;freeze reasons. Key Params:no subjective visual score. State/Dependencies:owned JSON receipt only."""
    index,row,binding=job;path=OUT/'screen'/row['dataset']/(row['sample_id']+'.json');signature=dict(binding_sha256=binding,source_hashes=row['output_sha256'])
    if path.exists():
        r=read(path);assert r['binding']==signature;return r
    measurements=measure(row);thresholds=read(OUT/'BINDING.json')['thresholds'];decision=decide(measurements,thresholds)
    record=dict(passed=True,id=f'{index+1:05d}',dataset=row['dataset'],sample_id=row['sample_id'],binding=signature,decision=decision,measurements=measurements)
    save(path,record);return record


def repair_one(job):
    """@luna-comment Input:retained published row and frozen quality receipt. Output:separate trusted GT/updated provenance row. Key Steps:strict original/published/mask hashes;approved equivalent repair;atomic PNG and receipt. Key Params:region-only GT preserved,source unchanged. State/Dependencies:owned target files only."""
    from tools.trusted_small_background_repair import repair
    row,quality,binding=job;cv2.setNumThreads(1);folder=OUT/'targets'/row['dataset']/row['sample_id'];path=folder/'receipt.json'
    signature=dict(binding_sha256=binding,gt_sha256=row['output_sha256']['gt'],mask_sha256=row['output_sha256']['text_mask'],decision=quality['decision'])
    if path.exists():
        r=read(path);assert r['binding']==signature and digest(r['row']['files']['gt'])==r['row']['output_sha256']['gt'];return r
    assert quality['decision']['keep']
    new=copy.deepcopy(row)
    if row['repair_mode']=='preserve_region_mask':
        audit=dict(changed_pixels=0,repair_not_applicable='region-only mask; preserve original GT')
    else:
        for k in ['gt','text_mask','valid_mask']:assert digest(row['files'][k])==row['output_sha256'][k]
        original=row['previous_training_target'];assert digest(original['path'])==original['sha256']
        raw=cv2.cvtColor(cv2.imread(original['path']),cv2.COLOR_BGR2RGB);previous=cv2.cvtColor(cv2.imread(row['files']['gt']),cv2.COLOR_BGR2RGB)
        output,audit=repair(raw,previous,cv2.imread(row['files']['text_mask'],0),cv2.imread(row['files']['valid_mask'],0))
        folder.mkdir(parents=True,exist_ok=True);target=folder/'gt.png';tmp=folder/'gt.tmp.png';assert cv2.imwrite(str(tmp),cv2.cvtColor(output,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]);tmp.replace(target)
        new['previous_mask_repaired_target']=dict(path=row['files']['gt'],sha256=row['output_sha256']['gt']);new['files']['gt']=str(target);new['output_sha256']['gt']=digest(target);new['target_version']='quality_filtered_trusted_small_bg_v1_20261105'
    new['quality_screen']=dict(id=quality['id'],receipt=str(OUT/'screen'/row['dataset']/(row['sample_id']+'.json')),decision=quality['decision'])
    record=dict(passed=True,id=quality['id'],binding=signature,row=new,audit=audit);save(path,record);return record


def publish(phase,completed,total,started,counts=None,error=None,cached=0):
    """@luna-comment Input:stage counters/reason totals. Output:readable live Chinese monitor and measured stage ETA. Key Steps:separate scan and correction phases;show keep/reject reasons,links. Key Params:stage ETA only,not training promise. State/Dependencies:owned atomic metadata/page."""
    tz=timezone(timedelta(hours=8));now=time.time();elapsed=now-started;measured=max(0,completed-cached);eta=(total-completed)*elapsed/measured if measured else None
    stamp=datetime.fromtimestamp(now+eta,tz).strftime('%m-%d %H:%M') if eta is not None else '首批完成后更新'
    labels={'screening':'全量质量筛选','repairing':'保留集 · 可信小区域背景修复','complete':'全部完成','error':'异常，等待自动恢复'};label=labels[phase]
    record=dict(phase=phase,completed=completed,total=total,cached=cached,elapsed_seconds=elapsed,eta_seconds=eta,estimated_finish=stamp,counts=counts or {},updated=now,pid=os.getpid(),workers=WORKERS,error=error)
    save(OUT/'STATUS.json',record)
    details=['8个CPU进程；复用已有GT / V3 LQ / masks，不跑OCR和教师模型','当前阶段预计完成：'+stamp,'错位最终要求原GT与LQ的局部偏移互相印证；筛选过程中的数量为初判','旧版数据和正在运行的训练保持原版本']
    if counts:details.append('统计：'+json.dumps({REASONS.get(k,k):v for k,v in counts.items()},ensure_ascii=False))
    snap=dict(updated=datetime.now(tz).isoformat(),runs=[],datasets=[dict(label='筛除错位/严重失焦 → 可信背景GT',status=label,current=completed,total=total,details=details)])
    save(OUT/'monitor.json',snap)
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>质量筛选与可信背景GT</title><style>body{font:18px/1.7 system-ui;background:#eef3f8;color:#17324d;max-width:1100px;margin:30px auto}article{background:white;border-radius:14px;padding:25px}progress{width:100%;height:22px}a{color:#146a9a}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}</style><h1>全量质量筛选 · 可信背景GT</h1><article><h2>'+label+'</h2><p>'+str(completed)+' / '+str(total)+'</p><progress value="'+str(completed)+'" max="'+str(max(1,total))+'"></progress><p>'+'</p><p>'.join(html.escape(x) for x in details)+'</p>'
    if error:body+='<p>'+html.escape(error)+'</p>'
    body+='</article><p><a href="decisions.html">查看保留/剔除清单及原因</a>　<a href="compare_retained.html">保留集新旧对比（完成后）</a>　<a href="compare_rejected.html">剔除样本检查（筛选完成后）</a></p><p>更新：'+datetime.now(tz).strftime('%m-%d %H:%M:%S')+'</p>'
    tmp=OUT/'index.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')


def process_stage(phase,jobs,worker):
    """@luna-comment Input:stage jobs and accepted worker. Output:all receipts or explicit failure. Key Steps:eight processes,progress heartbeat,record each error without silent sample loss. Key Params:resumable per-parent receipts. State/Dependencies:owned child pool,immutable inputs."""
    started=time.time();records=[];counts=Counter();errors=[]
    if phase=='repairing':
        summary=read(OUT/'FINAL_DECISIONS.json')['counts'];counts.update({'保留':summary['retained'],'剔除':summary['rejected']})
    cached=0
    for j in jobs:
        row=j[1] if phase=='screening' else j[0]
        path=(OUT/'screen'/row['dataset']/(row['sample_id']+'.json')) if phase=='screening' else OUT/'targets'/row['dataset']/row['sample_id']/'receipt.json'
        cached+=path.exists()
    publish(phase,0,len(jobs),started,dict(counts),cached=cached)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(worker,j):i for i,j in enumerate(jobs)}
        for future in as_completed(futures):
            try:
                record=future.result();records.append(record)
                if phase=='screening':counts['保留' if record['decision']['keep'] else '剔除']+=1;counts.update(record['decision']['reasons'])
            except Exception as error:
                errors.append(dict(job=futures[future],error=repr(error)));save(OUT/'STAGE_ERRORS.json',dict(phase=phase,errors=errors))
            n=len(records)+len(errors)
            if n%10==0 or n==len(jobs):publish(phase,n,len(jobs),started,dict(counts),cached=cached);print(phase,n,len(jobs),dict(counts),flush=True)
    assert not errors,errors[:3]
    return sorted(records,key=lambda r:r['id'])


def decision_page(records):
    """@luna-comment Input:all quality decisions. Output:filterable reason list and source-viewer links. Key Steps:preserve IDs and both outcomes,escape all names,embed simple filters. Key Params:physical originals never deleted. State/Dependencies:standalone local-file HTML."""
    body='<!doctype html><meta charset="utf-8"><title>筛选清单</title><style>body{font:15px system-ui;margin:25px}input,select{font:inherit;padding:7px;margin:8px}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}.reject{background:#fff0ed}</style><h1>质量筛选清单</h1><p>同一张图可能同时触发两种原因；原始数据未删除。点击编号回看原GT、已有修复、LQ与mask。</p><input id="q" placeholder="编号 / 文件名"><select id="s"><option value="">全部</option><option value="keep">保留</option><option value="reject">剔除</option></select><table><thead><tr><th>编号 / 文件</th><th>结果</th><th>原因 / 边界标记</th><th>错位窗口占比</th><th>有效高频Q75(DN)</th></tr></thead><tbody>'
    for r in records:
        m=r['measurements'];d=r['decision'];state='keep' if d['keep'] else 'reject';name=r['id']+' · '+r['dataset']+'/'+r['sample_id'];reason='；'.join(REASONS.get(x,x) for x in d['reasons']+d['flags']) or '未触发剔除规则'
        link='file:///E:/workspace/AISR/Real-ESRGAN/evaluation/vosr_s22_mask_repair_mix50_20261104/compare.html#'+r['id']+';lq,repaired,input,mask_original;lq'
        body+='<tr data-state="'+state+'" class="'+state+'"><td><a href="'+html.escape(link,quote=True)+'">'+html.escape(name)+'</a></td><td>'+('保留' if d['keep'] else '剔除')+'</td><td>'+html.escape(reason)+'</td><td>'+f'{m["misaligned_fraction"]:.1%}'+'</td><td>'+f'{m["coherent_hf_q75_dn"]:.2f}'+'</td></tr>'
    body+='</tbody></table><script>function filterRows(){const q=document.getElementById("q").value.toLowerCase(),s=document.getElementById("s").value;for(const r of document.querySelectorAll("tbody tr"))r.hidden=!!((s&&r.dataset.state!==s)||!r.textContent.toLowerCase().includes(q));}document.getElementById("q").oninput=filterRows;document.getElementById("s").onchange=filterRows;</script>'
    tmp=OUT/'decisions.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'decisions.html')


def publish_viewer(records,targets,retained):
    """@luna-comment Input:completed quality/target receipts. Output:native retained or rejected four-panel review. Key Steps:original row pairing,exact NAS URLs,explicit rejection labels,shared renderer. Key Params:no copying images or platform registration. State/Dependencies:existing native viewer assets."""
    from tools.build_camera_2x2_viewer import render_html,file_url
    from PIL import Image
    original=json.loads((SOURCE/'SOURCE_ROWS.json').read_text());lookup={(r['dataset'],r['sample_id']):r for r in original};old=[json.loads(l) for l in (SOURCE/'source_view/manifests/train.jsonl').read_text().splitlines() if l];prior={(r['dataset'],r['sample_id']):r for r in old};target_lookup={r['id']:r for r in targets};rows=[]
    for r in records:
        if r['decision']['keep']!=retained:continue
        key=(r['dataset'],r['sample_id']);a=lookup[key];b=prior[key];h,w=r['measurements']['shape_hw'];images=dict(repaired=file_url(b['files']['gt']),input=file_url(a['files']['lq']),mask=file_url(a['files']['text_mask']))
        if retained:images['trusted']=file_url(target_lookup[r['id']]['row']['files']['gt'])
        reason='保留' if retained else '；'.join(REASONS[x] for x in r['decision']['reasons'])
        rows.append(dict(id=r['id'],name=r['dataset']+'/'+r['sample_id']+'【'+reason+'】',width=w,height=h,lq=file_url(a['files']['gt']),images=images))
    if not rows:return
    models=[dict(key='lq',label='原 VOSR+S2.2 GT',note='原教师GT'),dict(key='repaired',label='旧版 · 近邻背景修复GT',note='上一版本，保留作比较'),dict(key='input',label='LQ · 已有V3输入',note='用于失焦/细节判断，不重新预处理'),dict(key='mask',label='原 TextMask M0',note='用于检查空间对齐')]
    if retained:models.append(dict(key='trusted',label='新版 · 可信小区域背景GT',note='仅保留集；Mojito区域mask来源保留原GT'))
    data=dict(title=('保留集 · 可信背景GT' if retained else '剔除集 · 对齐/失焦检查'),subtitle=str(len(rows))+'张 · 原编号保留',notes='自动筛选规则与指标见BINDING.json和决策清单。错位判断使用原mask和原教师图；失焦判断使用LQ的文字区域细节。\n保留/剔除不删除原数据；当前训练仍使用原冻结版本。',models=models,defaults='lq,repaired,'+('trusted,input' if retained else 'input,mask'),rows=rows)
    name='compare_retained' if retained else 'compare_rejected';text=render_html(data).replace('与 LQ','与原 GT').replace('与LQ','与原GT');tmp=OUT/(name+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(OUT/(name+'.html'))
    save(OUT/(name+'_manifest.json'),data)


def main():
    """@luna-comment Input:user-authorized filter/repair task. Output:separate complete filtered GT version. Key Steps:exclusive lock,hash freeze,resumable screening then retained repair,complete manifests/viewers. Key Params:8CPU,no training launch. State/Dependencies:owned output and cron recovery only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if read(OUT/'COMPLETE.json').get('passed'):return
    started=time.time()
    try:
        configure();binding=digest(OUT/'BINDING.json');original=json.loads((SOURCE/'SOURCE_ROWS.json').read_text());published=[json.loads(l) for l in (SOURCE/'source_view/manifests/train.jsonl').read_text().splitlines() if l];lookup={(r['dataset'],r['sample_id']):r for r in published}
        if not (OUT/'SCREEN_COMPLETE.json').exists():
            records=process_stage('screening',[(i,r,binding) for i,r in enumerate(original)],screen_one)
            save(OUT/'SCREEN_RECORDS.json',records);save(OUT/'SCREEN_COMPLETE.json',dict(passed=True,count=len(records),records_sha256=digest(OUT/'SCREEN_RECORDS.json'),kept=sum(r['decision']['keep'] for r in records)))
        else:records=read(OUT/'SCREEN_RECORDS.json');assert digest(OUT/'SCREEN_RECORDS.json')==read(OUT/'SCREEN_COMPLETE.json')['records_sha256']
        assert len(records)==len(original)==8352
        records=final_decisions(records)
        decision_page(records);publish_viewer(records,[],False)
        retained=[r for r in records if r['decision']['keep']]
        repair_binding=hashlib.sha256((binding+digest(OUT/'FINAL_RULES.json')).encode()).hexdigest()
        jobs=[(lookup[r['dataset'],r['sample_id']],r,repair_binding) for r in retained];targets=process_stage('repairing',jobs,repair_one)
        view=OUT/'source_view/manifests';view.mkdir(parents=True,exist_ok=True);tmp=view/'train.tmp';tmp.write_text(''.join(json.dumps(r['row'])+'\n' for r in targets));tmp.replace(view/'train.jsonl')
        save(OUT/'REJECTED.json',[dict(id=r['id'],dataset=r['dataset'],sample_id=r['sample_id'],**r['decision']) for r in records if not r['decision']['keep']])
        save(OUT/'REPAIR_RECORDS.json',targets);publish_viewer(records,targets,True)
        counts=Counter(x for r in records for x in r['decision']['reasons']);result=dict(passed=True,total=8352,retained=len(retained),rejected=8352-len(retained),reason_counts=dict(counts),binding_sha256=binding,final_rules_sha256=digest(OUT/'FINAL_RULES.json'),manifest_sha256=digest(view/'train.jsonl'),repair_records_sha256=digest(OUT/'REPAIR_RECORDS.json'),original_data_deleted=False,training_changed=False,elapsed_seconds=time.time()-started)
        save(OUT/'COMPLETE.json',result);publish('complete',len(retained),len(retained),started,dict(counts,保留=len(retained),剔除=8352-len(retained)))
    except Exception as error:
        record=dict(error=repr(error),traceback=traceback.format_exc(),updated=time.time());save(OUT/'LAST_ERROR.json',record);publish('error',0,8352,started,error=repr(error));raise


if __name__=='__main__':main()
