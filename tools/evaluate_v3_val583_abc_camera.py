"""Sequential completed VOSR2u V3 full/L1 validation and original A/B/C camera evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
BASE=DATA/'vosr_2u_block256_v1_20261023'
OUT=DATA/'v3_val583_abc_camera280_20261026'
VAL=BASE/'source_view/manifests/val.jsonl'
VAL_ARMS=('a_v3','l1_v3')
CAM_ARMS=('a_v3','b_v3','c_sr')
LABELS=dict(a_v3='A组 V3 · VOSR2u · 完整损失 · 原始权重起训',b_v3='B组 V3 · VOSR2u · 完整损失 · 上版分块权重起训',c_sr='C组 V5compact X2SR · VOSR2u · 完整损失',l1_v3='V3 · VOSR2u · L1-only')


def prepare():
    """@luna-comment
    Input: exact completed Oct23 A/B/C and L1 configs. Output: frozen four-model bindings and583-val inventory.
    Key Steps: resolve final250K only, copy immutable snapshots, verify full/L1 identities and target versions.
    Key Params: exclude stopped Oct26 runs; no optimizer/partial checkpoint fallback.
    State/Dependencies: yaml/SHA/source manifests; writes owned output only, no original mutation.
    """
    import yaml
    from tools import evaluate_six_final_models as six
    from tools.evaluate_offline_camera_matrix import sha,save_json
    refs=dict(a_v3=BASE/'training_comparison_20261023/configs/06_v3_block256_raw.yml',b_v3=BASE/'training_comparison_prev_pretrain_20261023/configs/06_v3_block256_raw.yml',c_sr=DATA/'vosr2u_compact_sr_x2_20261023/training_20261023/configs/07_v5_compact_sr2x.yml',l1_v3=DATA/'original_gt_vosr2u_l1_six_20261023/training_20261023/configs/vosr2u_l1_v3.yml')
    models=OUT/'camera/models';models.mkdir(parents=True,exist_ok=True)
    bindings={};configs={}
    for arm,config in refs.items():
        cfg=yaml.safe_load(config.read_text());configs[arm]=cfg
        checkpoint=Path(cfg['path']['experiments_root'])/cfg['name']/'models/net_g_250000.pth'
        assert checkpoint.is_file() and 'vosr2u_full_three_20261026' not in str(checkpoint)
        for source,target in [(config,models/(arm+'.yml')),(checkpoint,models/(arm+'.pth'))]:
            if target.exists():assert sha(target)==sha(source)
            else:shutil.copy2(source,target)
        bindings[arm]=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=sha(config),checkpoint_sha256=sha(checkpoint),adapter_sha256=sha(six.__file__),controller_sha256=sha(__file__))
    assert configs['a_v3']['train']==configs['b_v3']['train']
    assert configs['a_v3']['network_g']==configs['b_v3']['network_g']==configs['l1_v3']['network_g']
    assert configs['a_v3']['datasets']==configs['b_v3']['datasets']==configs['l1_v3']['datasets']
    assert configs['a_v3']['path']['pretrain_network_g']==configs['l1_v3']['path']['pretrain_network_g']
    assert configs['a_v3']['path']['pretrain_network_g']!=configs['b_v3']['path']['pretrain_network_g']
    assert configs['a_v3']['train']['cp_weight']>0 and configs['l1_v3']['train']['cp_weight']==0
    assert configs['c_sr']['network_g']['type']=='BasicTextLineR4StudentY4UVV5CompactSRHRSkip'
    save_json(OUT/'camera/MODEL_BINDINGS.json',bindings)
    rows=[json.loads(x) for x in VAL.read_text().splitlines()];assert len(rows)==583 and all(r['bucket']=='val' for r in rows)
    save_json(OUT/'validation/selection.json',dict(source_manifest_sha256=sha(VAL),samples=[dict(id=f'{i+1:03d}',row=r) for i,r in enumerate(rows)]))


def worker(arm,suite,rank):
    """@luna-comment
    Input: frozen model/suite/rank0or1. Output: native predictions and source-bound receipts.
    Key Steps: strict existing load, aligned validation whole blocks or accepted camera geometry/HR-skip adapter.
    Key Params:256/224/batch8; no tone or GT input; repaired017; disjoint583/280 rank partition.
    State/Dependencies: process-local six-worker output binding, CUDA and immutable caches; own files only.
    """
    import cv2,torch
    from tools import evaluate_six_final_models as six
    six.OUT=OUT
    if suite=='camera':six.worker(arm,'camera',rank);return
    cv2.setNumThreads(1);torch.set_num_threads(2)
    model,binding=six.load(arm)
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    selection=OUT/'validation/selection.json';inventory=json.loads(selection.read_text())['samples']
    for done,entry in enumerate(inventory[rank::2],1):
        row=entry['row'];identity=entry['id'];root=OUT/'validation'/arm
        receipt=root/'receipts'/(identity+'.json')
        if receipt.exists():
            cached=json.loads(receipt.read_text());assert cached['binding']==binding and sha(OUT/cached['final'])==cached['final_sha256'];continue
        hashes={k:sha(v) for k,v in row['files'].items()};assert hashes==row['output_sha256']
        image=cv2.imread(row['files']['lq']);assert image is not None
        rgb,details=restore_block(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),model,'cuda')
        final=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);assert final.shape==image.shape
        path=root/'final'/(identity+'.png');image_write(path,final)
        save_json(receipt,dict(id=identity,sample_id=row['sample_id'],binding=binding,source_sha256=hashes['lq'],source_hashes=hashes,source_hw=list(image.shape[:2]),selection_sha256=sha(selection),final=str(path.relative_to(OUT)),final_sha256=sha(path),restoration=details,controller_sha256=sha(__file__)))
        save_json(root/f'progress.{rank}.json',dict(done=done,total=len(inventory[rank::2]),last=identity))
    save_json(OUT/'validation'/arm/f'COMPLETE.{rank}.json',dict(images=len(inventory[rank::2]),binding=binding))


def publish(suite):
    """@luna-comment
    Input: complete native suite/receipts. Output: verified four-choice synchronized viewer and coverage audit.
    Key Steps: verify all model/config/source/final hashes and dimensions; preserve explicit val target/camera geometry.
    Key Params:583 LQ/oldGT/full/L1 or280 LQ/A/B/C; no metrics report or subjective ranking.
    State/Dependencies: PIL and existing inline renderer/mapped-Y paths; no prior gallery changes.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import file_url,render_html
    from tools.evaluate_offline_camera_matrix import sha,save_json
    camera=suite=='camera';arms=CAM_ARMS if camera else VAL_ARMS
    binding=json.loads((OUT/'camera/MODEL_BINDINGS.json').read_text())
    if camera:
        old=json.loads((DATA/'camera_compare_2x2_20261022/manifest.json').read_text())
        rows=[{k:r[k] for k in ('id','name','width','height','lq')} for r in old['rows']]
    else:
        entries=json.loads((OUT/'validation/selection.json').read_text())['samples'];rows=[]
        for entry in entries:
            r=entry['row']
            assert sha(r['files']['lq'])==r['output_sha256']['lq'] and sha(r['files']['gt'])==r['output_sha256']['gt']
            with Image.open(r['files']['lq']) as image:w,h=image.size
            with Image.open(r['files']['gt']) as image:assert image.size==(w,h)
            rows.append(dict(id=entry['id'],name=r['dataset']+' · '+r['sample_id'],width=w,height=h,lq=file_url(r['files']['lq']),images=dict(gt=file_url(r['files']['gt']))))
    definitions=[dict(key='lq',label='LQ · 原始输入',note='原尺寸输入')]
    if not camera:definitions.append(dict(key='gt',label='GT · 旧版GT_VOSR验证目标',note='固定583对旧验证目标，仅展示；未提供新版VOSR2u验证GT'))
    definitions+=[dict(key=a,label=LABELS[a],note='Oct23 · 250K最终模型 · 无亮度预处理/后处理'+(' · 特征128＋原尺寸256直通' if a=='c_sr' else ' · 原生256分块')) for a in arms]
    audit=[]
    for row in rows:
        row.setdefault('images',{});receipts=[]
        for arm in arms:
            root=OUT/suite/('results' if camera else '')/arm
            receipt=json.loads((root/'receipts'/(row['id']+'.json')).read_text())
            assert receipt['binding']['checkpoint_sha256']==binding[arm]['checkpoint_sha256'] and receipt['binding']['config_sha256']==binding[arm]['config_sha256']
            path=(OUT/'camera' if camera else OUT)/receipt['final']
            assert sha(path)==receipt['final_sha256']
            with Image.open(path) as image:assert image.size==(row['width'],row['height'])
            if camera:
                assert receipt['outside_owned_unchanged']
                if row['id']=='017':assert receipt['geometry017_fixed']
                if arm=='c_sr':
                    for stage in receipt['stages']:
                        detail=stage['restoration'];assert detail['adapter_sha256']==binding[arm]['adapter_sha256']
                        assert detail['feature_rgb_hw']==[128,128] and detail['original_lq_skip_hw']==[256,256]
            row['images'][arm]=file_url(path);receipts.append(receipt)
        assert len({r['source_sha256'] for r in receipts})==1
        if camera:
            assert len({r['document_record_sha256'] for r in receipts})==1
            assert all([(s['ocr_sha256'],s['warp_sha256']) for s in r['stages']]==[(s['ocr_sha256'],s['warp_sha256']) for s in receipts[0]['stages']] for r in receipts)
        else:assert receipts[0]['source_hashes']==receipts[1]['source_hashes']
        audit.append(dict(id=row['id'],source_sha256=receipts[0]['source_sha256'],finals={a:r['final_sha256'] for a,r in zip(arms,receipts)}))
    assert len(rows)==(280 if camera else 583)
    for arm in arms:
        root=OUT/suite/('results' if camera else '')/arm
        assert {p.stem for p in (root/'final').glob('*.png')}=={r['id'] for r in rows}
    notes=('仅使用Oct23已完成的A/B组V3与C组X2SR；未使用已停止的Oct26重复训练。\n复用文档检测→warp→256/224分块融合→残差unwarp；017使用修复区域。' if camera else '完整583对验证集；A组V3完整损失 vs VOSR2u L1-only V3。\n同一新版VOSR2u训练数据、相同原始初始化，使用250K最终权重；验证GT仍为旧版GT_VOSR。\n已对齐整图直接256/224分块，无文档warp；GT未输入模型。')+'\n无亮度预处理/后处理，仅看图，不生成指标报告。'
    manifest=dict(title='A/B V3＋C X2SR · 280张' if camera else 'VOSR2u V3 · 完整损失/L1 · 验证583张',subtitle='四窗同步缩放 / 切图 / 选图列表 / 前后翻页',notes=notes,defaults='lq,a_v3,b_v3,c_sr' if camera else 'lq,gt,a_v3,l1_v3',models=definitions,rows=rows)
    save_json(OUT/suite/'manifest.json',manifest)
    (OUT/suite/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save_json(OUT/suite/'AUDIT.json',dict(passed=True,rows=len(rows),finals=len(rows)*len(arms),records=audit,published_at=datetime.now(timezone.utc).isoformat()))


def execute(job):
    """@luna-comment
    Input: arm/suite/rank/GPU job. Output: child zero/nonzero exit receipt. Key Steps: explicit isolated command/log.
    Key Params: four concurrent workers max, threads2; ranks disjoint. State/Dependencies: owned source/output only.
    """
    arm,suite,rank,gpu=job;log=OUT/'logs'/f'{suite}.{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--suite',suite,'--rank',str(rank)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,suite=suite,rank=rank,gpu=gpu,exit_code=result.returncode)


def run():
    """@luna-comment
    Input: authorized validation-first then camera scope. Output: completed ordered suites and root entry.
    Key Steps: lock/snapshot, run four validation shards, require complete audit/publication, then six camera shards.
    Key Params:1166 val finals before840 camera finals; previous/stopped runs untouched.
    State/Dependencies: Linux flock/CUDA/NAS, exact frozen source; resumable outputs enforce binding identities.
    """
    import fcntl
    from tools.evaluate_offline_camera_matrix import sha,save_json
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
        for suite,arms in [('validation',VAL_ARMS),('camera',CAM_ARMS)]:
            save_json(OUT/'PIPELINE.json',dict(phase=suite+'_running',controller_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
            jobs=[(arm,suite,rank,i%4) for i,(arm,rank) in enumerate((a,r) for a in arms for r in (0,1))]
            with ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(execute,jobs))
            save_json(OUT/(suite+'_QUEUE.json'),completed);assert all(r['exit_code']==0 for r in completed),completed
            publish(suite)
            save_json(OUT/(suite+'_COMPLETE.json'),dict(passed=True,finished_at=datetime.now(timezone.utc).isoformat()))
        for binding in json.loads((OUT/'camera/MODEL_BINDINGS.json').read_text()).values():
            assert sha(binding['checkpoint'])==binding['checkpoint_sha256'] and sha(binding['config'])==binding['config_sha256']
        (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>VOSR2u V3对比与A/B/C测试</h1><p><a href="validation/index.html">完整583张验证集：V3完整损失 / L1-only</a></p><p><a href="camera/index.html">280张相机：A组V3 / B组V3 / C组X2SR</a></p>',encoding='utf-8')
        save_json(OUT/'PIPELINE.json',dict(phase='complete',validation_finals=1166,camera_finals=840))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','worker','publish'));parser.add_argument('--arm',choices=('a_v3','b_v3','c_sr','l1_v3'));parser.add_argument('--suite',choices=('validation','camera'));parser.add_argument('--rank',type=int,default=0)
    args=parser.parse_args()
    if args.action=='worker':worker(args.arm,args.suite,args.rank)
    elif args.action=='publish':publish(args.suite)
    else:run()
