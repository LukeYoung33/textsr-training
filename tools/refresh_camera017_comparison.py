"""Recompute all ten original-camera017 strategies against the accepted fixed geometry."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.append('/home/guang.yang/workspace/Real-ESRGAN')
from tools.run_camera_compare_missing import DATA
from tools.evaluate_offline_camera_matrix import sha,save_json

FIX=DATA/'camera_robust_pre_native_v22_017_fix_20261022'
OUT=DATA/'camera017_comparison_refresh_20261022'
GALLERY=DATA/'camera_compare_2x2_20261022'
EXTRA={'vosr_v3_raw':'04_v3_raw','vosr_v5_raw':'03_v5_raw','vosr_v3_block':'06_v3_block256_raw','vosr_v5_block':'05_v5_block256_raw'}


def validate_fix():
    """@luna-comment Input:accepted017 fix artifacts. Output:verified record/pre receipt. Key Steps:bind source, fresh OCR, warp/pre/context hashes and geometry. Key Params:20 boxes, exterior approximate branch. State/Dependencies:read-only published fix; no old OCR reuse."""
    receipt=json.loads((FIX/'receipts/017.json').read_text())
    record_file=FIX/'geometry_cache/documents/017.jpg/result.json'
    assert sha(record_file)==receipt['corrected_document_record_sha256']
    record=json.loads(record_file.read_text())
    assert [record['image_height'],record['image_width']]==[3120,4160]
    assert len(record['instances'])==len(receipt['instances'])==1
    item=record['instances'][0]
    assert item['selection_reason']=='exterior_observed_line_consensus_perspective'
    for stage in receipt['instances']:
        ocr_file=FIX/'geometry_cache/ocr/017'/(f"{stage['instance_id']:03d}.json")
        ocr=json.loads(ocr_file.read_text())
        assert sha(ocr_file)==stage['ocr_sha256'] and len(ocr['boxes'])==20
        assert ocr['source_sha256']==receipt['source_sha256']
        assert sha(ocr['warp_file'])==stage['warp_sha256']==ocr['warp_sha256']
        assert ocr['transform']==item['warp']['image_to_output']
        for key in ('pre','context'):
            assert sha(FIX/stage[key])==stage[key+'_sha256']
    assert sha(FIX/receipt['image'])==receipt['image_sha256']
    return receipt


def worker(arm):
    """@luna-comment Input:one existing model choice. Output:isolated017 final/receipt. Key Steps:bind accepted fixed cache, select exact prior final weight, adapt only block route, call existing checked single-frame runner. Key Params:rank16/shards280 selects017; native versus historical height96 retained. State/Dependencies:process-local bindings only; original outputs/weights unchanged."""
    import yaml
    import tools.run_camera_compare_missing as base
    validate_fix()
    original_paths=base.model_paths
    # @luna-comment Input:arm. Output:original final config/checkpoint. Key Steps:explicit extra raw/block mapping. Key Params:250K. State/Dependencies:read-only original training roots.
    def paths(key):
        if key not in EXTRA:return original_paths(key)
        train=(DATA/'vosr_fullimage_block256_v1_20261021' if 'block' in key else DATA/'adaptive_vosr_preprocessed_augmented_v1_20261021')/'training_comparison_20261021'
        config=train/'configs'/(EXTRA[key]+'.yml')
        return config,train/'experiments'/yaml.safe_load(config.read_text())['name']/'models/net_g_250000.pth'
    base.model_paths=paths
    base.DOC=FIX/'geometry_cache';base.PRE=FIX;base.OUT=OUT
    if 'block' in arm:
        import tools.auto_hx_v3_native as native
        # @luna-comment Input:RGB/native-runner signature. Output:raw whole-warp256 restoration. Key Steps:delegate accepted overlap block implementation. Key Params:ignore OCR boxes, no tonal pre/post. State/Dependencies:eager network/CUDA.
        def block(rgb,boxes,network,device,blend_source=None):
            from tools.evaluate_vosr_six_models import restore_block
            return restore_block(rgb,network,device)
        native.restore=block
    base.run(SimpleNamespace(arm=arm,rank=16,shards=280,limit=0))


def publish():
    """@luna-comment Input:ten fresh finals and current280 manifest. Output:017-only replacement HTML/manifest and patch audit. Key Steps:verify exact original checkpoints/source/geometry, snapshot previous manifest, change eleven URLs, compare all other rows, render. Key Params:fixed pre image plus ten models; source/GT stay unchanged. State/Dependencies:existing model outputs immutable; mapped-Y file URLs."""
    from PIL import Image
    from tools.build_camera_2x2_viewer import render_html,file_url
    pre=validate_fix()
    audit=json.loads((GALLERY/'COMPARISON_AUDIT.json').read_text())
    manifest=json.loads((GALLERY/'manifest.json').read_text(encoding='utf-8'))
    old=json.loads(json.dumps(manifest))
    assert len(manifest['rows'])==280
    row=next(v for v in manifest['rows'] if v['id']=='017')
    assert [row['height'],row['width']]==[3120,4160]
    entries=[]
    for arm,expected in audit['bindings'].items():
        receipt_path=OUT/'results'/arm/'receipts/017.json'
        receipt=json.loads(receipt_path.read_text())
        for key in ('checkpoint_sha256','config_sha256'):assert receipt['binding'][key]==expected[key]
        assert receipt['source_sha256']==pre['source_sha256'] and receipt['outside_owned_unchanged']
        assert receipt['document_record_sha256']==pre['corrected_document_record_sha256']
        assert all(v['ocr_sha256']==pre['instances'][0]['ocr_sha256'] and v['warp_sha256']==pre['instances'][0]['warp_sha256'] for v in receipt['stages'])
        image=OUT/receipt['final'];assert sha(image)==receipt['final_sha256']
        with Image.open(image) as pixels:assert pixels.size==(4160,3120)
        row['images'][arm]=file_url(image)
        entries.append(dict(id='017',model=arm,receipt=str(receipt_path),receipt_sha256=sha(receipt_path),final_sha256=receipt['final_sha256']))
    assert len(entries)==10
    row['images']['pre']=file_url(FIX/pre['image'])
    for before,after in zip(old['rows'],manifest['rows']):
        if before['id']!='017':assert before==after
        else:assert before['lq']==after['lq'] and before['name']==after['name']
    assert old['models']==manifest['models']
    backup=OUT/'manifest.before.json'
    if not backup.exists():save_json(backup,old)
    target=GALLERY/'index.html';target.write_text(render_html(manifest),encoding='utf-8')
    save_json(GALLERY/'manifest.json',manifest)
    save_json(OUT/'REFRESH_AUDIT.json',dict(status='passed',image='017',models=10,unchanged_other_rows=279,entries=entries,pre_sha256=pre['image_sha256'],fixed_record_sha256=pre['corrected_document_record_sha256'],adapter_sha256=sha(__file__),html_sha256=sha(target)))
    audit['records']=[v for v in audit['records'] if v['id']!='017']+entries
    audit.update(html_sha256=sha(target),geometry017_refresh=str(OUT/'REFRESH_AUDIT.json'))
    save_json(GALLERY/'COMPARISON_AUDIT.json',audit)
    pre_audit_path=GALLERY/'PRE_COLUMN_AUDIT.json'
    pre_audit=json.loads(pre_audit_path.read_text())
    pre_backup=OUT/'pre_audit.before.json'
    if not pre_backup.exists():save_json(pre_backup,pre_audit)
    pre_audit['records']=[dict(id='017',receipt_sha256=sha(FIX/'receipts/017.json'),image_sha256=pre['image_sha256']) if value['id']=='017' else value for value in pre_audit['records']]
    pre_audit.update(html_sha256=sha(target),geometry017_refresh=str(OUT/'REFRESH_AUDIT.json'))
    save_json(pre_audit_path,pre_audit)
    print('PASS published017 / ten models + pre; other279 unchanged',flush=True)


def main(args):
    """@luna-comment Input:worker/publish/all CLI. Output:checked ten-arm replay and updated viewer. Key Steps:bounded four-process queue, collect every exit, publish after success. Key Params:existing four GPUs; no changes to HX batch. State/Dependencies:approved remote environment and task-owned logs."""
    if args.mode=='worker':return worker(args.arm)
    if args.mode=='publish':return publish()
    bindings=json.loads((GALLERY/'COMPARISON_AUDIT.json').read_text())['bindings']
    # @luna-comment Input:index/arm. Output:successful child status. Key Steps:redirect isolated logs and wait. Key Params:GPU index modulo4. State/Dependencies:own subprocess only.
    def execute(job):
        i,arm=job
        log=OUT/'logs'/(arm+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable,__file__,'worker','--arm',arm],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%4),OMP_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT,check=True)
        return arm
    with ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(execute,enumerate(bindings)))
    save_json(OUT/'COMPLETE.json',dict(arms=completed))
    publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('worker','publish','all'));parser.add_argument('--arm')
    main(parser.parse_args())
