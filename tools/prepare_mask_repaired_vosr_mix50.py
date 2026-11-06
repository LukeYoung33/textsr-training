"""Frozen complete-source GT repair, native banks and original validation references."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import sys, os, json, copy, time
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.prepare_r1_width_bank import save, digest
from tools.repair_vosr_mask_full import VERSION, repair, invariant_smoke
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'vosr_s22_mask_repair_mix50_20261104'
WORKERS = 8


def read_rows(path):
    """@luna-comment Input:manifest path. Output:JSONL rows. Key Steps:read nonblank lines. Key Params:immutable source. State/Dependencies:filesystem."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inventory():
    """@luna-comment Input:all completed source receipts. Output:frozen original-mask source rows and scope. Key Steps:refresh Lenovo population without old partial snapshot;preserve heldouts/dedup;bind current teacher inputs. Key Params:base4873,Lenovo3236,Mojito243 retained unmodified. State/Dependencies:read-only historical artifacts."""
    path = OUT/'SOURCE_ROWS.json'; OUT.mkdir(exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    from tools import prepare_latest_vosr_500k as historical
    from unittest.mock import patch
    # Redirect eligibility publication into this experiment; do not rewrite old frozen audits.
    with patch.object(historical, 'OUT', OUT):
        sets, audit = historical.eligibility()
    result = []
    for key, root in [('base', historical.BASE/'source_view'), ('mojito', historical.MOJITO/'source_view')]:
        for row in sets[key]['train']:
            if row['sample_id'] not in set(audit['allowed'][key]):
                continue
            r = copy.deepcopy(row); r['repair_mode'] = 'preserve_region_mask' if key == 'mojito' else 'stroke_repair'
            r['files'] = {k:str(root/v) for k,v in r['files'].items()}; result.append(r)
    complete = json.loads((historical.LENOVO/'COMPLETE.json').read_text()); assert complete['passed'] and complete['count'] == 3236
    found = {}
    for f in sorted(historical.LENOVO.glob('*/*/COMPLETE.json')):
        for r in json.loads(f.read_text()).get('records', []):
            found[Path(r['filename']).stem] = (r, f.parent/'images'/r['id'])
    for r in json.loads((historical.LENOVO/'RECOVERED_TEACHER.json').read_text())['records']:
        found.setdefault(Path(r['filename']).stem, (r, Path(r['stage_folder'])))
    for row in sets['lenovo']['train']:
        if row['sample_id'] not in set(audit['allowed']['lenovo']):
            continue
        record, folder = found[row['sample_id']]
        assert record['source_sha256'] == row['output_sha256']['lq']
        r = copy.deepcopy(row); r['dataset'] = 'lenovo0912'; r['repair_mode'] = 'stroke_repair'
        r['files'] = {k:str(historical.OLD/v) for k,v in r['files'].items()}
        r['files'].update(lq=str(folder/'pre_lq.png'), gt=record['candidate_gt'])
        r['output_sha256'].update(lq=record['pre_lq_sha256'], gt=record['arms']['steep_s']['sha256']); result.append(r)
    assert len(result) == sum(audit['train_counts'].values()) == 8352
    assert len({(r['dataset'],r['sample_id']) for r in result}) == len(result)
    save(path, result)
    save(OUT/'SOURCE_BINDING.json', dict(version=VERSION, source_rows_sha256=digest(path), counts=audit['train_counts'], total=len(result), repairable=8109, region_mask_preserved=243, old_sources_immutable=True, validation='unchanged620', training_mask='M2 for repaired sources; original region mask for preserved Mojito'))
    return result


def process(row):
    """@luna-comment Input:frozen row. Output:source-bound repaired row/receipt. Key Steps:verify GT/mask/valid;expand original mask;repair;atomic PNG/hash publication;resume strict binding. Key Params:no original-GT color,no OCR. State/Dependencies:owned paths only."""
    from tools.sync_dilate2_text_masks import expand
    cv2.setNumThreads(1)
    key = row['dataset']+'/'+row['sample_id']; folder = OUT/'targets'/key; receipt = folder/'receipt.json'
    binding = dict(version=VERSION, mode=row['repair_mode'], source_hashes=row['output_sha256'])
    if receipt.exists():
        old = json.loads(receipt.read_text()); assert old['binding'] == binding
        for k in ['gt', 'text_mask']:
            assert digest(old['row']['files'][k]) == old['row']['output_sha256'][k]
        return old
    for k in ['gt', 'text_mask', 'valid_mask']:
        assert digest(row['files'][k]) == row['output_sha256'][k], (key, k)
    new = copy.deepcopy(row)
    if row['repair_mode'] == 'preserve_region_mask':
        audit = dict(changed_pixels=0, reason='region-only mask, preserve existing GT and original mask')
    else:
        raw = cv2.imread(row['files']['gt']); mask = cv2.imread(row['files']['text_mask'], 0); valid = cv2.imread(row['files']['valid_mask'], 0)
        assert raw is not None and mask is not None and valid is not None and raw.shape[:2] == mask.shape == valid.shape
        m2, _ = expand(mask, valid)
        output, audit = repair(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), m2, valid)
        folder.mkdir(parents=True, exist_ok=True)
        for k, image in [('gt', cv2.cvtColor(output, cv2.COLOR_RGB2BGR)), ('text_mask', m2)]:
            dest = folder/(k+'.png'); tmp = dest.with_name(k+'.tmp.png')
            assert cv2.imwrite(str(tmp), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]); tmp.replace(dest)
            new['files'][k] = str(dest); new['output_sha256'][k] = digest(dest)
        new['previous_training_target'] = dict(path=row['files']['gt'], sha256=row['output_sha256']['gt'])
        new['original_text_mask_lineage'] = dict(path=row['files']['text_mask'], sha256=row['output_sha256']['text_mask'])
        new['target_version'] = VERSION
    record = dict(passed=True, binding=binding, row=new, audit=audit)
    save(receipt, record); return record


def publish(done, total, started, phase, **extra):
    """@luna-comment Input:real counters and stage. Output:atomic progress/ETA. Key Steps:elapsed throughput extrapolation with explicit stage. Key Params:no fabricated aggregate duration. State/Dependencies:owned status."""
    elapsed = time.time()-started; eta = (total-done)*elapsed/done if done else None
    save(OUT/'DATA_STATUS.json', dict(phase=phase, completed=done, total=total, elapsed_seconds=elapsed, eta_seconds=eta, estimated_finish=time.strftime('%m-%d %H:%M',time.localtime(time.time()+eta)) if eta is not None else None, updated=time.time(), pid=os.getpid(), workers=WORKERS, **extra))


def prepare_gt():
    """@luna-comment Input:full frozen inventory. Output:all receipts/new source manifest. Key Steps:CPU invariant gate,eight workers,fail-closed inventory completion. Key Params:shared repaired targets for both arms. State/Dependencies:resumable per-image receipts."""
    rows = inventory(); source = OUT/'source_view'; complete = OUT/'GT_COMPLETE.json'
    if complete.exists():
        assert digest(source/'manifests/train.jsonl') == json.loads(complete.read_text())['manifest_sha256']; return
    save(OUT/'CPU_SMOKE.json', invariant_smoke()); start = time.time(); records = {}
    publish(0, len(rows), start, 'repairing_gt')
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process, row):(row['dataset'],row['sample_id']) for row in rows}
        for future in as_completed(futures):
            key = futures[future]; records[key] = future.result()
            if len(records)%5 == 0 or len(records) == len(rows):
                publish(len(records), len(rows), start, 'repairing_gt'); print('GT',len(records),len(rows),flush=True)
    manifest = [records[(r['dataset'],r['sample_id'])]['row'] for r in rows]
    (source/'manifests').mkdir(parents=True,exist_ok=True)
    (source/'manifests/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (source/'manifests/val.jsonl').write_text('')
    audits = [v['audit'] for v in records.values()]
    save(complete, dict(passed=True, count=len(rows), changed_parents=sum(a['changed_pixels']>0 for a in audits), unresolved_background_pixels=sum(a.get('unresolved_background_pixels',0) for a in audits), skipped_components=sum(a.get('skipped_components',0) for a in audits), manifest_sha256=digest(source/'manifests/train.jsonl'), source_rows_sha256=digest(OUT/'SOURCE_ROWS.json'), version=VERSION))


def attach_validation(bank):
    """@luna-comment Input:owned bank root. Output:old620 validation references and matched line sidecars. Key Steps:prefix original arrays/indices;link immutable bank and its sidecars. Key Params:no validation GT changes or extra materialization. State/Dependencies:read-only old validation."""
    old = DATA/'latest_vosr_four_500k_20261102/bank'
    link = bank/'historical_validation'
    if not link.exists(): link.symlink_to(old,target_is_directory=True)
    val = json.loads((old/'val.json').read_text())
    val['original_bank_build_id'] = val['build_id']
    val['build_id'] = json.loads((bank/'COMPLETE.json').read_text())['build_id']
    for shard in val['shards']:
        for key in ['array','metadata']: shard[key] = 'historical_validation/'+shard[key]
    for key in ['index','unique_index']:val[key]='historical_validation/'+val[key]
    # Runtime line sidecars follow the array-relative path.
    line_link=bank/'line_regions/historical_validation'
    line_link.parent.mkdir(exist_ok=True)
    if not line_link.exists():line_link.symlink_to(old/'line_regions',target_is_directory=True)
    save(bank/'val.json',val)
    return val


def finish_bank():
    """@luna-comment Input:complete native bank/sidecars. Output:effective-text filtered train and old620 validation manifests. Key Steps:preserve balanced draws;hash sidecars;validation symlink isolation. Key Params:NAS bank only. State/Dependencies:owned metadata,old val immutable."""
    from tools.run_s22_nocp_text_guard import read_support, filter_sampling_index
    bank = OUT/'bank'
    manifest = json.loads((bank/'train_balanced.json').read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        flags = list(pool.map(read_support, [(bank,s) for s in manifest['shards']]))
    for key in ['index', 'unique_index']:
        source = np.load(bank/manifest[key],allow_pickle=False); filtered,_ = filter_sampling_index(source,flags)
        target = 'effective.'+key+'.npy'; np.save(bank/target,filtered); manifest[key] = target
    save(bank/'train.json',manifest)
    val = attach_validation(bank)
    assert len(val['shards'])==620 and len(manifest['shards'])==8352
    assert not ({s['source_identity'] for s in manifest['shards']}&{s['source_identity'] for s in val['shards']})
    save(OUT/'DATA_READY.json',dict(passed=True,train_parents=8352,val_parents=620,gt_complete_sha256=digest(OUT/'GT_COMPLETE.json'),train_sha256=digest(bank/'train.json'),val_sha256=digest(bank/'val.json'),source_binding_sha256=digest(OUT/'SOURCE_BINDING.json')))


def main():
    """@luna-comment Input:authorized dataset job. Output:repair stage or downstream fully ready bank. Key Steps:separate stages for controller heartbeats;resume completed stages. Key Params:--stage gt/bank/finalize. State/Dependencies:existing known builders only."""
    import argparse, subprocess
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['gt','bank','finalize'],required=True);args=parser.parse_args()
    if args.stage=='gt': prepare_gt()
    elif args.stage=='bank':
        assert json.loads((OUT/'GT_COMPLETE.json').read_text())['passed']
        if not (OUT/'bank/COMPLETE.json').exists():
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(OUT/'source_view'),'--output',str(OUT/'bank'),'--workers','4','--preprocess-mode','none'],check=True)
    else:
        if not (OUT/'bank/LINE_REGIONS_COMPLETE.json').exists():
            from tools.run_v3pre_training import sidecars
            sidecars(OUT/'bank',OUT/'source_view',4)
        finish_bank()


if __name__=='__main__': main()
