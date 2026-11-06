"""Nine-case source-bound bright-guard pilot; first case is a separate gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_v3_reference_post_pilot import (BASE, OUTPUT as OLD_OUTPUT, IDS,
    decode_bound, encode_png, load_fields, preflight_bindings, sha256, source_cases,
    verify_case_sources)
from tools.v3_reference_post import canonical_to_stored, render_post, stored_to_canonical
from tools.v3_reference_bright_guard import guarded_node_tables, repair_evidence

OUTPUT = ROOT/'output/v3_reference_bright_guard_20261028'
ARMS = {'guard_weak': .35, 'guard_full': 1.}


def guard_bindings(selection_bytes, manifest_bytes):
    """@luna-comment Input: frozen nine-case selection and gallery bytes. Output: exact current and prior code/data SHA bindings. Key Steps: rerun prior binding gate, compare old stage snapshots, then bind new sources. Key Params: SHA256; no source writes. State/Dependencies: old core/results remain immutable."""
    prior = preflight_bindings(selection_bytes, manifest_bytes)
    weak = json.loads((OLD_OUTPUT/'weak_summary.json').read_text(encoding='utf-8'))
    full = json.loads((OLD_OUTPUT/'full_summary.json').read_text(encoding='utf-8'))
    if weak['bindings_sha256'] != prior or full['bindings_sha256'] != prior:
        raise AssertionError('Frozen prior stage bindings differ')
    return {'old_bindings': prior,
            'guard_core': sha256(ROOT/'tools/v3_reference_bright_guard.py'),
            'guard_runner': sha256(ROOT/'tools/run_v3_reference_bright_guard.py'),
            'prior_weak_summary': sha256(OLD_OUTPUT/'weak_summary.json'),
            'prior_full_summary': sha256(OLD_OUTPUT/'full_summary.json')}


def load_guard_fields(case, old_bindings):
    """@luna-comment Input: immutable old field cache and source receipt. Output: all seven V3 node fields and actual repair evidence. Key Steps: use prior cache hash gate, then rerun frozen repair and require exact L3 equality. Key Params: movement allowance12/scale24. State/Dependencies: no model-output rail or calibration."""
    arrays, receipt, receipt_path = load_fields(case, old_bindings)
    cache = Path(receipt['field_cache_path'])
    with np.load(cache, allow_pickle=False) as fields:
        original_low = fields['original_low'].copy()
        quantiles = fields['quantiles'].copy()
        small_tail = fields['small_tail'].copy()
    amount, movement, evidence = repair_evidence(original_low, arrays['low3'],
        arrays['high'], quantiles, small_tail, arrays['nodes'])
    return arrays, receipt, receipt_path, amount, movement, evidence


def write_guard_arm(case, arm, image, telemetry, field_receipt, field_path,
                    bindings, old_sha, model_receipt_sha, v3_receipt_sha):
    """@luna-comment Input: one native guarded preview and bound provenance. Output: exclusive PNG, JSON receipt and summary record. Key Steps: source rehash, native size, PNG encode/hash, no overwrite. Key Params: compression3; RGB-clipped preview. State/Dependencies: only new repository output is written."""
    path = OUTPUT/arm/(case['key']+'.png')
    receipt_path = OUTPUT/'receipts'/(case['key']+'_'+arm+'.json')
    if path.exists() or receipt_path.exists():
        raise FileExistsError('Guard output already exists '+case['key']+' '+arm)
    if image.shape[:2] != (case['row']['height'], case['row']['width']):
        raise AssertionError('Guard native geometry mismatch')
    source_sha = sha256(case['source_path'])
    if source_sha != json.loads(case['model_receipt_path'].read_text(encoding='utf-8'))['final_sha256']:
        raise AssertionError('Original model source changed')
    png, encode_seconds = encode_png(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(png)
    record = {'key':case['key'], 'id':case['id'], 'arm':arm,
              'eta':ARMS[arm], 'lambda':1., 'render_strength':1.,
              'source_url':case['source_url'], 'source_sha256':source_sha,
              'model_receipt_sha256':model_receipt_sha,
              'v3_receipt_sha256':v3_receipt_sha,
              'original_lq_sha256':case['entry']['source_sha256'],
              'v3_stored_sha256':case['entry']['input_sha256'],
              'exact_v3_replay_max_abs':field_receipt['exact_v3_replay_max_abs'],
              'field_receipt_path':str(field_path), 'field_receipt_sha256':sha256(field_path),
              'field_cache_path':field_receipt['field_cache_path'],
              'field_cache_sha256':field_receipt['field_cache_sha256'],
              'old_reuse_full_sha256':old_sha,
              'display_hw':[case['row']['height'],case['row']['width']],
              'orientation':case['entry']['orientation'],
              'output_calibration_enabled':False,
              'output_statistics_used_for_candidate_parameters':False,
              'native_uv_identity_proven':False, 'device_latency_proven':False,
              'bindings_sha256':bindings, 'output_path':str(path),
              'output_sha256':hashlib.sha256(png).hexdigest(),
              'encode_seconds_host':encode_seconds, 'telemetry':telemetry}
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    return {'path':str(path),'sha256':record['output_sha256'],
            'receipt_path':str(receipt_path),
            'field_cache_sha256':field_receipt['field_cache_sha256']}


def generate_guard_case(case, bindings, model_bindings, prior_full):
    """@luna-comment Input: one exact V3-preprocessed full-model result and old source fields. Output: weak/full native guard PNG pair. Key Steps: verify lineage and cache, exact old eta0 preview, render both tables and enforce channelwise ordering. Key Params: eta.35/1, strength1. State/Dependencies: old results read-only; no output-based parameters."""
    model, model_receipt_sha, v3_receipt_sha = verify_case_sources(
        case, bindings['old_bindings'], model_bindings)
    fields, field_receipt, field_path, amount, movement, evidence = load_guard_fields(
        case, bindings['old_bindings'])
    stored = canonical_to_stored(model, case['entry']['orientation'])
    if list(stored.shape) != case['entry']['stored_shape'] or not np.array_equal(
            stored_to_canonical(stored,case['entry']['orientation']), model):
        raise AssertionError('Stored/display permutation mismatch')
    old_record = prior_full[case['key']]['outputs']['reuse_full']
    old_path = Path(old_record['path'])
    if sha256(old_path) != old_record['sha256']:
        raise AssertionError('Old reuse full output changed')
    old_bgr = decode_bound(old_path, old_record['sha256'], model.shape[:2])
    old_table, tables, node_stats = guarded_node_tables(fields['low3'], fields['high'],
        fields['population'], evidence, etas=(0., .35, 1.))
    zero_stored, _ = render_post(stored,fields['nodes'],tables[0.])
    zero = stored_to_canonical(zero_stored,case['entry']['orientation'])
    if not np.array_equal(zero,old_bgr):
        raise AssertionError('Eta0 preview differs from frozen reuse_full '+case['id'])
    outputs = {}
    rendered = {}
    telemetry_base = {'nodes':node_stats,
        'amount_nonzero_nodes':int(np.count_nonzero(amount)),
        'amount_gt_half_zero_movement_nodes':int(np.count_nonzero((amount>.5)&(movement<.01))),
        'movement_nonzero_nodes':int(np.count_nonzero(movement>.01)),
        'movement_max':float(movement.max()),
        'evidence_active_nodes':int(np.count_nonzero(evidence)),
        'evidence_max':float(evidence.max()),
        'evidence_mean':float(evidence.mean()),
        'old_reuse_full_exact_eta0':True}
    for arm,eta in ARMS.items():
        candidate_stored, render_stats = render_post(stored, fields['nodes'], tables[eta])
        candidate = stored_to_canonical(candidate_stored,case['entry']['orientation'])
        rendered[arm] = candidate
        if np.any(candidate.astype(np.int16)>old_bgr.astype(np.int16)):
            raise AssertionError('Guard increased old preview channel')
        difference = old_bgr.astype(np.int16)-candidate.astype(np.int16)
        telemetry = dict(telemetry_base)
        telemetry.update(render=render_stats,
            old_minus_guard_changed_pixels=int(np.count_nonzero(np.any(difference,axis=2))),
            old_minus_guard_max_channel=int(difference.max()),
            old_minus_guard_sum_channels=int(difference.sum()),
            guard_minus_model_changed_pixels=int(np.count_nonzero(np.any(candidate!=model,axis=2))))
        outputs[arm] = write_guard_arm(case,arm,candidate,telemetry,field_receipt,
            field_path,bindings,old_record['sha256'],model_receipt_sha,v3_receipt_sha)
    if np.any(rendered['guard_full'].astype(np.int16)>rendered['guard_weak'].astype(np.int16)):
        raise AssertionError('Full guard exceeds weak preview channel')
    print(json.dumps({'done':case['id'],'changed_weak':
        int(np.count_nonzero(np.any(rendered['guard_weak']!=old_bgr,axis=2))),
        'changed_full':int(np.count_nonzero(np.any(rendered['guard_full']!=old_bgr,axis=2)))}),flush=True)
    return {'key':case['key'],'id':case['id'],'name':case['name'],
            'width':case['row']['width'],'height':case['row']['height'],
            'source_url':case['source_url'],'source_sha256':sha256(case['source_path']),
            'lq_url':case['row']['lq'],'v3_input_url':case['row']['images']['v3_input'],
            'field_receipt_path':str(field_path),
            'field_cache_sha256':field_receipt['field_cache_sha256'],
            'old_reuse_full':old_record,'outputs':outputs}


def main():
    """@luna-comment Input: frozen nine-case V3 selection/full results plus pilot/rest CLI. Output: two stage summaries and final merged9 summary. Key Steps: run089 first only, then remaining8 with immutable hash checks. Key Params: one cv thread, no inference. State/Dependencies: only writes new repo output, never replaces prior results."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','rest'),required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    selection_bytes=(BASE/'selection.json').read_bytes()
    manifest_bytes=(BASE/'manifest.json').read_bytes()
    bindings=guard_bindings(selection_bytes,manifest_bytes)
    selection=json.loads(selection_bytes)
    manifest=json.loads(manifest_bytes)
    cases=source_cases(selection,manifest)
    chosen=[case for case in cases if (case['id']=='089')==(args.stage=='pilot')]
    model_bindings=json.loads((BASE/'MODEL_BINDINGS.json').read_text(encoding='utf-8'))
    prior_full_summary=json.loads((OLD_OUTPUT/'full_summary.json').read_text(encoding='utf-8'))
    prior_full={item['key']:item for item in prior_full_summary['cases']}
    summary_path=OUTPUT/(args.stage+'_summary.json')
    if summary_path.exists() or any((OUTPUT/arm/(case['key']+'.png')).exists()
        for arm in ARMS for case in chosen):
        raise FileExistsError('Fresh stage output required')
    records=[generate_guard_case(case,bindings,model_bindings,prior_full) for case in chosen]
    if guard_bindings((BASE/'selection.json').read_bytes(),
                      (BASE/'manifest.json').read_bytes()) != bindings:
        raise AssertionError('Code/data changed during guard stage')
    summary={'status':'experimental_visual_pending','stage':args.stage,
             'arms':list(ARMS),'case_count':len(records),
             'new_png_count':2*len(records),'bindings_sha256':bindings,
             'domain':'PNG-derived Y8 RGB-clipped preview; not native UV/NEON/140ms proof',
             'output_calibration_enabled':False,
             'output_statistics_used_for_candidate_parameters':False,'cases':records}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with summary_path.open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)
    if args.stage=='rest':
        pilot=json.loads((OUTPUT/'pilot_summary.json').read_text(encoding='utf-8'))
        if pilot['bindings_sha256']!=bindings or [r['id'] for r in pilot['cases']]!=['089']:
            raise AssertionError('Pilot binding/case mismatch')
        full=dict(summary,stage='complete',case_count=9,new_png_count=18,
                  cases=pilot['cases']+records)
        with (OUTPUT/'summary.json').open('x',encoding='utf-8') as stream:
            json.dump(full,stream,ensure_ascii=False,indent=2)
    print(str(summary_path),flush=True)


if __name__=='__main__':
    main()
