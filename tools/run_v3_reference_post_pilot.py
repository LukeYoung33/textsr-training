"""Bounded nine-image direct reuse of source V3 fields on V3-preprocessed full results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.document_box_pre640_revision2_20261027 import revision2
from evaluation.document_box_neighbor_20261027 import neighbor
from evaluation.document_box_pre640_astra_20261027 import pre640
from tools.anchored_y_contrast import ARMS,process_arm
from tools.v3_reference_post import (NODES,canonical_to_stored,node_tables,
    render_post,stored_to_canonical)


BASE=ROOT/'output/v3pre_three_mojito280_20261028'
OUTPUT=ROOT/'output/v3_reference_post_pilot_20261028'
IDS=('006','017','051','052','089','099','100','197','251')
PRE_MODULES=('evaluation/document_box_pre640_revision2_20261027/revision2.py',
             'evaluation/document_box_neighbor_20261027/neighbor.py',
             'evaluation/document_box_pre640_astra_20261027/pre640.py',
             'tools/prepare_v3pre_training.py',
             'tools/adaptive_scurve_robust_v2.py',
             'evaluation/adaptive_prepost_frozen_v1_20261020/lowres_curve.py')
V3_BINDING_KEYS=('revision2','neighbor','pre640','prepare_v3pre_training',
                 'cluster_module','lowres_module')


def sha256(path):
    """@luna-comment Input: existing file. Output: exact SHA256. Key Steps: bounded byte streaming. Key Params:1MiB chunks. State/Dependencies: source and code read-only."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def remote_path(path):
    """@luna-comment Input: exact approved Linux TextSR data path. Output: mapped read-only Y path. Key Steps: reject all other roots. Key Params: /AI_DNS/guang.yang/TextSR/data. State/Dependencies: never writes Y."""
    prefix='/AI_DNS/guang.yang/TextSR/data/'
    if not path.startswith(prefix):
        raise ValueError('Unexpected remote source path '+path)
    return Path('Y:/guang.yang/TextSR/data')/path[len(prefix):]


def file_url_path(url):
    """@luna-comment Input: manifest file URL. Output: mapped read-only Y Path. Key Steps: validate scheme/drive and decode percent escapes. Key Params: file:///Y: only. State/Dependencies: source model image is immutable."""
    parsed=urlparse(url)
    if parsed.scheme!='file' or parsed.netloc or not parsed.path.lower().startswith('/y:/'):
        raise ValueError('Unexpected model URL '+url)
    return Path(unquote(parsed.path.lstrip('/')))


def decode_bound(path, expected_hash, expected_hw=None):
    """@luna-comment Input: exact source file/hash and optional native H,W. Output: decoded BGR8. Key Steps: hash bytes then OpenCV imdecode; reject shape mismatch. Key Params: PNG/JPEG unchanged. State/Dependencies: avoids Unicode-path imread ambiguity."""
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_hash:
        raise AssertionError('Source hash mismatch '+str(path))
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise AssertionError('Expected BGR8 source '+str(path))
    if expected_hw is not None and tuple(image.shape[:2])!=tuple(expected_hw):
        raise AssertionError('Source dimensions mismatch '+str(path))
    return image


def source_cases(selection,manifest):
    """@luna-comment Input: frozen V3 selection/gallery. Output: nine exact full-model provenance cases. Key Steps: match IDs, raw/input hashes, canonical orientation and source URLs. Key Params: no old no-V3 or PS outputs. State/Dependencies: no output-derived source substitution."""
    selected={row['id']:row for row in selection['samples'] if row['id'] in IDS}
    displayed={row['id']:row for row in manifest['rows'] if row['id'] in IDS}
    if set(selected)!=set(IDS) or set(displayed)!=set(IDS):
        raise AssertionError('Nine-case selection/manifest mismatch')
    cases=[]
    for identity in IDS:
        entry,row=selected[identity],displayed[identity]
        if entry['preprocess']['version']!='document_preprocessing_V3' or entry['preprocess']['source_hashes']['lq']!=entry['source_sha256']:
            raise AssertionError('V3 source lineage mismatch '+identity)
        if (entry['width'],entry['height'])!=(row['width'],row['height']):
            raise AssertionError('Display geometry mismatch '+identity)
        final_url=row['images']['full']
        final_path=file_url_path(final_url)
        cases.append({'key':'cam'+identity,'id':identity,'name':row['name'],
                      'entry':entry,'row':row,'source_url':final_url,'source_path':final_path,
                      'model_receipt_path':final_path.parent.parent/'receipts'/(identity+'.json'),
                      'stored_lq_path':remote_path(entry['preprocess']['source_files']['lq']),
                      'v3_input_path':remote_path(entry['input']),
                      'v3_receipt_path':remote_path('/AI_DNS/guang.yang/TextSR/data/v3pre_v3m_three_20261027/finetune/source_view/receipts/'+
                                                    entry['split']+'/lenovo_mojito/'+entry['sample_id']+'.json')})
    return cases


def preflight_bindings(selection_bytes,manifest_bytes):
    """@luna-comment Input: frozen selection/gallery bytes. Output: source/core/runner hashes. Key Steps: compare live V3 modules to original source manifest, freeze all behavior inputs. Key Params: SHA256. State/Dependencies: fail closed on V3 implementation drift."""
    source_snapshot=json.loads((BASE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for relative in PRE_MODULES:
        if sha256(ROOT/relative)!=source_snapshot[relative]:
            raise AssertionError('V3 code differs from original preprocess '+relative)
    paths={'candidate_core':'tools/v3_reference_post.py',
           'runner':'tools/run_v3_reference_post_pilot.py',
           'old_post_core':'tools/anchored_y_contrast.py',
           'old_current_core':'tools/native_y_curve_four_images.py',
           'revision2':PRE_MODULES[0],'neighbor':PRE_MODULES[1],
           'pre640':PRE_MODULES[2],'prepare_v3pre_training':PRE_MODULES[3],
           'cluster_module':PRE_MODULES[4],'lowres_module':PRE_MODULES[5]}
    binding={name:sha256(ROOT/path) for name,path in paths.items()}
    binding.update(selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
                   manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   source_manifest_sha256=sha256(BASE/'SOURCE_MANIFEST.json'),
                   model_bindings_sha256=sha256(BASE/'MODEL_BINDINGS.json'))
    return binding


def verify_case_sources(case,bindings,model_bindings):
    """@luna-comment Input: one exact V3 case and code bindings. Output: native displayed model BGR8 and receipt hashes. Key Steps: validate original LQ, stored V3 input, model receipt/checkpoint and orientation. Key Params: full250K only. State/Dependencies: read-only Y; no model rerun."""
    entry=case['entry']
    if sha256(remote_path(entry['source']))!=entry['source_sha256'] or sha256(case['stored_lq_path'])!=entry['source_sha256']:
        raise AssertionError('Original source hash mismatch '+case['id'])
    if sha256(case['v3_input_path'])!=entry['input_sha256']:
        raise AssertionError('V3 stored input hash mismatch '+case['id'])
    v3_receipt=json.loads(case['v3_receipt_path'].read_text(encoding='utf-8'))
    if (v3_receipt['row']['output_sha256']['lq']!=entry['input_sha256'] or
            v3_receipt['row']['preprocess']['source_hashes']['lq']!=entry['source_sha256']):
        raise AssertionError('V3 receipt lineage mismatch '+case['id'])
    receipt=json.loads(case['model_receipt_path'].read_text(encoding='utf-8'))
    if (receipt['id']!=case['id'] or receipt['binding']!=model_bindings['full'] or
            receipt['selection_sha256']!=bindings['selection_sha256'] or
            receipt['source_sha256']!=entry['source_sha256'] or
            receipt['input_sha256']!=entry['input_sha256'] or
            receipt['display_orientation']!=entry['orientation'] or
            receipt['preprocess_version']!='document_preprocessing_V3'):
        raise AssertionError('Full-model receipt lineage mismatch '+case['id'])
    model=decode_bound(case['source_path'],receipt['final_sha256'],(entry['height'],entry['width']))
    if model.shape[:2]!=(case['row']['height'],case['row']['width']):
        raise AssertionError('Gallery/model dimensions mismatch '+case['id'])
    return model,sha256(case['model_receipt_path']),sha256(case['v3_receipt_path'])


def prepare_fields(case,bindings):
    """@luna-comment Input: original stored LQ/V3 input and frozen V3 code. Output: source-bound node NPZ plus exact-replay receipt. Key Steps: recompute revision2/descriptor/repair; frozen V3 replay must equal stored input pixelwise before caching. Key Params:81x81 nodes, strength1/chunk128. State/Dependencies: no model output enters estimation; only writes new repo cache."""
    identity=case['id'];entry=case['entry']
    cache=OUTPUT/'fields'/('cam'+identity+'.npz')
    receipt_path=OUTPUT/'fields'/('cam'+identity+'.json')
    if cache.exists() or receipt_path.exists():
        raise FileExistsError('Existing field cache '+identity)
    began=time.perf_counter()
    raw=decode_bound(case['stored_lq_path'],entry['source_sha256'],entry['stored_shape'][:2])
    estimate=revision2.estimate(raw)
    quantiles,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
    repaired,repair_info=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],
                                         quantiles,tail,estimate['nodes'])
    original_tables=neighbor.tables(repaired,estimate['fields']['high'])
    packed={'nodes':estimate['nodes'],'tables':original_tables,'native_shape':list(raw.shape[:2])}
    replay,_=pre640.render(raw,packed,strength=1.,chunk=128)
    cached=decode_bound(case['v3_input_path'],entry['input_sha256'],entry['stored_shape'][:2])
    if not np.array_equal(replay,cached):
        raise AssertionError('Exact original V3 replay failed '+identity)
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('xb') as stream:
        np.savez_compressed(stream,nodes=estimate['nodes'],low3=repaired,high=estimate['fields']['high'],
                            population=estimate['fields']['high_fraction'],quantiles=quantiles,
                            small_tail=tail,original_low=estimate['fields']['low'])
    receipt={'id':identity,'source_sha256':entry['source_sha256'],'v3_stored_sha256':entry['input_sha256'],
             'selection_sha256':bindings['selection_sha256'],'v3_code_sha256':
                 {key:bindings[key] for key in V3_BINDING_KEYS},
             'field_cache_path':str(cache),'field_cache_sha256':sha256(cache),
             'exact_v3_replay_equal':True,'exact_v3_replay_max_abs':0,
             'stored_shape':entry['stored_shape'],'orientation':entry['orientation'],
             'node_count':len(estimate['nodes']),'corrected_nodes':repair_info['corrected_nodes'],
             'source_only_prepare_seconds_host':time.perf_counter()-began}
    with receipt_path.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    return receipt,receipt_path


def load_fields(case,bindings):
    """@luna-comment Input: previously verified source-only field cache. Output: immutable node arrays and receipt. Key Steps: hash cache/receipt, require same selection/source/V3 code/orientation. Key Params: stage reuse with no re-estimation. State/Dependencies: fails if weak-stage provenance differs."""
    path=OUTPUT/'fields'/(case['key']+'.json')
    receipt=json.loads(path.read_text(encoding='utf-8'))
    entry=case['entry']
    if (receipt['id']!=case['id'] or receipt['source_sha256']!=entry['source_sha256'] or
            receipt['v3_stored_sha256']!=entry['input_sha256'] or
            receipt['selection_sha256']!=bindings['selection_sha256'] or
            receipt['orientation']!=entry['orientation'] or not receipt['exact_v3_replay_equal'] or
            receipt['exact_v3_replay_max_abs']!=0 or
            receipt['v3_code_sha256']!={key:bindings[key] for key in V3_BINDING_KEYS}):
        raise AssertionError('Field cache receipt binding mismatch '+case['id'])
    cache=Path(receipt['field_cache_path'])
    if sha256(cache)!=receipt['field_cache_sha256']:
        raise AssertionError('Field cache bytes changed '+case['id'])
    with np.load(cache,allow_pickle=False) as fields:
        arrays={key:fields[key].copy() for key in ('nodes','low3','high','population')}
    if not np.array_equal(arrays['nodes'],NODES):
        raise AssertionError('V3 node geometry changed '+case['id'])
    return arrays,receipt,path


def encode_png(image):
    """@luna-comment Input: display-native BGR8 preview. Output: lossless PNG bytes and encode time. Key Steps: OpenCV compression3. Key Params: no scaling/sharpening. State/Dependencies: result only under new repository output."""
    began=time.perf_counter()
    ok,png=cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not ok:
        raise RuntimeError('PNG encode failed')
    return png.tobytes(),time.perf_counter()-began


def write_arm(case,arm,image,telemetry,field_receipt,field_path,bindings,model_receipt_sha,v3_receipt_sha):
    """@luna-comment Input: one native candidate/control and complete provenance. Output: exclusive PNG/receipt and summary arm record. Key Steps: recheck source, encode/hash, record calibration boundary. Key Params: candidate reuse_*, controls current/mean_g3. State/Dependencies: never alters gallery, source, model or V3 caches."""
    path=OUTPUT/arm/(case['key']+'.png')
    receipt_path=OUTPUT/'receipts'/(case['key']+'_'+arm+'.json')
    if path.exists() or receipt_path.exists():
        raise FileExistsError('Existing arm output '+case['key']+' '+arm)
    if image.shape[:2]!=(case['row']['height'],case['row']['width']):
        raise AssertionError('Output not display-native '+case['key'])
    if sha256(case['source_path'])!=json.loads(case['model_receipt_path'].read_text())['final_sha256']:
        raise AssertionError('Full model source mutated '+case['key'])
    png,encode_seconds=encode_png(image)
    path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(png)
    source=case['entry'];candidate=arm.startswith('reuse_')
    record={'key':case['key'],'id':case['id'],'arm':arm,
            'source_url':case['source_url'],'source_sha256':sha256(case['source_path']),
            'model_receipt_path':str(case['model_receipt_path']),
            'model_receipt_sha256':model_receipt_sha,
            'original_lq_sha256':source['source_sha256'],'v3_stored_sha256':source['input_sha256'],
            'v3_receipt_path':str(case['v3_receipt_path']),'v3_receipt_sha256':v3_receipt_sha,
            'exact_v3_replay_max_abs':field_receipt['exact_v3_replay_max_abs'],
            'field_receipt_path':str(field_path),'field_receipt_sha256':sha256(field_path),
            'field_cache_path':field_receipt['field_cache_path'],
            'field_cache_sha256':field_receipt['field_cache_sha256'],
            'stored_shape':source['stored_shape'],'display_hw':[case['row']['height'],case['row']['width']],
            'orientation':source['orientation'],'output_calibration_enabled':False,
            'output_statistics_used_for_candidate_parameters':False,
            'control_uses_model_output_statistics':not candidate,
            'native_uv_identity_proven':False,'device_latency_proven':False,
            'bindings_sha256':bindings,'output_path':str(path),
            'output_sha256':hashlib.sha256(png).hexdigest(),
            'encode_seconds_host':encode_seconds,'telemetry':telemetry}
    with receipt_path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    return {'path':str(path),'sha256':record['output_sha256'],'receipt_path':str(receipt_path),
            'field_cache_sha256':field_receipt['field_cache_sha256']}


def generate_case(case,stage,bindings,model_bindings):
    """@luna-comment Input: one fully bound full-model final and weak/full stage. Output: case summary plus new arm PNGs. Key Steps: verify source/model lineage, exact V3 replay/cache, lossless orientation, fixed-field LUT candidate and isolated controls. Key Params: weak current/mean_g3/reuse_weak; full reuse_full. State/Dependencies: candidate never estimates model-output rails."""
    model,model_receipt_sha,v3_receipt_sha=verify_case_sources(case,bindings,model_bindings)
    if stage=='weak':
        prepare_fields(case,bindings)
    fields,field_receipt,field_path=load_fields(case,bindings)
    stored=canonical_to_stored(model,case['entry']['orientation'])
    if list(stored.shape)!=case['entry']['stored_shape'] or not np.array_equal(stored_to_canonical(stored,case['entry']['orientation']),model):
        raise AssertionError('Stored/canonical model permutation mismatch '+case['key'])
    arms=['current','mean_g3','reuse_weak'] if stage=='weak' else ['reuse_full']
    outputs={}
    if stage=='weak':
        current,mean,control_stats=process_arm(model,ARMS['mean_g3'])
        outputs['current']=write_arm(case,'current',current,{'control':control_stats},field_receipt,field_path,
                                     bindings,model_receipt_sha,v3_receipt_sha)
        outputs['mean_g3']=write_arm(case,'mean_g3',mean,{'control':control_stats},field_receipt,field_path,
                                     bindings,model_receipt_sha,v3_receipt_sha)
    strength=.35 if stage=='weak' else 1.0
    tables,field_stats=node_tables(fields['low3'],fields['high'],fields['population'],strengths=(strength,))
    candidate_stored,render_stats=render_post(stored,fields['nodes'],tables[strength])
    candidate=stored_to_canonical(candidate_stored,case['entry']['orientation'])
    arm='reuse_weak' if stage=='weak' else 'reuse_full'
    outputs[arm]=write_arm(case,arm,candidate,{'fields':field_stats,'render':render_stats,
        'candidate_minus_model_pixels':int(np.count_nonzero(np.any(candidate!=model,axis=2)))},
        field_receipt,field_path,bindings,model_receipt_sha,v3_receipt_sha)
    print(json.dumps({'done':case['id'],'stage':stage,'arms':arms,
                      'candidate_changed_pixels':render_stats['changed_pixels']}),flush=True)
    return {'key':case['key'],'id':case['id'],'name':case['name'],
            'width':case['row']['width'],'height':case['row']['height'],
            'source_url':case['source_url'],'source_sha256':sha256(case['source_path']),
            'lq_url':case['row']['lq'],'v3_input_url':case['row']['images']['v3_input'],
            'field_receipt_path':str(field_path),'field_cache_sha256':field_receipt['field_cache_sha256'],
            'outputs':outputs}


def main():
    """@luna-comment Input: frozen nine-case V3 selection/full model and stage CLI. Output: stage PNGs/receipts/summary. Key Steps: source-code snapshot gate, serial full lineage checks, exact V3 replay before weak, same cached fields for full, final binding check. Key Params: one cv thread, no inference. State/Dependencies: source Y read-only; only new repo output writes."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('weak','full'),default='weak')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    selection_bytes=(BASE/'selection.json').read_bytes()
    manifest_bytes=(BASE/'manifest.json').read_bytes()
    selection=json.loads(selection_bytes)
    manifest=json.loads(manifest_bytes)
    cases=source_cases(selection,manifest)
    bindings=preflight_bindings(selection_bytes,manifest_bytes)
    model_bindings=json.loads((BASE/'MODEL_BINDINGS.json').read_text(encoding='utf-8'))
    arms=['current','mean_g3','reuse_weak'] if args.stage=='weak' else ['reuse_full']
    summary_path=OUTPUT/(args.stage+'_summary.json')
    if summary_path.exists() or any((OUTPUT/arm/(case['key']+'.png')).exists() for arm in arms for case in cases):
        raise FileExistsError('Fresh stage outputs required')
    records=[generate_case(case,args.stage,bindings,model_bindings) for case in cases]
    if preflight_bindings((BASE/'selection.json').read_bytes(),(BASE/'manifest.json').read_bytes())!=bindings:
        raise AssertionError('Code or selection/manifest changed during stage')
    summary={'status':'experimental_visual_pending','stage':args.stage,'arms':arms,
             'case_count':len(records),'new_png_count':len(records)*len(arms),
             'bindings_sha256':bindings,
             'domain':'PNG-derived Y8 RGB-clipped preview; not native UV/NEON/140ms proof',
             'output_calibration_enabled':False,
             'output_statistics_used_for_candidate_parameters':False,'cases':records}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with summary_path.open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)
    print(str(summary_path),flush=True)


if __name__=='__main__':
    main()
