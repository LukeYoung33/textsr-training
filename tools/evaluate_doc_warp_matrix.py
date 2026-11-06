"""Frozen document detection -> fresh warped OCR -> native restoration -> inverse residual."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_offline_camera_matrix import ARMS,sha,save_json,image_write,preview,verify_frozen
from tools.doc_warp_composition import source_support,inverse_geometry,inverse_residual

TRAIN = Path('/AI_DNS/guang.yang/TextSR/data/adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020')
BASE = TRAIN / 'evaluation/camera_final_20261021'
OUT = Path(os.environ.get('DOC_WARP_OUTPUT',str(TRAIN / 'evaluation/camera_docwarp_20261021')))
INVENTORY = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016/inventory.json')
REPO = '/home/guang.yang/workspace/Real-ESRGAN'


def source_path(row):
    """@luna-comment
    Input: original inventory row. Output: prior verified portable original path.
    Key Steps: bind stable ID and original suffix. Key Params: no source substitutions.
    State/Dependencies: immutable baseline source directory.
    """
    return BASE / 'source' / (row['id']+Path(row['name']).suffix.lower())


def document_record(row):
    """@luna-comment
    Input: original row. Output: released detector record and containing directory.
    Key Steps: locate by copied filename; bind to baseline's actual decoded frame
    dimensions and source hash, never raw JPEG-header dimensions before EXIF rotation.
    Key Params: missing detector record is an error, not no_document.
    State/Dependencies: frozen detector's actual result JSON.
    """
    directory = OUT / 'documents' / source_path(row).name
    record = json.loads((directory/'result.json').read_text())
    baseline=json.loads((BASE/'prepared'/(row['id']+'.receipt.json')).read_text())
    if baseline['source_sha256']!=row['sha256'] or [record['image_height'],record['image_width']] != baseline['source_hw']:
        raise ValueError('Document/source decode shape mismatch')
    return record,directory


def await_document(row):
    """@luna-comment
    Input: one original row. Output: complete released detector JSON when ready.
    Key Steps: retry absent/in-progress JSON while the recorded detector is alive;
    fail on detector exit/error instead of fabricating an empty document result.
    Key Params: .5s polls and2h deadline. State/Dependencies: own detector receipt/process.
    """
    deadline=time.monotonic()+7200
    while True:
        try:
            return document_record(row)
        except (FileNotFoundError,json.JSONDecodeError):
            launch=json.loads((OUT/'DOC_LAUNCH.json').read_text())
            try:
                os.kill(launch['pid'],0)
            except ProcessLookupError:
                raise RuntimeError('Document detector exited without required result: '+row['id'])
            if time.monotonic()>deadline:
                raise TimeoutError('Document result not ready: '+row['id'])
            time.sleep(.5)


def detect_warped_text(args,rows):
    """@luna-comment
    Input: disjoint original-row partition and completed released document warps.
    Output: fresh source-bound per-instance OCR/overlay and valid-source masks.
    Key Steps: verify detector weights against baseline, detect in warped coordinates,
    reject empty/padding-only bounding rectangles; retain partial support and the
    warped detector's emitted order as clipped half-open pixel boxes (x0,y0,x1,y1).
    Key Params: PP-OCRv6_medium_det/max2400/.3/.6/unclip1.5, CPU threads2.
    State/Dependencies: local Paddle model only; no downloads, no original OCR reuse.
    """
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    from paddlex import create_model
    cv2.setNumThreads(1)
    model_root=Path('/home/guang.yang/workspace/PaddleOCR-3.7.0/deploy/official_models/PP-OCRv6_medium_det')
    hashes={p.name:sha(p) for p in model_root.iterdir() if p.is_file()}
    reference=Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016/detections/IMG_20260605_011836092.json')
    if hashes != json.loads(reference.read_text())['model_hashes']:
        raise ValueError('Warped OCR detector differs from the prior baseline')
    detector=create_model(model_name='PP-OCRv6_medium_det',model_dir=str(model_root),device='cpu',enable_mkldnn=True,cpu_threads=2)
    count=0
    for frame_index,row in enumerate(rows):
        if frame_index%args.shards != args.rank:
            continue
        if sha(source_path(row)) != row['sha256']:
            raise ValueError('Portable original changed')
        record,directory=await_document(row)
        for instance in record['instances']:
            identifier=f"{instance['instance_id']:03d}"
            output=OUT/'ocr'/row['id']/(identifier+'.json')
            path=directory/instance['image_file']
            warp_hash=sha(path)
            if output.exists():
                cached=json.loads(output.read_text())
                if cached['warp_sha256'] != warp_hash or cached['model_hashes'] != hashes:
                    raise ValueError('Existing warped OCR belongs to another input/model')
                count+=1
                continue
            image=cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable document warp')
            height,width=image.shape[:2]
            matrix=np.asarray(instance['warp']['image_to_output'],np.float64)
            support=source_support((record['image_height'],record['image_width']),matrix,(height,width))
            inverse,error=inverse_geometry(matrix,instance['quad'])
            predictions=list(detector.predict(input=image,batch_size=1,limit_side_len=2400,limit_type='max',thresh=.3,box_thresh=.6,unclip_ratio=1.5))
            if len(predictions)!=1:
                raise ValueError('Expected exactly one warped OCR result')
            raw=predictions[0].json
            raw=json.loads(raw) if isinstance(raw,str) else raw
            data=raw.get('res',raw)
            boxes,polygons,rejected=[],[],[]
            overlay=image.copy()
            for polygon in data['dt_polys']:
                points=np.asarray(polygon,np.float64).reshape(-1,2)
                if not np.isfinite(points).all():
                    raise ValueError('Nonfinite OCR polygon')
                x0,y0=np.maximum(np.floor(points.min(0)).astype(int),0)
                x1,y1=np.minimum(np.ceil(points.max(0)).astype(int),[width,height])
                if x1<=x0 or y1<=y0 or not np.any(support[y0:y1,x0:x1]==255):
                    rejected.append(dict(polygon=points.tolist(),reason='empty_or_unknown_padding_only'))
                    continue
                boxes.append(list(map(int,(x0,y0,x1,y1))))
                polygons.append(points.tolist())
                cv2.polylines(overlay,[np.rint(points).astype(np.int32)],True,(0,200,0),2,cv2.LINE_AA)
            valid_relative=f"ocr/{row['id']}/{identifier}.valid.png"
            overlay_relative=f"ocr/{row['id']}/{identifier}.overlay.png"
            image_write(OUT/valid_relative,support)
            image_write(OUT/overlay_relative,overlay)
            save_json(output,dict(id=row['id'],instance_id=instance['instance_id'],source_sha256=row['sha256'],
                     warp_file=str(path),warp_sha256=warp_hash,warp_hw=[height,width],boxes=boxes,polygons=polygons,
                     rejected=rejected,valid_file=valid_relative,valid_sha256=sha(OUT/valid_relative),overlay=overlay_relative,
                     transform=matrix.tolist(),inverse=inverse.tolist(),roundtrip_error=error,
                     geometry_type=instance['geometry_type'],model_hashes=hashes,
                     parameters=dict(max_side=2400,thresh=.3,box_thresh=.6,unclip_ratio=1.5),fresh_warp_detection=True))
            count+=1
            print(json.dumps(dict(ocr_rank=args.rank,frame=row['id'],instance=instance['instance_id'],text_boxes=len(boxes))),flush=True)
        save_json(OUT/f'ocr_progress.rank{args.rank}.json',dict(last_frame=row['id'],instances=count))
    save_json(OUT/f'OCR_COMPLETE.rank{args.rank}.json',dict(complete=True,instances=count,model_hashes=hashes))


def prepare_instance(job):
    """@luna-comment
    Input: frame ID/instance ID/frozen manifest hash. Output: shared warp preprocessing.
    Key Steps: verify OCR/warp/valid hashes, estimate original-reference context,
    apply gamma1.8 once and preserve pixels outside ordered OCR boxes.
    Key Params: unknown padding remains excluded from eventual inverse pasteback.
    State/Dependencies: frozen CPU API; no model/geometry mutation.
    """
    from evaluation.adaptive_prepost_frozen_v1_20261020.preprocess import prepare_context,apply_preprocess,make_post_context
    identity,identifier,frozen_hash=job
    cv2.setNumThreads(1)
    ocr_path=OUT/'ocr'/identity/(identifier+'.json')
    ocr=json.loads(ocr_path.read_text())
    receipt_path=OUT/'prepared'/identity/(identifier+'.receipt.json')
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['ocr_sha256'] != sha(ocr_path) or receipt['frozen_manifest_sha256'] != frozen_hash:
            raise ValueError('Existing preparation differs from current OCR/frozen package')
        if sha(OUT/receipt['image']) != receipt['image_sha256'] or sha(OUT/receipt['context']) != receipt['context_sha256']:
            raise ValueError('Prepared artifact changed')
        return receipt
    if sha(ocr['warp_file']) != ocr['warp_sha256'] or sha(OUT/ocr['valid_file']) != ocr['valid_sha256']:
        raise ValueError('Warp/support hash mismatch')
    image=cv2.imread(ocr['warp_file'])
    context=prepare_context(image,ocr['boxes'],reference_id=ocr['source_sha256']+':'+identifier+':'+ocr['warp_sha256'])
    processed=apply_preprocess(image,context)
    mask=np.zeros(image.shape[:2],bool)
    for x0,y0,x1,y1 in ocr['boxes']:
        mask[y0:y1,x0:x1]=True
    if not np.array_equal(processed[~mask],image[~mask]):
        raise ValueError('Preprocessing changed outside warped OCR union')
    image_relative=f'prepared/{identity}/{identifier}.png'
    context_relative=f'prepared/{identity}/{identifier}.context.json'
    image_write(OUT/image_relative,processed)
    save_json(OUT/context_relative,make_post_context(context))
    receipt=dict(frame=identity,instance=identifier,source_sha256=ocr['source_sha256'],warp_sha256=ocr['warp_sha256'],
                 ocr_sha256=sha(ocr_path),image=image_relative,image_sha256=sha(OUT/image_relative),
                 context=context_relative,context_sha256=sha(OUT/context_relative),frozen_manifest_sha256=frozen_hash,
                 preprocess_gamma=1.8,outside_ocr_unchanged=True)
    save_json(receipt_path,receipt)
    return receipt


def infer(args,rows):
    """@luna-comment
    Input: one arm/disjoint frame partition with shared document/OCR/preparation.
    Output: model/post warps, cumulative original-frame composites, diagnostic
    uint8 alpha PNGs and receipts; float residual/alpha are transient composition state.
    Key Steps: strict baseline-identical checkpoint, native restore, frozen post,
    valid weighted residual inverse warp and confidence-ordered first ownership.
    Key Params:96x480/stride64x448/post1.2; no document -> explicit unchanged original.
    State/Dependencies: one inherited CUDA card, frozen package, immutable source/baseline.
    """
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap(REPO)
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore
    from evaluation.adaptive_prepost_frozen_v1_20261020.postprocess import postprocess_bgr
    cv2.setNumThreads(1);torch.set_num_threads(2)
    config=TRAIN/'configs'/(args.arm+'.yml')
    settings=yaml.safe_load(config.read_text())
    checkpoint=TRAIN/'experiments'/settings['name']/'models/net_g_250000.pth'
    checkpoint_hash=sha(checkpoint)
    baseline=json.loads((BASE/args.arm/'COMPLETE.json').read_text())
    if checkpoint_hash!=baseline['checkpoint_sha256']:
        raise ValueError('New pipeline is not using the exact previous checkpoint')
    model=build_network(settings['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    frozen_hash=verify_frozen()
    results=[]
    for row_index,row in enumerate(rows):
        if row_index%args.shards!=args.rank:
            continue
        started=time.perf_counter()
        record,directory=document_record(row)
        record_hash=sha(directory/'result.json')
        destination=OUT/args.arm
        receipt_path=destination/'receipts'/(row['id']+'.json')
        if receipt_path.exists():
            cached=json.loads(receipt_path.read_text())
            if cached['checkpoint_sha256']!=checkpoint_hash or cached['document_record_sha256']!=record_hash or cached['frozen_manifest_sha256']!=frozen_hash:
                raise ValueError('Existing corrected result differs in provenance')
            if sha(OUT/cached['final'])!=cached['final_sha256']:
                raise ValueError('Corrected final image changed')
            results.append(cached)
            continue
        source=cv2.imread(str(source_path(row)))
        if source is None or sha(source_path(row))!=row['sha256']:
            raise ValueError('Source original changed')
        canvas=source.astype(np.float32)
        owned=np.zeros(source.shape[:2],bool)
        instances=sorted(record['instances'],key=lambda item:(-item['confidence'],item['instance_id']))
        stages=[]
        for item in instances:
            identifier=f"{item['instance_id']:03d}"
            ocr_path=OUT/'ocr'/row['id']/(identifier+'.json')
            ocr=json.loads(ocr_path.read_text())
            prepared=json.loads((OUT/'prepared'/row['id']/(identifier+'.receipt.json')).read_text())
            if prepared['frozen_manifest_sha256']!=frozen_hash or prepared['source_sha256']!=row['sha256'] or prepared['ocr_sha256']!=sha(ocr_path):
                raise ValueError('Wrong warp preparation/context')
            if sha(OUT/prepared['image'])!=prepared['image_sha256'] or sha(OUT/prepared['context'])!=prepared['context_sha256']:
                raise ValueError('Warp preparation changed')
            original_warp=cv2.imread(ocr['warp_file'])
            processed=cv2.imread(str(OUT/prepared['image']))
            valid=cv2.imread(str(OUT/ocr['valid_file']),cv2.IMREAD_GRAYSCALE)
            if sha(ocr['warp_file'])!=ocr['warp_sha256'] or sha(OUT/ocr['valid_file'])!=ocr['valid_sha256']:
                raise ValueError('Original warp/support changed')
            prediction,details=restore(cv2.cvtColor(processed,cv2.COLOR_BGR2RGB),ocr['boxes'],model,'cuda',blend_source=cv2.cvtColor(original_warp,cv2.COLOR_BGR2RGB))
            model_bgr=cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR)
            context=json.loads((OUT/prepared['context']).read_text())
            post,reports=postprocess_bgr(model_bgr,context,gamma=1.2)
            delta,alpha,take,geometry=inverse_residual(source,original_warp,post,item['warp']['image_to_output'],item['quad'],valid,owned)
            canvas+=delta;owned|=take
            composite=np.clip(np.rint(canvas),0,255).astype(np.uint8)
            stem=f"{args.arm}/instances/{row['id']}/{identifier}"
            paths=dict(model=stem+'.model.png',post=stem+'.post.png',unwarp=stem+'.unwarp.png',alpha=stem+'.alpha.png')
            for key,pixels in (('model',model_bgr),('post',post),('unwarp',composite),('alpha',np.clip(np.rint(alpha*255),0,255).astype(np.uint8))):
                image_write(OUT/paths[key],pixels)
            stage=dict(instance_id=item['instance_id'],confidence=item['confidence'],geometry_type=item['geometry_type'],
                       warp_hw=list(original_warp.shape[:2]),text_boxes=len(ocr['boxes']),image_to_warp=item['warp']['image_to_output'],
                       geometry=geometry,postprocess_gamma=1.2,paths=paths,restoration=details,post_lines=len(reports),
                       alpha_file_encoding='uint8_diagnostic_only_float32_used_for_composition')
            save_json(OUT/(stem+'.json'),stage);stages.append(stage)
        final=np.clip(np.rint(canvas),0,255).astype(np.uint8)
        if final.shape!=source.shape or not np.array_equal(final[~owned],source[~owned]):
            raise ValueError('Unwarp modified pixels outside owned valid support')
        if not instances and not np.array_equal(final,source):
            raise ValueError('No-document frame must remain exactly original')
        final_relative=f"{args.arm}/final/{row['id']}.png"
        image_write(OUT/final_relative,final)
        image_write(destination/'ownership'/(row['id']+'.png'),owned.astype(np.uint8)*255)
        preview(destination/'previews'/(row['id']+'.jpg'),final)
        receipt=dict(id=row['id'],name=row['name'],source_sha256=row['sha256'],source_hw=list(source.shape[:2]),arm=args.arm,
                     checkpoint_sha256=checkpoint_hash,config_sha256=sha(config),frozen_manifest_sha256=frozen_hash,
                     document_record_sha256=record_hash,status='processed' if instances else 'no_document_original_unchanged',
                     final=final_relative,final_sha256=sha(OUT/final_relative),documents=len(instances),instances=stages,
                     outside_owned_support_unchanged=True,overlap_policy='descending_confidence_stable_first_owner',
                     seconds=time.perf_counter()-started)
        save_json(receipt_path,receipt);results.append(receipt)
        save_json(destination/f'progress.rank{args.rank}.json',dict(done=len(results),expected=len(rows[args.rank::args.shards]),last=row['id']))
        print(json.dumps(dict(arm=args.arm,rank=args.rank,done=len(results),frame=row['id'],instances=len(instances),seconds=receipt['seconds'])),flush=True)
    save_json(OUT/args.arm/f'progress.rank{args.rank}.json',dict(done=len(results),expected=len(rows[args.rank::args.shards]),last=results[-1]['id'] if results else None))
    save_json(OUT/args.arm/f'COMPLETE.rank{args.rank}.json',dict(complete=True,images=len(results),checkpoint_sha256=checkpoint_hash,frozen_manifest_sha256=frozen_hash))


def main():
    """@luna-comment
    Input: stage, partition options and optional pilot limit. Output: explicit stage artifacts.
    Key Steps: load exact original inventory; dispatch fresh OCR, shared preparation or inference.
    Key Params: full280 by default; CLI defaults to one shard, while the launcher
    selects16 OCR shards and3 inference shards per model/GPU.
    State/Dependencies: task-isolated output (optional DOC_WARP_OUTPUT override);
    immutable baseline/release untouched.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('ocr','prepare','infer'))
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if not 0<=args.rank<args.shards:
        parser.error('Invalid partition')
    rows=json.loads(INVENTORY.read_text())['images'][:args.limit or None]
    if args.mode=='ocr':
        detect_warped_text(args,rows)
    elif args.mode=='prepare':
        frozen=verify_frozen()
        jobs=[]
        for row in rows:
            record,_=document_record(row)
            jobs.extend((row['id'],f"{item['instance_id']:03d}",frozen) for item in record['instances'])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count,receipt in enumerate(pool.map(prepare_instance,jobs),1):
                save_json(OUT/'prepare_progress.json',dict(done=count,expected=len(jobs),last=receipt['frame']))
        marker='PREPARED_PILOT.json' if args.limit else 'PREPARED.json'
        save_json(OUT/marker,dict(complete=True,images=len(rows),instances=len(jobs),frozen_manifest_sha256=frozen))
    else:
        if not args.arm:
            parser.error('--arm required')
        infer(args,rows)


if __name__=='__main__':
    main()
