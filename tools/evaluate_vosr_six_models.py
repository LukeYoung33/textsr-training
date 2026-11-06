"""Hash-bound six-arm camera and aligned validation evaluation; no training mutations."""
import argparse
import base64
import html
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/guang.yang/workspace/Real-ESRGAN')
from tools.evaluate_offline_camera_matrix import sha, save_json, image_write, preview, verify_frozen
from tools.auto_hx_v3_native import restore, positions
from tools.doc_warp_composition import inverse_residual
from tools.build_offline_augmented_dataset import read_annotations, clipped_boxes
from evaluation.adaptive_prepost_frozen_v1_20261020.preprocess import prepare_context, apply_preprocess, make_post_context
from evaluation.adaptive_prepost_frozen_v1_20261020.postprocess import postprocess_bgr

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
NEW = DATA/'adaptive_vosr_preprocessed_augmented_v1_20261021'
OLD = DATA/'adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020/evaluation'
DOC = OLD/'camera_docwarp_robust_20261021'
OUT = DATA/'vosr_final_evaluation_20261022'
ARMS = ('01_v5_preprocessed', '02_v3_preprocessed', '03_v5_raw', '04_v3_raw', '05_v5_block256_raw', '06_v3_block256_raw')
INVENTORY = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016/inventory.json')


def read_image(path):
    """@luna-comment Input:path. Output:BGR8. Key Steps:checked decode. Key Params:none. State/Dependencies:read-only OpenCV."""
    value = cv2.imread(str(path))
    if value is None:
        raise ValueError('Cannot decode '+str(path))
    return value


def restore_block(rgb, network, device):
    """@luna-comment
    Input:RGB8 full frame/network/device. Output:RGB8 and tile telemetry.
    Key Steps:even-origin256 tiles, edge padding, Y4UV model, positive feather accumulation.
    Key Params:224 stride,16 feather,batch8. State/Dependencies:CUDA; no OCR crop or rescale.
    """
    import torch
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    h,w = rgb.shape[:2]
    grid = [(x,y) for y in positions(h,256,224) for x in positions(w,256,224)]
    total = np.zeros((h,w,3),np.float32)
    weights = np.zeros((h,w),np.float32)
    axis = np.minimum(np.minimum(np.arange(256)+1,256-np.arange(256))/16,1).astype(np.float32)
    window = axis[:,None]*axis[None,:]
    for start in range(0,len(grid),8):
        selected = grid[start:start+8]
        patches = []
        for x,y in selected:
            patch = rgb[y:y+256,x:x+256]
            patches.append(np.pad(patch,((0,256-patch.shape[0]),(0,256-patch.shape[1]),(0,0)),mode='edge'))
        batch = torch.from_numpy(np.stack(patches).transpose(0,3,1,2).copy()).to(device).float()/255
        with torch.inference_mode():
            decoded = YUV2RGB(y4uv_to_yuv444(network(rgb_to_y4uv(batch))),is_train=True).clamp(0,1)
        if decoded.shape != batch.shape or not torch.isfinite(decoded).all():
            raise ValueError('Invalid block prediction')
        for (x,y),value in zip(selected,decoded.permute(0,2,3,1).cpu().numpy()):
            hh,ww = min(256,h-y),min(256,w-x)
            total[y:y+hh,x:x+ww] += value[:hh,:ww]*window[:hh,:ww,None]
            weights[y:y+hh,x:x+ww] += window[:hh,:ww]
    if not np.all(weights>0):
        raise ValueError('Uncovered full-frame pixels')
    return np.clip(np.rint(total/weights[...,None]*255),0,255).astype(np.uint8),dict(canvas_hw=[256,256],stride_hw=[224,224],patches=len(grid),ocr_cropping=False)


def restore_strategy(source, boxes, network, arm, prepared=None, context=None):
    """@luna-comment
    Input:LQ/boxes/network/arm, optional hash-verified pre-cache. Output:input/model/final/telemetry.
    Key Steps:training-matched tonal routing, native geometry, LQ-only post context.
    Key Params:frozen1.8/1.2 only for preprocessed arms. State/Dependencies:no GT access.
    """
    use_pre = 'preprocessed' in arm
    if use_pre and prepared is None:
        ctx = prepare_context(source,boxes)
        prepared,context = apply_preprocess(source,ctx),make_post_context(ctx)
    inputs = prepared if use_pre else source
    rgb = cv2.cvtColor(inputs,cv2.COLOR_BGR2RGB)
    if 'block256' in arm:
        prediction,details = restore_block(rgb,network,'cuda')
    else:
        prediction,details = restore(rgb,boxes,network,'cuda',blend_source=cv2.cvtColor(source,cv2.COLOR_BGR2RGB))
    model = cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR)
    final = postprocess_bgr(model,context,gamma=1.2)[0] if use_pre else model
    details['preprocessing'] = 'frozen_gamma1.8' if use_pre else 'none'
    details['postprocessing'] = 'frozen_gamma1.2' if use_pre else 'none'
    if use_pre:
        details['post_context'] = context
    return inputs,model,final,details


def metrics(prediction, target, valid, text):
    """@luna-comment
    Input:BGR8 pair and binary masks. Output:RGB SSE/count/PSNR, Y SSIM and subgroup SSE.
    Key Steps:float64 SSE; Gaussian11 sigma1.5 SSIM; only wholly valid windows count.
    Key Params:255 range,C1=6.5025,C2=58.5225. State/Dependencies:GT evaluation-only.
    """
    squared = np.square(prediction.astype(np.float64)-target.astype(np.float64)).sum(2)
    result = {}
    for name,mask in (('all',valid),('text',valid & text),('background',valid & ~text)):
        count = int(mask.sum())*3
        sse = float(squared[mask].sum())
        result[name] = dict(sse=sse,count=count,psnr=(10*math.log10(255**2/(sse/count)) if count and sse else None),exact=bool(count and sse==0))
    a = cv2.cvtColor(prediction,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float64)
    b = cv2.cvtColor(target,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float64)
    ma,mb = cv2.GaussianBlur(a,(11,11),1.5),cv2.GaussianBlur(b,(11,11),1.5)
    va = cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma
    vb = cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov = cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    ssim = ((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    support = cv2.erode(valid.astype(np.uint8),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    result['ssim_y'] = dict(sum=float(ssim[support].sum()),count=int(support.sum()))
    return result


def validation_source(row):
    """@luna-comment
    Input:fixed manifest row. Output:verified aligned LQ/GT/masks/ordered boxes and source hashes.
    Key Steps:verify five assets, decode, merge supplementary OCR like training builder.
    Key Params:no crop/warp/GT-derived inference context. State/Dependencies:immutable583 pairs.
    """
    files = row['files']
    hashes = {key:sha(path) for key,path in files.items()}
    if hashes != row['output_sha256']:
        raise ValueError('Validation source changed '+row['sample_id'])
    source,target = read_image(files['lq']),read_image(files['gt'])
    valid = cv2.imread(files['valid_mask'],0)>0
    text = cv2.imread(files['text_mask'],0)>0
    if source.shape != target.shape or valid.shape != source.shape[:2] or text.shape != valid.shape:
        raise ValueError('Aligned geometry mismatch')
    boxes = clipped_boxes(read_annotations(json.loads(Path(files['ocr']).read_text())),source.shape)
    extra = NEW/'source_view/OCR_ppocrv6_medium/val'/(row['sample_id']+'.json')
    if extra.exists():
        hashes['supplemental_ocr'] = sha(extra)
        for box in clipped_boxes(read_annotations(json.loads(extra.read_text())),source.shape):
            if not any(abs((box[1]+box[3])-(b[1]+b[3])) < .5*min(box[3]-box[1],b[3]-b[1]) and min(box[2],b[2])>max(box[0],b[0]) for b in boxes):
                boxes.append(box)
    return source,target,valid,text,boxes,hashes


def run(args):
    """@luna-comment
    Input:arm/suite/partition. Output:lossless intermediates and resumable provenance receipts.
    Key Steps:strict final250K load, source verification, own strategy, aligned scoring or residual unwarp.
    Key Params:isolated output, no checkpoint substitution. State/Dependencies:read-only caches/CUDA.
    """
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    train = (DATA/'vosr_fullimage_block256_v1_20261021' if 'block256' in args.arm else NEW)/'training_comparison_20261021'
    config = train/'configs'/(args.arm+'.yml')
    options = yaml.safe_load(config.read_text())
    checkpoint = train/'experiments'/options['name']/'models/net_g_250000.pth'
    logs = list(checkpoint.parent.parent.glob('train_*.log'))
    if not logs or not any('End of training' in path.read_text(errors='replace') for path in logs):
        raise ValueError('Final training completion log missing')
    binding = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(config),frozen_manifest_sha256=verify_frozen(),evaluator_sha256=sha(__file__))
    model = build_network(options['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    if args.suite == 'camera':
        rows = json.loads(INVENTORY.read_text())['images']
    else:
        rows = [json.loads(line) for line in (NEW/'source_view/manifests/val.jsonl').read_text().splitlines()]
    if len(rows) != (280 if args.suite=='camera' else 583):
        raise ValueError('Wrong evaluation inventory size')
    selected = rows[:args.limit or None][args.rank::args.shards]
    destination = OUT/args.suite/args.arm
    for index,row in enumerate(selected,1):
        started = time.perf_counter()
        identity = row['id'] if args.suite=='camera' else row['sample_id']
        receipt_file = destination/'receipts'/(identity+'.json')
        if receipt_file.exists():
            cached = json.loads(receipt_file.read_text())
            if cached['binding'] != binding or sha(OUT/cached['final']) != cached['final_sha256']:
                raise ValueError('Resume provenance mismatch')
            continue
        stem = destination/'stages'/identity
        stages = []
        if args.suite == 'validation':
            source,target,valid,text,boxes,hashes = validation_source(row)
            inputs,model_image,final,details = restore_strategy(source,boxes,model,args.arm)
            for label,pixels in (('input',inputs),('model',model_image)):
                image_write(stem/(label+'.png'),pixels)
            score = metrics(final,target,valid,text)
            baseline = metrics(source,target,valid,text)
            shared = OUT/'validation/reference'/identity
            # Arm-independent references use identical immutable source pixels.
            preview(shared/'lq.jpg',source)
            preview(shared/'gt.jpg',target)
            stages = [dict(restoration=details,boxes=boxes)]
            provenance = dict(source_hashes=hashes,source_files=row['files'],dataset=row['dataset'],source_identity=row['source_identity'],metrics=score,baseline=baseline)
        else:
            source_file = OLD/'camera_final_20261021/source'/(identity+Path(row['name']).suffix.lower())
            if sha(source_file) != row['sha256']:
                raise ValueError('Original camera source changed')
            source = read_image(source_file)
            record_file = DOC/'documents'/source_file.name/'result.json'
            record = json.loads(record_file.read_text())
            if [record['image_height'],record['image_width']] != list(source.shape[:2]):
                raise ValueError('Document/source decoded geometry mismatch')
            canvas = source.astype(np.float32)
            owned = np.zeros(source.shape[:2],bool)
            for item in sorted(record['instances'],key=lambda v:(-v['confidence'],v['instance_id'])):
                iid = f"{item['instance_id']:03d}"
                ocr_file = DOC/'ocr'/identity/(iid+'.json')
                ocr = json.loads(ocr_file.read_text())
                if ocr['source_sha256'] != row['sha256'] or sha(ocr['warp_file']) != ocr['warp_sha256'] or sha(DOC/ocr['valid_file']) != ocr['valid_sha256']:
                    raise ValueError('Warp/OCR support changed')
                original = read_image(ocr['warp_file'])
                valid = cv2.imread(str(DOC/ocr['valid_file']),0)
                pre,ctx = None,None
                if 'preprocessed' in args.arm:
                    prepared = json.loads((DOC/'prepared'/identity/(iid+'.receipt.json')).read_text())
                    if prepared['frozen_manifest_sha256'] != binding['frozen_manifest_sha256'] or prepared['source_sha256'] != row['sha256'] or prepared['ocr_sha256'] != sha(ocr_file):
                        raise ValueError('Preparation provenance changed')
                    for key in ('image','context'):
                        if sha(DOC/prepared[key]) != prepared[key+'_sha256']:
                            raise ValueError('Preparation asset changed')
                    pre = read_image(DOC/prepared['image'])
                    ctx = json.loads((DOC/prepared['context']).read_text())
                inputs,prediction,post,details = restore_strategy(original,ocr['boxes'],model,args.arm,pre,ctx)
                delta,alpha,take,geometry = inverse_residual(source,original,post,item['warp']['image_to_output'],item['quad'],valid,owned)
                canvas += delta
                owned |= take
                for label,pixels in (('input',inputs),('model',prediction),('post',post),('alpha',np.clip(np.rint(alpha*255),0,255).astype(np.uint8))):
                    image_write(stem/iid/(label+'.png'),pixels)
                stages.append(dict(instance_id=item['instance_id'],ocr_sha256=sha(ocr_file),warp_sha256=ocr['warp_sha256'],restoration=details,geometry=geometry))
            final = np.clip(np.rint(canvas),0,255).astype(np.uint8)
            if not np.array_equal(final[~owned],source[~owned]):
                raise ValueError('Changed outside owned document support')
            image_write(stem/'ownership.png',owned.astype(np.uint8)*255)
            preview(OUT/'camera/reference'/(identity+'.jpg'),source)
            provenance = dict(source_file=str(source_file),source_sha256=row['sha256'],document_record=str(record_file),document_record_sha256=sha(record_file),outside_owned_unchanged=True)
        final_file = destination/'final'/(identity+'.png')
        image_write(final_file,final)
        preview(destination/'previews'/(identity+'.jpg'),final)
        save_json(receipt_file,dict(id=identity,arm=args.arm,suite=args.suite,binding=binding,final=str(final_file.relative_to(OUT)),final_sha256=sha(final_file),stages=stages,seconds=time.perf_counter()-started,**provenance))
        save_json(destination/f'progress.{args.rank}.json',dict(done=index,expected=len(selected),last=identity))
        print(json.dumps(dict(arm=args.arm,suite=args.suite,done=index,total=len(selected),id=identity,seconds=round(time.perf_counter()-started,2))),flush=True)
    save_json(destination/f'COMPLETE.{args.rank}.json',dict(images=len(selected),binding=binding,pilot=bool(args.limit)))


def report():
    """@luna-comment
    Input:all immutable receipts/previews. Output:finite pooled metrics and self-contained comparison HTML.
    Key Steps:verify counts/final hashes, pool identical-support SSE and SSIM, embed previews.
    Key Params:280/583 per arm. State/Dependencies:completed jobs; no inferred quality labels.
    """
    summaries = {}
    gallery = {}
    for suite,expected in (('camera',280),('validation',583)):
        gallery[suite] = {}
        for arm in ARMS:
            receipts = [json.loads(p.read_text()) for p in sorted((OUT/suite/arm/'receipts').glob('*.json'))]
            if len(receipts) != expected:
                raise ValueError(f'Incomplete {suite}/{arm}: {len(receipts)}/{expected}')
            for receipt in receipts:
                if sha(OUT/receipt['final']) != receipt['final_sha256']:
                    raise ValueError('Final checksum mismatch')
                identifier = receipt['id']
                entry = gallery[suite].setdefault(identifier,{})
                if suite=='validation':
                    for previous in entry.values():
                        if previous['metrics'] and previous['metrics']['all']['count'] != receipt['metrics']['all']['count']:
                            raise ValueError('Cross-arm valid-support mismatch')
                entry[arm] = dict(preview='data:image/jpeg;base64,'+base64.b64encode((OUT/suite/arm/'previews'/(identifier+'.jpg')).read_bytes()).decode(),final=receipt['final'],metrics=receipt.get('metrics'))
            if suite=='validation':
                for key in ('metrics','baseline'):
                    pooled = {}
                    for region in ('all','text','background'):
                        sse = sum(r[key][region]['sse'] for r in receipts)
                        count = sum(r[key][region]['count'] for r in receipts)
                        pooled[region] = dict(sse=sse,count=count,psnr=10*math.log10(255**2/(sse/count)) if sse else None)
                    count = sum(r[key]['ssim_y']['count'] for r in receipts)
                    pooled['ssim_y'] = sum(r[key]['ssim_y']['sum'] for r in receipts)/count
                    summaries[arm if key=='metrics' else 'LQ_baseline'] = pooled
        for identity,entry in gallery[suite].items():
            refs = {'LQ':OUT/'camera/reference'/(identity+'.jpg')} if suite=='camera' else {name:OUT/'validation/reference'/identity/(name.lower()+'.jpg') for name in ('LQ','GT')}
            entry['references'] = {k:'data:image/jpeg;base64,'+base64.b64encode(v.read_bytes()).decode() for k,v in refs.items()}
    save_json(OUT/'METRICS.json',summaries)
    table = '<table><tr><th>Arm</th><th>RGB PSNR</th><th>Text PSNR</th><th>Background PSNR</th><th>Y SSIM</th></tr>'
    for arm,value in summaries.items():
        table += '<tr><td>'+html.escape(arm)+'</td>'+''.join('<td>'+f'{value[r]["psnr"]:.4f}'+'</td>' for r in ('all','text','background'))+f'<td>{value["ssim_y"]:.6f}</td></tr>'
    table += '</table>'
    pages = OUT/'pages'
    pages.mkdir(exist_ok=True)
    catalog = {}
    for suite,entries in gallery.items():
        catalog[suite] = list(entries)
        for identity,entry in entries.items():
            cells = ''.join('<div><h3>'+label+'</h3><img src="'+url+'"></div>' for label,url in entry['references'].items())
            for arm in ARMS:
                value = entry[arm]
                score = value['metrics']
                label = (' | RGB PSNR '+str(round(score['all']['psnr'],4))) if score and score['all']['psnr'] is not None else ''
                cells += '<div><h3>'+arm+label+'</h3><a href="../'+value['final']+'">Full resolution</a> | <a href="../'+suite+'/'+arm+'/stages/'+identity+'/">Intermediate stages</a><img src="'+value['preview']+'"></div>'
            content = '<!doctype html><meta charset="utf-8"><style>body{font:15px sans-serif}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}img{width:100%}</style><h2>'+suite+' / '+html.escape(identity)+'</h2><div class="grid">'+cells+'</div>'
            (pages/(suite+'_'+identity+'.html')).write_text(content,encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><title>VOSR six models</title><style>body{font:15px sans-serif}td,th{border:1px solid #bbb;padding:7px}iframe{width:100%;height:85vh;border:0}select,button{padding:8px}</style><h1>VOSR six final models</h1><p>Common raw GT_VOSR domain. Pixel-pooled RGB PSNR; Gaussian11 sigma1.5 Y SSIM, wholly valid windows. No paired GT for camera frames. Native96x480 OCR-region versus whole-document256 tiles. Preprocessed arms: gamma1.8 input + gamma1.2 LQ-context postprocess. Raw arms: no tonal operators.</p>'+table
    page += '<p><select id="suite"><option>camera</option><option>validation</option></select><button id="prev">Previous</button><select id="sample"></select><button id="next">Next</button><button onclick="location.reload()">Refresh</button></p><iframe id="viewer"></iframe>'
    page += '<script>const data='+json.dumps(catalog)+';const s=document.querySelector("#suite"),n=document.querySelector("#sample");function draw(){document.querySelector("#viewer").src="pages/"+s.value+"_"+n.value+".html"}function populate(){n.replaceChildren(...data[s.value].map(k=>new Option(k,k)));draw()}s.onchange=populate;n.onchange=draw;document.querySelector("#prev").onclick=()=>{n.selectedIndex=Math.max(0,n.selectedIndex-1);draw()};document.querySelector("#next").onclick=()=>{n.selectedIndex=Math.min(n.length-1,n.selectedIndex+1);draw()};populate();</script>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    save_json(OUT/'COMPLETE.json',dict(camera_per_arm=280,validation_per_arm=583,arms=list(ARMS),metrics=summaries))


def main():
    """@luna-comment Input:CLI. Output:isolated run/report. Key Steps:validate partition, dispatch. Key Params:explicit arm/suite. State/Dependencies:remote paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','report'))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--suite',choices=('camera','validation'))
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    if not 0<=args.rank<args.shards or (args.mode=='run' and (not args.arm or not args.suite)):
        parser.error('Specify arm/suite and valid partition')
    report() if args.mode=='report' else run(args)


if __name__=='__main__':
    main()
