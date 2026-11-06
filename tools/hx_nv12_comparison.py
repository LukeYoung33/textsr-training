"""Inventory, decode and evaluate the explicitly selected HX NV12 camera dumps."""
import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/guang.yang/workspace/Real-ESRGAN')
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'hx_nv12_three_pipeline_20261022'
BUNDLE = DATA/'releases/textsr_v22_v3_bundle_20261022/textsr_v22_v3'


def sha(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream chunks. Key Params:4MiB. State/Dependencies:read-only asset."""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    """@luna-comment Input:path/value. Output:UTF8 finite JSON. Key Steps:create owned parent/write. Key Params:no NaN. State/Dependencies:task outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def pack(args):
    """@luna-comment
    Input:authorized dump directory/archive destination. Output:122-file archive and61-pair hashed inventory.
    Key Steps:validate alternating input/output timestamps, geometry/stride/bytes; stream compression.
    Key Params:NV12 single frame; maximum following-output gap3s; gzip level1.
    State/Dependencies:read-only originals; user-approved export destination; no unrelated files included.
    """
    files = sorted(args.source.glob('*.yuv'), key=lambda p:int(p.name.split('_')[0]))
    assert files and len(files)%2 == 0
    rows = []
    for i in range(0, len(files), 2):
        a,b = files[i:i+2]
        assert 'bst_textsr_input_0_' in a.name and 'bst_textsr_output_' in b.name
        gap = int(b.name.split('_')[0])-int(a.name.split('_')[0])
        assert 0 < gap < 3000
        assets = {}
        for key,path in [('input',a), ('output',b)]:
            width,height,stride = map(int,re.search(r'_(\d+)x(\d+)_(\d+)',path.name).groups())
            assert width%2==0 and height%2==0 and stride>=width and stride%2==0
            assert path.stat().st_size == stride*height*3//2
            assets[key] = dict(name=path.name, width=width, height=height, stride=stride, bytes=path.stat().st_size, sha256=sha(path))
        assert all(assets['input'][k]==assets['output'][k] for k in ('width','height','stride'))
        rows.append(dict(id=f'{i//2+1:03d}', gap_ms=gap, **assets))
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(format='NV12', conversion='OpenCV COLOR_YUV2BGR_NV12; BT.601 limited range; identical for input/output', pairs=rows)
    inventory = args.archive.with_suffix('.json')
    save(inventory,manifest)
    with tarfile.open(args.archive, 'w:gz', compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname=path.name, recursive=False)
        archive.add(inventory, arcname='inventory.json')
    print(json.dumps(dict(pairs=len(rows), archive=str(args.archive), bytes=args.archive.stat().st_size)), flush=True)


def decode(args):
    """@luna-comment
    Input:transferred archive of explicitly paired dumps. Output:lossless LQ/device PNGs and decoded manifest.
    Key Steps:safe flat extraction; hash/stride validation; crop planes before NV12 conversion.
    Key Params:BT.601 limited-range OpenCV conversion, same for both assets; no enhancement applied.
    State/Dependencies:owned output tree/OpenCV; preserve original YUV bytes.
    """
    import cv2
    import numpy as np
    from tools.evaluate_offline_camera_matrix import image_write
    raw = OUT/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'r:gz') as archive:
        for item in archive.getmembers():
            assert item.isfile() and Path(item.name).name==item.name and item.name not in ('.','..')
            with archive.extractfile(item) as source:
                (raw/item.name).write_bytes(source.read())
    inventory = json.loads((raw/'inventory.json').read_text(encoding='utf-8'))
    for row in inventory['pairs']:
        for key,folder in [('input','lq'),('output','device_output')]:
            asset = row[key]
            path = raw/asset['name']
            assert sha(path)==asset['sha256']
            w,h,s = (asset[k] for k in ('width','height','stride'))
            data = np.fromfile(path,np.uint8)
            assert data.size==s*h*3//2
            y = data[:s*h].reshape(h,s)[:,:w]
            uv = data[s*h:].reshape(h//2,s)[:,:w]
            image = cv2.cvtColor(np.concatenate([y,uv],axis=0), cv2.COLOR_YUV2BGR_NV12)
            target = OUT/folder/(row['id']+'.png')
            image_write(target,image)
            asset.update(png=str(target),png_sha256=sha(target))
        print('decoded '+row['id'],flush=True)
    save(OUT/'inventory.json', inventory)


def infer(args, final_blending=False):
    """@luna-comment
    Input:shared fresh document/OCR/V2_2 cache and one selected arm. Output:stages/native final/provenance.
    Key Steps:load exact prior final checkpoint; native V3/pre/post or raw V5/block; confidence ownership/unwarp.
    Key Params:V3 native96x480; V5 whole-warp256 stride224; pre-only bypasses model/post.
    final_blending selects unblended pre inputs and defers the only box feather until after post.
    State/Dependencies:immutable released geometry/tonal algorithms; original eager networks; task-owned outputs.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.run_camera_compare_missing import model_paths
    from tools.auto_hx_v3_native import restore
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import image_write
    from tools.doc_warp_composition import inverse_residual
    from evaluation.adaptive_prepost_frozen_v1_20261020.postprocess import postprocess_bgr
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    model, binding = None, dict(preprocessing='V2_2' if args.arm!='vosr_v5_block' else 'none')
    if final_blending:binding.update(revision='upright_postblend_v2',input_preblend=False,final_box_feather=16 if args.arm in ('original_v3_pre','vosr_v3_pre') else 0)
    if args.arm!='pre':
        if args.arm=='vosr_v5_block':
            train = DATA/'vosr_fullimage_block256_v1_20261021/training_comparison_20261021'
            config = train/'configs/05_v5_block256_raw.yml'
            checkpoint = train/'experiments'/yaml.safe_load(config.read_text())['name']/'models/net_g_250000.pth'
        else:
            config, checkpoint = model_paths(args.arm)
        binding.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(config))
        model = build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    records = json.loads((OUT/'run/documents/run.json').read_text())
    assert records['status']=='completed' and not records['errors']
    for i,entry in enumerate(records['results'],1):
        if (i-1)%args.shards!=args.rank:
            continue
        identity = Path(entry['source']).stem
        receipt_path = OUT/'results'/args.arm/'receipts'/(identity+'.json')
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            assert old['binding']==binding and sha(OUT/old['final'])==old['final_sha256']
            continue
        started = time.perf_counter()
        source = cv2.imread(entry['source'])
        document = json.loads(Path(entry['result_json']).read_text())
        assert list(source.shape[:2])==[document['image_height'],document['image_width']]
        canvas,owned,stages = source.astype(np.float32),np.zeros(source.shape[:2],bool),[]
        for item in sorted(document['instances'],key=lambda v:(-v['confidence'],v['instance_id'])):
            iid = f"{item['instance_id']:03d}"
            shared = OUT/'run/shared'/f'{i:06d}'/iid
            ocr = json.loads((shared/'ocr.json').read_text())
            warp = Path(entry['result_json']).parent/item['image_file']
            assert ocr['warp_sha256']==sha(warp) and ocr['source_sha256']==sha(entry['source'])
            assert ocr['transform']==item['warp']['image_to_output']
            original = cv2.imread(str(warp))
            use_pre = args.arm!='vosr_v5_block'
            inputs = cv2.imread(str(shared/('pre_unblended.png' if final_blending else 'pre.png'))) if use_pre else original
            details = dict(preprocessing='V2_2' if use_pre else 'none',postprocessing='none')
            if args.arm=='pre':
                prediction = post = inputs
            else:
                rgb = cv2.cvtColor(inputs,cv2.COLOR_BGR2RGB)
                if use_pre:
                    options={'box_feather':0} if final_blending else {}
                    value,details = restore(rgb,ocr['boxes'],model,'cuda',blend_source=cv2.cvtColor(original,cv2.COLOR_BGR2RGB),**options)
                else:
                    value,details = restore_block(rgb,model,'cuda')
                    if final_blending:details.update(preprocessing='none',postprocessing='none',final_box_feather=0)
                prediction = cv2.cvtColor(value,cv2.COLOR_RGB2BGR)
                post = prediction
                if use_pre:
                    context = json.loads((shared/'context.json').read_text())
                    assert context['reference_estimator']=='adaptive_scurve_robust_pre_v2_2'
                    post,_ = postprocess_bgr(prediction,context,gamma=1.2)
                    details.update(preprocessing='V2_2',postprocessing='gamma1.2')
            if final_blending and use_pre and args.arm!='pre':
                from tools.hx_upright_geometry import final_blend
                stage = OUT/'results'/args.arm/'stages'/identity/iid
                image_write(stage/'post_unblended.png',post)
                post,box_alpha=final_blend(original,post,ocr['boxes'])
                image_write(stage/'final_box_alpha.png',box_alpha)
                details['final_box_feather']=16
            valid = cv2.imread(str(shared/'valid.png'),0)
            delta,alpha,take,geometry = inverse_residual(source,original,post,item['warp']['image_to_output'],item['quad'],valid,owned)
            canvas+=delta;owned|=take
            stage = OUT/'results'/args.arm/'stages'/identity/iid
            for label,pixels in [('input',inputs),('model',prediction),('post',post)]:
                image_write(stage/(label+'.png'),pixels)
            stages.append(dict(instance_id=item['instance_id'],ocr_sha256=sha(shared/'ocr.json'),warp_sha256=sha(warp),restoration=details,geometry=geometry,processing_rotation_ccw=item.get('processing_rotation_ccw',0)))
        final = np.clip(np.rint(canvas),0,255).astype(np.uint8)
        assert np.array_equal(final[~owned],source[~owned])
        target = OUT/'results'/args.arm/'final'/(identity+'.png')
        image_write(target,final)
        save(receipt_path,dict(id=identity,binding=binding,source_sha256=sha(entry['source']),final=str(target.relative_to(OUT)),final_sha256=sha(target),source_hw=list(source.shape[:2]),outside_owned_unchanged=True,stages=stages,seconds=time.perf_counter()-started))
        print(json.dumps(dict(arm=args.arm,id=identity,seconds=round(time.perf_counter()-started,2))),flush=True)


def shared(args):
    """@luna-comment Input:OCR/prepare stage and disjoint partition. Output:shared release-compatible cache. Key Steps:load immutable released worker; filter records retaining global IDs; invoke accepted stage. Key Params:CPU2 per worker. State/Dependencies:separate Paddle/text environments; no release mutation."""
    import importlib.util
    specification = importlib.util.spec_from_file_location('hx_release_worker', BUNDLE/'worker.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    original = module.records
    module.records = lambda output: (row for index,row in enumerate(original(output)) if index%args.shards==args.rank)
    getattr(module,args.mode)(OUT/'run')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('pack','decode','infer','ocr','prepare'))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--arm',choices=('pre','original_v3_pre','vosr_v3_pre','vosr_v5_block'))
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    options = parser.parse_args()
    if options.mode in ('ocr','prepare'):
        shared(options)
    else:
        globals()[options.mode](options)
