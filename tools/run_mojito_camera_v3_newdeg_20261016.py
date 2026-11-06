"""Detect native camera text regions and restore all280 images with original V3."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016')
CHECKPOINT = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/v3_native_four_20261015/runs/03_displayfit_degraded/models/net_g_250000.pth')


def save_json(path, value):
    """@luna-comment Input: output path/object. Output: atomic UTF8 receipt.
    Key Steps: temporary sibling and replace. Key Params: finite JSON only.
    State/Dependencies: isolated evaluation output tree.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))
    temporary.replace(path)


def wait_file(path, size=None):
    """@luna-comment Input: staged asset path/optional expected byte size. Output: available file.
    Key Steps: bounded2hour wait for upstream stream/detection. Key Params:2second polls.
    State/Dependencies: filesystem only; missing prerequisites fail explicitly.
    """
    deadline = time.monotonic()+7200
    while not path.exists() or (size is not None and path.stat().st_size != size):
        if time.monotonic() > deadline:raise TimeoutError(path)
        time.sleep(2)


def detect(limit, rank=0, shards=1):
    """@luna-comment
    Input:280 original JPEG inventory/frozen PP-OCRv6_medium_det.
    Output: one hashed original-coordinate detection receipt per image.
    Key Steps: await exact staging size, verify hash, CPU detect, clamp native boxes.
    Key Params: max2400,DB.3/box.6/unclip1.5; no source resize for restoration.
    State/Dependencies: Paddle CPU environment and explicit local model, no downloads.
    """
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    import cv2
    import numpy as np
    import hashlib
    from paddlex import create_model
    cv2.setNumThreads(1)
    model_dir = Path('/home/guang.yang/workspace/PaddleOCR-3.7.0/deploy/official_models/PP-OCRv6_medium_det')
    model_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.iterdir() if p.is_file()}
    engine = create_model(model_name='PP-OCRv6_medium_det', model_dir=str(model_dir), device='cpu',
                          enable_mkldnn=True, cpu_threads=4)
    rows = json.loads((OUT/'inventory.json').read_text())['images']
    for n, row in enumerate(rows[:limit], 1):
        if (n-1)%shards != rank:continue
        path = Path(row['remote'])
        wait_file(path, row['bytes'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        output = OUT/'detections'/(path.stem+'.json')
        if output.exists():
            cached = json.loads(output.read_text())
            assert cached['source_sha256'] == row['sha256']
            continue
        started = time.perf_counter()
        # Match restoration's OpenCV decoding/orientation exactly.
        image = cv2.imread(str(path))
        assert image is not None
        h, w = image.shape[:2]
        result = list(engine.predict(input=image, batch_size=1, limit_side_len=2400,
            limit_type='max', thresh=.3, box_thresh=.6, unclip_ratio=1.5))
        assert len(result) == 1
        raw = result[0].json
        if isinstance(raw, str):raw = json.loads(raw)
        data = raw.get('res', raw)
        boxes, polygons = [], []
        for poly in data['dt_polys']:
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            assert np.isfinite(pts).all()
            x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int), [w, h])
            if x1 > x0 and y1 > y0:
                boxes.append(list(map(int, (x0, y0, x1, y1))))
                polygons.append(pts.tolist())
        save_json(output, dict(source=str(path), source_sha256=row['sha256'], source_hw=[h,w], boxes=boxes,
            polygons=polygons, model='PP-OCRv6_medium_det', model_hashes=model_hashes,
            parameters=dict(max_side=2400,thresh=.3,box_thresh=.6,unclip_ratio=1.5), seconds=time.perf_counter()-started))
        save_json(OUT/f'detection_progress_{rank}.json', dict(last_index=n,expected=len(rows),last=path.name))
        print('DETECTED', n, path.name, len(boxes), flush=True)


def infer(limit, rank=0, shards=1):
    """@luna-comment
    Input: exact original250K V3 and detector receipts. Output:280 native PNGs/previews.
    Key Steps: strict checkpoint, same shared96x480 restore as HX, byte-exact outside
    boxes, finite output/geometry checks, hash receipts and counts.
    Key Params: strict_combined_v1 before model; batch8/stride64x448;16px feather to original RGB.
    State/Dependencies: torch/CUDA, original staged source, independent output folders.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore, sha
    sys.path.insert(0,str(OUT/'preprocessing'))
    from contrast_curve import prepare_variant,render
    frozen = json.loads((OUT/'preprocessing/freeze_manifest.json').read_text())
    frozen_hashes = {}
    for name in ('contrast_curve.py','edge_curve.py','mode_curve.py','block_curve.py','simple_curve.py'):
        frozen_hashes[name] = sha(OUT/'preprocessing'/name)
        assert frozen_hashes[name] == frozen['files'][name]['sha256']
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert sha(CHECKPOINT) == '115fdaf386447c03d61681710149620555e178b8921b5f0bd2892f81e25a6937'
    config = CHECKPOINT.parents[3]/'configs/03_displayfit_degraded.yml'
    settings = yaml.safe_load(config.read_text())
    net = build_network(settings['network_g']).cuda().eval()
    net.load_state_dict(torch.load(CHECKPOINT,map_location='cpu')['params'],strict=True)
    rows = json.loads((OUT/'inventory.json').read_text())['images']
    for folder in ('final','preprocessed','previews','receipts'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    results = []
    for n, row in enumerate(rows[:limit], 1):
        if (n-1)%shards != rank:continue
        path = Path(row['remote']); receipt = OUT/'receipts'/(path.stem+'.json')
        if receipt.exists() and (OUT/'preprocessed'/(path.stem+'.png')).exists():
            cached = json.loads(receipt.read_text());assert cached['source_sha256'] == row['sha256']
            assert cached['preprocessing']['snapshot']=='strict_combined_v1'
            results.append(cached);continue
        detection = OUT/'detections'/(path.stem+'.json')
        wait_file(detection)
        meta = json.loads(detection.read_text())
        assert meta['source_sha256'] == row['sha256'] and sha(path) == row['sha256']
        started = time.perf_counter()
        bgr = cv2.imread(str(path));assert bgr is not None
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        assert list(rgb.shape[:2]) == meta['source_hw']
        preprocessed_bgr = bgr.copy()
        source_y = cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)[:,:,0]
        owner = np.full(source_y.shape,-1,np.int16)
        ledger=[];preprocess_lines=[]
        assert len(meta['boxes']) < 32767
        for line_index,bbox in enumerate(meta['boxes']):
            x0,y0,x1,y1=bbox
            free=owner[y0:y1,x0:x1]<0
            luts,xc,yc,cells=prepare_variant(source_y,bbox,owner,ledger,'combined')
            corrected=render(bgr[y0:y1,x0:x1],source_y,bbox,luts,xc,yc)
            ledger.append(dict(bbox=bbox,luts=luts,xc=xc,yc=yc,cells=cells))
            preprocessed_bgr[y0:y1,x0:x1][free]=corrected[free]
            owner[y0:y1,x0:x1][free]=line_index
            preprocess_lines.append(dict(bbox=bbox,written_pixels=int(free.sum()),cells=cells))
        assert np.array_equal(preprocessed_bgr[owner<0],bgr[owner<0])
        preprocessed_rgb=cv2.cvtColor(preprocessed_bgr,cv2.COLOR_BGR2RGB)
        output, details = restore(preprocessed_rgb,meta['boxes'],net,'cuda',blend_source=rgb)
        assert output.shape == rgb.shape
        if not meta['boxes']:assert np.array_equal(output,rgb)
        final = OUT/'final'/(path.stem+'.png')
        assert cv2.imwrite(str(final),cv2.cvtColor(output,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
        assert cv2.imwrite(str(OUT/'preprocessed'/(path.stem+'.png')),preprocessed_bgr,[cv2.IMWRITE_PNG_COMPRESSION,3])
        factor = min(1,960/max(rgb.shape[:2]));size=(round(rgb.shape[1]*factor),round(rgb.shape[0]*factor))
        for name, pixels in [('source',rgb),('preprocessed',preprocessed_rgb),('result',output)]:
            preview = cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(OUT/'previews'/(row['id']+'_'+name+'.jpg')),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
        item = dict(id=row['id'],name=path.name,source=str(path),source_sha256=row['sha256'],
            output=str(final),output_sha256=sha(final),width=rgb.shape[1],height=rgb.shape[0],
            detection_sha256=sha(detection),text_boxes=meta['boxes'],seconds=time.perf_counter()-started,**details)
        import hashlib
        item['preprocessing']=dict(snapshot='strict_combined_v1',variant='combined',module_hashes=frozen_hashes,
            changed_pixels=int(np.any(preprocessed_bgr!=bgr,axis=2).sum()),outside_boxes_unchanged=True,
            source_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            model_input_pixel_sha256=hashlib.sha256(preprocessed_rgb.tobytes()).hexdigest(),lines=preprocess_lines)
        save_json(receipt,item);results.append(item)
        save_json(OUT/f'inference_progress_{rank}.json',dict(completed=len(results),expected=len(rows[:limit][rank::shards]),last=path.name))
        print('RESTORED',len(results),path.name,len(meta['boxes']),round(item['seconds'],2),flush=True)
    expected=len(rows[:limit][rank::shards])
    save_json(OUT/f'summary_infer_{rank}.json',dict(status='passed',images=len(results),expected=expected,
        complete=len(results)==expected,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),results=results))


def main():
    """@luna-comment Input: detect/infer and optional pilot limit. Output: chosen stage.
    Key Steps: parse CLI and dispatch. Key Params: limit defaults280.
    State/Dependencies: explicitly separate Paddle/Torch processes and environments.
    """
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['detect','infer']);parser.add_argument('--limit',type=int,default=280)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();assert 0<=args.rank<args.shards
    (detect if args.mode=='detect' else infer)(args.limit,args.rank,args.shards)


if __name__=='__main__':main()
