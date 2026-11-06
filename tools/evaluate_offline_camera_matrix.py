"""Evaluate final native V3/V5 checkpoints with frozen camera pre/postprocessing."""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ('01_v5_real', '02_v3_real', '03_v5_synthetic', '04_v3_synthetic')


def sha(path):
    """@luna-comment
    Input: asset path. Output: streamed SHA256. Key Steps: bounded file reads.
    Key Params:4MiB chunks. State/Dependencies: immutable source filesystem.
    """
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    """@luna-comment
    Input: output path/object. Output: atomic JSON. Key Steps: sibling temp replacement.
    Key Params: finite values only. State/Dependencies: isolated output tree.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def verify_frozen():
    """@luna-comment
    Input: staged frozen package. Output: manifest hash after all-file validation.
    Key Steps: hash every frozen file. Key Params: current gamma1.8/default post1.2.
    State/Dependencies: frozen package never modified.
    """
    root = ROOT / 'evaluation/adaptive_prepost_frozen_v1_20261020'
    manifest = json.loads((root / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen integrity failure: ' + relative)
    return sha(root / 'manifest.json')


def image_write(path, image):
    """@luna-comment
    Input: image path/BGR8 pixels. Output: checked image file.
    Key Steps: create directory and encode. Key Params:PNG compression3.
    State/Dependencies: OpenCV and own output files only.
    """
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3] if path.suffix == '.png' else []):
        raise IOError('Image write failed: ' + str(path))


def preview(path, image):
    """@luna-comment
    Input: destination and BGR frame. Output: bounded display JPEG.
    Key Steps: proportional area resize. Key Params: longest edge1200.
    State/Dependencies: OpenCV; native full-resolution images retained separately.
    """
    import cv2
    factor = min(1., 1200 / max(image.shape[:2]))
    image_write(path, cv2.resize(image, (round(image.shape[1]*factor), round(image.shape[0]*factor)), interpolation=cv2.INTER_AREA))


def prepare_one(job):
    """@luna-comment
    Input: source row, exact detection receipt, output root and frozen hash.
    Output: verified shared preprocessed PNG/reference context and receipt.
    Key Steps: hash source/detection, clip-check boxes, freeze once, preserve outside union.
    Key Params: same original geometry and gamma1.8. State/Dependencies: CPU/frozen API.
    """
    import cv2
    import numpy as np
    from evaluation.adaptive_prepost_frozen_v1_20261020.preprocess import prepare_context, apply_preprocess, make_post_context
    row, detections, output, frozen_hash = job
    output = Path(output)
    receipt_path = output / 'prepared' / (row['id'] + '.receipt.json')
    source = Path(row['remote'])
    detection = Path(detections) / (source.stem + '.json')
    cv2.setNumThreads(1)
    if sha(source) != row['sha256']:
        raise ValueError('Original source differs from verified inventory')
    meta = json.loads(detection.read_text())
    if meta['source_sha256'] != row['sha256']:
        raise ValueError('OCR belongs to another image')
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved['source_sha256'] != row['sha256'] or saved['frozen_manifest_sha256'] != frozen_hash or saved['detection_sha256'] != sha(detection):
            raise ValueError('Existing preparation has different provenance')
        for key in ('preprocessed', 'context'):
            if sha(output / saved[key]) != saved[key + '_sha256']:
                raise ValueError('Prepared artifact changed')
        return saved
    started = time.perf_counter()
    bgr = cv2.imread(str(source))
    if bgr is None or list(bgr.shape[:2]) != meta['source_hw']:
        raise ValueError('Original decode/shape mismatch')
    context = prepare_context(bgr, meta['boxes'], reference_id=row['sha256'])
    processed = apply_preprocess(bgr, context)
    union = np.zeros(bgr.shape[:2], bool)
    for x0, y0, x1, y1 in meta['boxes']:
        union[y0:y1, x0:x1] = True
    if not np.array_equal(processed[~union], bgr[~union]):
        raise ValueError('Preprocessing modified outside OCR union')
    relative = 'prepared/' + row['id']
    image_write(output / (relative + '.png'), processed)
    save_json(output / (relative + '.context.json'), make_post_context(context))
    preview(output / 'previews' / (row['id'] + '_source.jpg'), bgr)
    preview(output / 'previews' / (row['id'] + '_preprocessed.jpg'), processed)
    receipt = dict(id=row['id'], name=row['name'], source=str(source), source_sha256=row['sha256'],
                   source_hw=list(bgr.shape[:2]), boxes=meta['boxes'], detection_sha256=sha(detection),
                   frozen_manifest_sha256=frozen_hash, preprocess_gamma=1.8, outside_unchanged=True,
                   preprocessed=relative + '.png', context=relative + '.context.json', seconds=time.perf_counter()-started)
    for key in ('preprocessed', 'context'):
        receipt[key + '_sha256'] = sha(output / receipt[key])
    save_json(receipt_path, receipt)
    return receipt


def evaluate(args, rows, frozen_hash):
    """@luna-comment
    Input: one arm, validated shared preparation and exact final training run.
    Output: native model/final PNGs, previews and hashed per-image receipts.
    Key Steps: require clean250K completion, strict load, native restore, frozen post.
    Key Params:96x480 native grid, batch8/stride64x448, post gamma1.2, no height resize.
    State/Dependencies: one CUDA card, read-only weights/sources, frozen context.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap(args.repo_root)
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore
    from evaluation.adaptive_prepost_frozen_v1_20261020.postprocess import postprocess_bgr
    output, training = Path(args.output), Path(args.training_root)
    config = training / 'configs' / (args.arm + '.yml')
    settings = yaml.safe_load(config.read_text())
    checkpoint = training / 'experiments' / settings['name'] / 'models/net_g_250000.pth'
    log = (training / 'logs' / (args.arm + '.log')).read_text()
    if 'End of training' not in log or not checkpoint.exists():
        raise ValueError('Final250K training completion is not verified')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = build_network(settings['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['params'], strict=True)
    checkpoint_hash, config_hash = sha(checkpoint), sha(config)
    results = []
    for row in rows:
        started = time.perf_counter()
        prepared = json.loads((output / 'prepared' / (row['id'] + '.receipt.json')).read_text())
        if prepared['frozen_manifest_sha256'] != frozen_hash or prepared['source_sha256'] != row['sha256']:
            raise ValueError('Wrong frame/frozen context')
        receipt_path = output / args.arm / 'receipts' / (row['id'] + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['checkpoint_sha256'] != checkpoint_hash or receipt['source_sha256'] != row['sha256'] or receipt['frozen_manifest_sha256'] != frozen_hash:
                raise ValueError('Existing evaluation belongs to another run')
            if sha(output / receipt['final']) != receipt['final_sha256']:
                raise ValueError('Existing final image changed')
            results.append(receipt)
            continue
        source = cv2.imread(row['remote'])
        preprocessed = cv2.imread(str(output / prepared['preprocessed']))
        if source is None or preprocessed is None or source.shape != preprocessed.shape:
            raise ValueError('Missing or misaligned frame')
        if sha(output / prepared['context']) != prepared['context_sha256'] or sha(output / prepared['preprocessed']) != prepared['preprocessed_sha256']:
            raise ValueError('Prepared artifact changed')
        rgb, details = restore(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), prepared['boxes'], model, 'cuda',
                               blend_source=cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        model_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        context = json.loads((output / prepared['context']).read_text())
        final, reports = postprocess_bgr(model_bgr, context, gamma=1.2)
        union = np.zeros(source.shape[:2], bool)
        for x0, y0, x1, y1 in prepared['boxes']:
            union[y0:y1, x0:x1] = True
        if final.shape != source.shape or not np.array_equal(final[~union], source[~union]):
            raise ValueError('Final geometry/outside-union preservation failed')
        model_path, final_path = (f'{args.arm}/{stage}/{row["id"]}.png' for stage in ('model', 'final'))
        image_write(output / model_path, model_bgr)
        image_write(output / final_path, final)
        preview(output / args.arm / 'previews' / (row['id'] + '.jpg'), final)
        receipt = dict(id=row['id'], name=row['name'], source_sha256=row['sha256'], source_hw=list(source.shape[:2]),
                       arm=args.arm, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, config_sha256=config_hash,
                       frozen_manifest_sha256=frozen_hash, preprocess_gamma=1.8, postprocess_gamma=1.2,
                       post_lines=len(reports), outside_unchanged=True, model=model_path, final=final_path,
                       final_sha256=sha(output / final_path), seconds=time.perf_counter()-started, geometry=details)
        save_json(receipt_path, receipt)
        results.append(receipt)
        suffix = '' if args.partition_rank is None else f'.rank{args.partition_rank}'
        save_json(output / args.arm / ('progress' + suffix + '.json'), dict(done=len(results), expected=len(rows), last=row['id']))
        print(json.dumps(dict(arm=args.arm, done=len(results), expected=len(rows), seconds=receipt['seconds'])), flush=True)
    suffix = '' if args.partition_rank is None else f'.rank{args.partition_rank}'
    save_json(output / args.arm / ('COMPLETE' + suffix + '.json'), dict(complete=True, images=len(results), checkpoint_sha256=checkpoint_hash, config_sha256=config_hash, frozen_manifest_sha256=frozen_hash))


def parallel_infer(args, rows):
    """@luna-comment
    Input: one arm and full exact source list. Output: three disjoint GPU-worker results.
    Key Steps: spawn source-index partitions on inherited GPU, aggregate progress,
    require all exits/counts/hashes before publishing arm completion.
    Key Params: three workers per GPU; independent progress and image outputs.
    State/Dependencies: own child processes only; no existing jobs stopped.
    """
    output = Path(args.output) / args.arm
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rank in range(3):
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--partition-rank', str(rank)]
        jobs.append(subprocess.Popen(command, env=env))
    while any(job.poll() is None for job in jobs):
        done = 0
        for rank in range(3):
            progress = output / f'progress.rank{rank}.json'
            if progress.exists():
                done += json.loads(progress.read_text())['done']
        save_json(output / 'progress.json', dict(done=done, expected=len(rows), workers=3))
        time.sleep(3)
    if any(job.returncode for job in jobs):
        raise RuntimeError('One or more partition workers failed')
    receipts = [json.loads((output / f'COMPLETE.rank{rank}.json').read_text()) for rank in range(3)]
    if sum(r['images'] for r in receipts) != len(rows) or len({r['checkpoint_sha256'] for r in receipts}) != 1:
        raise ValueError('Worker source count/checkpoint mismatch')
    save_json(output / 'progress.json', dict(done=len(rows), expected=len(rows), workers=3))
    save_json(output / 'COMPLETE.json', dict(receipts[0], images=len(rows), workers=3))


def gallery(output, rows):
    """@luna-comment
    Input: completed four-arm receipts and source inventory. Output: linked HTML gallery.
    Key Steps: verify hashes/outputs, copy originals, embed per-frame JPEG previews
    and previous/next/select navigation for mapped-drive-compatible comparison.
    Key Params: embedded800px previews; native originals/model/final outputs retained.
    State/Dependencies: output-only OpenCV/JPEG encoding; no web service or asset fetch.
    """
    output = Path(output)
    import cv2
    names = ['Original', 'Preprocessed', 'V5 real', 'V3 real', 'V5 synthetic', 'V3 synthetic']
    (output / 'pages').mkdir(exist_ok=True)
    for row_index, row in enumerate(rows):
        relative_source = f'source/{row["id"]}{Path(row["remote"]).suffix.lower()}'
        source_copy = output / relative_source
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        if not source_copy.exists():
            shutil.copyfile(row['remote'], source_copy)
        if sha(source_copy) != row['sha256']:
            raise ValueError('Portable original copy differs from verified source')
        cells = [(f'previews/{row["id"]}_source.jpg', relative_source),
                 (f'previews/{row["id"]}_preprocessed.jpg', f'prepared/{row["id"]}.png')]
        for arm in ARMS:
            receipt = json.loads((output / arm / 'receipts' / (row['id'] + '.json')).read_text())
            if receipt['source_sha256'] != row['sha256'] or not (output / receipt['final']).exists():
                raise ValueError('Incomplete/mismatched comparison')
            cells.append((f'{arm}/previews/{row["id"]}.jpg', receipt['final']))
        figures = []
        for name, (image, link) in zip(names, cells):
            pixels = cv2.imread(str(output / image))
            if pixels is None:
                raise ValueError('Missing comparison preview')
            factor = min(1., 800/max(pixels.shape[:2]))
            pixels = cv2.resize(pixels, (round(pixels.shape[1]*factor),round(pixels.shape[0]*factor)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise IOError('Preview embedding failed')
            data = base64.b64encode(encoded).decode('ascii')
            figures.append(f'<figure><figcaption>{name}</figcaption><a href="../{html.escape(link,quote=True)}" target="_blank"><img src="data:image/jpeg;base64,{data}"></a></figure>')
        options = ''.join(f'<option value="{r["id"]}.html"'+(' selected' if r['id']==row['id'] else '')+'>'+html.escape(r['id']+' · '+r['name'])+'</option>' for r in rows)
        previous = rows[max(0,row_index-1)]['id']+'.html'
        following = rows[min(len(rows)-1,row_index+1)]['id']+'.html'
        navigation = f'<nav><a href="{previous}">← Previous</a><select onchange="location.href=this.value">{options}</select><a href="{following}">Next →</a><span>{row_index+1} / {len(rows)}</span></nav>'
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Final camera comparison</title><style>body{font:14px Segoe UI,sans-serif;background:#f1f5f9;color:#183247;margin:24px}h1{font-size:28px}section{background:white;border-radius:12px;padding:18px;margin:20px 0}h2{font-size:16px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}figcaption{margin-bottom:8px;font-weight:600}img{width:100%;height:340px;object-fit:contain;background:#e8edf1}nav{display:flex;align-items:center;gap:20px;flex-wrap:wrap}nav a,select{padding:10px;border:1px solid #b8ccd8;border-radius:6px;color:#23556c;background:white}a{text-decoration:none}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><h1>Final V3/V5 camera comparison</h1><p>All four final250K checkpoints · Native96×480 inference · Frozen preprocessing gamma1.8 / postprocessing gamma1.2. Click previews for full resolution. No paired GT: visual comparison only.</p>' + navigation + '<section><h2>'+html.escape(row['id']+' · '+row['name'])+'</h2><div class="grid">'+''.join(figures)+'</div></section></html>'
        (output / 'pages' / (row['id']+'.html')).write_text(page, encoding='utf-8')
        if row_index == 0:
            # Root entry has its own base to preserve links into the pages directory.
            root_page = page.replace('<meta charset="utf-8">', '<meta charset="utf-8"><base href="pages/">', 1)
            (output / 'index.html').write_text(root_page, encoding='utf-8')
        if (row_index+1) % 20 == 0:
            print('GALLERY',row_index+1,len(rows),flush=True)
    save_json(output / 'COMPLETE.json', dict(complete=True, images=len(rows), arms=list(ARMS), outputs=len(rows)*4))


def main():
    """@luna-comment
    Input: explicit approved inventory/detection/training/output roots and mode.
    Output: shared preparation, one model arm or completed gallery.
    Key Steps: verify frozen package, select exact rows, dispatch requested stage.
    Key Params: limit only for pilot; workers4 CPU preparation. State/Dependencies: isolated run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'infer', 'gallery'))
    for key in ('inventory', 'detections', 'training-root', 'output', 'repo-root'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--partition-rank', type=int, choices=range(3))
    args = parser.parse_args()
    rows = json.loads(Path(args.inventory).read_text())['images'][:args.limit or None]
    frozen_hash = verify_frozen()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        jobs = [(row, args.detections, args.output, frozen_hash) for row in rows]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for number, receipt in enumerate(pool.map(prepare_one, jobs), 1):
                print(json.dumps(dict(prepared=number, expected=len(rows), id=receipt['id'])), flush=True)
        save_json(Path(args.output) / 'PREPARED.json', dict(images=len(rows), frozen_manifest_sha256=frozen_hash))
    elif args.mode == 'infer':
        if not args.arm:
            parser.error('--arm required for inference')
        if args.partition_rank is None:
            parallel_infer(args, rows)
        else:
            evaluate(args, rows[args.partition_rank::3], frozen_hash)
    else:
        gallery(args.output, rows)


if __name__ == '__main__':
    main()
