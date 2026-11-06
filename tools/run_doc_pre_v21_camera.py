"""Released document warp, fresh OCR and latest preprocessing; no restoration model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_adaptive_pre_v21_camera_batch import sha, save_json
from tools.adaptive_scurve_robust_v2 import prepare_context, apply_preprocess, make_post_context
from tools.doc_warp_composition import source_support, inverse_geometry

INVENTORY = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016/inventory.json')
OCR_MODEL = Path('/home/guang.yang/workspace/PaddleOCR-3.7.0/deploy/official_models/PP-OCRv6_medium_det')
PYTHON_DOC = '/home/guang.yang/doc_yolo11_env/bin/python'
PYTHON_OCR = '/home/guang.yang/deployTfLite/bin/python'


def write_image(path, value):
    """@luna-comment
    Input: Destination and pixel array. Output: Checked image file.
    Key Steps: Create parent, encode, reject failures. Key Params: PNG compression3.
    State/Dependencies: Own output only; no in-place source mutation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_PNG_COMPRESSION, 3] if path.suffix == '.png' else [cv2.IMWRITE_JPEG_QUALITY, 90]
    if not cv2.imwrite(str(path), value, options):
        raise IOError(path)


def preview(path, value):
    """@luna-comment
    Input: Native BGR and preview path. Output: 1200px maximum-side JPEG.
    Key Steps: Aspect-preserving area resize and checked write; Key Params: no upscale.
    State/Dependencies: Preview only, never feeds processing.
    """
    scale = min(1., 1200 / max(value.shape[:2]))
    write_image(path, cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))


def detect_text(output, rank, shards):
    """@luna-comment
    Input: Completed warp jobs and disjoint worker rank. Output: Fresh OCR records.
    Key Steps: Verify local detector hash; detect actual warp; retain clipped boxes,
    polygons and source support; exclude only invalid/padding-only geometry.
    Key Params: PP-OCRv6_medium_det max2400, .3/.6, unclip1.5, CPU threads2.
    State/Dependencies: Existing Paddle environment; no download or old OCR reuse.
    """
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    from paddlex import create_model
    cv2.setNumThreads(1)
    hashes = {p.name: sha(p) for p in OCR_MODEL.iterdir() if p.is_file()}
    reference = INVENTORY.parent / 'detections/IMG_20260605_011836092.json'
    assert hashes == json.loads(reference.read_text())['model_hashes']
    engine = create_model(model_name='PP-OCRv6_medium_det', model_dir=str(OCR_MODEL),
                          device='cpu', enable_mkldnn=True, cpu_threads=2)
    jobs = json.loads((output / 'jobs.json').read_text())
    completed = 0
    for job in jobs[rank::shards]:
        started = time.perf_counter()
        source = output / job['warp']
        assert sha(source) == job['warp_sha256']
        image = cv2.imread(str(source))
        height, width = image.shape[:2]
        support = source_support(job['source_hw'], job['transform'], (height, width))
        inverse, error = inverse_geometry(job['transform'], job['quad'])
        predictions = list(engine.predict(input=image, batch_size=1, limit_side_len=2400,
                           limit_type='max', thresh=.3, box_thresh=.6, unclip_ratio=1.5))
        assert len(predictions) == 1
        raw = predictions[0].json
        raw = json.loads(raw) if isinstance(raw, str) else raw
        boxes, polygons, rejected = [], [], []
        overlay = image.copy()
        for polygon in raw.get('res', raw)['dt_polys']:
            points = np.asarray(polygon, np.float64).reshape(-1, 2)
            assert np.isfinite(points).all()
            x0, y0 = np.maximum(np.floor(points.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(points.max(0)).astype(int), [width, height])
            if x1 <= x0 or y1 <= y0 or not np.any(support[y0:y1, x0:x1] == 255):
                rejected.append(dict(polygon=points.tolist(), reason='empty_or_unknown_padding_only'))
                continue
            boxes.append(list(map(int, (x0, y0, x1, y1))))
            polygons.append(points.tolist())
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
        key = job['key']
        write_image(output / 'ocr' / (key + '.overlay.png'), overlay)
        write_image(output / 'ocr' / (key + '.valid.png'), support)
        preview(output / 'previews' / (key + '_ocr.jpg'), overlay)
        save_json(output / 'ocr' / (key + '.json'), dict(job, boxes=boxes, polygons=polygons,
                  rejected=rejected, model_hashes=hashes, fresh_warp_detection=True,
                  parameters=dict(max_side=2400, thresh=.3, box_thresh=.6, unclip_ratio=1.5),
                  inverse=inverse.tolist(), roundtrip_error=error, seconds=time.perf_counter()-started))
        completed += 1
        save_json(output / ('ocr_progress_' + str(rank) + '.json'), dict(completed=completed, last=key))
        print('OCR', key, len(boxes), flush=True)
    save_json(output / ('ocr_complete_' + str(rank) + '.json'), dict(completed=completed, status='passed'))


def prepare_instance(args):
    """@luna-comment
    Input: Output root and source-bound instance job. Output: Native pre/context/receipt.
    Key Steps: Bind fresh OCR, run unchanged V2.1, validate ownership/chroma and
    PNG roundtrip, persist source-to-warp provenance for future model/post use.
    Key Params: Foreground-core=True; one OpenCV thread; no additional quality gate.
    State/Dependencies: Immutable warp and latest preprocessing snapshot.
    """
    output, job = args
    output = Path(output)
    cv2.setNumThreads(1)
    key = job['key']
    record = output / 'ocr' / (key + '.json')
    ocr = json.loads(record.read_text())
    assert ocr['warp_sha256'] == job['warp_sha256'] == sha(output / job['warp'])
    image = cv2.imread(str(output / job['warp']))
    started = time.perf_counter()
    context = prepare_context(image, ocr['boxes'], reference_id=job['source_sha256'] + ':' + key + ':' + job['warp_sha256'])
    processed = apply_preprocess(image, context)
    core_ms = (time.perf_counter()-started)*1000
    assert processed.shape == image.shape and context['foreground_core']
    owned = np.zeros(image.shape[:2], bool)
    for x0, y0, x1, y1 in ocr['boxes']:
        owned[y0:y1, x0:x1] = True
    assert np.array_equal(processed[~owned], image[~owned])
    for y0 in range(0, image.shape[0], 128):
        delta = processed[y0:y0+128].astype(np.int16)-image[y0:y0+128].astype(np.int16)
        assert np.array_equal(delta[:, :, 0], delta[:, :, 1]) and np.array_equal(delta[:, :, 1], delta[:, :, 2])
    native = output / 'images' / (job['output_stem'] + '.png')
    reference = output / 'contexts' / (job['output_stem'] + '.context.json')
    write_image(native, processed)
    assert np.array_equal(cv2.imread(str(native)), processed)
    save_json(reference, make_post_context(context))
    preview(output / 'previews' / (key + '_warp.jpg'), image)
    preview(output / 'previews' / (key + '_pre.jpg'), processed)
    receipt = dict(job, image=str(native.relative_to(output)), context=str(reference.relative_to(output)),
                   image_sha256=sha(native), context_sha256=sha(reference), ocr_sha256=sha(record),
                   boxes=len(ocr['boxes']), shape=list(processed.shape), core_ms=core_ms,
                   version=context['version'], foreground_core=True, gamma=1.8,
                   outside_unchanged=True, equal_channel_delta=True, png_roundtrip_exact=True)
    save_json(output / 'receipts' / (key + '.json'), receipt)
    return receipt


def gallery(output, rows, jobs, records):
    """@luna-comment
    Input: Complete source/instance records. Output: 280 stage pages and link audit.
    Key Steps: Bind all output hashes, expose document/OCR/pre stages and no-doc cases;
    validate every generated local navigation/image target before publishing index.
    Key Params: Preview JPEGs with native links; no model-output claim.
    State/Dependencies: Own immutable stage files only.
    """
    style = '<meta charset="utf-8"><style>body{font:17px sans-serif;background:#181c22;color:#eee;margin:25px}a{color:#8cd8ff}img{max-width:100%}.pair{position:relative;max-width:1200px}.pair img{display:block;width:100%}.pair .after{position:absolute;top:0;left:0;clip-path:inset(0 50% 0 0)}</style>'
    all_links = []
    for r in records:
        assert sha(output / r['image']) == r['image_sha256']
        assert sha(output / r['context']) == r['context_sha256']
        assert sha(output / 'ocr' / (r['key'] + '.json')) == r['ocr_sha256']
    for i, row in enumerate(rows):
        own = [r for r in records if r['id'] == row['id']]
        record_path = 'documents/' + row['name'] + '/result.json'
        document = json.loads((output / record_path).read_text())
        assert len(own) == len(document['instances'])
        body = style + '<h1>' + html.escape(row['id'] + ' | ' + row['name']) + '</h1><a href="../index.html">All images</a>'
        for label, j in [('Previous', i-1), ('Next', i+1)]:
            if 0 <= j < len(rows):
                body += ' | <a href="' + rows[j]['id'] + '.html">' + label + '</a>'
        body += '<p>Released doc_quad_v1.0.0 -> fresh warped OCR -> latest V2.1 preprocessing. No TextSR model/postprocessing.</p>'
        body += '<p>Documents: ' + str(len(own)) + '; rejected geometries: ' + str(len(document.get('rejected', []))) + '</p>'
        body += '<a href="../' + record_path + '">Document geometry JSON</a><h2>Document detection</h2><a href="../documents/' + row['name'] + '/overlay.png"><img src="../previews/' + row['id'] + '_document.jpg"></a>'
        body += '<p><a href="../source/' + row['name'] + '">Original image</a></p>'
        if not own:
            body += '<p>No valid document. Original retained; no OCR/preprocessing output fabricated.</p>'
        for r in own:
            key = r['key']
            body += '<h2>Document ' + str(r['instance']) + ' | ' + r['geometry_type'] + ' | ' + str(r['boxes']) + ' OCR boxes</h2>'
            body += '<p><a href="../' + r['warp'] + '">Native warp</a> | <a href="../ocr/' + key + '.overlay.png">Native OCR overlay</a> | <a href="../' + r['image'] + '">Native preprocessed PNG</a> | <a href="../' + r['context'] + '">Post context</a></p>'
            body += '<img src="../previews/' + key + '_ocr.jpg"><p>Slider: left = V2.1 preprocessed; right = original warp.</p>'
            body += '<input type="range" min="0" max="100" value="50" style="width:90%" oninput="document.getElementById(\'' + key + '\').style.clipPath=\'inset(0 \'+(100-this.value)+\'% 0 0)\'">'
            body += '<div class="pair"><img src="../previews/' + key + '_warp.jpg"><img class="after" id="' + key + '" src="../previews/' + key + '_pre.jpg"></div>'
        page = output / 'pages' / (row['id'] + '.html')
        page.write_text(body, encoding='utf-8')
        all_links.append(page)
    index = style + '<h1>Document-first camera V2.1 preprocessing</h1><p>280 original frames; ' + str(len(jobs)) + ' document instances. Native model inputs: images/. Matching post references: contexts/.</p><ul>'
    index += ''.join('<li><a href="pages/' + r['id'] + '.html">' + html.escape(r['id'] + ' ' + r['name']) + '</a></li>' for r in rows) + '</ul>'
    (output / 'index.html').write_text(index, encoding='utf-8')
    all_links.append(output / 'index.html')
    import re
    count = 0
    for page in all_links:
        for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
            assert (page.parent / html.unescape(link)).is_file(), (page, link)
            count += 1
    save_json(output / 'gallery_audit.json', dict(status='passed', pages=len(all_links), links=count))


def run(output, shards):
    """@luna-comment
    Input: Fresh output root and OCR parallelism. Output: Complete isolated pipeline.
    Key Steps: Hash-bound copies; unchanged detector subprocess; geometry jobs;
    fresh OCR subprocesses; parallel V2.1 export; gallery and final coverage audit.
    Key Params: Frozen detector640/.25/.8, allocation8192/33554432, sixteen OCR workers.
    State/Dependencies: Known remote environments; own children/logs; no training or model SR.
    """
    output.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'logs', 'ocr', 'images', 'contexts', 'receipts', 'previews', 'pages'):
        (output / name).mkdir()
    started = time.perf_counter()
    try:
        rows = json.loads(INVENTORY.read_text())['images']
        assert len(rows) == len({r['id'] for r in rows}) == 280
        release = ROOT / 'release'
        assert sha(release / 'best.pt') == '1921fbad80063489fed5b87428a4c28ce6132482feba3c5fd8a0d5b9caa36c77'
        assert sha(release / 'doc_quad_infer.py') == '2106c82547263bb2faa85cd82865c7207063a8784235837869c4f292af9f2d87'
        save_json(output / 'source_manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py', '.pt')})
        save_json(output / 'inventory.json', dict(images=rows, upstream_sha256=sha(INVENTORY)))
        save_json(output / 'status.json', dict(phase='source_validation', frames=280))
        for row in rows:
            assert sha(row['remote']) == row['sha256']
            shutil.copyfile(row['remote'], output / 'source' / row['name'])
            assert sha(output / 'source' / row['name']) == row['sha256']
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        save_json(output / 'status.json', dict(phase='document_detection', frames=280))
        command = [PYTHON_DOC, str(release / 'doc_quad_infer.py'), '--weights', str(release / 'best.pt'),
                   '--source', str(output / 'source'), '--output', str(output / 'documents'), '--device', '0',
                   '--max-side', '8192', '--max-pixels', '33554432']
        with (output / 'logs/documents.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        detector = json.loads((output / 'documents/run.json').read_text())
        assert detector['status'] == 'completed' and not detector['errors'] and len(detector['results']) == 280
        jobs, no_document = [], []
        for row in rows:
            directory = output / 'documents' / row['name']
            record = json.loads((directory / 'result.json').read_text())
            source = cv2.imread(str(output / 'source' / row['name']))
            assert list(source.shape[:2]) == [record['image_height'], record['image_width']]
            preview(output / 'previews' / (row['id'] + '_document.jpg'), cv2.imread(str(directory / 'overlay.png')))
            if not record['instances']:
                no_document.append(row['id'])
            for instance in record['instances']:
                key = row['id'] + '_doc' + str(instance['instance_id']).zfill(3)
                warp = directory / instance['image_file']
                inverse, error = inverse_geometry(instance['warp']['image_to_output'], instance['quad'])
                jobs.append(dict(id=row['id'], key=key, instance=instance['instance_id'], source_sha256=row['sha256'],
                                 source_hw=list(source.shape[:2]), warp=str(warp.relative_to(output)), warp_sha256=sha(warp),
                                 quad=instance['quad'], transform=instance['warp']['image_to_output'],
                                 geometry_type=instance['geometry_type'], output_stem=Path(row['name']).stem+'__doc'+str(instance['instance_id']).zfill(3)))
        assert len({j['key'] for j in jobs}) == len(jobs)
        save_json(output / 'jobs.json', jobs)
        document_seconds = time.perf_counter()-started
        save_json(output / 'status.json', dict(phase='fresh_warped_ocr', frames=280, instances=len(jobs), no_document=no_document))
        children = []
        for rank in range(shards):
            with (output / 'logs' / ('ocr_' + str(rank) + '.log')).open('w') as log:
                children.append(subprocess.Popen([PYTHON_OCR, str(Path(__file__).resolve()), 'ocr', '--output', str(output),
                    '--rank', str(rank), '--shards', str(shards)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env))
        codes = [child.wait() for child in children]
        save_json(output / 'ocr_exit_codes.json', codes)
        assert not any(codes), codes
        assert len(list((output / 'ocr').glob('*.json'))) == len(jobs)
        ocr_seconds = time.perf_counter()-started-document_seconds
        save_json(output / 'status.json', dict(phase='preprocessing', instances=len(jobs)))
        records = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for r in pool.map(prepare_instance, [(str(output), j) for j in jobs]):
                records.append(r)
                save_json(output / 'status.json', dict(phase='preprocessing', completed=len(records), expected=len(jobs)))
                print('PRE', len(records), '/', len(jobs), r['key'], flush=True)
        assert len(records) == len(jobs)
        gallery(output, rows, jobs, records)
        no_text = [r['key'] for r in records if r['boxes'] == 0]
        summary = dict(status='passed', frames=280, instances=len(jobs), no_document=no_document, no_text=no_text,
                       ocr_boxes=sum(r['boxes'] for r in records), document_seconds=document_seconds, ocr_seconds=ocr_seconds,
                       wall_seconds=time.perf_counter()-started, version='adaptive_scurve_robust_pre_v2_1',
                       detector_release='doc_quad_v1.0.0', model_inference=False, records=records)
        save_json(output / 'summary.json', summary)
        save_json(output / 'status.json', {k:v for k,v in summary.items() if k != 'records'})
        print('PASS', 280, 'frames', len(jobs), 'instances', flush=True)
    except Exception as error:
        save_json(output / 'status.json', dict(phase='failed', error=repr(error)))
        raise


def main():
    """@luna-comment
    Input: run/ocr mode and explicit output. Output: Selected stage artifacts.
    Key Steps: Parse, limit OpenCV threading, dispatch; Key Params: 16 OCR shards default.
    State/Dependencies: Run controller is separate from CPU OCR child environments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'ocr'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--shards', type=int, default=16)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.mode == 'ocr':
        detect_text(args.output, args.rank, args.shards)
    else:
        run(args.output, args.shards)


if __name__ == '__main__':
    main()
