"""Versioned blending export reusing verified document/OCR/V2.1 artifacts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.preprocess_boundary_blend import feather_preprocess
from tools.run_adaptive_pre_v21_camera_batch import sha, save_json
from tools.run_doc_pre_v21_camera import write_image, preview, gallery


def blend_one(args):
    """@luna-comment
    Input: Base/output roots and verified V2.1 receipt. Output: Blended native image/receipt.
    Key Steps: Hash-bind inputs; blend once; verify alpha endpoints, common delta and
    exact PNG roundtrip; retain original reference context. Key Params: default12percent.
    State/Dependencies: Source and prior outputs immutable; disjoint destination files.
    """
    base, output, record = args
    base, output = Path(base), Path(output)
    cv2.setNumThreads(1)
    assert sha(base/record['image']) == record['image_sha256']
    assert sha(base/record['warp']) == record['warp_sha256']
    ocr = base/'ocr'/(record['key']+'.json')
    assert sha(ocr) == record['ocr_sha256']
    source, previous = cv2.imread(str(base/record['warp'])), cv2.imread(str(base/record['image']))
    boxes = json.loads(ocr.read_text())['boxes']
    tick = time.perf_counter()
    blended, alpha = feather_preprocess(source, previous, boxes)
    blend_ms = (time.perf_counter()-tick)*1000
    assert np.array_equal(blended[alpha == 0], source[alpha == 0])
    assert np.array_equal(blended[alpha == 255], previous[alpha == 255])
    native = output/record['image']
    write_image(native, blended)
    assert np.array_equal(cv2.imread(str(native)), blended)
    preview(output/'previews'/(record['key']+'_pre.jpg'), blended)
    preview(output/'before'/(record['key']+'.jpg'), previous)
    receipt = dict(record, image_sha256=sha(native), previous_image_sha256=record['image_sha256'],
                   blend_ms=blend_ms, boundary_blend=dict(fraction=.12,max_width=24,curve='smoothstep',overlap='max'),
                   boundary_source_exact=True, interior_pre_exact=True,
                   changed_by_blend=int(np.count_nonzero(np.any(blended != previous, axis=2))))
    save_json(output/'receipts'/(record['key']+'.json'), receipt)
    return receipt


def main():
    """@luna-comment
    Input: Explicit immutable batch and new output root; optional three-image pilot.
    Output: New blended PNGs/gallery with unchanged source geometry and reference contexts.
    Key Steps: Link immutable stage files (copy fallback), run workers, rehash output,
    generate full gallery and annotate geometric blend; save timing/coverage evidence.
    Key Params: Four CPU workers, pilot008/016/072; never overwrite baseline.
    State/Dependencies: Existing local snapshot; no OCR, detector, TextSR or post execution.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = json.loads((args.base/'summary.json').read_text())
    assert summary['status'] == 'passed' and summary['instances'] == 307
    records = summary['records']
    if args.pilot:
        records = [r for r in records if r['id'] in ('008','016','072')]
    for name in ('images','receipts','previews','pages','before'):
        (args.output/name).mkdir()
    save_json(args.output/'status.json', dict(status='running',completed=0,expected=len(records)))
    if not args.pilot:
        for name in ('source','documents','ocr','contexts','previews'):
            for source in (args.base/name).rglob('*'):
                if not source.is_file() or (name == 'previews' and source.name.endswith('_pre.jpg')):
                    continue
                dest = args.output/source.relative_to(args.base)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, dest)
                except OSError:
                    shutil.copyfile(source, dest)
        for name in ('inventory.json','jobs.json','source_manifest.json'):
            shutil.copyfile(args.base/name, args.output/name)
    try:
        results = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for receipt in pool.map(blend_one, [(str(args.base), str(args.output), r) for r in records]):
                results.append(receipt)
                save_json(args.output/'status.json', dict(status='running', completed=len(results), expected=len(records)))
                print('BLEND',len(results),len(records),receipt['key'],flush=True)
        for r in results:
            assert sha(args.output/r['image']) == r['image_sha256']
        if not args.pilot:
            rows = json.loads((args.output/'inventory.json').read_text())['images']
            jobs = json.loads((args.output/'jobs.json').read_text())
            gallery(args.output, rows, jobs, results)
            for page in [args.output/'index.html', *sorted((args.output/'pages').glob('*.html'))]:
                body = page.read_text()
                body = '<p style="color:#eee;background:#181c22;padding:12px">Boundary-blended V2.1:12% short-side smoothstep, maximum24 native pixels. Original at region edges; full V2.1 inside. Reference contexts unchanged.</p>'+body
                if page.parent.name == 'pages':
                    for item in results:
                        if item['id'] != page.stem:
                            continue
                        key = item['key']
                        body += '<h2>Before / after boundary blending: '+key+'</h2><p>Left: blended. Right: previous V2.1.</p>'
                        body += '<input type="range" min="0" max="100" value="50" style="width:90%" oninput="document.getElementById(\'blend_'+key+'\').style.clipPath=\'inset(0 \'+(100-this.value)+\'% 0 0)\'">'
                        body += '<div class="pair"><img src="../before/'+key+'.jpg"><img class="after" id="blend_'+key+'" src="../previews/'+key+'_pre.jpg"></div>'
                page.write_text(body, encoding='utf-8')
            pages = [args.output/'index.html', *sorted((args.output/'pages').glob('*.html'))]
            links = 0
            for page in pages:
                for link in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
                    assert (page.parent/link).is_file(), (page,link)
                    links += 1
            save_json(args.output/'gallery_audit.json',dict(status='passed',pages=len(pages),links=links))
        final = dict(summary, records=results, instances=len(results), boundary_blended=True,
                     base=str(args.base), wall_seconds=time.perf_counter()-started,
                     blend_ms_median=float(np.median([r['blend_ms'] for r in results])),
                     blend_ms_p95=float(np.percentile([r['blend_ms'] for r in results],95)),
                     timing_note='Concurrent CPU blending only, excluding decode, PNG and checks.',
                     code_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').glob('*.py')})
        save_json(args.output/'summary.json',final)
        save_json(args.output/'status.json',dict(status='passed',completed=len(results),expected=len(records)))
        print('PASS',len(results),flush=True)
    except Exception as error:
        save_json(args.output/'status.json',dict(status='failed',error=repr(error)))
        raise


if __name__ == '__main__':
    main()
