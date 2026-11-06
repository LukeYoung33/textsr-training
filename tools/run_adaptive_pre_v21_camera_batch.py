"""Export the approved 280-image camera inventory with V2.1 preprocessing only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.adaptive_scurve_robust_v2 import prepare_context, apply_preprocess, make_post_context


def sha(path):
    """@luna-comment
    Input: Existing file. Output: SHA256 hex digest.
    Key Steps: Stream bytes; Key Params: 4MiB chunks.
    State/Dependencies: Read-only filesystem access.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    """@luna-comment
    Input: Output path and JSON-compatible value. Output: Atomic JSON file.
    Key Steps: Write sibling temporary and replace; Key Params: UTF8.
    State/Dependencies: Only own output directory is mutated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def process_one(job):
    """@luna-comment
    Input: Inventory row, cached detection directory and fresh export root.
    Output: Verified native PNG, reference context, previews and receipt.
    Key Steps: Bind source hashes; run unchanged V2.1; validate geometry,
    ownership, equal-channel delta and exact PNG roundtrip before publishing.
    Key Params: Foreground core enabled, gamma1.8, one OpenCV thread.
    State/Dependencies: CPU only; immutable inputs; disjoint per-image outputs.
    """
    cv2.setNumThreads(1)
    row, detections, output = job
    output = Path(output)
    started = time.perf_counter()
    source = Path(row['remote'])
    detection = Path(detections) / (source.stem + '.json')
    meta = json.loads(detection.read_text())
    assert sha(source) == row['sha256'] == meta['source_sha256'], source
    image = cv2.imread(str(source))
    assert image is not None and list(image.shape[:2]) == meta['source_hw'], source
    tick = time.perf_counter()
    context = prepare_context(image, meta['boxes'], reference_id=row['sha256'], foreground_core=True)
    processed = apply_preprocess(image, context)
    core_ms = (time.perf_counter() - tick) * 1000
    assert processed.shape == image.shape and processed.dtype == np.uint8
    union = np.zeros(image.shape[:2], bool)
    for x0, y0, x1, y1 in meta['boxes']:
        union[y0:y1, x0:x1] = True
    assert np.array_equal(processed[~union], image[~union])
    # Check chroma invariants in row chunks to bound native temporary memory.
    for y0 in range(0, image.shape[0], 128):
        delta = processed[y0:y0+128].astype(np.int16) - image[y0:y0+128].astype(np.int16)
        assert np.array_equal(delta[:, :, 0], delta[:, :, 1])
        assert np.array_equal(delta[:, :, 1], delta[:, :, 2])
    name = source.stem + '.png'
    native = output / 'images' / name
    assert cv2.imwrite(str(native), processed, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert np.array_equal(cv2.imread(str(native)), processed), native
    reference = output / 'contexts' / (source.stem + '.context.json')
    save_json(reference, make_post_context(context))
    for suffix, value in [('source', image), ('pre', processed)]:
        scale = min(1., 1200 / max(value.shape[:2]))
        preview = cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(output / 'previews' / (row['id'] + '_' + suffix + '.jpg')), preview)
    receipt = dict(id=row['id'], name=row['name'], source=str(source), source_sha256=row['sha256'],
                   detection_sha256=sha(detection), shape=list(image.shape), boxes=len(meta['boxes']),
                   version=context['version'], foreground_core=True, gamma=1.8,
                   image='images/' + name, context='contexts/' + reference.name,
                   image_sha256=sha(native), context_sha256=sha(reference),
                   outside_unchanged=True, equal_channel_delta=True, png_roundtrip_exact=True,
                   core_ms=core_ms, prepare_timings=context['timings'],
                   total_seconds=time.perf_counter()-started)
    save_json(output / 'receipts' / (row['id'] + '.json'), receipt)
    return receipt


def main():
    """@luna-comment
    Input: Explicit inventory/detection paths and isolated output directory.
    Output: Complete hash-verified 280-image batch, gallery, timing and status.
    Key Steps: Validate unique inventory; bounded process pool; rehash all exports;
    publish passed only after exact coverage, otherwise persist failed status.
    Key Params: Four workers default; limit supports a separate smoke directory.
    State/Dependencies: No model/GPU; refuses existing output to avoid mixed versions.
    """
    parser = argparse.ArgumentParser()
    for key in ('inventory', 'detections', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=280)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'contexts', 'receipts', 'previews', 'pages'):
        (output / name).mkdir()
    started = time.perf_counter()
    rows = json.loads(Path(args.inventory).read_text())['images']
    assert len(rows) == 280 and len({r['id'] for r in rows}) == 280
    rows = rows[:args.limit]
    assert len({Path(r['remote']).stem for r in rows}) == len(rows)
    save_json(output / 'inventory.json', dict(images=rows, upstream=str(args.inventory), upstream_sha256=sha(args.inventory)))
    save_json(output / 'code_manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*.py'))})
    records = []
    try:
        save_json(output / 'status.json', dict(status='running', completed=0, expected=len(rows)))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for receipt in pool.map(process_one, [(r, args.detections, str(output)) for r in rows]):
                records.append(receipt)
                save_json(output / 'status.json', dict(status='running', completed=len(records), expected=len(rows)))
                print(f"{len(records)}/{len(rows)} {receipt['id']} {receipt['name']} core={receipt['core_ms']:.1f}ms", flush=True)
        assert {r['id'] for r in records} == {r['id'] for r in rows}
        for r in records:
            assert sha(output / r['image']) == r['image_sha256']
            assert sha(output / r['context']) == r['context_sha256']
            assert json.loads((output / 'receipts' / (r['id'] + '.json')).read_text()) == r
            page = '<meta charset="utf-8"><h1>' + html.escape(r['id'] + ' ' + r['name']) + '</h1>'
            page += '<p>Latest V2.1 preprocessing only. Original full-frame OCR boxes. No model/postprocessing/warp.</p>'
            page += '<p><a href="../index.html">All images</a> | <a href="../' + r['image'] + '">Native PNG</a> | <a href="../' + r['context'] + '">Post context</a></p>'
            for label, suffix in [('Original', 'source'), ('V2.1 preprocessing', 'pre')]:
                page += '<h2>' + label + '</h2><img style="max-width:100%" src="../previews/' + r['id'] + '_' + suffix + '.jpg">'
            (output / 'pages' / (r['id'] + '.html')).write_text(page, encoding='utf-8')
        index = '<meta charset="utf-8"><h1>Camera 280: adaptive preprocessing V2.1</h1><p>Native PNG files: images/; reusable frozen-post reference: contexts/. No new model inference.</p><ul>'
        index += ''.join('<li><a href="pages/' + r['id'] + '.html">' + html.escape(r['id'] + ' ' + r['name']) + '</a></li>' for r in records) + '</ul>'
        (output / 'index.html').write_text(index, encoding='utf-8')
        summary = dict(status='passed', images=len(records), workers=args.workers,
                       algorithm='adaptive_scurve_robust_pre_v2_1', foreground_core=True,
                       wall_seconds=time.perf_counter()-started,
                       core_ms_median=float(np.median([r['core_ms'] for r in records])),
                       core_ms_p95=float(np.percentile([r['core_ms'] for r in records], 95)),
                       timing_note='Concurrent CPU export, not a single-worker deployment benchmark.',
                       records=records)
        save_json(output / 'summary.json', summary)
        save_json(output / 'status.json', dict(status='passed', completed=len(records), expected=len(rows), wall_seconds=summary['wall_seconds']))
        print('PASS', len(records), flush=True)
    except Exception as error:
        save_json(output / 'status.json', dict(status='failed', error=repr(error), completed=len(records), expected=len(rows)))
        raise


if __name__ == '__main__':
    main()
