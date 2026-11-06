"""Hash-bound multiprocess V3 preprocessing of pretrain and Mojito full-image pairs."""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA / 'v3pre_v3m_three_20261027'
TZ = timezone(timedelta(hours=8))
SOURCES = {'pretrain': DATA / 'vosr_2u_block256_v1_20261023/source_view',
           'finetune': DATA / 'six_models_mojito_finetune_20261027/source_view'}
REFERENCES = {'evaluation/document_box_neighbor_20261027/neighbor.py': 'dca994d66ee78bdd3d84892ffd5ad09d06c26a7a2c16a40c3f4c0501662f31ed',
              'evaluation/document_box_pre640_revision2_20261027/revision2.py': '09e340a85a96c146bebcdf6db074ff13c5734cf48262ace0f4ec14528211cbff',
              'evaluation/document_box_pre640_astra_20261027/pre640.py': '0dadc5ce47faea782a07b8f9d3d4cd39dd909618be23a82f1e59065a181195d8',
              'tools/adaptive_scurve_robust_v2.py': '1bb266d490a4cb1cf8162066c10ebf8bf190403a22f5d5c6d1ce0f052cf70612',
              'evaluation/adaptive_prepost_frozen_v1_20261020/lowres_curve.py': '10cd1144c77fd98c813373960ac415846978ece14ad28fb1049efd810ddb27fa'}


def sha(path):
    """@luna-comment Input:file path. Output:SHA256. Key Steps:stream4MiB.
    Key Params:read-only bytes. State/Dependencies:hashlib/filesystem.
    """
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    """@luna-comment Input:owned path/JSON. Output:atomic UTF8 receipt.
    Key Steps:write sibling then replace. Key Params:finite values. State/Dependencies:owned run only.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def status(phase, **values):
    """@luna-comment Input:phase/progress/rates. Output:persistent JSON and refreshing HTML.
    Key Steps:merge task state and publish timestamps/ETA. Key Params:30s reload.
    State/Dependencies:single controller writes; no external web service required.
    """
    path = OUT / 'STATUS.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(phase=phase, updated_at=datetime.now(TZ).isoformat(), **values)
    save(path, state)
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>V3 preprocessing and V3M training</title><style>body{font:16px system-ui;margin:28px}pre{white-space:pre-wrap}</style><h1>V3 preprocessing / V3M three training chains</h1><button onclick="location.reload()">Refresh</button><pre>'
    (OUT / 'index.html').write_text(page + html.escape(json.dumps(state, indent=2)) + '</pre>', encoding='utf-8')


def process(job):
    """@luna-comment Input:stage/split/frozen parent/build identity. Output:complete pair receipt.
    Key Steps:verify source hashes; exact V3 independently on GT/LQ; check native
    shape/shared delta; lossless PNG and provenance. Key Params:STANDARD repair1/slope.1.
    State/Dependencies:immutable source masks/OCR; resume only matching verified outputs.
    """
    import cv2
    import numpy as np
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    cv2.setNumThreads(1)
    stage, split, original, build = job
    row = copy.deepcopy(original)
    base = OUT / stage / 'source_view'
    receipt = base / 'receipts' / split / row['dataset'] / (row['sample_id'] + '.json')
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r['build_id'] == build
        for key, path in r['row']['files'].items():
            assert sha(path) == r['row']['output_sha256'][key]
        return r
    started = time.time()
    for key, path in row['files'].items():
        assert sha(path) == row['output_sha256'][key], (key, path)
    pair_shape = None
    for key in ('lq', 'gt'):
        source = Path(row['files'][key])
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        assert image is not None and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        if pair_shape is None:
            pair_shape = image.shape
        assert image.shape == pair_shape
        estimate = revision2.estimate(image)
        q, tail = neighbor.descriptors(estimate['canvas'], estimate['nodes'])
        low, info = neighbor.repair(estimate['fields']['low'], estimate['fields']['high'], q, tail, estimate['nodes'])
        packed = dict(nodes=estimate['nodes'], tables=neighbor.tables(low, estimate['fields']['high']), native_shape=list(image.shape[:2]))
        result, telemetry = pre640.render(image, packed, strength=1., chunk=128)
        assert result.shape == image.shape and result.dtype == np.uint8
        delta = result.astype(np.int16) - image.astype(np.int16)
        assert np.array_equal(delta[..., 0], delta[..., 1]) and np.array_equal(delta[..., 1], delta[..., 2])
        path = base / 'images' / split / row['dataset'] / key / (row['sample_id'] + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        assert ok
        temp = path.with_suffix('.png.tmp')
        temp.write_bytes(encoded.tobytes())
        temp.replace(path)
        row['files'][key] = str(path)
        row['output_sha256'][key] = sha(path)
    row['preprocess'] = dict(version='document_preprocessing_V3', full_repair=1., independent_gt_lq=True,
                             source_files=original['files'], source_hashes=original['output_sha256'])
    result = dict(build_id=build, stage=stage, split=split, row=row, seconds=time.time() - started, shape=list(pair_shape))
    save(receipt, result)
    return result


def main():
    """@luna-comment Input:complete original source views. Output:parallel V3 views/progress/ETA.
    Key Steps:freeze reference/selection;24-process parent pairs; only publish complete
    split manifests; fail closed. Key Params:4873/583 pretrain,380/67 finetune.
    State/Dependencies:owned snapshot/outputs; does not launch training itself.
    """
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert {key: sha(ROOT / key) for key in REFERENCES} == REFERENCES
    build = hashlib.sha256(json.dumps(dict(references=REFERENCES, worker=sha(__file__)), sort_keys=True).encode()).hexdigest()
    save(OUT / 'PREPROCESS_BINDING.json', dict(build_id=build, references=REFERENCES, workers=24))
    jobs = []
    for stage, source in SOURCES.items():
        for split, count in zip(('train', 'val'), (4873, 583) if stage == 'pretrain' else (380, 67)):
            rows = [json.loads(line) for line in (source / 'manifests' / (split + '.jsonl')).read_text().splitlines() if line.strip()]
            assert len(rows) == count
            jobs.extend((stage, split, row, build) for row in rows)
    save(OUT / 'SOURCE_SELECTION.json', dict(build_id=build, jobs=jobs))
    started = time.time()
    done = []
    status('preprocessing', workers=24, completed_pairs=0, total_pairs=len(jobs), total_images=2 * len(jobs))
    try:
        with ProcessPoolExecutor(max_workers=24) as pool:
            futures = [pool.submit(process, job) for job in jobs]
            for future in as_completed(futures):
                done.append(future.result())
                if len(done) % 12 == 0 or len(done) == len(jobs):
                    elapsed = time.time() - started
                    remaining = elapsed / len(done) * (len(jobs) - len(done))
                    status('preprocessing', completed_pairs=len(done), completed_images=2*len(done),
                           elapsed_seconds=elapsed, pairs_per_minute=60*len(done)/elapsed,
                           preprocessing_eta=(datetime.now(TZ) + timedelta(seconds=remaining)).isoformat(),
                           eta_basis='observed aggregate wall rate; bank generation and checks additional')
                    print(json.dumps(dict(done=len(done), total=len(jobs), eta_seconds=remaining)), flush=True)
        for stage in SOURCES:
            for split in ('train', 'val'):
                rows = [r['row'] for r in done if r['stage'] == stage and r['split'] == split]
                rows.sort(key=lambda r: (r['dataset'], r['sample_id']))
                path = OUT / stage / 'source_view/manifests' / (split + '.jsonl')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        save(OUT / 'PREPROCESS_COMPLETE.json', dict(passed=True, build_id=build, pairs=len(done), seconds=time.time()-started))
        status('preprocessing_complete', completed_pairs=len(done))
    except BaseException as error:
        status('preprocessing_failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
