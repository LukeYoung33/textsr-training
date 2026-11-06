"""Frozen V14 batch: explicit reviewed routes and bounded isolated workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    """@luna-comment Input:file. Output:SHA256. Steps:stream chunks.
    Params:1MiB. Dependencies:read-only filesystem.
    """
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    """@luna-comment Input:path/object. Output:atomic JSON. Steps:replace temp.
    Params:UTF8. Dependencies:one writer per destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def process(folder, manifest, index):
    """@luna-comment Input:frozen manifest job. Output:PNG, debug, receipt.
    Steps:hash gates, declared ROI, unchanged V14, roundtrip/source checks.
    Params:4 OpenCV threads. Dependencies:isolated CPU child, no model.
    """
    import cv2
    import numpy as np
    from tools.postprocess_ppt_polarity_v14 import process_image
    from tools.postprocess_ppt_contour_v6 import read_image
    from tools.postprocess_ppt_dehalo_20260924 import write_png
    cv2.setNumThreads(4)
    job = manifest['jobs'][index]
    receipt = folder/'receipts'/f'{index:03d}.json'
    start = time.time()
    try:
        save(receipt, dict(state='running', pid=os.getpid(), **job))
        for name, expected in manifest['code_hashes'].items():
            assert digest(ROOT/name) == expected, name
        source = Path(job['source'])
        assert digest(source) == job['sha256']
        image = read_image(source)
        assert list(image.shape) == job['shape']
        roi = None
        if job['polygon']:
            points = np.array(job['polygon'], np.int32)
            assert len(points) >= 3 and np.all(points >= 0)
            assert np.all(points[:, 0] < image.shape[1]) and np.all(points[:, 1] < image.shape[0])
            roi = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(roi, [points], 1)
            roi = roi.astype(bool)
            assert roi.any()
        result, debug, report = process_image(image, job['polarity'], roi)
        dest = folder/job['output']
        assert not dest.exists()
        temporary = dest.with_suffix('.partial.png')
        write_png(temporary, result)
        assert np.array_equal(read_image(temporary), result)
        assert digest(source) == job['sha256']
        temporary.replace(dest)
        d = folder/'debug'/job['group']/source.stem
        for key in ('roi', 'core', 'actual_strokes', 'weight'):
            write_png(d/(key+'.png'), np.clip(np.rint(debug[key]*255), 0, 255).astype(np.uint8))
        preview = cv2.resize(result, (640, 480), interpolation=cv2.INTER_AREA)
        write_png(folder/'previews'/job['group']/(source.stem+'.png'), preview)
        save(receipt, dict(state='complete', pid=os.getpid(), **job, report=report,
                           elapsed_seconds=time.time()-start, output_sha256=digest(dest)))
    except Exception:
        save(receipt, dict(state='failed', **job, error=traceback.format_exc()))
        raise


def child(folder, index):
    """@luna-comment Input:folder/index. Output:child exit. Steps:run/log process.
    Params:single image lifecycle. Dependencies:bounded parent pool.
    """
    with (folder/'logs'/f'{index:03d}.log').open('w') as log:
        return subprocess.run([sys.executable, __file__, '--output', str(folder), '--job', str(index)], stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    """@luna-comment Input:prepare/launch/run CLI. Output:72 frozen outputs/status.
    Steps:manifest, detached parent, isolated jobs, atomic progress receipts.
    Params:8 workers maximum, explicit route file. Dependencies:Linux runtime.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--routes', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--launch', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--job', type=int)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    folder = args.output.resolve()
    assert 1 <= args.workers <= 8
    if args.prepare:
        import cv2
        from tools.postprocess_ppt_contour_v6 import read_image
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'logs').mkdir()
        routes = json.loads(args.routes.read_text())
        jobs = []
        for group in ('blackwhite', 'native96x480', 'paste_fixed'):
            paths = sorted((args.input/group).glob('*.png'))
            assert len(paths) == 24
            for source in paths:
                route = routes[group+'/'+source.name]
                image = read_image(source)
                jobs.append(dict(group=group, source=str(source), sha256=digest(source), shape=list(image.shape),
                                 output=f'{group}/{source.stem}_dehalo_v14.png', **route))
        hashes = {str(f.relative_to(ROOT)): digest(f) for f in (ROOT/'tools').glob('*.py')}
        save(folder/'manifest.json', dict(algorithm='V14/P16a', jobs=jobs, code_hashes=hashes,
                                         python=sys.version, opencv=cv2.__version__))
        print('PREPARED', len(jobs), flush=True)
        return
    manifest = json.loads((folder/'manifest.json').read_text())
    if args.job is not None:
        process(folder, manifest, args.job)
        return
    if args.launch:
        assert not (folder/'launch.json').exists()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        with (folder/'batch.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, __file__, '--output', str(folder), '--run', '--workers', str(args.workers)],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        save(folder/'launch.json', dict(pid=proc.pid, workers=args.workers))
        print('STARTED', proc.pid, flush=True)
        return
    if args.run:
        start = time.time()
        done = [i for i in range(len(manifest['jobs'])) if (folder/'receipts'/f'{i:03d}.json').exists()
                and json.loads((folder/'receipts'/f'{i:03d}.json').read_text())['state'] == 'complete']
        todo = [i for i in range(len(manifest['jobs'])) if i not in done]
        failed = []
        save(folder/'status.json', dict(state='running', total=72, completed=len(done), failed=0, workers=args.workers, pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(child, folder, i): i for i in todo}
            for future in as_completed(futures):
                index = futures[future]
                (done if future.result() == 0 else failed).append(index)
                save(folder/'status.json', dict(state='running', total=72, completed=len(done), failed=len(failed),
                                               completed_ids=done, failed_ids=failed, elapsed_seconds=time.time()-start, workers=args.workers))
                print('DONE', len(done), 'FAILED', len(failed), flush=True)
        save(folder/'status.json', dict(state='finished' if not failed else 'finished_with_errors', total=72, completed=len(done), failed=len(failed),
                                       failed_ids=failed, elapsed_seconds=time.time()-start, workers=args.workers))


if __name__ == '__main__':
    main()
