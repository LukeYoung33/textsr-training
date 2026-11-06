"""Run the frozen V13b exterior-cap cleanup over explicit Remote58 PNG groups."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    """@luna-comment Input:read-only file. Output:SHA256 string.
    Key Steps:stream fixed binary chunks. Key Params:1 MiB chunks.
    State/Dependencies:does not mutate source or result files.
    """
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path, data):
    """@luna-comment Input:destination path and JSON-safe record. Output:atomic JSON.
    Key Steps:write sibling temporary then replace. Key Params:UTF-8/indent2.
    State/Dependencies:status writer is parent-only; each receipt path is unique.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def build_jobs(inputs):
    """@luna-comment Input:two immutable input group directories. Output:hashed job list.
    Key Steps:enumerate top-level PNG files and make collision-free group outputs.
    Key Params:group directory basename; sorted deterministic ordering.
    State/Dependencies:input roots stay read-only and must each be non-empty.
    """
    jobs = []
    for input_root in inputs:
        paths = sorted(input_root.glob('*.png'))
        if not paths:
            raise ValueError(f'No PNG inputs in {input_root}')
        for source in paths:
            jobs.append(dict(
                group=input_root.name,
                source=str(source.resolve()),
                source_sha256=digest(source),
                output=f'{input_root.name}/{source.stem}_dehalo_v13b.png',
            ))
    return jobs


def process_job(output_root, job):
    """@luna-comment Input:one frozen V13b job. Output:lossless result and receipt.
    Key Steps:source hash gate, V4 reconstruction, P15b exterior cap, decode/hash gates.
    Key Params:robust_seed=true; exterior_cap=true; four OpenCV threads.
    State/Dependencies:one process owns one output and receipt; source must remain unchanged.
    """
    import cv2
    import numpy as np
    from tools.postprocess_ppt_clean_background_v13 import clean_background_feather
    from tools.postprocess_ppt_contour_v6 import read_image
    from tools.postprocess_ppt_dehalo_20260924 import prepare, write_png
    from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct

    cv2.setNumThreads(4)
    source_path = Path(job['source'])
    destination = output_root / job['output']
    receipt_path = output_root / 'diagnostics' / job['group'] / f'{source_path.stem}.json'
    started = time.perf_counter()
    try:
        if digest(source_path) != job['source_sha256']:
            raise RuntimeError(f'Source hash changed before processing: {source_path}')
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {destination}')
        source = read_image(source_path)
        prepared = prepare(source, robust_seed=True)
        _, maps, _ = normal_correct(prepared)
        base = prepared[0] + maps['correction_weight'][..., None] * (prepared[4] - prepared[0])
        result, _, report = clean_background_feather(base, prepared, exterior_cap=True)
        if result.shape != source.shape:
            raise RuntimeError(f'Output shape mismatch: {result.shape} != {source.shape}')
        write_png(destination, result)
        if not np.array_equal(read_image(destination), result):
            raise RuntimeError(f'Lossless PNG roundtrip failed: {destination}')
        if digest(source_path) != job['source_sha256']:
            raise RuntimeError(f'Source hash changed after processing: {source_path}')
        receipt = dict(status='complete', algorithm='V13b/P15b_positive_exterior', pid=os.getpid(),
                       shape=list(result.shape), output_sha256=digest(destination),
                       elapsed_seconds=time.perf_counter() - started, report=report, **job)
        save_json(receipt_path, receipt)
        return receipt
    except Exception:
        save_json(receipt_path, dict(status='failed', elapsed_seconds=time.perf_counter() - started,
                                     error=traceback.format_exc(), **job))
        raise


def main():
    """@luna-comment Input:explicit source groups/output/workers CLI. Output:parallel V13b batch.
    Key Steps:create exclusive manifest; process pool; atomically update status and receipts.
    Key Params:two user-selected groups; default six workers; no input fallback.
    State/Dependencies:result directory must not exist; no source or existing-output overwrite.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f'Output root already exists: {output_root}')
    inputs = [path.resolve() for path in args.input]
    if len({path.name for path in inputs}) != len(inputs):
        raise ValueError('Input group directory names must be unique')
    jobs = build_jobs(inputs)
    output_root.mkdir(parents=True)
    script_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (
        ROOT / 'tools' / 'postprocess_ppt_clean_background_v13.py',
        ROOT / 'tools' / 'postprocess_ppt_dehalo_20260924.py',
        ROOT / 'tools' / 'postprocess_ppt_normal_dehalo_v4.py',
        ROOT / 'tools' / 'postprocess_ppt_contour_v6.py',
        ROOT / 'tools' / 'postprocess_ppt_contour_v5.py',
        ROOT / 'tools' / 'postprocess_ppt_crossseam_v9.py',
        Path(__file__),
    )}
    save_json(output_root / 'manifest.json', dict(algorithm='V13b/P15b_positive_exterior',
              exterior_cap=True, inputs=[str(path) for path in inputs], workers=args.workers,
              code_hashes=script_hashes, jobs=jobs))
    started = time.perf_counter()
    completed, failures = [], []
    save_json(output_root / 'status.json', dict(state='running', completed=0, failed=0,
              total=len(jobs), workers=args.workers, elapsed_seconds=0.0, errors=[]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_job, output_root, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(dict(source=job['source'], error=repr(exc)))
            save_json(output_root / 'status.json', dict(state='running', completed=len(completed),
                      failed=len(failures), total=len(jobs), workers=args.workers,
                      elapsed_seconds=time.perf_counter() - started, errors=failures))
            print(f'PROGRESS {len(completed)}/{len(jobs)} failed={len(failures)}', flush=True)
    final = dict(state='finished' if not failures else 'finished_with_errors', completed=len(completed),
                 failed=len(failures), total=len(jobs), workers=args.workers,
                 elapsed_seconds=time.perf_counter() - started, errors=failures)
    save_json(output_root / 'status.json', final)
    if failures:
        raise RuntimeError(f'V13b batch failed for {len(failures)} jobs')


if __name__ == '__main__':
    main()
