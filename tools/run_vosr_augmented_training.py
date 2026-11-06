"""Build the isolated VOSR-target bank and start the approved matched training matrix."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data/adaptive_vosr_preprocessed_augmented_v1_20261021')
SOURCE = Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261014/full_images')
OLD = Path('/AI_DNS/guang.yang/TextSR/data/adaptive_preprocessed_augmented_v1_20261020')
RUN = DATA / 'training_comparison_20261021'
VIEW = DATA / 'source_view'
REPO = '/home/guang.yang/workspace/Real-ESRGAN'


def digest(path):
    """@luna-comment
    Input: immutable file. Output: streamed SHA256. Key Steps: bounded read/hash.
    Key Params:4MiB. State/Dependencies: hashlib and read-only source files.
    """
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    """@luna-comment
    Input: build-owned JSON path/value. Output: atomic status receipt.
    Key Steps: write sibling then replace. Key Params:UTF8.
    State/Dependencies: isolated version root; never writes source assets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare_sources():
    """@luna-comment
    Input: immutable published full-image source and GT_VOSR targets.
    Output: versioned absolute-path manifests and target lineage receipt.
    Key Steps: verify dimensions/splits, hash targets, retain original hashes,
      bind matching LQ/OCR/masks and expose existing supplemental OCR read-only.
    Key Params:all published parents; no target resampling or line cropping.
    State/Dependencies: PNG headers/source manifests; writes VIEW plus DATA progress/readiness receipts.
    """
    from tools.inventory_vosr_training import image_hw
    rows_by_split = {}
    manifest_dir = VIEW / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in (SOURCE / 'manifests' / (split + '.jsonl')).read_text().splitlines() if line.strip()]
        output = []
        target_names = {p.name for p in (SOURCE / split / 'GT_VOSR').iterdir() if p.is_file()}
        if target_names != {Path(row['files']['gt']).name for row in rows}:
            raise ValueError('GT_VOSR inventory does not exactly match published ' + split)
        for ordinal, original in enumerate(rows):
            row = copy.deepcopy(original)
            if row.get('preprocess', {}).get('applied'):
                raise ValueError('Source already preprocessed')
            target = SOURCE / split / 'GT_VOSR' / Path(row['files']['gt']).name
            paths = {key: SOURCE / relative for key, relative in row['files'].items()}
            if len({image_hw(paths[key]) for key in ('lq', 'gt', 'text_mask', 'valid_mask')} | {image_hw(target)}) != 1:
                raise ValueError('Source geometry mismatch: ' + row['sample_id'])
            row['original_gt_lineage'] = dict(path=str(paths['gt']), sha256=row['output_sha256']['gt'])
            paths['gt'] = target
            row['files'] = {key: str(path) for key, path in paths.items()}
            row['output_sha256']['gt'] = digest(target)
            row['target_version'] = 'GT_VOSR_20261014'
            output.append(row)
            if (ordinal + 1) % 250 == 0:
                save(DATA / 'source_progress.json', dict(split=split, completed=ordinal + 1, total=len(rows)))
        rows_by_split[split] = output
    if {r['source_identity'] for r in rows_by_split['train']} & {r['source_identity'] for r in rows_by_split['val']}:
        raise ValueError('Capture split overlap')
    for split, rows in rows_by_split.items():
        path = manifest_dir / (split + '.jsonl')
        payload = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
        if path.exists() and path.read_text() != payload:
            raise ValueError('Refusing changed versioned source manifest')
        path.write_text(payload, encoding='utf-8')
    excluded = SOURCE / 'manifests/review_excluded.jsonl'
    (manifest_dir / excluded.name).write_bytes(excluded.read_bytes())
    supplemental = VIEW / 'OCR_ppocrv6_medium'
    if not supplemental.exists():
        supplemental.symlink_to(SOURCE / 'OCR_ppocrv6_medium', target_is_directory=True)
    receipt = dict(source=str(SOURCE), target='GT_VOSR', geometry_verified=True, capture_overlap=0,
                   counts={s: len(rows) for s, rows in rows_by_split.items()},
                   manifests={s: digest(manifest_dir / (s + '.jsonl')) for s in rows_by_split})
    save(DATA / 'SOURCE_READY.json', receipt)
    print(json.dumps(receipt), flush=True)


def command(arguments, log_name, environment=None):
    """@luna-comment
    Input: explicit Python arguments, log name and optional environment overrides.
    Output: successful subprocess or raised error. Key Steps: append log and check exit.
    Key Params:unbuffered Python, bounded threading. State/Dependencies: owned logs.
    """
    env = dict(os.environ, **(environment or {}))
    with (DATA / 'logs' / log_name).open('ab') as stream:
        subprocess.run([sys.executable, '-u', *arguments], env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, cwd=str(ROOT))


def detach(arguments, log_name):
    """@luna-comment
    Input: explicit approved worker arguments and log filename. Output: detached PID.
    Key Steps: spawn with inherited versioned environment and owned log.
    Key Params:start_new_session. State/Dependencies: never terminates existing processes.
    """
    with (DATA / 'logs' / log_name).open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-u', *arguments], stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True, cwd=str(ROOT))
    return process.pid


def main():
    """@luna-comment
    Input: explicit new-round invocation. Output: two complete banks and four real-paired arms.
    Key Steps: single-instance lock, signature-bound resume, offline build/verification/coverage,
      exact cross-mode geometry/mask/index matching, strict CUDA smokes and launches, HTML monitor.
    Key Params:16 offline CPU workers;250K/batch4 inherited; original architecture warm starts.
    State/Dependencies: versioned outputs, unchanged frozen package and historical losses.
    """
    import fcntl
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'logs').mkdir(exist_ok=True)
    RUN.mkdir(exist_ok=True)
    os.environ.update(OFFLINE_MATRIX_DATA=str(DATA), OFFLINE_MATRIX_RUN=str(RUN),
                      OFFLINE_MATRIX_TAG='vosr_20261021', OFFLINE_MATRIX_SOURCE=str(VIEW),
                      OFFLINE_MATRIX_KIND='preprocess_ab',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (DATA / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        try:
            save(DATA / 'PIPELINE.json', dict(phase='preparing_sources', pid=os.getpid(), started=started))
            if not (DATA / 'SOURCE_READY.json').exists():
                prepare_sources()
            for mode, folder in (('none', 'bank_raw_v1'), ('both', 'bank_both_v1')):
                bank = DATA / folder
                environment = dict(OFFLINE_PREPROCESS_MODE=mode)
                save(DATA / 'PIPELINE.json', dict(phase='building_' + mode, pid=os.getpid(), started=started))
                if not (bank / 'COMPLETE.json').exists():
                    command([str(ROOT / 'tools/build_offline_augmented_dataset.py'), '--source', str(VIEW), '--output', str(bank),
                             '--workers', '16', '--preprocess-mode', mode], 'build_' + mode + '.log', environment)
                else:
                    signature = json.loads((bank / 'build_signature.json').read_text())
                    complete = json.loads((bank / 'COMPLETE.json').read_text())
                    expected = {split: digest(VIEW / 'manifests' / (split + '.jsonl')) for split in ('train', 'val')}
                    if not (complete['complete'] and complete['build_id'] == signature['build_id']
                            and signature['builder'] == digest(ROOT / 'tools/build_offline_augmented_dataset.py')
                            and signature['source_manifests'] == expected and signature['preprocess_mode'] == mode):
                        raise ValueError('Completed bank resume signature mismatch')
                save(DATA / 'PIPELINE.json', dict(phase='verifying_' + mode, pid=os.getpid(), started=started))
                previous = json.loads((bank / 'verification.json').read_text()) if (bank / 'verification.json').exists() else {}
                if not (previous.get('passed') and previous.get('build_id') == json.loads((bank / 'COMPLETE.json').read_text())['build_id']):
                    command([str(ROOT / 'tools/verify_offline_augmented_dataset.py'), '--bank', str(bank), '--repo-root', REPO,
                             '--source', str(VIEW), '--config', str(OLD / 'bank_v1/train_paired.yml'),
                             '--gallery', str(DATA / ('gallery_' + mode))], 'verification_' + mode + '.log', environment)
                command([str(ROOT / 'offline_parameter_coverage.py'), str(bank)], 'coverage_' + mode + '.log')
                coverage = json.loads((bank / 'parameter_coverage.json').read_text())
                if not coverage['all_parameters_cover_10_deciles']:
                    raise ValueError('Incomplete augmentation parameter coverage')
            for split in ('train', 'val'):
                raw = json.loads((DATA / 'bank_raw_v1' / (split + '.json')).read_text())
                both = json.loads((DATA / 'bank_both_v1' / (split + '.json')).read_text())
                if len(raw['shards']) != len(both['shards']):
                    raise ValueError('Cross-mode parent count mismatch')
                for left, right in zip(raw['shards'], both['shards']):
                    for key in ('sample_id', 'source_identity', 'count', 'metadata_sha256', 'mask_sha256'):
                        if left[key] != right[key]:
                            raise ValueError('Cross-mode mismatch: ' + key)
                for filename in (split + '.index.npy', split + '.unique.npy'):
                    if digest(DATA / 'bank_raw_v1' / filename) != digest(DATA / 'bank_both_v1' / filename):
                        raise ValueError('Cross-mode sampling index mismatch')
            if digest(DATA / 'bank_raw_v1/train_balanced.index.npy') != digest(DATA / 'bank_both_v1/train_balanced.index.npy'):
                raise ValueError('Cross-mode balanced index mismatch')
            save(DATA / 'CROSS_MODE_VERIFIED.json', dict(passed=True, metadata_masks_and_indices_exact=True))
            manager = str(ROOT / 'tools/manage_offline_comparison.py')
            command([manager, 'prepare'], 'prepare_training.log')
            save(DATA / 'PIPELINE.json', dict(phase='smoking_and_launching', pid=os.getpid(), started=started))
            for arm, gpu in (('01_v5_preprocessed', '0'), ('02_v3_preprocessed', '1'), ('03_v5_raw', '2'), ('04_v3_raw', '3')):
                command([manager, 'smoke', '--arm', arm], arm + '_smoke.log', dict(CUDA_VISIBLE_DEVICES=gpu))
                command([manager, 'launch', '--arm', arm], arm + '_launch.log')
            monitor_pid = detach([str(ROOT / 'tools/watch_offline_comparison.py')], 'monitor.log')
            save(DATA / 'PIPELINE.json', dict(phase='four_real_paired_arms_launched', pid=os.getpid(), started=started,
                                           monitor_pid=monitor_pid, run=str(RUN)))
        except Exception as error:
            save(DATA / 'PIPELINE.json', dict(phase='failed', error=repr(error), pid=os.getpid(), started=started))
            raise


if __name__ == '__main__':
    main()
