"""Isolated VOSR-2u target refresh and approved block256 V3/V5 launch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data/vosr_2u_block256_v1_20261023')
ARCHIVE = DATA.parent / 'LQ_2x_vosr_2u.zip'
OLD = DATA.parent / 'vosr_fullimage_block256_v1_20261021'
SOURCE = DATA.parent / 'adaptive_vosr_preprocessed_augmented_v1_20261021/source_view'
RUN = DATA / 'training_comparison_20261023'
BANK = DATA / 'bank_raw_v1'
ARMS = ('05_v5_block256_raw', '06_v3_block256_raw')


def source_parent(job):
    """@luna-comment
    Input: published split/row and extracted target map. Output: verified row/resize provenance.
    Key Steps: verify all original hashes/geometry, use new train GT, resize only GT and hash output.
    Key Params: INTER_CUBIC to exact LQ width/height; old val unchanged; eight bounded threads.
    State/Dependencies: cv2/numpy, immutable originals; writes only isolated aligned_gt directory.
    """
    import cv2
    import numpy as np
    from tools.build_offline_augmented_dataset import digest
    split, original, targets = job
    row = copy.deepcopy(original)
    shapes = {}
    for key, filename in row['files'].items():
        path = SOURCE / filename
        if digest(path) != row['output_sha256'][key]:
            raise ValueError('Original hash mismatch: ' + str(path))
        if key != 'ocr':
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint8:
                raise ValueError('Invalid source pixels: ' + str(path))
            shapes[key] = image.shape[:2]
        row['files'][key] = str(path)
    if len(set(shapes.values())) != 1:
        raise ValueError('Original source geometry mismatch')
    record = dict(sample_id=row['sample_id'], split=split, original_hw=list(shapes['lq']))
    if split == 'train':
        target = targets[row['sample_id']]
        raw_hash = digest(target)
        image = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid new target: ' + str(target))
        height, width = shapes['lq']
        old_shape = list(image.shape[:2])
        if old_shape != [height, width]:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
            aligned = DATA / 'aligned_gt' / (row['sample_id'] + '.png')
            if aligned.exists():
                raise ValueError('Refusing to overwrite aligned GT: ' + str(aligned))
            if not cv2.imwrite(str(aligned), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise IOError(str(aligned))
            target = aligned
        row['previous_gt'] = dict(path=row['files']['gt'], sha256=row['output_sha256']['gt'])
        row['files']['gt'] = str(target)
        row['output_sha256']['gt'] = digest(target)
        row['target_version'] = 'VOSR_2u_20261023'
        row['target_refresh'] = dict(raw_path=str(targets[row['sample_id']]), raw_sha256=raw_hash,
                                    raw_hw=old_shape, output_hw=[height, width],
                                    interpolation='INTER_CUBIC' if old_shape != [height, width] else 'identity')
        record.update(row['target_refresh'])
    else:
        record['target_version'] = 'old_GT_VOSR_20261014_validation_only'
    return row, record


def prepare_source():
    """@luna-comment
    Input: user archive and previous immutable source manifests. Output: CRC-checked extraction/source view.
    Key Steps: reject unsafe/duplicate/foreign names; extract exactly4873 PNGs; verify source families/splits.
    Key Params: excludes two absent train IDs; old583 validation; native originals retained.
    State/Dependencies: ZIP CRC/read hashes, cv2 threads1; isolated output must be fresh.
    """
    import cv2
    from tools.build_offline_augmented_dataset import digest, write_json
    cv2.setNumThreads(1)
    marker = DATA / 'source_view' / 'COMPLETE.json'
    if marker.exists():
        raise ValueError('Source already published; use train action to continue')
    rows = {s: [json.loads(x) for x in (SOURCE / 'manifests' / (s + '.jsonl')).read_text().splitlines()
                if x.strip()] for s in ('train', 'val')}
    extracted = DATA / 'extracted'
    extracted.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        paths = [Path(m.filename) for m in members]
        if any(p.is_absolute() or '..' in p.parts or len(p.parts) != 2 or p.parts[0] != 'LQ_2x_vosr_2u'
               or p.suffix.lower() != '.png' or '\\' in str(p) or ':' in str(p) for p in paths):
            raise ValueError('Unexpected archive member')
        names = [p.stem for p in paths]
        known = {r['sample_id'] for r in rows['train']}
        if len(names) != 4873 or len(set(names)) != len(names) or not set(names) <= known:
            raise ValueError('Archive identity/count mismatch')
        excluded = sorted(known - set(names))
        if excluded != ['capture_261011_004394', 'capture_261011_004871']:
            raise ValueError('Unexpected missing train sources')
        archive.extractall(extracted)
    targets = {p.stem: extracted / p for p in paths}
    rows['train'] = [r for r in rows['train'] if r['sample_id'] in targets]
    if {r['source_identity'] for r in rows['train']} & {r['source_identity'] for r in rows['val']}:
        raise ValueError('Capture split overlap')
    (DATA / 'aligned_gt').mkdir(exist_ok=True)
    view = DATA / 'source_view'
    (view / 'manifests').mkdir(parents=True, exist_ok=True)
    records = []
    for split in ('train', 'val'):
        jobs = [(split, r, targets) for r in rows[split]]
        with ThreadPoolExecutor(max_workers=8) as pool:
            processed = list(pool.map(source_parent, jobs))
        (view / 'manifests' / (split + '.jsonl')).write_text(
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r, _ in processed), encoding='utf-8')
        records.extend(record for _, record in processed)
        print(json.dumps(dict(phase='source_ready', split=split, parents=len(processed))), flush=True)
    supplemental = SOURCE / 'OCR_ppocrv6_medium'
    if supplemental.exists():
        (view / supplemental.name).symlink_to(supplemental, target_is_directory=True)
    write_json(DATA / 'SOURCE_AUDIT.json', dict(archive=str(ARCHIVE), archive_sha256=digest(ARCHIVE),
        excluded_train=excluded, capture_overlap=0, records=records,
        train_target='new VOSR2u; target-only cubic alignment', validation_target='old GT_VOSR; 583 pairs'))
    write_json(marker, dict(complete=True, train=4873, val=len(rows['val']), preprocessing='none'))


def bind_manager():
    """@luna-comment
    Input: isolated DATA constants. Output: reused block manager with owned run/bank bindings.
    Key Steps: override output globals only; keep approved train/bootstrap/source reference modules.
    Key Params: GPU2/3; original warm-start configs. State/Dependencies: process-local module state.
    """
    from tools import manage_vosr_block256 as manager
    manager.DATA, manager.RUN, manager.BANK = DATA, RUN, BANK
    manager.MODE, manager.SUFFIX, manager.ARMS = 'none', 'raw', ARMS
    return manager


def train():
    """@luna-comment
    Input: complete new source view. Output: complete offline bank, two strict-smoked jobs and loss monitor.
    Key Steps: unchanged offline builder, clone block configs with new identities, smoke/launch, detach observer.
    Key Params:16workers;250K,batch4,seed/loss/original warm starts retained; old validation explicitly labeled.
    State/Dependencies: torch/CUDA, staged old source closure; never terminate or overwrite other jobs.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest, write_json
    manager = bind_manager()
    if not (DATA / 'source_view/COMPLETE.json').exists():
        raise ValueError('New source not complete')
    if not (BANK / 'COMPLETE.json').exists():
        with (DATA / 'build.log').open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(ROOT / 'tools/build_vosr_block256_dataset.py'),
                '--source', str(DATA / 'source_view'), '--output', str(BANK), '--workers', '16',
                '--preprocess-mode', 'none'], stdout=stream, stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OFFLINE_PREPROCESS_MODE='none'))
    complete = json.loads((BANK / 'COMPLETE.json').read_text())
    assert complete['complete'] and complete['splits']['train']['nonempty_parents'] == 4873
    RUN.mkdir(exist_ok=True)
    provenance = {}
    for arm, gpu in zip(ARMS, (2, 3)):
        cfg = yaml.safe_load((OLD / 'training_comparison_20261021/configs' / (arm + '.yml')).read_text())
        cfg['name'] = 'vosr2u_' + arm + '_250k_20261023'
        cfg['path']['experiments_root'] = str(RUN / 'experiments')
        for split in ('train', 'val'):
            cfg['datasets'][split].update(root=str(BANK), manifest=str(BANK / ('train_balanced.json' if split == 'train' else 'val.json')))
        path = RUN / 'configs' / (arm + '.yml')
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != payload:
            raise ValueError('Refusing modified training config')
        path.write_text(payload)
        provenance[arm] = dict(gpu=gpu, config_sha256=digest(path), checkpoint=cfg['path']['pretrain_network_g'],
                              checkpoint_sha256=digest(cfg['path']['pretrain_network_g']), build_id=complete['build_id'],
                              validation_target='old_GT_VOSR_20261014')
    write_json(RUN / 'provenance.json', provenance)
    for arm, gpu in zip(ARMS, (2, 3)):
        if (RUN / 'launch' / (arm + '.json')).exists():
            raise ValueError('Already launched: ' + arm)
        with (DATA / (arm + '_smoke.log')).open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(__file__), 'smoke', '--arm', arm],
                stdout=stream, stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        manager.launch(arm)
    with (DATA / 'monitor.log').open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(__file__), 'monitor'],
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(DATA / 'PIPELINE.json', dict(phase='both_launched', monitor_pid=process.pid,
                                          validation_target='old GT_VOSR; not the refreshed target'))


def main():
    """@luna-comment
    Input: run/train/smoke/monitor CLI. Output: locked controller or isolated delegated action.
    Key Steps: dispatch child actions; lock owned controller and preserve failure status/traceback.
    Key Params: run includes extraction, train resumes only after source publication.
    State/Dependencies: Linux flock; no background shell, subprocesses own exact output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'train', 'smoke', 'monitor'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action in ('smoke', 'monitor'):
        manager = bind_manager()
        if args.action == 'smoke':
            if not args.arm:
                parser.error('--arm required')
            manager.smoke(args.arm)
        else:
            manager.MODE = 'none; train GT: VOSR2u 20261023; validation GT: old VOSR 20261014 (583 pairs)'
            manager.monitor()
        return
    import fcntl
    from tools.build_offline_augmented_dataset import write_json
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(DATA / 'PIPELINE.json', dict(phase='extracting_and_preparing' if args.action == 'run' else 'building', pid=os.getpid()))
            if args.action == 'run':
                prepare_source()
            write_json(DATA / 'PIPELINE.json', dict(phase='building_then_smoking', pid=os.getpid()))
            train()
        except Exception as error:
            write_json(DATA / 'PIPELINE.json', dict(phase='failed', error=repr(error), pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
