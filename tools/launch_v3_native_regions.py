"""Launch only the four verified V3 jobs, without touching other GPU processes."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    """@luna-comment
    Input: verified experiment root. Output: four detached processes and launch receipts.
    Key Steps: reject duplicate launch; snapshot task sources/configs; assign two jobs
    per idle GPU; persist PID/log/config hashes after each successful spawn.
    Key Params: GPUs0/3, original batch4, bounded CPU threads; no existing job stops.
    State/Dependencies: subprocess environment, experiment logs and immutable snapshots.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    root = Path(__file__).resolve().parents[1]
    if (out/'launch.json').exists() or (out/'runs').exists():
        raise RuntimeError('Refusing duplicate launch into an existing run root')
    verified = json.loads((out/'verification.json').read_text())
    if len(verified) != 4 or not all(v['strict_checkpoint_loaded'] and v['weights_changed'] for v in verified.values()):
        raise RuntimeError('Four-arm GPU verification required')
    snapshot = out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    files = ['tools/train_v3_native_regions.py', 'tools/prepare_v3_native_regions.py',
             'tools/verify_v3_native_regions.py', 'tools/launch_v3_native_regions.py',
             'realesrgan/data/native_text_region_dataset.py', 'realesrgan/models/native_region_0921_model.py',
             'realesrgan/models/basic_text_line_y4uv_v4_gtdeg_model.py',
             'realesrgan/models/basic_text_line_y4uv_training_model_v3.py',
             'realesrgan/models/realesrnet_model.py', 'realesrgan/archs/basic_text_line_r4_student_y4uv_v3_arch.py']
    hashes = {}
    for name in files:
        source = root/name
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = {'started_at': datetime.now().isoformat(), 'source_sha256': hashes, 'jobs': []}
    for config, gpu in zip(sorted((out/'configs').glob('*.yml')), [0, 0, 3, 3]):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        log = out/(config.stem+'.launch.log')
        command = [sys.executable, '-u', str(root/'tools/train_v3_native_regions.py'), '-opt', str(config), '--launcher', 'none']
        with log.open('wb') as stream:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['jobs'].append(dict(arm=config.stem, pid=process.pid, gpu=gpu, log=str(log), command=command,
                                   config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
        (out/'launch.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
