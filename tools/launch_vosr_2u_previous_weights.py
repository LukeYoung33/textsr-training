"""Additional VOSR2u comparisons initialized from the previous trained block models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import manage_vosr_2u_block256 as current

RUN = current.DATA / 'training_comparison_prev_pretrain_20261023'


def manager():
    """@luna-comment
    Input: isolated previous-weight run constant. Output: reused smoke/observer with owned bindings.
    Key Steps: reuse current bank binding, override only run root and observer description.
    Key Params: same two architecture identifiers. State/Dependencies: process-local module globals.
    """
    result = current.bind_manager()
    result.RUN = RUN
    result.MODE = 'none; pretrain: previous block250K; train GT: VOSR2u; validation: old GT_VOSR (583 pairs)'
    return result


def run():
    """@luna-comment
    Input: today's complete bank/configs and old corresponding block250K checkpoints.
    Output: two strict-smoked weights-only runs, provenance, detached monitor and four-arm overview.
    Key Steps: lock own run, clone only three configuration fields, verify original jobs, smoke then launch.
    Key Params: GPU0/1 >=4096MiB free;250K/batch4 unchanged; no optimizer or scheduler resume.
    State/Dependencies: CUDA, yaml, old checkpoints read-only; existing processes never stopped.
    """
    import fcntl
    import yaml
    from tools.build_offline_augmented_dataset import digest, write_json
    RUN.mkdir(exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = current.RUN
        provenance = {}
        for arm, gpu in zip(current.ARMS, (0, 1)):
            old_receipt = json.loads((original / 'launch' / (arm + '.json')).read_text())
            command = (Path('/proc') / str(old_receipt['pid']) / 'cmdline').read_bytes()
            if str(original / 'configs' / (arm + '.yml')).encode() not in command:
                raise ValueError('Original-start process identity changed')
            cfg = yaml.safe_load((original / 'configs' / (arm + '.yml')).read_text())
            previous = yaml.safe_load((current.OLD / 'training_comparison_20261021/configs' / (arm + '.yml')).read_text())
            checkpoint = current.OLD / 'training_comparison_20261021/experiments' / previous['name'] / 'models/net_g_250000.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            cfg['name'] = 'vosr2u_prevblock_' + arm + '_250k_20261023'
            cfg['path']['pretrain_network_g'] = str(checkpoint)
            cfg['path']['experiments_root'] = str(RUN / 'experiments')
            assert cfg['path']['resume_state'] is None and cfg['path']['strict_load_g']
            path = RUN / 'configs' / (arm + '.yml')
            path.parent.mkdir(exist_ok=True)
            payload = yaml.safe_dump(cfg, sort_keys=False)
            if path.exists() and path.read_text() != payload:
                raise ValueError('Refusing changed comparison config')
            path.write_text(payload)
            provenance[arm] = dict(gpu=gpu, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                config_sha256=digest(path), initialization='weights_only_previous_block250K',
                validation_target='old_GT_VOSR_20261014', original_pair_pid=old_receipt['pid'])
        write_json(RUN / 'provenance.json', provenance)
        for arm, gpu in zip(current.ARMS, (0, 1)):
            receipt = RUN / 'launch' / (arm + '.json')
            if receipt.exists():
                raise ValueError('Already launched: ' + arm)
            free = int(subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free < 4096:
                raise RuntimeError('Insufficient free GPU memory: ' + str(gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (RUN / (arm + '_smoke.log')).open('ab') as stream:
                subprocess.run([sys.executable, '-u', str(__file__), 'smoke', '--arm', arm],
                    stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
            checked = json.loads((RUN / 'smoke' / (arm + '.json')).read_text())
            path = RUN / 'configs' / (arm + '.yml')
            assert checked['passed'] and checked['config_sha256'] == digest(path)
            cfg = yaml.safe_load(path.read_text())
            if (RUN / 'experiments' / cfg['name']).exists():
                raise ValueError('Existing experiment output')
            log = RUN / 'logs' / (arm + '.log')
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, str(ROOT / 'tools/train_vosr_block256.py'),
                       '--repo-root', '/home/guang.yang/workspace/Real-ESRGAN', '-opt', str(path)]
            with log.open('xb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                    stdin=subprocess.DEVNULL, cwd=str(RUN), start_new_session=True)
            write_json(receipt, dict(arm=arm, gpu=gpu, pid=process.pid, log=str(log), command=command,
                                    config_sha256=digest(path)))
        with (RUN / 'monitor.log').open('ab') as stream:
            observer = subprocess.Popen([sys.executable, '-u', str(__file__), 'monitor'],
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(RUN / 'PIPELINE.json', dict(phase='both_launched', monitor_pid=observer.pid))
        overview = '''<!doctype html><meta charset="utf-8"><title>VOSR2u four-arm comparison</title>
<style>body{font:16px system-ui;margin:24px;background:#eef2f6}iframe{width:100%;height:850px;border:1px solid #ccd5df;background:white}h2{margin-top:30px}</style>
<h1>VOSR2u: V3/V5 initialization comparison</h1><p>Same bank, augmentation,250K schedule and losses. Weights-only initialization; no optimizer resume. Validation uses old583 GT_VOSR pairs.</p>
<h2>A: Original architecture warm starts — GPU2/3</h2><a href="training_comparison_20261023/STATUS.html">Open A monitor</a><iframe src="training_comparison_20261023/STATUS.html"></iframe>
<h2>B: Previous trained block250K warm starts — GPU0/1</h2><a href="training_comparison_prev_pretrain_20261023/STATUS.html">Open B monitor</a><iframe src="training_comparison_prev_pretrain_20261023/STATUS.html"></iframe>'''
        (current.DATA / 'OVERVIEW.html').write_text(overview, encoding='utf-8')


def main():
    """@luna-comment
    Input: run/smoke/monitor action and optional architecture. Output: isolated launch or delegated check.
    Key Steps: validate smoke arm; delegate observer/optimizer; run controller otherwise.
    Key Params: original run remains unchanged. State/Dependencies: process-local manager overrides.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'smoke', 'monitor'))
    parser.add_argument('--arm', choices=current.ARMS)
    args = parser.parse_args()
    if args.action == 'run':
        run()
    elif args.action == 'smoke':
        if not args.arm:
            parser.error('--arm required')
        manager().smoke(args.arm)
    else:
        manager().monitor()


if __name__ == '__main__':
    main()
