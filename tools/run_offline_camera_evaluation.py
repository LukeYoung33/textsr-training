"""Run approved camera pilot and full four-arm evaluation without training changes."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_offline_camera_matrix import ARMS, save_json


def main():
    """@luna-comment
    Input: staged approved evaluation package. Output: pilot/full results and completion page.
    Key Steps: prepare two-image pilot, strict-load all four arms, then full280 preparation
    and inference; require zero child exits before gallery publication.
    Key Params: one arm per GPU, original ViT contexts preserved; fixed frozen defaults.
    State/Dependencies: approved source inventory, remote Python/CUDA and own run logs.
    """
    training = Path('/AI_DNS/guang.yang/TextSR/data/adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020')
    output = training / 'evaluation/camera_final_20261021'
    old = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016')
    common = [sys.executable, str(ROOT / 'tools/evaluate_offline_camera_matrix.py')]
    options = ['--inventory', str(old / 'inventory.json'), '--detections', str(old / 'detections'),
               '--training-root', str(training), '--repo-root', '/home/guang.yang/workspace/Real-ESRGAN']
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    try:
        for phase, destination, limit in (('pilot', output / 'pilot', ['--limit', '2']), ('full', output, [])):
            save_json(output / 'pipeline_status.json', dict(phase=phase + '_preparation'))
            with (output / 'logs' / (phase + '_prepare.log')).open('ab') as stream:
                subprocess.run(common + ['prepare', *options, '--output', str(destination), *limit],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            jobs = []
            for gpu, arm in enumerate(ARMS):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                with (output / 'logs' / (phase + '_' + arm + '.log')).open('ab') as stream:
                    proc = subprocess.Popen(common + ['infer', *options, '--output', str(destination), '--arm', arm, *limit],
                                            env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                jobs.append((arm, proc))
            save_json(output / 'pipeline_status.json', dict(phase=phase + '_inference', jobs={arm:proc.pid for arm,proc in jobs}))
            codes = {arm:proc.wait() for arm,proc in jobs}
            if any(codes.values()):
                raise RuntimeError(phase + ' inference failed: ' + json.dumps(codes))
        subprocess.run(common + ['gallery', *options, '--output', str(output)], check=True)
        save_json(output / 'pipeline_status.json', dict(phase='complete', images=280, outputs=1120, gallery='index.html'))
    except Exception as error:
        save_json(output / 'pipeline_status.json', dict(phase='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
