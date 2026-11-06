"""Prepare, smoke-test and launch only explicitly selected offline matrix arms."""
import argparse
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
DATA = Path(os.environ.get('OFFLINE_MATRIX_DATA', '/AI_DNS/guang.yang/TextSR/data/adaptive_preprocessed_augmented_v1_20261020'))
RUN = Path(os.environ.get('OFFLINE_MATRIX_RUN', str(DATA / 'training_comparison_20261020')))
TAG = os.environ.get('OFFLINE_MATRIX_TAG', '20261020')
REPO = '/home/guang.yang/workspace/Real-ESRGAN'
MATRIX_KIND = os.environ.get('OFFLINE_MATRIX_KIND', 'real_synthetic')
ARMS = (('01_v5_preprocessed', '02_v3_preprocessed', '03_v5_raw', '04_v3_raw')
        if MATRIX_KIND == 'preprocess_ab' else ('01_v5_real', '02_v3_real', '03_v5_synthetic', '04_v3_synthetic'))


def write_json(path, value):
    """@luna-comment
    Input: receipt path/value. Output: atomic JSON. Key Steps: replace sibling temp.
    Key Params: UTF8. State/Dependencies: own run directory only.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    """@luna-comment
    Input: file. Output: streamed SHA256. Key Steps: bounded read.
    Key Params: 4MiB. State/Dependencies: immutable inputs.
    """
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    """@luna-comment
    Input: selected matrix kind, approved bank configs and historical V5 configuration.
    Output: four matched immutable configs and provenance.
    Key Steps: retain losses/optimizer; select architecture and matching train/validation bank.
    Key Params: 250K steps, batch4, seed20260828; DATA/RUN/TAG and MATRIX_KIND environment settings.
    State/Dependencies: read-only warm starts; historical defaults remain available.
    """
    import yaml
    base_bank = 'bank_both_v1' if MATRIX_KIND == 'preprocess_ab' else 'bank_v1'
    base = yaml.safe_load((DATA / base_bank / 'train_paired.yml').read_text())
    v5 = yaml.safe_load((Path(REPO) / 'options/train_bucket_3/lenovo_v5_p17_effect_iteration_20260920/train_original_r1_bglf_uv_250k_20260920.yml').read_text())
    v5_checkpoint = '/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v5_p17_effect_iteration_20260920/lenovo_v5_p17_original_r1_bglf_uv_250k_20260920/models/net_g_250000.pth'
    provenance = {}
    (RUN / 'configs').mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        config = copy.deepcopy(base)
        if MATRIX_KIND == 'preprocess_ab' and arm.endswith('_raw'):
            config = yaml.safe_load((DATA / 'bank_raw_v1/train_paired.yml').read_text())
        config['name'] = 'adaptive_' + arm + '_250k_' + TAG
        config['path']['experiments_root'] = str(RUN / 'experiments')
        config['logger']['use_tb_logger'] = False
        config['logger']['print_freq'] = 100
        if '_v5_' in arm:
            config['network_g'] = v5['network_g']
            config['path']['pretrain_network_g'] = v5_checkpoint
        config['path']['strict_load_g'] = True
        if 'synthetic' in arm:
            synthetic = RUN / 'synthetic_bank_v1'
            config['datasets']['train'].update(root=str(synthetic), manifest=str(synthetic / 'train_balanced.json'))
        path = RUN / 'configs' / (arm + '.yml')
        payload = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != payload:
            raise ValueError('Refusing to overwrite a different matrix configuration')
        path.write_text(payload)
        checkpoint = config['path']['pretrain_network_g']
        provenance[arm] = dict(config_sha256=digest(path), checkpoint=checkpoint, checkpoint_sha256=digest(checkpoint), gpu=ARMS.index(arm))
    write_json(RUN / 'provenance.json', provenance)
    print(json.dumps(provenance), flush=True)


def smoke(arm):
    """@luna-comment
    Input: one immutable arm. Output: full active-loss optimizer and input-route receipt.
    Key Steps: strict-load model, select eligible and excluded metadata, run post-ramp update.
    Key Params: current_iter8000 exercises enabled auxiliary losses.
    State/Dependencies: CUDA; weights updated in memory only, no formal checkpoint saved.
    """
    from tools.train_offline_comparison import bootstrap
    bootstrap(REPO)
    import numpy as np
    import torch
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    config_path = RUN / 'configs' / (arm + '.yml')
    sys.argv = [sys.argv[0], '-opt', str(config_path)]
    opt, _ = parse_options(REPO, is_train=True)
    opt['dist'], opt['rank'], opt['world_size'] = False, 0, 1
    dataset = OfflineAugmentedRegionDataset(opt['datasets']['train'])
    refs = {}
    for shard_id, shard in enumerate(dataset.shards):
        records = json.loads((dataset.root / shard['metadata']).read_text())
        for sample_id, meta in enumerate(records):
            key = 'eligible' if meta['cp_eligible'] else ('vertical' if meta['orientation'] == 'VERTICAL' else meta['category'])
            if key in ('eligible', 'direction', 'neighbors', 'vertical') and key not in refs:
                refs[key] = (shard_id, sample_id)
        if len(refs) == 4:
            break
    if not {'eligible', 'direction', 'neighbors', 'vertical'} <= refs.keys():
        raise ValueError('Smoke requires ordinary, neighbor, direction and vertical examples')
    dataset.index = np.asarray(list(refs.values()), dtype=np.uint32)
    selected = [dataset[index] for index in range(len(refs))]
    while len(selected) < 4:
        selected.append(selected[0])
    batch = dataset.collate_fn(selected)
    model = build_model(opt)
    model.feed_data(batch)
    torch.testing.assert_close(model.lq, rgb_to_y4uv(batch['lq_rgb'].cuda()), rtol=0, atol=0)
    tracked = next(p for p in model.net_g.parameters() if p.requires_grad)
    before = tracked.detach().clone()
    started = time.time()
    model.optimize_parameters(8000)
    torch.cuda.synchronize()
    logs = {k: float(v) for k, v in model.get_current_log().items()}
    if not all(np.isfinite(value) for value in logs.values()):
        raise ValueError('Nonfinite active loss')
    if torch.equal(before, tracked.detach()):
        raise ValueError('Optimizer did not change weights')
    receipt = dict(arm=arm, passed=True, strict_load=True, optimizer_changed_weights=True,
                   exact_stored_input=True, current_iter=8000, losses=logs,
                   sample_categories=[s['augmentation_category'] for s in selected],
                   cp_eligible=[bool(s['cp_eligible']) for s in selected],
                   seconds=time.time()-started, max_memory_bytes=torch.cuda.max_memory_allocated(),
                   config_sha256=digest(config_path))
    write_json(RUN / 'smoke' / (arm + '.json'), receipt)
    print(json.dumps(receipt), flush=True)


def launch(arm):
    """@luna-comment
    Input: approved arm with matching passed smoke. Output: detached PID receipt.
    Key Steps: reject reused outputs; set one GPU; spawn isolated exact entry/config.
    Key Params: no auto_resume; threads2; experiment name read from exact config.
    State/Dependencies: preserves all prior jobs and versioned run outputs.
    """
    config = RUN / 'configs' / (arm + '.yml')
    checked = json.loads((RUN / 'smoke' / (arm + '.json')).read_text())
    if not checked['passed'] or checked['config_sha256'] != digest(config):
        raise ValueError('No matching smoke')
    receipt_path = RUN / 'launch' / (arm + '.json')
    import yaml
    experiment_name = yaml.safe_load(config.read_text())['name']
    if receipt_path.exists() or (RUN / 'experiments' / experiment_name).exists():
        raise ValueError('Arm already launched or experiment exists')
    log = RUN / 'logs' / (arm + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ARMS.index(arm)), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = [sys.executable, str(ROOT / 'tools/train_offline_comparison.py'), '--repo-root', REPO, '-opt', str(config)]
    with log.open('xb') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, start_new_session=True, cwd=str(RUN))
    receipt = dict(arm=arm, pid=proc.pid, gpu=ARMS.index(arm), log=str(log), command=command, config_sha256=digest(config), launched_at=time.time())
    write_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def main():
    """@luna-comment
    Input: action and optional arm. Output: selected concrete operation.
    Key Steps: dispatch prepare/smoke/launch. Key Params: four approved arm names.
    State/Dependencies: fixed isolated run root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'launch'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif not args.arm:
        parser.error('--arm required')
    elif args.action == 'smoke':
        smoke(args.arm)
    else:
        launch(args.arm)


if __name__ == '__main__':
    main()

