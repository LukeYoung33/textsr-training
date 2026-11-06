"""Prepare, verify, launch and observe the dedicated Compact128-to256 SR comparison."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data/vosr2u_compact_sr_x2_20261023')
BASE = DATA.parent / 'vosr_2u_block256_v1_20261023'
BANK = BASE / 'bank_raw_v1'
RUN = DATA / 'training_20261023'
CONFIG = RUN / 'configs/07_v5_compact_sr2x.yml'
ARM = '07_v5_compact_sr2x'
REPO = '/home/guang.yang/workspace/Real-ESRGAN'
GPU = 1


def prepare():
    """@luna-comment
    Input: complete newest HR bank and current V5 block config. Output: SR config and hashed fresh initial state.
    Key Steps: keep data/loss schedule, select original-HR skip variant, seed zero residual heads and count fused MACs.
    Key Params:RGB128->256,250K,batch4; no old restoration weights; original validation583 targets.
    State/Dependencies:staged bootstrap, torch/yaml; writes only owned SR root.
    """
    import torch
    import yaml
    from tools.train_vosr2u_sr import bootstrap
    from tools.build_offline_augmented_dataset import digest, write_json
    bootstrap(REPO)
    from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_compact_sr_hrskip_arch import BasicTextLineR4StudentY4UVV5CompactSRHRSkip
    complete = json.loads((BANK / 'COMPLETE.json').read_text())
    assert complete['complete'] and complete['shape_hw'] == [256, 256]
    cfg = yaml.safe_load((BASE / 'training_comparison_20261023/configs/05_v5_block256_raw.yml').read_text())
    cfg['name'] = 'vosr2u_v5_compact_sr128_to256_hrskip_250k_20261023'
    cfg['model_type'] = cfg['train']['model_type'] = 'SR2uBlockModel'
    cfg['scale'] = 2
    cfg['network_g'] = dict(type='BasicTextLineR4StudentY4UVV5CompactSRHRSkip', canvas_hw=[128, 128])
    initial = DATA / 'initial_state.pth'
    if initial.exists() or CONFIG.exists():
        raise ValueError('Refusing to overwrite an existing SR initialization/config')
    torch.manual_seed(cfg['manual_seed'])
    net = BasicTextLineR4StudentY4UVV5CompactSRHRSkip((128, 128))
    torch.save(dict(params=net.state_dict()), initial)
    fused = copy.deepcopy(net).eval().switch_to_deploy()
    assert fused.deployment_audit()['passed']
    macs = []

    def count(layer, inputs, output):
        """@luna-comment
        Input: executed fused convolution and tensors. Output: appended dense convolution MAC count.
        Key Steps: use output volume for regular conv and input volume for transpose conv, include groups.
        Key Params:batch1; fixed zeros counted; excludes interpolation/additions/memory.
        State/Dependencies:prepare-local mac list and temporary torch hooks only.
        """
        kernel = layer.kernel_size[0]*layer.kernel_size[1]
        macs.append((inputs[0].numel()*layer.out_channels if isinstance(layer, torch.nn.ConvTranspose2d)
                     else output.numel()*layer.in_channels)*kernel//layer.groups)

    hooks = [m.register_forward_hook(count) for m in fused.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    with torch.no_grad():
        fused(torch.rand(1,6,64,64), skip_input_y4uv=torch.rand(1,6,128,128))
    for hook in hooks:
        hook.remove()
    assert sum(macs) == 23470080
    cfg['path'].update(pretrain_network_g=str(initial), resume_state=None, strict_load_g=True,
                       experiments_root=str(RUN / 'experiments'))
    for split in ('train', 'val'):
        cfg['datasets'][split].update(type='OfflineSR2uDataset', name='vosr2u_sr_'+split)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(RUN / 'provenance.json', dict(config_sha256=digest(CONFIG), initial_sha256=digest(initial),
        bank_build_id=complete['build_id'], initialization='fresh_seeded_features_zero_sr_heads_original_HR_LQ_identity',
        downsample='RGB float2x area average before packing; LQ only', input_rgb=[128, 128], output_rgb=[256, 256],
        gpu=GPU, validation_target='old_GT_VOSR_20261014_583_pairs', macs=sum(macs), skip='original_HR_LQ_no_interpolation',
        comparable_v3_output256_macs=24293376))


def smoke():
    """@luna-comment
    Input: strict initialized SR config and real augmented bank. Output: actual LR/HR CUDA learning/validation receipt.
    Key Steps: cover CP/background/multiline/border; verify exact original-HR identity and area input; two updates.
    Key Params:iter0 then8000; verify first heads then deep feature gradients, all HR losses and validation.
    State/Dependencies:CUDA/real data; smoke never saves its updated weights into formal initialization.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tools.train_vosr2u_sr import bootstrap
    from tools.build_offline_augmented_dataset import digest, write_json
    bootstrap(REPO)
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.data.offline_sr2u_dataset import OfflineSR2uDataset
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    sys.argv = [sys.argv[0], '-opt', str(CONFIG)]
    opt, _ = parse_options(REPO, is_train=True)
    opt.update(dist=False, rank=0, world_size=1)
    dataset = OfflineSR2uDataset(opt['datasets']['train'])
    refs = {}
    for si, shard in enumerate(dataset.shards):
        for ri, record in enumerate(json.loads((BANK / shard['metadata']).read_text())):
            for key, yes in [('cp', bool(record['cp_lines'])), ('background', record['background_only']),
                             ('multiline', record['visible_ocr_rows'] > 3), ('border', record['real_hw'] != [256, 256])]:
                if yes:
                    refs.setdefault(key, (si, ri))
        if len(refs) == 4:
            break
    assert len(refs) == 4
    dataset.index = np.asarray([refs[k] for k in ('cp', 'background', 'multiline', 'border')], np.uint32)
    samples = [dataset[i] for i in range(4)]
    for i, sample in enumerate(samples):
        si, ri = map(int, dataset.index[i])
        array, _ = dataset._load(si)
        rgb = torch.from_numpy(array[ri][..., [2, 1, 0]].copy().transpose(2, 0, 1)).float()/255
        reference = (rgb[:, 0::2, 0::2] + rgb[:, 0::2, 1::2] + rgb[:, 1::2, 0::2] + rgb[:, 1::2, 1::2])/4
        torch.testing.assert_close(sample['lq_rgb'], reference, atol=1e-7, rtol=1e-6)
    batch = dataset.collate_fn(samples)
    model = build_model(opt)
    model.feed_data(batch)
    assert tuple(model.lq.shape) == (4, 6, 64, 64) and tuple(model.gt.shape) == (4, 6, 128, 128)
    torch.testing.assert_close(model.lq, rgb_to_y4uv(batch['lq_rgb'].cuda()), atol=0, rtol=0)
    torch.testing.assert_close(model.gt, rgb_to_y4uv(batch['gt_rgb'].cuda()), atol=0, rtol=0)
    with torch.no_grad():
        reference = rgb_to_y4uv(batch['original_lq_rgb'].cuda())
        torch.testing.assert_close(model.original_lq, reference, atol=0, rtol=0)
        torch.testing.assert_close(model._forward_v5(model.lq), reference, atol=0, rtol=0)
    heads = [model.net_g.local_head.weight, model.net_g.context_head.weight]
    before = [p.detach().clone() for p in heads]
    model.optimize_parameters(0)
    assert all(not torch.equal(a, b.detach()) for a, b in zip(before, heads))
    model.optimize_parameters(8000)
    logs = {k: float(v) for k, v in model.get_current_log().items()}
    assert all(np.isfinite(v) for v in logs.values()) and logs['cp_accepted_lines'] > 0
    for block in (model.net_g.local_pre, model.net_g.stem, *model.net_g.stage1, *model.net_g.stage2):
        grad = block.conv3x3.weight.grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0
    assert tuple(model.output.shape) == (4, 6, 128, 128)
    validation = OfflineSR2uDataset(opt['datasets']['val'])
    validation.index = validation.index[:4]
    loader = torch.utils.data.DataLoader(validation, batch_size=2, collate_fn=validation.collate_fn)
    model.nondist_validation(loader, 8000, None, False)
    torch.cuda.synchronize()
    write_json(RUN / 'smoke.json', dict(passed=True, config_sha256=digest(CONFIG),
        strict_load=True, lr_rgb_shape=[4, 3, 128, 128], packed_input=[4, 6, 64, 64],
        packed_output_target=[4, 6, 128, 128], original_hr_identity=True, hr_masks_cp=True, actual_optimizer_updates=2,
        head_then_feature_learning=True, validation_samples=4, losses=logs,
        max_memory_bytes=torch.cuda.max_memory_allocated()))


def monitor():
    """@luna-comment
    Input:owned SR launch/log and checkpoint. Output: live loss/ETA JSON/HTML every30seconds.
    Key Steps:validate PID command, parse progress, estimate speed, publish atomic snapshots.
    Key Params:250K; completion requires end log and final checkpoint; old validation target labeled.
    State/Dependencies:read-only train process/log probes; writes own status files only.
    """
    import fcntl
    from tools.manage_vosr_block256 import estimate_remaining
    from tools.offline_loss_charts import read_loss_history, render_loss_charts
    from tools.build_offline_augmented_dataset import write_json
    with (RUN / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(RUN / 'monitor.json', dict(pid=os.getpid()))
        while True:
            receipt = json.loads((RUN / 'launch.json').read_text())
            path = Path(receipt['log'])
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 262144))
                text = stream.read().decode(errors='replace')
            matches = re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)', text)
            step = int(matches[-1][0].replace(',', '')) if matches else 0
            command = Path('/proc') / str(receipt['pid']) / 'cmdline'
            alive = command.exists() and str(CONFIG).encode() in command.read_bytes()
            status = ('running' if step else 'starting') if alive else 'exited'
            if 'Traceback (most recent call last)' in text:
                status = 'failed'
            final = RUN / 'experiments/vosr2u_v5_compact_sr128_to256_hrskip_250k_20261023/models/net_g_250000.pth'
            if not alive and 'End of training' in text and final.exists():
                status, step = 'completed', 250000
            now = datetime.now(timezone(timedelta(hours=8)))
            item = dict(pid=receipt['pid'], gpu=GPU, iteration=step, status=status)
            item.update(estimate_remaining(step, status, [m[1] for m in matches], now))
            state = dict(updated_at=now.isoformat(), arms={ARM: item}, loss_history={ARM: read_loss_history(path)})
            write_json(RUN / 'STATUS.json', state)
            seconds = item['estimated_remaining_seconds']
            minutes = (seconds + 59)//60 if seconds is not None else None
            duration = f'{minutes//60}h {minutes%60}m' if minutes is not None else 'pending'
            finish = item['estimated_finish_at'] or 'pending'
            page = '<!doctype html><meta charset="utf-8"><title>Compact SR128 to256</title><style>body{font:16px system-ui;margin:28px;background:#f4f7fa}button{padding:10px}</style><h1>Compact X2 SR: RGB128 →256</h1><p>Output = original RGB256 LQ + learned residual. Only feature input is downsampled; no resized skip. Newest VOSR2u GT; old583-pair validation; fresh seeded initialization.</p>'
            page += f'<p>GPU{GPU} · {step:,}/250,000 · {status}</p><p>Estimated remaining: <b>{duration}</b> · Finish (Beijing): {html.escape(finish)}</p><p>Recent-speed estimate; validation/save/load variability may delay completion.</p>'
            page += '<button onclick="location.reload()">Refresh now</button><p>Auto refresh30s · '+state['updated_at']+'</p>'+render_loss_charts(state['loss_history'])+'<script>setTimeout(()=>location.reload(),30000)</script>'
            temp = RUN / 'STATUS.html.tmp'
            temp.write_text(page, encoding='utf-8')
            temp.replace(RUN / 'STATUS.html')
            if status in ('completed', 'failed', 'exited'):
                return
            time.sleep(30)


def run():
    """@luna-comment
    Input:approved single SR comparison. Output: strict-smoked detached training/observer and receipts.
    Key Steps:lock isolated root, prepare, smoke on shared GPU1, preserve four original command identities.
    Key Params:>=4096MiB GPU headroom; no existing training stopped; fresh250K run.
    State/Dependencies:Linux flock/CUDA; immutable original bank and own staged source.
    """
    import fcntl
    from tools.build_offline_augmented_dataset import write_json
    DATA.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(exist_ok=True)
    with (DATA / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = {}
        for run_name in ('training_comparison_20261023', 'training_comparison_prev_pretrain_20261023'):
            for receipt in (BASE / run_name / 'launch').glob('*.json'):
                value = json.loads(receipt.read_text())
                prior[value['pid']] = (Path('/proc') / str(value['pid']) / 'cmdline').read_bytes()
                assert str(BASE / run_name / 'configs' / (value['arm']+'.yml')).encode() in prior[value['pid']]
        free = int(subprocess.check_output(['nvidia-smi', '--id='+str(GPU), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 4096:
            raise RuntimeError('Insufficient GPU headroom')
        prepare()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (RUN / 'smoke.log').open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(__file__), 'smoke'], stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        with (RUN / 'train.log').open('xb') as stream:
            cmd = [sys.executable, '-u', str(ROOT / 'tools/train_vosr2u_sr.py'), '--repo-root', REPO, '-opt', str(CONFIG)]
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=env, start_new_session=True, cwd=str(RUN))
        write_json(RUN / 'launch.json', dict(pid=process.pid, gpu=GPU, command=cmd, log=str(RUN / 'train.log')))
        for pid, cmdline in prior.items():
            assert (Path('/proc') / str(pid) / 'cmdline').read_bytes() == cmdline
        with (RUN / 'monitor.log').open('ab') as stream:
            observer = subprocess.Popen([sys.executable, '-u', str(__file__), 'monitor'], stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(DATA / 'PIPELINE.json', dict(phase='launched', pid=process.pid, monitor_pid=observer.pid,
                                               preserved_training_pids=list(prior)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'smoke', 'monitor'))
    args = parser.parse_args()
    dict(run=run, smoke=smoke, monitor=monitor)[args.action]()
