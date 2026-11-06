"""Watch four V3 runs and evaluate final checkpoints on the frozen 52 HX photos."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/v3_native_four_20261015')
SOURCE = Path('/AI_DNS/guang.yang/TextSR/data/HX_testset')
TRACES = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/v4_formal_hx_20260903/c0_cp45/traces')
OUT = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/v3_native_four_hx_20261015')


def sha(path):
    """@luna-comment Input: asset path. Output: SHA256. Key Steps: stream1MiB chunks.
    Key Params: fixed chunk size. State/Dependencies: read-only file.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    """@luna-comment Input: destination/object. Output: atomic UTF8 JSON.
    Key Steps: temporary sibling then replace. Key Params: reject NaN.
    State/Dependencies: isolated evaluation outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def positions(length, canvas, stride):
    """@luna-comment Input: native axis extent/canvas/stride. Output: even covering origins.
    Key Steps: regular grid plus even-ceil last origin. Key Params: stride<=canvas.
    State/Dependencies: deterministic integer geometry.
    """
    last = max(0, ((length-canvas+1)//2)*2)
    return sorted(set(list(range(0, last, stride)) + [last]))


def restore(rgb, boxes, network, device, blend_source=None, box_feather=16):
    """@luna-comment
    Input: uint8 model-input RGB, clipped OCR rectangles, trained V3, torch device;
    optional same-size original RGB supplies the final feather blending base.
    Output: same-size RGB and diagnostic counts, untouched outside full boxes.
    Key Steps: shared native grid, overlap feather accumulation, full-box alpha blend.
    Key Params:96x480 stride64x448 batch8, patch16px ramp; box_feather defaults16px,
    zero defers source blending until after tonal postprocessing in the upright HX route.
    State/Dependencies: torch training color transforms, OpenCV/numpy; no GT or glyph mask.
    """
    import cv2
    import numpy as np
    import torch
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    h, w = rgb.shape[:2]
    base = rgb if blend_source is None else blend_source
    assert base.shape == rgb.shape and base.dtype == rgb.dtype
    mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = 1
    grid = [(x, y) for y in positions(h, 96, 64) for x in positions(w, 480, 448)
            if mask[y:min(h, y+96), x:min(w, x+480)].any()]
    totals = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    wy = np.minimum(np.minimum(np.arange(96)+1, 96-np.arange(96))/16, 1).astype(np.float32)
    wx = np.minimum(np.minimum(np.arange(480)+1, 480-np.arange(480))/16, 1).astype(np.float32)
    window = wy[:, None]*wx[None, :]
    for offset in range(0, len(grid), 8):
        selected = grid[offset:offset+8]
        patches = []
        for x, y in selected:
            patch = rgb[y:y+96, x:x+480]
            patch = np.pad(patch, ((0, 96-patch.shape[0]), (0, 480-patch.shape[1]), (0, 0)), mode='edge')
            patches.append(patch)
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2).copy()).to(device).float()/255
        with torch.inference_mode():
            prediction = network(rgb_to_y4uv(batch))
            decoded = YUV2RGB(y4uv_to_yuv444(prediction), is_train=True).clamp(0, 1)
        assert decoded.shape == batch.shape and torch.isfinite(decoded).all()
        values = decoded.permute(0, 2, 3, 1).cpu().numpy()
        for (x, y), value in zip(selected, values):
            hh, ww = min(96, h-y), min(480, w-x)
            totals[y:y+hh, x:x+ww] += value[:hh, :ww]*window[:hh, :ww, None]
            weights[y:y+hh, x:x+ww] += window[:hh, :ww]
    assert np.all(weights[mask > 0] > 0), 'Uncovered OCR support'
    predicted = totals/np.maximum(weights[..., None], 1e-12)
    # Explicit zero border gives deterministic distance even for boxes touching frame edges.
    distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    if box_feather < 0:
        raise ValueError('Negative box feather')
    t = np.clip((distance-1)/max(box_feather,1), 0, 1)
    alpha = t*t*(3-2*t) if box_feather else mask.astype(np.float32)
    result = np.clip(np.rint(base.astype(np.float32)*(1-alpha[..., None])+predicted*255*alpha[..., None]), 0, 255).astype(np.uint8)
    assert np.array_equal(result[mask == 0], base[mask == 0])
    return result, dict(patches=len(grid), boxes=len(boxes), outside_unchanged=True,
                        canvas_hw=[96, 480], stride_hw=[64, 448], box_feather=box_feather)


def evaluate(arm, checkpoint, *, preflight=False):
    """@luna-comment Input: arm/fixed checkpoint. Output:52 native results and receipts.
    Key Steps: explicit registry bootstrap, strict load, frozen traces, native inference.
    Key Params: preflight isolates first-image smoke. State/Dependencies: CUDA/read-only models.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    from basicsr.archs import build_network
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    files = sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    assert len(files) == 52, len(files)
    config = TRAIN/'configs'/f'{arm}.yml'
    opt = yaml.safe_load(config.read_text())
    net = build_network(opt['network_g']).cuda().eval()
    state = torch.load(checkpoint, map_location='cpu')
    net.load_state_dict(state['params'], strict=True)
    checkpoint_hash = sha(checkpoint)
    destination = OUT/('_preflight' if preflight else arm)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files[:1] if preflight else files, 1):
        trace_path = TRACES/(path.stem+'.json')
        trace = json.loads(trace_path.read_text())
        bgr = cv2.imread(str(path))
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        assert trace['source_hw'] == [h, w] and Path(trace['input']).resolve() == path.resolve()
        boxes = []
        for detection in trace['detections']:
            if detection['status'] != 'accepted':
                continue
            p = np.asarray(detection['polygon']).reshape(-1, 2)
            x0, y0 = np.maximum(np.floor(p.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int), [w, h])
            if x1 > x0 and y1 > y0:
                boxes.append([int(x0), int(y0), int(x1), int(y1)])
        result, details = restore(rgb, boxes, net, 'cuda')
        output = destination/(path.stem+'.png')
        assert cv2.imwrite(str(output), cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        scale = min(1, 960/max(h, w))
        preview = cv2.resize(result, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(destination/(path.stem+'.jpg')), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        rows.append(dict(source=str(path), source_sha256=sha(path), trace_sha256=sha(trace_path),
                         output=str(output), output_sha256=sha(output), **details))
        save_json(destination/'progress.json', dict(done=len(rows), expected=1 if preflight else 52))
    save_json(destination/'summary.json', dict(status='passed', arm=arm, checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash, images=len(rows), results=rows))


def gallery(jobs):
    """@luna-comment Input: completed job statuses. Output: Chinese linked comparison Markdown.
    Key Steps: collect52 source stems and four result columns. Key Params: previews link native PNG.
    State/Dependencies: completed summary receipts; no quality claims without paired GT.
    """
    names = {'01_0921data_degraded': '旧数据退化', '02_displayfit_paired': '新数据配对',
             '03_displayfit_degraded': '新数据退化', '04_displayfit_mixed75': '新数据混合75%'}
    lines = ['# V3 四组 HX 原图验证', '', '四组均使用最终250K权重。原尺寸96×480重叠块推理，完整文字框贴回，16像素边界羽化。',
             '', '使用固定OCR框。这里展示52张原始整图；没有配对GT，不报告PSNR或OCR准确率。', '',
             '| 输入原图 | '+' | '.join(names[a] for a in jobs)+' |', '|---|'+'---|'*len(jobs)]
    files = sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    for path in files:
        row = [f'[原图 {path.stem}]({str(path).replace("/AI_DNS/", "Y:/", 1)})']
        for arm in jobs:
            row.append(f'[![结果]({arm}/{path.stem}.jpg)]({arm}/{path.stem}.png)')
        lines.append('| '+' | '.join(row)+' |')
    (OUT/'对比报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def watch():
    """@luna-comment Input: four training launch receipts. Output: automatic final HX jobs/status.
    Key Steps: poll60s, require exited training plus completion log/exact250K, run each arm;
    report failures without substituting checkpoint, then generate comparison after all pass.
    Key Params: assigned original GPU, bounded threads; idempotent successful summaries.
    State/Dependencies: child PIDs, local status JSON; never stops or edits training jobs.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    launch = json.loads((TRAIN/'launch.json').read_text())
    states = {j['arm']: {'status': 'waiting_training', 'training_pid': j['pid']} for j in launch['jobs']}
    processes = {}
    while True:
        for job in launch['jobs']:
            arm = job['arm']
            if states[arm]['status'] in ('passed', 'failed'):
                continue
            summary = OUT/arm/'summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('status') == 'passed':
                states[arm]['status'] = 'passed'
                continue
            if arm in processes:
                code = processes[arm].poll()
                if code is not None:
                    states[arm].update(status='passed' if code == 0 and summary.exists() else 'failed', exit_code=code)
                continue
            process_stat = Path('/proc')/str(job['pid'])/'stat'
            if process_stat.exists() and process_stat.read_text().split()[2] != 'Z':
                command = process_stat.with_name('cmdline').read_bytes().decode(errors='replace')
                if 'train_v3_native_regions.py' in command and str(TRAIN/'configs'/f'{arm}.yml') in command:
                    continue
            log = Path(job['log']).read_text(errors='replace')
            checkpoint = TRAIN/'runs'/arm/'models/net_g_250000.pth'
            if 'End of training.' not in log or 'Traceback (most recent' in log or not checkpoint.exists():
                states[arm].update(status='failed', reason='training exited without clean250K completion')
                continue
            if time.time()-checkpoint.stat().st_mtime < 60:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            with (OUT/(arm+'.log')).open('ab') as stream:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--arm', arm,
                                         '--checkpoint', str(checkpoint)], cwd=ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            processes[arm] = child
            states[arm].update(status='evaluating', evaluation_pid=child.pid)
        save_json(OUT/'watch_status.json', dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), jobs=states))
        if all(s['status'] in ('passed', 'failed') for s in states.values()):
            if all(s['status'] == 'passed' for s in states.values()):
                gallery(states)
            return
        time.sleep(60)


def main():
    """@luna-comment Input: watch or arm/checkpoint/preflight CLI. Output: selected action.
    Key Steps: parse then dispatch. Key Params: explicit checkpoint for inference.
    State/Dependencies: watcher or CUDA evaluation, isolated output directories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm')
    parser.add_argument('--checkpoint')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.arm:
        if not args.checkpoint:
            parser.error('--checkpoint required with --arm')
        evaluate(args.arm, Path(args.checkpoint), preflight=args.preflight)
    else:
        watch()


if __name__ == '__main__':
    main()
