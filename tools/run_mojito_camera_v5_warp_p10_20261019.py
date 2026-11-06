"""Rectify supplied camera text lines, run V5 and historical postprocessing, unwarp."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v5_warp_p10_20261019')
OLD = OUT.with_name('mojito_camera_v3_newdeg_20261016')
TRAIN = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/displayfit_4arm_20261014')
CHECKPOINT = TRAIN/'displayfit_v5_r1_bglf_uv_strict_combined_h96_250k_20261014/models/net_g_250000.pth'
CHECKPOINT_SHA = 'de3af40f67646e5335cfdfe6b794ea75ce6f666e1396fac305550a12a3fc9403'


def order_quad(points):
    """@luna-comment
    Input: four finite original-frame XY vertices forming a convex quadrilateral.
    Output: float32 clockwise TL/TR/BR/BL points in image coordinates.
    Key Steps: convex-hull validation, centroid angle order, stable upper-left start.
    Key Params: reject degenerate/nonconvex detections instead of silent box fallback.
    State/Dependencies: pure OpenCV/NumPy geometry, original points unchanged.
    """
    p = np.asarray(points, np.float32).reshape(4, 2)
    assert np.isfinite(p).all()
    hull = cv2.convexHull(p).reshape(-1, 2)
    assert len(hull) == 4 and abs(cv2.contourArea(hull)) > 1
    center = hull.mean(0)
    p = hull[np.argsort(np.arctan2(hull[:, 1]-center[1], hull[:, 0]-center[0]))]
    start = int(np.argmin(p.sum(1)))
    return np.roll(p, -start, axis=0).astype(np.float32)


def rectify(image, points):
    """@luna-comment
    Input: native HWC uint8 image and original-coordinate quadrilateral.
    Output: horizontalized crop and reversible geometry/rotation receipt.
    Key Steps: derive native edge-length canvas, perspective warp, rotate tall lines CW.
    Key Params: height>=1.5*width; cubic interpolation, replicated border, no height fit.
    State/Dependencies: pure OpenCV geometry; inverse transform retained for pasteback.
    """
    p = order_quad(points)
    w = max(2, int(np.ceil(max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))))+1)
    h = max(2, int(np.ceil(max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))))+1)
    dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    matrix = cv2.getPerspectiveTransform(p, dst)
    assert np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-12
    patch = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = h >= 1.5*w
    if rotated:patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch, dict(quad=p.tolist(), native_hw=[h, w], model_line_hw=list(patch.shape[:2]),
                       rotated_clockwise=rotated, matrix=matrix.tolist(), inverse=np.linalg.inv(matrix).tolist())


def feather(height, width, radius=16):
    """@luna-comment
    Input: positive rectangle dimensions and native feather radius.
    Output: float32 full-rectangle smoothstep alpha, zero on outer boundary.
    Key Steps: nearest-edge distance with radius bounded for narrow text lines.
    Key Params:16pixels; tiny2pixel rectangles retain a positive half-pixel window.
    State/Dependencies: deterministic NumPy arrays, no glyph support mask.
    """
    yy, xx = np.mgrid[:height, :width]
    distance = np.minimum.reduce((xx, yy, width-1-xx, height-1-yy)).astype(np.float32)
    if min(height, width) <= 2:distance += .5
    effective = max(.5, min(float(radius), (min(height, width)-1)/2))
    t = np.clip(distance/effective, 0, 1)
    return t*t*(3-2*t)


def inverse_layers(model, postprocessed, geometry, frame_hw):
    """@luna-comment
    Input: model/postprocessed horizontal BGR lines, geometry and original frame HW.
    Output: bounded ROI, premultiplied model+alpha, premultiplied post delta, support.
    Key Steps: reverse vertical rotation, rectangle feather, inverse warp to small ROI,
    clamp support to original full quadrilateral; interpolate premultiplied values.
    Key Params: linear inverse interpolation, constant-zero border, native16pixel feather.
    State/Dependencies: saved inverse matrix, no original pixels or global state modified.
    """
    if geometry['rotated_clockwise']:
        model = cv2.rotate(model, cv2.ROTATE_90_COUNTERCLOCKWISE)
        postprocessed = cv2.rotate(postprocessed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = geometry['native_hw']
    assert model.shape[:2] == postprocessed.shape[:2] == (h, w)
    points = np.asarray(geometry['quad'], np.float32)
    fh, fw = frame_hw
    lo = np.maximum(np.floor(points.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(points.max(0)).astype(int)+1, [fw, fh])
    x0, y0 = map(int, lo);x1, y1 = map(int, hi)
    assert x1 > x0 and y1 > y0
    matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)@np.asarray(geometry['inverse'])
    alpha = feather(h, w)
    rgba = np.dstack((model.astype(np.float32)*alpha[:, :, None], alpha))
    delta = (postprocessed.astype(np.float32)-model.astype(np.float32))*alpha[:, :, None]
    warped = cv2.warpPerspective(rgba, matrix, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_delta = cv2.warpPerspective(delta, matrix, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    support = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.fillConvexPoly(support, np.rint(points-[x0, y0]).astype(np.int32), 1)
    warped *= support[:, :, None];warped_delta *= support[:, :, None]
    return (x0, y0, x1, y1), warped, warped_delta, support


def infer_line(image, network, torch):
    """@luna-comment
    Input: horizontal uint8 BGR text line, strict V5 network and torch module.
    Output: same-size BGR restoration and count of native model patches.
    Key Steps: native overlapping96x480 crops, edge pad, training color transforms,
    batched inference, finite checks and positive weighted patch accumulation.
    Key Params: stride64x448, batch8,16pixel positive tile ramps, no scale normalization.
    State/Dependencies: CUDA process and unchanged training RGB/Y4UV transforms.
    """
    from tools.auto_hx_v3_native import positions
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    grid = [(x, y) for y in positions(h, 96, 64) for x in positions(w, 480, 448)]
    sums = np.zeros((h, w, 3), np.float32);weights = np.zeros((h, w), np.float32)
    wy = np.minimum(np.minimum(np.arange(96)+1, 96-np.arange(96))/16, 1)
    wx = np.minimum(np.minimum(np.arange(480)+1, 480-np.arange(480))/16, 1)
    window = (wy[:, None]*wx[None, :]).astype(np.float32)
    for offset in range(0, len(grid), 8):
        chosen = grid[offset:offset+8];patches = []
        for x, y in chosen:
            crop = rgb[y:y+96, x:x+480]
            patches.append(np.pad(crop, ((0, 96-crop.shape[0]), (0, 480-crop.shape[1]), (0, 0)), mode='edge'))
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2).copy()).cuda().float()/255
        with torch.inference_mode():
            result = YUV2RGB(y4uv_to_yuv444(network(rgb_to_y4uv(batch))), is_train=True).clamp(0, 1)
        assert result.shape == batch.shape and torch.isfinite(result).all()
        for (x, y), pixels in zip(chosen, result.permute(0, 2, 3, 1).cpu().numpy()):
            hh, ww = min(96, h-y), min(480, w-x)
            sums[y:y+hh, x:x+ww] += pixels[:hh, :ww]*window[:hh, :ww, None]
            weights[y:y+hh, x:x+ww] += window[:hh, :ww]
    assert (weights > 0).all()
    pixels = np.clip(np.rint(sums/weights[:, :, None]*255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), len(grid)


def worker(rank, shards, limit):
    """@luna-comment
    Input: hash-verified supplied P10 PNGs, source-bound cached quads and final V5.
    Output: native model/final PNGs, preview stages, line debug and geometry receipts.
    Key Steps: strict bootstrap/load; rectify/rotate/infer; historical conservative
    V5 postprocessing and32px residual fade; inverse-rotate/warp; weighted overlap;
    full-frame board protection for post delta and exact unchanged outside support.
    Key Params: user-approved native geometry, vertical1.5ratio, final250K checkpoint.
    State/Dependencies: isolated worker outputs; read-only sources/historical helpers.
    """
    import importlib
    import torch
    import yaml
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch')
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import sha, save_json
    post_path = OUT/'code/postprocess_lenovo_borderrail_normal_v5_20260929.py'
    spec = importlib.util.spec_from_file_location('camera_historical_post', post_path)
    post = importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
    _process_every_box, _blend_weight, _board_frame_weight = post._process_every_box, post._blend_weight, post._board_frame_weight
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    config = TRAIN/'configs/train_strict_combined_h96_250k_20261014.yml'
    net = build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    net.load_state_dict(torch.load(CHECKPOINT, map_location='cpu')['params'], strict=True)
    assert sha(ROOT/'tools/text_lq_scurve_mild.py') == '950b2a4854041e09bc7394982d6f741b181ef1c35f2dedfd31f3d4a74fdd9fd5'
    snapshot = json.loads((OUT/'code/snapshot.json').read_text())
    assert sha(post_path) == snapshot['postprocess_sha256']
    code_hashes = {str(p): sha(p) for p in [Path(__file__), ROOT/'tools/text_lq_scurve_mild.py', post_path]}
    rows = json.loads((OUT/'inventory.json').read_text())['images'][:limit]
    for name in ('model', 'final', 'receipts', 'previews', 'lines'):(OUT/name).mkdir(parents=True, exist_ok=True)
    for row in rows[rank::shards]:
        started = time.perf_counter()
        receipt = OUT/'receipts'/(Path(row['name']).stem+'.json')
        original_path = OLD/'input'/row['original_name'];input_path = OUT/'input'/row['name']
        assert sha(original_path) == row['original_sha256'] and sha(input_path) == row['sha256']
        source = cv2.imread(str(original_path));image = cv2.imread(str(input_path))
        assert source is not None and image is not None and source.shape == image.shape
        assert list(image.shape[:2]) == [row['height'], row['width']]
        detection_path = OLD/'detections'/(Path(row['name']).stem+'.json')
        detection = json.loads(detection_path.read_text())
        assert detection['source_sha256'] == row['original_sha256'] and detection['source_hw'] == list(source.shape[:2])
        h, w = image.shape[:2]
        sums = np.zeros((h, w, 3), np.float32);post_sums = np.zeros_like(sums)
        weights = np.zeros((h, w), np.float32);support = np.zeros((h, w), np.uint8)
        lines = []
        for index, points in enumerate(detection['polygons']):
            line, geometry = rectify(image, points)
            model, tiles = infer_line(line, net, torch)
            candidate, report = _process_every_box(model, background_sigma=4.0, ambiguity_mode='conservative_edge', delta_filter=False)
            post_weight = _blend_weight(model)
            after = np.clip(np.rint(model.astype(np.float32)+(candidate.astype(np.float32)-model)*post_weight[:, :, None]), 0, 255).astype(np.uint8)
            box, rgba, delta, mask = inverse_layers(model, after, geometry, (h, w))
            x0, y0, x1, y1 = box
            sums[y0:y1, x0:x1] += rgba[:, :, :3];weights[y0:y1, x0:x1] += rgba[:, :, 3]
            post_sums[y0:y1, x0:x1] += delta;support[y0:y1, x0:x1] |= mask
            line_report = dict(index=index, **geometry, patches=tiles, postprocess=report,
                               post_raw_changed_pixels=int(np.any(candidate != model, axis=2).sum()),
                               post_blended_changed_pixels=int(np.any(after != model, axis=2).sum()))
            if int(row['id']) <= 2 or (geometry['rotated_clockwise'] and sum(r['rotated_clockwise'] for r in lines) < 3):
                folder = OUT/'lines'/row['id'];folder.mkdir(exist_ok=True)
                for label, pixels in [('input', line), ('model', model), ('post', after)]:
                    assert cv2.imwrite(str(folder/f'{index:03d}_{label}.png'), pixels)
                line_report['debug_dir'] = str(folder)
            lines.append(line_report)
        denominator = np.maximum(weights, 1)[:, :, None]
        base = source.astype(np.float32)
        model_float = base+(sums-base*weights[:, :, None])/denominator
        model_output = np.clip(np.rint(model_float), 0, 255).astype(np.uint8)
        frame_weight, board_report = _board_frame_weight(model_output)
        final = np.clip(np.rint(model_float+post_sums/denominator*frame_weight[:, :, None]), 0, 255).astype(np.uint8)
        assert np.array_equal(model_output[support == 0], source[support == 0])
        assert np.array_equal(final[support == 0], source[support == 0])
        for label, pixels in [('model', model_output), ('final', final)]:
            assert cv2.imwrite(str(OUT/label/row['name']), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        factor = min(1, 960/max(h, w));size = (round(w*factor), round(h*factor))
        for label, pixels in [('source', source), ('preprocessed', image), ('model', model_output), ('final', final)]:
            assert cv2.imwrite(str(OUT/'previews'/(row['id']+'_'+label+'.jpg')), cv2.resize(pixels, size, interpolation=cv2.INTER_AREA))
        item = dict(id=row['id'], name=row['original_name'], width=w, height=h,
                    input=str(input_path), input_sha256=row['sha256'], source=str(original_path), source_sha256=row['original_sha256'],
                    model=str(OUT/'model'/row['name']), output=str(OUT/'final'/row['name']),
                    model_sha256=sha(OUT/'model'/row['name']), output_sha256=sha(OUT/'final'/row['name']),
                    checkpoint_sha256=CHECKPOINT_SHA, code_hashes=code_hashes, detection_sha256=sha(detection_path),
                    outside_unchanged=True, polygons=detection['polygons'], text_boxes=detection['boxes'],
                    lines=lines, rotated_lines=sum(r['rotated_clockwise'] for r in lines),
                    postprocess_changed_pixels=int(np.any(final != model_output, axis=2).sum()),
                    board_protection=board_report, seconds=time.perf_counter()-started)
        save_json(receipt, item)
        print('RESTORED', row['id'], 'lines', len(lines), 'vertical', item['rotated_lines'], 'seconds', round(item['seconds'], 2), flush=True)


def supervise():
    """@luna-comment
    Input: successful geometry/model pilot and complete hash-verified280 PNG inventory.
    Output: detached worker status and all280 audited completion summary.
    Key Steps: launch four isolated GPU shards, stop only owned children on failure,
    verify every native output hash/source identity/postprocess receipt, publish pass.
    Key Params: GPUs0..3, two CPU threads each; no skip/fallback on image failure.
    State/Dependencies: original assets read-only; isolated task logs/status files.
    """
    from tools.auto_hx_v3_native import save_json, sha
    jobs = []
    for rank in range(4):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        with (OUT/f'worker_{rank}.log').open('ab') as stream:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--mode', 'worker', '--rank', str(rank), '--shards', '4'],
                                     cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        jobs.append(child)
    save_json(OUT/'launch.json', dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), pids=[p.pid for p in jobs]))
    while True:
        states = [p.poll() for p in jobs];failed = any(code not in (None, 0) for code in states)
        save_json(OUT/'status.json', dict(status='failed' if failed else 'running', completed=len(list((OUT/'receipts').glob('*.json'))),
                  expected=280, exit_codes=states, updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        if failed:
            for child in jobs:
                if child.poll() is None:child.terminate()
            raise RuntimeError(states)
        if all(code == 0 for code in states):break
        time.sleep(15)
    inventory = json.loads((OUT/'inventory.json').read_text())['images']
    records = [json.loads((OUT/'receipts'/(Path(r['name']).stem+'.json')).read_text()) for r in inventory]
    assert len(records) == len({r['id'] for r in records}) == 280
    for row, record in zip(inventory, records):
        assert row['sha256'] == record['input_sha256'] and record['outside_unchanged']
        assert sha(record['input']) == record['input_sha256']
        assert sha(record['model']) == record['model_sha256'] and sha(record['output']) == record['output_sha256']
    save_json(OUT/'summary.json', dict(status='passed', images=280, checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
              postprocess='historical_v5_conservative_edge_raw_delta', rotated_lines=sum(r['rotated_lines'] for r in records),
              total_lines=sum(len(r['lines']) for r in records), results=records))
    save_json(OUT/'status.json', dict(status='passed', completed=280, expected=280, exit_codes=states, completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print('PASS all280 rectified V5/postprocess', flush=True)


def main():
    """@luna-comment
    Input: run/worker mode and disjoint shard/limit CLI.
    Output: selected evaluation stage and explicit exit status.
    Key Steps: validate shard indices then dispatch; no implicit model/preprocessor choice.
    Key Params:280image default; pilot limit supports real path validation.
    State/Dependencies: task-owned constants and checked original assets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['run', 'worker'], default='run')
    parser.add_argument('--rank', type=int, default=0);parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=280)
    args = parser.parse_args();assert 0 <= args.rank < args.shards
    if args.mode == 'run':supervise()
    else:worker(args.rank, args.shards, args.limit)


if __name__ == '__main__':main()
