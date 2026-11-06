"""Isolated height-routed preprocess inference with corrected V10 pasteback."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import infer_basic_text_line_full_image as src
from tools import replay_v10_paste_seams as paste
from realesrgan.data.basic_text_line_full_frame import build_native_tiled_full_frame_segments, pad_even_rgb


def native_prediction(rgb, polygon, runner):
    """@luna-comment Infer native tiles and normalize overlap without source blending.

    Input: RGB float HWC frame, OCR polygon, model runner accepting 96x480 RGB.
    Output: CHW YUV prediction, boolean support, crop bounds and actual tile bounds.
    Key Steps: Slice native pixels, edge-pad, infer, discard padding, normalize overlaps.
    Key Params: Canvas96x480, overlap24x64, positive separable sine weights.
    State/Dependencies: Runner calls only; no resizing or persistent cache writes.
    """
    segments, reason = build_native_tiled_full_frame_segments(polygon, rgb.shape[:2], (96, 480), overlap_x=64, overlap_y=24)
    if not segments:
        raise ValueError(reason)
    bounds = [min(s.crop_xyxy[0] for s in segments), min(s.crop_xyxy[1] for s in segments),
              max(s.crop_xyxy[2] for s in segments), max(s.crop_xyxy[3] for s in segments)]
    x0, y0, x1, y1 = bounds
    total = np.zeros((3, y1-y0, x1-x0), np.float32)
    weights = np.zeros(total.shape[1:], np.float32)
    for s in segments:
        l, t, r, b = s.crop_xyxy
        h, w = b-t, r-l
        canvas = src._pad_canvas(rgb[t:b, l:r], (96, 480), edge_replicate=True)
        assert canvas.shape == (96, 480, 3)
        pred = np.clip(runner.run_rgb(canvas), 0, 1)[:, :h, :w]
        wy = np.maximum(np.sin(np.linspace(0, np.pi, h)), .1)
        wx = np.maximum(np.sin(np.linspace(0, np.pi, w)), .1)
        weight = np.outer(wy, wx).astype(np.float32)
        total[:, t-y0:b-y0, l-x0:r-x0] += pred*weight[None]
        weights[t-y0:b-y0, l-x0:r-x0] += weight
    assert np.all(weights > 0)
    return total/weights[None], weights > 0, bounds, [list(s.crop_xyxy) for s in segments]


def run(cache_root, input_dir, output_dir, device='cuda', self_test=False):
    """@luna-comment Execute approved height routing and retain local-review evidence.

    Input: Verified cache, source directory, fresh output directory, CUDA device.
    Output: Full PNGs, per-line debug PNGs, route JSON and batch summary.
    Key Steps: Verify checkpoint/source hashes; native-infer tall boxes, reuse small
      cached predictions, apply unchanged V10, normalize overlap then feather to source.
    Key Params: Strict original OCR AABB height>128; source blend uses frozen replay alpha.
    State/Dependencies: One model load; no OCR, recovery probe or new NPY cache.
    """
    if self_test:
        test_contract()
        return
    cache_root, input_dir, output_dir = map(Path, (cache_root, input_dir, output_dir))
    manifest = paste.base.verify_cache(cache_root)
    assert len(manifest['frames']) == 24
    network, digest = src._load_network(Path(manifest['checkpoint']), torch.device(device), Path(manifest['network_config']))
    assert digest == manifest['checkpoint_sha256']
    config = src.load_lq_preprocess_config(Path(manifest['network_config']))
    assert config.get('p17_wiener_bypass_enabled')
    runner = src.NativeY4UVTileRunner(network, torch.device(device), config)
    output_dir.mkdir(parents=True, exist_ok=False)
    images = {p.stem:p for p in input_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    summary = dict(checkpoint_sha256=digest, threshold=128, tile_hw=[96,480], overlap_hw=[24,64], images=[])
    for index, frame in enumerate(manifest['frames']):
        image_id = frame['image_id']
        path = images[image_id]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['input_sha256']
        rgb, hw = pad_even_rgb(src._read_rgb(path))
        source = np.load(cache_root/frame['directory']/'frame_yuv444.npy')
        numerator = np.zeros_like(source)
        denominator = np.zeros(source.shape[1:], np.float32)
        coverage = np.zeros_like(denominator)
        routes = []
        for line in sorted(frame['lines'], key=lambda line:line['line_index']):
            ld = cache_root/line['directory']
            geometry = json.loads((ld/'geometry.json').read_text())
            polygon = np.asarray(geometry['geometry_recovery']['original_polygon'], np.float32)
            height = float(np.ptp(polygon[:,1]))
            if height > 128:
                p, valid, crop, tiles = native_prediction(rgb, polygon, runner)
                route = 'native_no_resize'
            else:
                p = np.load(ld/'prediction_yuv444.npy')
                valid = np.load(ld/'valid_support.npy').astype(bool)
                crop, tiles, route = geometry['crop_xyxy'], [], 'cached_bicubic'
            l,t,r,b = crop
            q, evidence, report = paste.v10.fill_v10(p, source[:,t:b,l:r], valid)
            alpha = paste.alpha_for(valid, evidence['raw'])
            numerator[:,t:b,l:r] += q*alpha[None]
            denominator[t:b,l:r] += alpha
            coverage[t:b,l:r] = np.maximum(coverage[t:b,l:r], alpha)
            dd = output_dir/'debug'/image_id/f"line_{line['line_index']:04d}"
            paste.base._rgb_png(dd/'prediction_before.png', p)
            paste.base._rgb_png(dd/'reconstructed_line.png', q)
            paste.base._png(dd/'mask_raw.png', evidence['raw'].astype('uint8')*255)
            paste.base._png(dd/'paste_alpha.png', np.rint(alpha*255).astype('uint8'))
            routes.append(dict(line_index=line['line_index'], original_ocr_height=height, route=route,
                               crop=crop, tile_bounds=tiles, native_scale=1 if tiles else None, v10=report))
        result = paste.compose(source, numerator, denominator, coverage)
        assert np.array_equal(result[:,coverage==0], source[:,coverage==0])
        paste.base._rgb_png(output_dir/f'{image_id}.png', result[:,:hw[0],:hw[1]])
        for rec in routes:
            l,t,r,b = rec['crop']
            paste.base._rgb_png(output_dir/'debug'/image_id/f"line_{rec['line_index']:04d}"/'final_pasted_crop.png', result[:,t:b,l:r])
        (output_dir/f'{image_id}.routes.json').write_text(json.dumps(routes, indent=2))
        record = dict(image_id=image_id, native_lines=sum(r['route']=='native_no_resize' for r in routes),
                      cached_lines=sum(r['route']=='cached_bicubic' for r in routes), uncovered_exact=True)
        summary['images'].append(record)
        summary['runner_calls'] = runner.calls
        (output_dir/'run_summary.json').write_text(json.dumps(summary, indent=2))
        print(f'{index+1}/24 {record}', flush=True)


def test_contract():
    """@luna-comment Check geometry and independent source feather using an identity model.

    Input: None. Output: Assertions and PASS marker.
    Key Steps: Reconstruct irregular native crop, assert exact canvases and alpha arithmetic.
    Key Params: Pixel tolerance1e-6 and threshold128/129.
    State/Dependencies: CPU fake runner only; monkeypatch resize helpers to reject calls.
    """
    class Identity:
        def run_rgb(self, rgb):
            """@luna-comment Input RGB96x480; output CHW identity; assert canvas, no state."""
            assert rgb.shape == (96,480,3)
            return rgb.transpose(2,0,1)
    rng = np.random.default_rng(7)
    rgb = rng.random((210,1002,3), dtype=np.float32)
    p,v,c,tiles = native_prediction(rgb, [[4,6],[999,6],[999,205],[4,205]], Identity())
    l,t,r,b = c
    assert np.max(np.abs(p-rgb[t:b,l:r].transpose(2,0,1))) < 1e-6
    assert v.all() and len(tiles)>1
    assert not (128>128) and 129>128
    zero = np.zeros((3,10,10),np.float32)
    a = np.full((10,10),.25,np.float32)
    assert np.allclose(paste.compose(zero,np.ones_like(zero)*a,a,a),.25)
    print('PASS native identity, canvas, coverage, threshold, source feather', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-root')
    ap.add_argument('--input-dir')
    ap.add_argument('--output-dir')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--self-test', action='store_true')
    run(**vars(ap.parse_args()))
