"""Isolated sample030 stitching A/B; never replaces existing evaluation outputs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/AI_DNS/guang.yang/TextSR/data/finetuned_four_val67_20261027')
OUT = Path('/AI_DNS/guang.yang/TextSR/data/finetuned030_feather_ab_20261027')
# Reuse the exact dependency snapshot of the completed evaluation.
if (BASE / '_build').is_dir():
    ROOT = BASE / '_build'
sys.path.insert(0, str(ROOT))


def axis_weights(origin, extent, mode):
    """@luna-comment Input:tile axis geometry/mode. Output:256 float weights.
    Key Steps:historical ramp or zero-ended overlap with optional context discard;
    retain image-facing boundary support. Key Params:224 stride,32 overlap,crop8.
    State/Dependencies:NumPy only; no image or model state mutation.
    """
    import numpy as np
    i = np.arange(256)
    if mode == 'old':
        return np.minimum(np.minimum(i + 1, 256 - i) / 16, 1).astype(np.float32)
    crop = 8 if mode == 'crop8' else 0
    t = np.clip((np.minimum(i, 255 - i) - crop) / (31 - 2 * crop), 0, 1)
    if mode == 'crop8':
        t = t * t * (3 - 2 * t)
    if origin == 0:
        t[:128] = 1
    if origin + 256 >= extent:
        t[128:] = 1
    return t.astype(np.float32)


def main():
    """@luna-comment Input:frozen030 assets/checkpoint. Output:isolated A/B finals/viewers/audit.
    Key Steps:strict load, shared tile predictions, three weight accumulators,
    exact baseline pixel replay, coverage and common metrics, native/fixed ROI delivery.
    Key Params:256/224/batch8; no tone/OCR/source blending. State/Dependencies:CUDA,
    accepted bootstrap/transforms/renderer; preserves historical files and aggregate scores.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    from tools.auto_hx_v3_native import positions
    from tools.evaluate_offline_camera_matrix import sha, image_write, save_json
    from tools.evaluate_vosr_six_models import metrics
    from tools.build_camera_2x2_viewer import render_html, file_url
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    entry = json.loads((BASE / 'selection.json').read_text())['samples'][29]
    assert entry['id'] == '030'
    files = entry['row']['files']
    assert {k: sha(v) for k, v in files.items()} == entry['row']['output_sha256']
    binding = json.loads((BASE / 'MODEL_BINDINGS.json').read_text())['text15_v3m']
    config, weight = BASE / 'models/text15_v3m.yml', BASE / 'models/text15_v3m.pth'
    assert sha(config) == binding['config_sha256'] and sha(weight) == binding['checkpoint_sha256']
    net = build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    net.load_state_dict(torch.load(weight, map_location='cpu')['params'], strict=True)
    source, gt = [cv2.imread(files[k], -1) for k in ('lq', 'gt')]
    valid, text = [cv2.imread(files[k], 0) > 0 for k in ('valid_mask', 'text_mask')]
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    h, w = source.shape[:2]
    grid = [(x, y) for y in positions(h, 256, 224) for x in positions(w, 256, 224)]
    modes = ['old', 'zero32', 'crop8']
    totals = {k: np.zeros((h, w, 3), np.float32) for k in modes}
    weights = {k: np.zeros((h, w), np.float32) for k in modes}
    for start in range(0, len(grid), 8):
        selected = grid[start:start + 8]
        patches = [rgb[y:y + 256, x:x + 256] for x, y in selected]
        assert all(p.shape == (256, 256, 3) for p in patches)
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
        with torch.inference_mode():
            decoded = YUV2RGB(y4uv_to_yuv444(net(rgb_to_y4uv(batch))), is_train=True).clamp(0, 1)
        assert decoded.shape == batch.shape and torch.isfinite(decoded).all()
        for (x, y), value in zip(selected, decoded.permute(0, 2, 3, 1).cpu().numpy()):
            for mode in modes:
                window = axis_weights(y, h, mode)[:, None] * axis_weights(x, w, mode)[None, :]
                totals[mode][y:y + 256, x:x + 256] += value * window[..., None]
                weights[mode][y:y + 256, x:x + 256] += window
        print(json.dumps(dict(done=min(start + 8, len(grid)), total=len(grid))), flush=True)
    results, evidence = {}, {}
    previous = cv2.imread(str(BASE / 'results/text15_v3m/final/030.png'), -1)
    for mode in modes:
        assert np.isfinite(weights[mode]).all() and np.all(weights[mode] > 0)
        final = cv2.cvtColor(np.clip(np.rint(totals[mode] / weights[mode][..., None] * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if mode == 'old':
            assert np.array_equal(final, previous), 'Baseline replay differs; stop candidate publication'
        path = OUT / (mode + '.png')
        image_write(path, final)
        results[mode] = path
        evidence[mode] = dict(final_sha256=sha(path), weight_min=float(weights[mode].min()),
                              metrics=metrics(final, gt, valid, text),
                              changed_pixels=int(np.any(final != previous, axis=2).sum()),
                              interior_axis=axis_weights(224, w, mode).tolist())
    labels = {'old': 'Baseline | positive16', 'zero32': 'Mild | zero-ended32', 'crop8': 'Candidate | crop8 + smooth16'}
    models = [dict(key='lq', label='LQ'), dict(key='gt', label='Teacher GT')]
    models += [dict(key=k, label=v, note='V3M Text x1.5 final150K | same256/224 predictions') for k, v in labels.items()]
    row = dict(id='030', name=entry['row']['sample_id'], width=w, height=h, lq=file_url(Path(files['lq'])),
               images={'gt': file_url(Path(files['gt'])), **{k: file_url(v) for k, v in results.items()}})
    manifest = dict(title='030 | Patch feather A/B', subtitle='Same model, source and tile predictions',
                    notes='Baseline exact replay verified. Candidates are pending user visual review. No original outputs or aggregate metrics replaced.',
                    defaults='lq,old,zero32,crop8', models=models, rows=[row])
    (OUT / 'index.html').write_text(render_html(manifest), encoding='utf-8')
    save_json(OUT / 'manifest.json', manifest)
    crop = dict(row, width=850, height=260, images={})
    for key, path in {'lq': Path(files['lq']), 'gt': Path(files['gt']), **results}.items():
        image = cv2.imread(str(path), -1)[1640:1900, 700:1550]
        dest = OUT / 'crop' / (key + '.png')
        image_write(dest, image)
        if key == 'lq':
            crop['lq'] = file_url(dest)
        else:
            crop['images'][key] = file_url(dest)
    manifest['rows'] = [crop]
    manifest['title'] = '030 | Screenshot ROI feather A/B'
    (OUT / 'crop.html').write_text(render_html(manifest), encoding='utf-8')
    save_json(OUT / 'AUDIT.json', dict(passed=True, baseline_pixel_equal=True,
              binding=binding, source_hashes=entry['row']['output_sha256'], script_sha256=sha(__file__),
              effective_path='strict network -> shared predictions -> three independent normalized windows',
              grid=grid, canvas=256, stride=224, patches=len(grid), modes=evidence,
              crop_xyxy=[700, 1640, 1550, 1900], visual_approval='pending'))
    print('A/B published; baseline exact replay and coverage passed', flush=True)


if __name__ == '__main__':
    main()
