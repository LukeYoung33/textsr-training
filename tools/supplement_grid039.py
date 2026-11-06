"""Fixed-mask sensitivity analysis and same-color old/new checkpoint controls."""
from pathlib import Path
import sys
import json
import shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_grid039 as core


def correlation(left, right):
    """@luna-comment Input:two phaseprofile dictionaries. Output:alllagbest plus0/112correlations peraxis. Key Steps:centered224cyclic Pearson scan. Key Params:interpretwithamplitudes/counts. State/Dependencies:NumPy read-only."""
    import numpy as np
    result = {}
    for axis in ['x', 'y']:
        a = np.array(left[axis])
        b = np.array(right[axis])
        values = [float(np.corrcoef(np.roll(a, -lag), b)[0, 1]) for lag in range(224)]
        result[axis] = dict(zero_lag=values[0], expected112=values[112], best_lag=int(np.argmax(values)), best_correlation=max(values))
    return result


def main():
    """@luna-comment Input:completed039diagnostic andsameimmutable checkpoints. Output:detrend/quadrant sensitivity andsameflatold/newprobe. Key Steps:fixedsigma56normalizedmaskedtrend,rawcounts,strictold/newflatfloatcontrols. Key Params:no newfullframeinference/fix;denominator>.25. State/Dependencies:completedraw/shiftPNGs andCUDA."""
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    out = core.OUT
    binding = json.loads((out/'BINDING.json').read_text())
    rgb = cv2.cvtColor(cv2.imread(binding['entry']['input']), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(out/'background_mask.png'), 0)>0
    h, w = mask.shape
    denominator_small = cv2.GaussianBlur(cv2.resize(mask.astype(np.float32), (w//4, h//4), interpolation=cv2.INTER_AREA), (0, 0), 14)
    denominator = cv2.resize(denominator_small, (w,h), interpolation=cv2.INTER_LINEAR)
    common = mask&(denominator>.25)
    paths = dict(new_original_grid=core.SOURCE/'results/v3m_dilate2_l1/final/039.png', new_shift112=out/'shift112.png',
                 old_l1_reference=core.DATA/'latest_six_500k_mojito280_20261103/results/v3m_l1/final/039.png')
    detrended = {}
    arrays = {}
    for key, path in paths.items():
        residual = core.luma(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))-core.luma(rgb)
        numerator_small = cv2.GaussianBlur(cv2.resize(residual*mask, (w//4,h//4), interpolation=cv2.INTER_AREA), (0, 0), 14)
        trend = cv2.resize(numerator_small, (w,h), interpolation=cv2.INTER_LINEAR)/np.maximum(denominator, 1e-6)
        arrays[key] = residual-trend
        detrended[key] = core.phases(arrays[key], common)
    quadrants = []
    for yy, xx in [(0,0),(0,1),(1,0),(1,1)]:
        support = common.copy()
        support[:h//2 if yy else 0] = False
        support[h if yy else h//2:] = False
        support[:, :w//2 if xx else 0] = False
        support[:, w if xx else w//2:] = False
        y,x = np.nonzero(support)
        counts = dict(x=[int(np.bincount(x%224,minlength=224).min()),int(np.bincount(x%224,minlength=224).max())],
                      y=[int(np.bincount(y%224,minlength=224).min()),int(np.bincount(y%224,minlength=224).max())])
        a = core.phases(arrays['new_original_grid'], support)
        b = core.phases(arrays['new_shift112'], support)
        quadrants.append(dict(y_half=yy,x_half=xx,pixels=int(support.sum()),phase_counts=counts,
                             original_amplitude=dict(x=a['x_peak_to_peak_dn'],y=a['y_peak_to_peak_dn']),correlation=correlation(a,b)))
    probes = {}
    model_bindings = dict(new=binding['model'], old_l1=json.loads((core.DATA/'latest_six_500k_mojito280_20261103/MODEL_BINDINGS.json').read_text())['v3m_l1'])
    for key, b in model_bindings.items():
        assert core.digest(b['checkpoint']) == b['checkpoint_sha256']
        model = build_network(yaml.safe_load(Path(b['config']).read_text())['network_g']).cuda().eval()
        model.load_state_dict(torch.load(b['checkpoint'],map_location='cpu')['params'],strict=True)
        probes[key] = core.patch_probe(model,rgb,mask)
        shutil.copy2(out/'flat_residual_x8.png',out/(key+'_flat_residual_x8.png'))
        flat = np.empty((256,256,3), np.uint8)
        flat[:] = probes[key]['flat_rgb']
        batch = torch.from_numpy(flat.transpose(2,0,1).copy()).unsqueeze(0).cuda().float()/255
        with torch.inference_mode():
            raw = YUV2RGB(y4uv_to_yuv444(model(rgb_to_y4uv(batch))),is_train=True)
        probes[key]['flat_clipped_channel_fraction'] = float(((raw<0)|(raw>1)).float().mean())
        probes[key]['flat_unclamped_min_max_dn'] = [float(raw.min()*255),float(raw.max()*255)]
        assert core.digest(b['checkpoint']) == b['checkpoint_sha256']
        del model
    shutil.copy2(out/'new_flat_residual_x8.png',out/'flat_residual_x8.png')
    core.save(out/'SUPPLEMENT.json', dict(detrend_sigma_native56=True,downsample_for_trend=4,denominator_threshold=.25,
              detrended_correlation=correlation(detrended['new_original_grid'],detrended['new_shift112']),
              quadrants=quadrants, probes=probes))
    core.save(out/'DETRENDED_PHASES.json', detrended)
    print(json.dumps(dict(detrended_correlation=correlation(detrended['new_original_grid'],detrended['new_shift112']),
          quadrants=quadrants,flat_summary={k:{f:v[f] for f in ['flat_center_residual_dn','flat_max_deviation_from_center_dn','flat_deviation_dn_at_distance','flat_clipped_channel_fraction']} for k,v in probes.items()})))


if __name__ == '__main__':
    main()
