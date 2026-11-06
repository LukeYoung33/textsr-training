"""Authorized no-update fixed039 tile-grid and background-response diagnosis."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_uniform20_steep_scurve import save, digest
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'grid039_diagnostic_20261104'
SOURCE = DATA/'v3m_original_dilate2_l1_mojito280_20261104'


def luma(rgb):
    """@luna-comment Input:RGB in0..255 units. Output:float32 lumaDN. Key Steps:fixedRGBweighted sum. Key Params:.299/.587/.114. State/Dependencies:NumPy;no normalization or tone change."""
    import numpy as np
    return np.asarray(rgb, np.float32) @ np.array([.299, .587, .114], np.float32)


def phases(residual, mask):
    """@luna-comment Input:lumaresidualDN/commonbackgroundmask. Output:224pxphaseprofiles and additiveexplainedvariance. Key Steps:bincountaxismeans,zero-center,predictphasefield. Key Params:regularinterioronly. State/Dependencies:read-onlysame mask for everyvariant."""
    import numpy as np
    y, x = np.nonzero(mask)
    value = residual[y, x]
    mean = float(value.mean())
    profiles = []
    for coord in [x, y]:
        counts = np.bincount(coord % 224, minlength=224)
        assert counts.min() > 0
        profiles.append(np.bincount(coord % 224, weights=value, minlength=224)/counts-mean)
    prediction = profiles[0][x % 224]+profiles[1][y % 224]
    mse = float(np.mean((value-mean)**2))
    return dict(mean_dn=mean, std_dn=float(value.std()), pixels=len(value),
                x=profiles[0].tolist(), y=profiles[1].tolist(),
                x_peak_to_peak_dn=float(np.ptp(profiles[0])), y_peak_to_peak_dn=float(np.ptp(profiles[1])),
                additive_phase_variance_explained=1-float(np.mean((value-mean-prediction)**2))/max(mse, 1e-12))


def decode(network, patches):
    """@luna-comment Input:owned strictmodel/listRGB8 patches. Output:floatRGBDN predictions. Key Steps:acceptedY4UVencode/network/decode/clamp withoutPNGquantization. Key Params:inferenceonly,batch<=8. State/Dependencies:CUDA;no optimizer/state update."""
    import numpy as np
    import torch
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2).copy()).cuda().float()/255
    with torch.inference_mode():
        output = YUV2RGB(y4uv_to_yuv444(network(rgb_to_y4uv(batch))), is_train=True).clamp(0, 1)
    assert torch.isfinite(output).all()
    return output.permute(0, 2, 3, 1).cpu().numpy()*255


def patch_probe(network, rgb, mask):
    """@luna-comment Input:strictnetwork/same039RGB/mask. Output:realpatchanduniformbackground edge-distance residuals. Key Steps:select8darkinteriorpatches,predictsamefloatpath,foldedge-distance;syntheticflatcolorprobe. Key Params:256canvas,16pxdeepcenter. State/Dependencies:isolatedno-updateCUDA."""
    import numpy as np
    import cv2
    candidates = []
    h, w = mask.shape
    for y in range(448, h-512, 224):
        for x in range(448, w-512, 224):
            support = float(mask[y:y+256, x:x+256].mean())
            if support > .7:
                candidates.append((support, x, y))
    assert len(candidates) >= 8
    candidates.sort(reverse=True)
    chosen = [candidates[i] for i in np.linspace(0, len(candidates)-1, 8, dtype=int)]
    patches = [rgb[y:y+256, x:x+256] for _, x, y in chosen]
    predicted = decode(network, patches)
    yy, xx = np.indices((256, 256))
    distance = np.minimum.reduce([xx, yy, 255-xx, 255-yy])
    real = []
    sums = np.zeros(128)
    counts = np.zeros(128)
    for (support, x, y), patch, output in zip(chosen, patches, predicted):
        r = luma(output)-luma(patch)
        m = mask[y:y+256, x:x+256]
        sums += np.bincount(distance[m], weights=r[m], minlength=128)
        counts += np.bincount(distance[m], minlength=128)
        real.append(dict(x=x, y=y, background_fraction=support, mean_residual_dn=float(r[m].mean())))
    real_profile = sums/np.maximum(counts, 1)
    color = np.rint(np.median(rgb[mask], axis=0)).astype(np.uint8)
    flat = np.empty((256, 256, 3), np.uint8)
    flat[:] = color
    flat_output = decode(network, [flat])[0]
    flat_residual = luma(flat_output)-luma(flat)
    flat_profile = np.bincount(distance.ravel(), weights=flat_residual.ravel(), minlength=128)/np.bincount(distance.ravel(), minlength=128)
    center = float(flat_residual[96:160, 96:160].mean())
    periodic = {}
    interior = flat_residual[64:192, 64:192]
    for period in [2, 4, 8, 16]:
        periodic[str(period)] = dict(x_peak_to_peak_dn=float(np.ptp([interior[:, i::period].mean() for i in range(period)])),
                                   y_peak_to_peak_dn=float(np.ptp([interior[i::period].mean() for i in range(period)])))
    error = np.abs(flat_profile-center)
    cv2.imwrite(str(OUT/'flat_residual_x8.png'), np.clip(128+(flat_residual-center)*8, 0, 255).astype(np.uint8))
    return dict(real_patches=real, real_distance_profile=real_profile.tolist(), flat_rgb=color.tolist(),
                flat_distance_profile=flat_profile.tolist(), flat_center_residual_dn=center,
                flat_max_deviation_from_center_dn=float(error.max()),
                flat_deviation_dn_at_distance={str(i):float(error[i]) for i in [0, 4, 8, 16, 24, 32, 48, 64, 96]},
                flat_small_periods=periodic)


def main():
    """@luna-comment Input:authorized039 source/checkpoint andunchangedacceptedstitcher. Output:exactbaseline replay,112shift,phase/probeJSON andfinitePNGevidence. Key Steps:hashgate,strictload,replay,evenphaseshift,commonmask,comparison/probe. Key Params:no optimizer/training/fix,exclude512pxphysicaledges. State/Dependencies:CUDA andimmutable prioroutputs."""
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    started = time.time()
    entry = next(e for e in json.loads((SOURCE/'selection.json').read_text())['samples'] if e['id'] == '039')
    b = json.loads((SOURCE/'MODEL_BINDINGS.json').read_text())['v3m_dilate2_l1']
    assert entry['orientation'] == 1 and digest(entry['input']) == entry['input_sha256']
    assert digest(b['checkpoint']) == b['checkpoint_sha256'] and digest(b['config']) == b['config_sha256']
    assert digest(ROOT/'tools/restore_crop8_block256.py') == b['stitcher_sha256']
    network = build_network(yaml.safe_load(Path(b['config']).read_text())['network_g']).cuda().eval()
    network.load_state_dict(torch.load(b['checkpoint'], map_location='cpu')['params'], strict=True)
    rgb = cv2.cvtColor(cv2.imread(entry['input']), cv2.COLOR_BGR2RGB)
    archived = cv2.cvtColor(cv2.imread(str(SOURCE/'results/v3m_dilate2_l1/final/039.png')), cv2.COLOR_BGR2RGB)
    replay, baseline_details = restore_block(rgb, network, 'cuda')
    assert np.array_equal(replay, archived), 'baseline replay is not bit exact'
    print('BASELINE BIT EXACT', flush=True)
    shifted_input = np.pad(rgb, ((112, 0), (112, 0), (0, 0)), mode='reflect')
    shifted, shifted_details = restore_block(shifted_input, network, 'cuda')
    shifted = shifted[112:, 112:]
    assert shifted.shape == rgb.shape
    cv2.imwrite(str(OUT/'shift112.png'), cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))
    print('SHIFT COMPLETE', flush=True)
    y = luma(rgb)
    soft = cv2.GaussianBlur(y, (0, 0), 1)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0)/8
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1)/8
    bright = cv2.dilate((rgb.max(axis=2)>140).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    mask = (y>15)&(y<100)&(~bright)&(np.hypot(gx, gy)<2)
    mask[:512] = False
    mask[-512:] = False
    mask[:, :512] = False
    mask[:, -512:] = False
    assert mask.sum() > 100000
    cv2.imwrite(str(OUT/'background_mask.png'), mask.astype(np.uint8)*255)
    old_path = DATA/'latest_six_500k_mojito280_20261103/results/v3m_l1/final/039.png'
    old = cv2.cvtColor(cv2.imread(str(old_path)), cv2.COLOR_BGR2RGB)
    stats = {}
    residuals = {}
    for key, output in [('new_original_grid', replay), ('new_shift112', shifted), ('old_l1_reference', old)]:
        residual = luma(output)-y
        residuals[key] = residual
        stats[key] = phases(residual, mask)
    correlation = {}
    for axis in ['x', 'y']:
        a = np.array(stats['new_original_grid'][axis])
        b1 = np.array(stats['new_shift112'][axis])
        values = [float(np.corrcoef(np.roll(a, -lag), b1)[0, 1]) for lag in range(224)]
        correlation[axis] = dict(zero_lag=values[0], expected112=values[112], best_lag=int(np.argmax(values)), best_correlation=max(values))
    probes = patch_probe(network, rgb, mask)
    for key, output in [('input', rgb), ('new_original_grid', replay), ('new_shift112', shifted), ('old_l1_reference', old)]:
        crop = output[768:1792, 2816:3840]
        cv2.imwrite(str(OUT/(key+'_crop.png')), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT/(key+'_preview.jpg')), cv2.cvtColor(cv2.resize(output, (1040, 780)), cv2.COLOR_RGB2BGR))
    for key, residual in residuals.items():
        cv2.imwrite(str(OUT/(key+'_residual_x8.png')), np.clip(128+residual[768:1792, 2816:3840]*8, 0, 255).astype(np.uint8))
    assert digest(b['checkpoint']) == b['checkpoint_sha256']
    save(OUT/'PHASES.json', stats)
    save(OUT/'PATCH_PROBES.json', probes)
    save(OUT/'METRICS.json', dict(sample='039', grid_stride=224, shift=[112,112], baseline_bit_exact=True,
         common_background_pixels=int(mask.sum()), mask_definition='input luma15..100,maxRGB<=140,dilatebright21,gradient<2DN,512pxphysicaledgesexcluded',
         phase_shift=correlation, variants={key:{k:v for k,v in value.items() if k not in ['x','y']} for key,value in stats.items()},
         flat_probe={k:v for k,v in probes.items() if k not in ['real_patches','real_distance_profile','flat_distance_profile']},
         baseline_stitch=baseline_details, shifted_stitch=shifted_details, seconds=time.time()-started))
    save(OUT/'BINDING.json', dict(entry=entry, model=b, input_sha256=digest(entry['input']), old_reference_sha256=digest(old_path),
         original_output_sha256=digest(SOURCE/'results/v3m_dilate2_l1/final/039.png'),shift_output_sha256=digest(OUT/'shift112.png'),
         script_sha256=digest(__file__), no_optimizer=True, no_training=True, deployment_unchanged=True))
    save(OUT/'COMPLETE.json', dict(passed=True, sample='039', seconds=time.time()-started))
    print(json.dumps(json.loads((OUT/'METRICS.json').read_text())), flush=True)


if __name__ == '__main__':
    main()
