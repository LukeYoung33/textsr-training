"""P15/P15a: clean background anchors and continuous stroke-safe feather."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare, smoothstep, write_png
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_crossseam_v9 import sample


def estimate_background(source, y, roi, anchor_distance):
    """@luna-comment Input: full-resolution float BGR/Y, ROI, stroke distance.
    Output: normalized background target and viewable diagnostic source buffers.
    Key Steps: flat distant anchors, weighted AREA, two Gaussian support scales.
    Key Params: anchor40..48; std2..5; quarter resolution sigma8/24.
    State/Dependencies: arrays immutable; finite OpenCV filters, no solver.
    """
    mean = cv2.GaussianBlur(y, (0, 0), 2)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(y*y, (0, 0), 2)-mean*mean, 0))
    anchors = roi*smoothstep((anchor_distance-40)/8)*(1-smoothstep((std-2)/3))
    if not np.any(anchors > 0):
        raise ValueError('P15: no clean background anchors')
    h, w = y.shape
    size = (max(1, w//4), max(1, h//4))
    small_weight = cv2.resize(anchors, size, interpolation=cv2.INTER_AREA)
    small_num = cv2.resize(source*anchors[..., None], size, interpolation=cv2.INTER_AREA)
    colors, supports = [], []
    for sigma in (8, 24):
        den = cv2.GaussianBlur(small_weight, (0, 0), sigma)
        num = cv2.GaussianBlur(small_num, (0, 0), sigma)
        colors.append(cv2.resize(num/np.maximum(den[..., None], 1e-6), (w, h), interpolation=cv2.INTER_LINEAR))
        supports.append(cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR))
    mix = smoothstep((supports[0]-.02)/.08)
    target = colors[0]*mix[..., None]+colors[1]*(1-mix[..., None])
    return target, dict(anchor_weight=anchors, anchor_std=std, support32=supports[0], support96=supports[1])


def support_confidence(support):
    """@luna-comment Input: finite support map. Output: float confidence [0,1].
    Key Steps: smooth support ramp and spatial distance ramp from invalid areas.
    Key Params: support1e-5..1e-4, spatial16px. Dependencies: no shared state.
    """
    bad = support < 1e-5
    confidence = smoothstep((support-1e-5)/9e-5)
    if bad.any():
        confidence *= smoothstep(distance_transform_edt(~bad).astype(np.float32)/16)
    return confidence


def clean_background_feather(base, prepared, exterior_cap=False, support_fallback=False):
    """@luna-comment Input: exact float V4 and immutable prepare tuple.
    Output: uint8 result, diagnostic maps and contract counters.
    Key Steps: conservative anchors; new target-relative guard; continuous blend.
    Key Params: contrast12; protect6; full10..32; fade32..80; optional positive exterior cap.
    State/Dependencies: original core retained; unsupported raises by default.
    Optional support_fallback gates complete correction against original input.
    """
    source, y, roi, core, old_background, _, _ = prepared
    original_strokes = roi & (cv2.cvtColor(old_background, cv2.COLOR_BGR2GRAY)-y > 12)
    original_distance = distance_transform_edt(~original_strokes).astype(np.float32)
    target, debug = estimate_background(source, y, roi, original_distance)
    confidence = support_confidence(debug['support96']) if support_fallback else None
    if support_fallback:
        target = target.copy()
        target[debug['support96'] < 1e-5] = old_background[debug['support96'] < 1e-5]
    deficit = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)-y
    actual_strokes = core | (roi & (deficit > 12))
    distance = distance_transform_edt(~actual_strokes).astype(np.float32)
    protected = core | (distance <= 6) | ~roi
    weight = smoothstep((distance-6)/4)*(1-smoothstep((distance-32)/48))
    weight *= smoothstep(distance_transform_edt(roi).astype(np.float32)/12)
    weight[protected] = 0
    excess = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    feather_weight = weight.copy()
    halo_weight = np.zeros_like(weight)
    if exterior_cap:
        halo_weight = smoothstep((excess-.5)/2)*(1-smoothstep((distance-10)/6))*smoothstep(distance)
        halo_weight *= smoothstep(distance_transform_edt(roi).astype(np.float32)/12)
        halo_weight[actual_strokes | core | ~roi] = 0
        weight = weight+(1-weight)*halo_weight
        protected = actual_strokes | core | ~roi | ((distance <= 6) & (excess <= .5))
    unsupported = (weight > 0) & (debug['support96'] < 1e-5)
    if unsupported.any() and not support_fallback:
        raise ValueError(f'P15: {int(unsupported.sum())} active pixels lack exterior anchors')
    floating = base+weight[..., None]*(target-base)
    if support_fallback:
        floating = source+confidence[..., None]*(floating-source)
        floating[core | ~roi] = source[core | ~roi]
    output = np.clip(np.rint(floating), 0, 255).astype(np.uint8)
    old = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    changed = np.any(output != old, axis=2)
    if not support_fallback:
        assert not changed[protected].any() and not changed[distance >= 80].any()
    report = dict(path='P15b_positive_exterior' if exterior_cap else 'P15a_clean_anchors_refreshed_guard', protected_changed=int(changed[protected].sum()),
                  core_changed=int(changed[core].sum()), clipping=int(np.count_nonzero((floating < 0) | (floating > 255))),
                  changed_pixels=int(changed.sum()), removed_candidates=int((original_strokes & ~actual_strokes).sum()),
                  added_candidates=int((~original_strokes & actual_strokes).sum()), unsupported_pixels=0,
                  positive_guard_changed=int((changed & (distance <= 6)).sum()))
    debug.update(original_strokes=original_strokes, actual_strokes=actual_strokes, protected=protected,
                 removed_candidates=original_strokes & ~actual_strokes, added_candidates=~original_strokes & actual_strokes,
                 original_distance=original_distance, distance=distance, target=target, deficit=deficit, weight=weight,
                 feather_weight=feather_weight, halo_weight=halo_weight, positive_excess=excess)
    if support_fallback:
        fallback = roi & (debug['support96'] < 1e-5)
        source_u8 = np.clip(np.rint(source), 0, 255).astype(np.uint8)
        source_changed = np.any(output != source_u8, axis=2)
        assert not source_changed[fallback | core | ~roi].any()
        assert np.isfinite(floating).all()
        debug.update(fallback_mask=fallback, fallback_confidence=confidence)
        report.update(fallback_enabled=True, fallback_pixels=int(fallback.sum()),
                      unsupported_active_before_fallback=int(unsupported.sum()),
                      fallback_changed=int(source_changed[fallback].sum()))
    return output, debug, report


def export_evidence(folder, stem, base, result, previous, debug, report, old_profiles):
    """@luna-comment Input: processed sample and P14a reference. Output: lossless evidence.
    Key Steps: save maps, fixed bar points/profiles and same-scale comparisons.
    Key Params: explicit display gains; full and half strength; native/2x crops.
    State/Dependencies: only new folder writes; no mutation of image buffers.
    """
    dest = folder/'debug'/stem
    gains = dict(anchor_weight=255, anchor_std=20, support32=255, support96=255,
                 original_strokes=255, actual_strokes=255, protected=255, removed_candidates=255,
                 added_candidates=255, original_distance=3, distance=3, weight=255, target=1,
                 feather_weight=255, halo_weight=255)
    for key, gain in gains.items():
        write_png(dest/(key+'.png'), np.clip(np.rint(debug[key]*gain), 0, 255).astype(np.uint8))
    write_png(dest/'deficit_signed.png', np.clip(128+debug['deficit']*4, 0, 255).astype(np.uint8))
    write_png(dest/'positive_excess.png', np.clip(debug['positive_excess']*8, 0, 255).astype(np.uint8))
    write_png(dest/'signed_rgb_change.png', np.clip(128+(result.astype(np.float32)-base)*8, 0, 255).astype(np.uint8))
    points = []
    for x in (1150,1190,1250,1300,1400,1500,1800,1900,2000):
        y0 = 495
        if y0 >= base.shape[0] or x >= base.shape[1]: continue
        point = dict(x=x, y=y0, old_distance=float(debug['original_distance'][y0,x]),
                     distance=float(debug['distance'][y0,x]), weight=float(debug['weight'][y0,x]),
                     protected=bool(debug['protected'][y0,x]))
        for key, im in [('v4',base),('v12a',previous),('v13',result),('target',debug['target'])]:
            color=im[y0,x].astype(float); point[key+'_bgr']=color.tolist(); point[key+'_y']=float(color@np.array([.114,.587,.299]))
        points.append(point)
    report['bar_points'] = points
    profiles=[]
    for old in old_profiles:
        px=np.array(old['x']); py=np.array(old['y']); profile=dict(pixel=old['pixel'],x=old['x'],y=old['y'])
        for key, im in [('v4',base),('v12a',previous),('v13',result)]:
            bgr=sample(im.astype(np.float32),px,py); yvalues=bgr@np.array([.114,.587,.299])
            profile[key+'_y']=yvalues.tolist(); profile[key+'_peak_step']=float(np.abs(np.diff(yvalues)).max())
        profiles.append(profile)
    report['profiles']=profiles
    half=np.clip(np.rint(base+.5*debug['weight'][...,None]*(debug['target']-base)),0,255).astype(np.uint8)
    write_png(folder/'low_strength'/(stem+'_dehalo.png'),half)
    crops=dict(rule_bars=(1050,330,2500,550), title=(950,450,2750,760), practice=(950,740,2150,990),
               marked_seam=(1780,495,2110,665), degrees=(1750,1260,2950,1480), color=(600,1690,2700,1890))
    for label,(x0,y0,x1,y1) in crops.items():
        for scale in (1,2):
            panels=[]
            for title,im in [('V4',base),('V12a',previous),('V13b clean background + exterior cap' if report['path'].startswith('P15b') else 'V13 clean background',result)]:
                crop=np.clip(np.rint(im[y0:y1,x0:x1]),0,255).astype(np.uint8)
                crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
                panel=cv2.copyMakeBorder(crop,26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
                cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1); panels.append(panel)
            write_png(folder/'comparisons'/f'{stem}_{label}_{scale}x.png', np.concatenate(panels))


def main():
    """@luna-comment Input: sample source and exclusive result path. Output: two results.
    Key Steps: hash/reproduce V4, P15a, export, verify every PNG and receipt.
    Key Params: exactly two approved IDs, cv2 four threads. State: local files only.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exterior-cap',action='store_true',help='Enable recorded P15b positive-only exterior correction')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); cv2.setNumThreads(4)
    root=Path('results/ppt_dehalo_20260924'); receipts=[]
    previous_receipt=json.loads((root/'sample_v12a_direct_rgb'/'receipt.json').read_text(encoding='utf-8'))
    for ident in ('044920599','052927799'):
        started=time.perf_counter(); path=next(args.input.glob(f'*{ident}_preprocessInput.png'))
        digest=hashlib.sha256(path.read_bytes()).hexdigest(); source=read_image(path)
        prepared=prepare(source,robust_seed=True); v4,maps,_=normal_correct(prepared)
        base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0]); name=path.stem+'_dehalo.png'
        assert np.array_equal(v4,read_image(root/'sample_v4_normal'/name))
        ready=time.perf_counter(); result,debug,report=clean_background_feather(base,prepared,args.exterior_cap); done=time.perf_counter()
        write_png(args.output/name,result); assert np.array_equal(result,read_image(args.output/name))
        previous=read_image(root/'sample_v12a_direct_rgb'/name)
        profiles=next(r['profiles'] for r in previous_receipt if ident in r['image'])
        export_evidence(args.output,path.stem,base,result,previous,debug,report,profiles)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        report.update(image=name,source_sha256=digest,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      exterior_cap=args.exterior_cap,prepare_seconds=ready-started,clean_feather_seconds=done-ready,total_seconds=time.perf_counter()-started)
        receipts.append(report); (args.output/'receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('bar_points','profiles')}),flush=True)
        del source,prepared,v4,maps,base,result,debug,previous; gc.collect()
    files=list(args.output.rglob('*.png'))
    for path in files: read_image(path)
    print(f'PNG decode verified: {len(files)}',flush=True)


if __name__ == '__main__': main()
