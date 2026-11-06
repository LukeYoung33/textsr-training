"""Reproducible synthetic mechanism probes for the anchored-Y research arms."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.anchored_y_contrast import ARMS, process_arm

OUTPUT_ROOT = ROOT/'output/anchored_y_contrast_pilot_20261028'


def grayscale(values):
    """@luna-comment Input: uint8 synthetic Y-like plane. Output: three equal BGR channels. Key Steps: channel replicate without display filtering. Key Params: native dimensions retained. State/Dependencies: no writes."""
    return np.repeat(values[:, :, None], 3, axis=2)


def ring_scene(amplitude):
    """@luna-comment Input: injected dark exterior ring amplitude. Output: known paper/stroke/ring BGR and masks. Key Steps: paper220, stroke80, two-pixel exterior ring. Key Params: amplitude0/4/8/16/24. State/Dependencies: synthetic label only, not real halo GT."""
    plane = np.full((256, 256), 220, np.uint8)
    plane[32:224, 115:141] = 80
    ring = np.zeros(plane.shape, bool)
    ring[32:224, 113:115] = True
    ring[32:224, 141:143] = True
    plane[ring] = 220-amplitude
    paper = np.zeros(plane.shape, bool)
    paper[32:224, 160:220] = True
    return grayscale(plane), ring, paper


def crossings(profile, low, high):
    """@luna-comment Input: one monotone synthetic edge row and fixed platforms. Output: interpolated 10/50/90 crossing positions. Key Steps: find first threshold crossing in increasing direction. Key Params: linear subpixel interpolation. State/Dependencies: measured output geometry, not a preservation guarantee."""
    values = np.asarray(profile, np.float32)
    result = []
    for fraction in (.1, .5, .9):
        threshold = low+(high-low)*fraction
        indexes = np.flatnonzero((values[:-1]<threshold)&(values[1:]>=threshold))
        if len(indexes) != 1:
            return None
        x = int(indexes[0])
        result.append(float(x+(threshold-values[x])/(values[x+1]-values[x])))
    return result


def injected_lobe_scene(paper_level, stroke_level, amplitude, side):
    """@luna-comment Input: two platforms, signed two-pixel ring and side. Output: synthetic BGR plus lobe/background masks. Key Steps: put lobe inside platform interval or beyond it without fitting. Key Params: paper220/160/40, stroke80/70/210. State/Dependencies: synthetic mechanism label, not a real halo annotation."""
    plane = np.full((256, 256), paper_level, np.uint8)
    plane[32:224, 115:141] = stroke_level
    lobe = np.zeros(plane.shape, bool)
    lobe[64:192, 113:115] = True
    lobe[64:192, 141:143] = True
    plane[lobe] = np.uint8(np.clip(paper_level+side*amplitude, 0, 255))
    paper = np.zeros(plane.shape, bool)
    paper[64:192, 160:208] = True
    return grayscale(plane), lobe, paper


def edge_geometry(current, candidate, source, left_slice, right_slice):
    """@luna-comment Input: one known edge row in source/current/candidate. Output: actual-platform and original-platform crossings plus bidirectional new extrema. Key Steps: estimate each output's own left/right platform medians; compare normal-side overshoot. Key Params: row128, x bands distant from transition. State/Dependencies: proxy of sampled geometry, not GT registration."""
    row = 128
    src = source[row, :, 0].astype(np.float32)
    cur = current[row, :, 0].astype(np.float32)
    can = candidate[row, :, 0].astype(np.float32)
    def platforms(values):
        """@luna-comment Input: row. Output: its observed low/high plateau medians. Key Steps: use fixed separated bands. Key Params: slanted synthetic known geometry. State/Dependencies: local only."""
        return float(np.median(values[left_slice])), float(np.median(values[right_slice]))
    s0,s1 = platforms(src)
    c0,c1 = platforms(cur)
    n0,n1 = platforms(can)
    actual_cur = crossings(cur,c0,c1)
    actual_new = crossings(can,n0,n1)
    fixed_cur = crossings(cur,s0,s1)
    fixed_new = crossings(can,s0,s1)
    # Relative to each arm's own two plateaus, measure both normal sides.
    cur_exc = max(0.0,c0-float(np.min(cur[111:124])),float(np.max(cur[111:124]))-c0,
                  c1-float(np.min(cur[136:149])),float(np.max(cur[136:149]))-c1)
    new_exc = max(0.0,n0-float(np.min(can[111:124])),float(np.max(can[111:124]))-n0,
                  n1-float(np.min(can[136:149])),float(np.max(can[136:149]))-n1)
    def delta_cross(a,b):
        """@luna-comment Input: two 10/50/90 crossing triples. Output: center/width changes. Key Steps: paired subtraction. Key Params: float subpixel coordinate. State/Dependencies: returns null if nonmonotone."""
        return None if a is None or b is None else {'center_50_delta':b[1]-a[1],
            'width_10_90_delta':(b[2]-b[0])-(a[2]-a[0])}
    return {'source_platforms':[s0,s1], 'current_platforms':[c0,c1],
            'candidate_platforms':[n0,n1],
            'actual_output_platform_geometry_delta':delta_cross(actual_cur,actual_new),
            'fixed_original_platform_geometry_delta':delta_cross(fixed_cur,fixed_new),
            'normal_side_extremum_current':cur_exc,'normal_side_extremum_candidate':new_exc,
            'normal_side_extremum_increase':new_exc-cur_exc}


def extended_controls(arm):
    """@luna-comment Input: one frozen research arm. Output: signed lobe, gray/dark-paper and shadow metrics. Key Steps: paired current/candidate known-scene comparisons; inspect both normal sides and actual output platforms. Key Params: lobe amplitudes8/16, paper220/160/40. State/Dependencies: objective synthetic falsification only, no quality grade."""
    lobes = {}
    for paper,ink in ((220,80),(160,70),(40,210)):
        for side in (-1,1):
            for amplitude in (8,16):
                source,lobe,paper_mask = injected_lobe_scene(paper,ink,amplitude,side)
                current,candidate,_ = process_arm(source,arm)
                # Signed departure from same arm's local paper baseline. Positive magnitude is injected lobe direction.
                cur_plane = current[:,:,0].astype(np.float32)
                can_plane = candidate[:,:,0].astype(np.float32)
                old = float(side*(cur_plane[lobe].mean()-cur_plane[paper_mask].mean()))
                new = float(side*(can_plane[lobe].mean()-can_plane[paper_mask].mean()))
                lobes[f'{paper}_{ink}_{side}_{amplitude}'] = {'current_lobe':old,'candidate_lobe':new,
                    'signed_lobe_increase':new-old,'paper_change':float(can_plane[paper_mask].mean()-cur_plane[paper_mask].mean())}
    yy,xx = np.mgrid[:256,:256]
    dist = xx-.27*yy-94.7
    edge = np.clip(80+140*(dist+3)/6,80,220).astype(np.uint8)
    current,candidate,_ = process_arm(grayscale(edge),arm)
    edge_info = edge_geometry(current,candidate,grayscale(edge),slice(90,108),slice(152,170))
    shadow = np.clip(170.0+55.0*xx/255.0 - 100.0*(dist<0),0,255).astype(np.uint8)
    shadow_current,shadow_candidate,_ = process_arm(grayscale(shadow),arm)
    shadow_diff = shadow_candidate[:,:,0].astype(np.int16)-shadow_current[:,:,0].astype(np.int16)
    return {'lobes':lobes,'slanted_edge':edge_info,
            'shadow':{'max_abs_change':int(np.max(np.abs(shadow_diff))),
                      'p95_abs_change':float(np.percentile(np.abs(shadow_diff),95)),
                      'changed_pixels':int(np.count_nonzero(shadow_diff))},
            'gates':{'lobe_increase_le2_all':all(v['signed_lobe_increase']<=2 for v in lobes.values()),
                     'normal_side_new_extremum_le1':edge_info['normal_side_extremum_increase']<=1}}


def evaluate_arm(key, arm=None):
    """@luna-comment Input: research arm key and optional offset override arm. Output: reproducible synthetic metrics and explicit artifact-gate booleans. Key Steps: halo injections, no-halo edge, phase/stroke widths, gradient geometry, clipping. Key Params: fixed test scenes and thresholds in child task. State/Dependencies: objective controls only, not visual quality judgment."""
    arm = ARMS[key] if arm is None else arm
    halo = {}
    for amplitude in (0,4,8,16,24):
        source, ring, paper = ring_scene(amplitude)
        current, candidate, branch = process_arm(source, arm)
        current_ring = float(current[:, :, 0][paper].mean()-current[:, :, 0][ring].mean())
        candidate_ring = float(candidate[:, :, 0][paper].mean()-candidate[:, :, 0][ring].mean())
        halo[str(amplitude)] = {'current_outer_dark_ring': current_ring,
                                'candidate_outer_dark_ring': candidate_ring,
                                'increase_vs_current': candidate_ring-current_ring,
                                'rgb_clip_candidate': branch['rgb_clipped_channels_candidate']}
    phase = {}
    for width in (8,16,32,96,144):
        deltas = []
        paper_changes = []
        for offset in range(16):
            plane = np.full((256, 256), 220, np.uint8)
            left = 48+offset
            plane[32:224, left:left+width] = 120
            current, candidate, _ = process_arm(grayscale(plane), arm)
            core = np.s_[64:192, left+max(0,width//2-2):left+min(width,width//2+2)]
            deltas.append(float((candidate[:, :, 0][core].astype(np.int16)-current[:, :, 0][core]).mean()))
            paper_roi = candidate[64:192, 208:240, 0].astype(np.int16)-current[64:192, 208:240, 0].astype(np.int16)
            paper_changes.append(float(np.percentile(np.abs(paper_roi),95)))
        phase[str(width)] = {'center_delta_range': [min(deltas),max(deltas)],
                             'center_delta_phase_span': max(deltas)-min(deltas),
                             'paper_p95_abs_change_max': max(paper_changes)}
    yy, xx = np.mgrid[:256,:256]
    distance = xx-.27*yy-94.7
    plane = np.clip(80+140*(distance+3)/6,80,220).astype(np.uint8)
    current, candidate, branch = process_arm(grayscale(plane),arm)
    c0 = crossings(current[128,:,0],80,220)
    c1 = crossings(candidate[128,:,0],80,220)
    geometry = {'current_crossings':c0,'candidate_crossings':c1,
                'width_10_90_change':None if c0 is None or c1 is None else (c1[2]-c1[0])-(c0[2]-c0[0]),
                'center_50_change':None if c0 is None or c1 is None else c1[1]-c0[1]}
    no_halo_new_extrema = max(0,int(current[128,160:220,0].min())-int(candidate[128,160:220,0].min()))
    halo_gate = all(halo[str(a)]['increase_vs_current'] <= 2.0 for a in (4,8,16,24))
    paper_gate = max(v['paper_p95_abs_change_max'] for v in phase.values()) <= 2.0
    return {'arm':key,'halo':halo,'phase':phase,'geometry':geometry,
            'no_halo_paper_new_dark_extremum':no_halo_new_extrema,
            'branch_on_slanted_edge':branch,
            'gates':{'injected_dark_halo_increase_le2':halo_gate,
                     'paper_p95_change_le2':paper_gate,
                     'no_new_paper_extremum_gt1':no_halo_new_extrema<=1}}


def main():
    """@luna-comment Input: selected research arms. Output: one synthetic JSON with measured mechanism gates. Key Steps: evaluate arms independently and store exact metrics. Key Params: no automatic real-image escalation. State/Dependencies: writes only approved pilot output subtree."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=('weak',))
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--offset', type=float, default=12.0, choices=(4.0,8.0,12.0,16.0,20.0))
    args = parser.parse_args()
    arm_config = {key:replace(ARMS[key], anchor_offset=args.offset) for key in args.arms}
    results = {key:(extended_controls(arm_config[key]) if args.extended else evaluate_arm(key,arm_config[key]))
               for key in args.arms}
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    destination = OUTPUT_ROOT/('synthetic_controls_'+('extended_' if args.extended else '')+
                               '_'.join(args.arms)+'_offset'+str(int(args.offset))+'.json')
    if destination.exists():
        raise FileExistsError('Synthetic evidence already exists '+str(destination))
    destination.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value['gates'] for key,value in results.items()}),flush=True)
    print(str(destination),flush=True)


if __name__ == '__main__':
    main()
