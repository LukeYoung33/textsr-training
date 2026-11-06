"""Synthetic failure controls for source-V3-field direct contrast reuse."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.v3_reference_post import NODES,node_tables,render_post
from evaluation.document_box_pre640_revision2_20261027 import revision2


OUTPUT=ROOT/'output/v3_reference_post_pilot_20261028/synthetic_controls.json'


def fixed_fields(low,high,population):
    """@luna-comment Input: known source L3/H/p scalars. Output:81x81 immutable controls. Key Steps: fill source-node grid. Key Params: fixed p is distribution evidence, not semantic label. State/Dependencies: no model output reads."""
    shape=(len(NODES),len(NODES))
    return (np.full(shape,low,np.float32),np.full(shape,high,np.float32),
            np.full(shape,population,np.float32))


def gray_scene(paper,ink,ring=0,shadow=False):
    """@luna-comment Input: paper/ink levels, signed ring and shadow switch. Output: known BGR and masks. Key Steps: construct wide stroke with optional two-pixel lobe and two-paper gradient. Key Params:256square. State/Dependencies: synthetic model-output proxy, not real GT."""
    yy,xx=np.mgrid[:256,:256]
    background=np.full((256,256),paper,np.float32)
    if shadow:
        background=np.clip(paper+24.*(xx>=128)+8.*xx/255.,0,255)
    plane=background.astype(np.uint8)
    stroke=np.zeros(plane.shape,bool)
    stroke[40:216,116:140]=True
    plane[stroke]=ink
    lobe=np.zeros(plane.shape,bool)
    lobe[56:200,114:116]=True
    lobe[56:200,140:142]=True
    plane[lobe]=np.clip(background[lobe]+ring,0,255).astype(np.uint8)
    paper_mask=np.zeros(plane.shape,bool)
    paper_mask[56:200,160:200]=True
    return np.repeat(plane[:,:,None],3,axis=2),lobe,paper_mask,stroke


def evaluate_scene(name,source,low,high,population,lobe=None,paper=None,stroke=None):
    """@luna-comment Input: one synthetic model proxy and fixed V3 rails. Output: weak/full native metrics and risk warnings. Key Steps: build once per-node tables, render with RGB clip, quantify signed ring/background/stroke changes. Key Params: lambda.35/1. State/Dependencies: source fields remain frozen across both arms; no output calibration."""
    fields=fixed_fields(low,high,population)
    tables,field_stats=node_tables(*fields)
    input_y=cv2.cvtColor(source,cv2.COLOR_BGR2YCrCb)[:,:,0]
    result={'name':name,'source_fields':{'low':low,'high':high,'population':population},
            'field_stats':field_stats,'arms':{}}
    for strength,key in ((.35,'reuse_weak'),(1.,'reuse_full')):
        target,telemetry=render_post(source,NODES,tables[strength])
        target_y=cv2.cvtColor(target,cv2.COLOR_BGR2YCrCb)[:,:,0]
        delta=target_y.astype(np.int16)-input_y.astype(np.int16)
        item={'changed_pixels':int(np.count_nonzero(np.any(target!=source,axis=2))),
              'signed_y_delta_minmax':[int(delta.min()),int(delta.max())],
              'rgb_clipped_channels':telemetry['rgb_clipped_channels'],
              'rgb_clipped_pixels':telemetry['rgb_clipped_pixels'],
              'y_headroom_limited_pixels':telemetry['y_headroom_limited_pixels']}
        if paper is not None:
            item['paper_mean_delta']=float(delta[paper].mean())
            item['paper_p95_abs_delta']=float(np.percentile(np.abs(delta[paper]),95))
        if stroke is not None:
            item['stroke_mean_delta']=float(delta[stroke].mean())
            item['stroke_black_code_merging']=int(np.count_nonzero((target_y[stroke]==0)&(input_y[stroke]>0)))
        if lobe is not None:
            item['ring_relative_to_paper_before']=float(input_y[lobe].mean()-input_y[paper].mean())
            item['ring_relative_to_paper_after']=float(target_y[lobe].mean()-target_y[paper].mean())
            item['ring_signed_amplitude_increase']=float(np.sign(item['ring_relative_to_paper_before'])*
                (item['ring_relative_to_paper_after']-item['ring_relative_to_paper_before']))
            item['ring_increase_gt2_warning']=item['ring_signed_amplitude_increase']>2.
        result['arms'][key]=item
    return result


def occupancy_controls():
    """@luna-comment Input: fixed p around.5 and65square synthetic histograms of widening dark bars. Output: scalar continuity and actual revision2 rail/occupancy counterexamples. Key Steps: evaluate sameY across p and frozen rails on bright220/ink80 strips. Key Params: widths8/24/40/56. State/Dependencies: rails are source-side estimates, not semantic text labels or repaired spatial L3."""
    rows=[]
    for p in (.1,.49,.4999,.5,.5001,.51,.9):
        tables,_=node_tables(*fixed_fields(50.,205.,p),strengths=(1.,))
        rows.append({'p':p,'Y100':float(tables[1.][0,0,100]),
                     'Y150':float(tables[1.][0,0,150])})
    rectangles=[]
    for width in (8,24,40,56):
        patch=np.full((65,65),220,np.uint8)
        left=(65-width)//2
        patch[:,left:left+width]=80
        rails=revision2.rails(np.bincount(patch.ravel(),minlength=256))
        tables,_=node_tables(*fixed_fields(rails['low'],rails['high'],rails['high_fraction']),strengths=(1.,))
        rectangles.append({'dark_bar_width':width,'high_fraction':rails['high_fraction'],
                           'L_unrepaired':rails['low'],'H':rails['high'],
                           'Y110_mapped':float(tables[1.][0,0,110]),
                           'Y150_mapped':float(tables[1.][0,0,150]),
                           'Y110_wrongly_brightened_vs_input':bool(tables[1.][0,0,110]>110.001)})
    return {'fixed_p_continuity':rows,'frozen_rails_dark_bar_occupancy':rectangles,
            'warning_repair_not_applied_to_synthetic_histogram':True}


def equal_luminance_colors():
    """@luna-comment Input: gray/green RGB materials with equal encoded Y. Output: candidate delta/clip distinction. Key Steps: search exact green code matching grayY110 then render half-field. Key Params: fixed high-background fields. State/Dependencies: Y cannot infer semantic material or UV identity."""
    gray=np.array([[[110,110,110]]],np.uint8)
    target_y=int(cv2.cvtColor(gray,cv2.COLOR_BGR2YCrCb)[0,0,0])
    green=next(code for code in range(256) if int(cv2.cvtColor(np.array([[[0,code,0]]],np.uint8),cv2.COLOR_BGR2YCrCb)[0,0,0])==target_y)
    source=np.empty((128,128,3),np.uint8)
    source[:,:64]=gray[0,0]
    source[:,64:]=[0,green,0]
    tables,_=node_tables(*fixed_fields(40.,220.,.9),strengths=(1.,))
    result,stats=render_post(source,NODES,tables[1.])
    return {'gray_bgr':[110,110,110],'green_bgr':[0,green,0],'common_input_y':target_y,
            'gray_output_bgr':result[64,32].tolist(),'green_output_bgr':result[64,96].tolist(),
            'rgb_clipped_channels':stats['rgb_clipped_channels']}


def extended_background_controls():
    """@luna-comment Input: two-paper shadow and a flat scene with spatially stepped source rails/occupancy. Output: both background-side changes and artificial-gradient metrics. Key Steps: exclude ink/lobe from background masks; vary source fields only, never inspect output for parameters. Key Params: lambda.35/1, node x40 step. State/Dependencies: diagnostic falsification, not calibrated source truth."""
    shadow,lobe,_,stroke=gray_scene(185,90,0,shadow=True)
    _,xx=np.mgrid[:shadow.shape[0],:shadow.shape[1]]
    valid=~(stroke|lobe)
    left=valid&(xx<128)
    right=valid&(xx>=128)
    table,_=node_tables(*fixed_fields(70.,220.,.55))
    source_y=cv2.cvtColor(shadow,cv2.COLOR_BGR2YCrCb)[:,:,0]
    result={'two_paper_background':{},'spatial_field_step':{}}
    for strength,key in ((.35,'reuse_weak'),(1.,'reuse_full')):
        target,_=render_post(shadow,NODES,table[strength])
        target_y=cv2.cvtColor(target,cv2.COLOR_BGR2YCrCb)[:,:,0]
        delta=target_y.astype(np.int16)-source_y.astype(np.int16)
        result['two_paper_background'][key]={side:{'mean_signed':float(delta[mask].mean()),
            'p95_abs':float(np.percentile(np.abs(delta[mask]),95)),
            'max_abs':int(np.max(np.abs(delta[mask])))} for side,mask in (('left',left),('right',right))}
    flat=np.full((256,256,3),180,np.uint8)
    low,high,p=fixed_fields(60.,210.,.1)
    low[:,40:]=100.;high[:,40:]=230.;p[:,40:]=.9
    step_tables,_=node_tables(low,high,p)
    for strength,key in ((.35,'reuse_weak'),(1.,'reuse_full')):
        target,_=render_post(flat,NODES,step_tables[strength])
        line=cv2.cvtColor(target,cv2.COLOR_BGR2YCrCb)[128,:,0].astype(np.int16)
        jump=np.diff(line)
        result['spatial_field_step'][key]={'source_constant_y':180,
            'output_minmax':[int(line.min()),int(line.max())],
            'max_adjacent_jump':int(np.max(np.abs(jump))),
            'changed_positions':int(np.count_nonzero(jump)),
            'flat_input_new_range':int(line.max()-line.min())}
    return result


def main():
    """@luna-comment Input: fixed synthetic source/node controls and extended switch. Output: separate immutable JSON of failure probes. Key Steps: basic polarity/span/ring/drift/color or extended background sides/field steps. Key Params: no quality-safe label or auto-parameter search. State/Dependencies: writes only new repo pilot directory."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    destination=OUTPUT.with_name('synthetic_controls_extended.json') if args.extended else OUTPUT
    if destination.exists():
        raise FileExistsError('Synthetic control evidence exists '+str(destination))
    if args.extended:
        result={'status':'diagnostic_only_no_quality_verdict',
                'source_fields_only':True,'output_calibration_enabled':False,
                'extended':extended_background_controls()}
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
        print(str(destination),flush=True)
        return
    cases=[]
    for paper,ink,population in ((220,80,.9),(145,65,.7),(40,180,.1)):
        for ring in (-16,0,16):
            source,lobe,paper_mask,stroke=gray_scene(paper,ink,ring)
            cases.append(evaluate_scene(f'paper{paper}_ink{ink}_ring{ring}',source,
                                        min(paper,ink)-5.,max(paper,ink)+5.,population,
                                        lobe,paper_mask,stroke))
    shadow,lobe,paper_mask,stroke=gray_scene(185,90,0,shadow=True)
    cases.append(evaluate_scene('two_paper_shadow',shadow,70.,220.,.55,lobe,paper_mask,stroke))
    baseline,baseline_lobe,baseline_paper,baseline_stroke=gray_scene(220,80,0)
    for gain,bias in ((.9,-8.),(1.1,12.)):
        drift=np.clip(baseline.astype(np.float32)*gain+bias,0,255).astype(np.uint8)
        cases.append(evaluate_scene(f'fixed_source_rails_model_gain{gain}_bias{bias}',drift,75.,225.,.9,
                                    baseline_lobe,baseline_paper,baseline_stroke))
    for span in (0.,4.,24.,120.):
        source,_,_,_=gray_scene(145,100,0)
        cases.append(evaluate_scene(f'span{span}',source,100.,100.+span,.9))
    result={'status':'diagnostic_only_no_quality_verdict','domain':'synthetic PNG-derived Y8 RGB-clipped preview',
            'source_fields_only':True,'output_calibration_enabled':False,
            'cases':cases,'occupancy':occupancy_controls(),
            'equal_luminance_colors':equal_luminance_colors(),
            'warnings':{'ring_increment_gt2':[(case['name'],arm) for case in cases for arm,detail in case['arms'].items()
                                            if detail.get('ring_increase_gt2_warning')]}}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'cases':len(cases),'ring_warnings':len(result['warnings']['ring_increment_gt2'])}),flush=True)
    print(str(destination),flush=True)


if __name__=='__main__':
    main()
