"""Read-only-input diagnostics for frozen V14 unsupported background anchors."""
import json
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_polarity_v14 import prepare_scoped
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_clean_background_v13 import estimate_background
from tools.postprocess_ppt_dehalo_20260924 import write_png,smoothstep
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct

if __name__=='__main__':
    cv2.setNumThreads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=int,default=10)
    args=parser.parse_args()
    manifest=json.loads(Path('Y:/guang.yang/TextSR/data/lenovo_test/results_dehalo_v14_20260924_v2/manifest.json').read_text())
    route=manifest['jobs'][args.index]
    source=read_image(Path(route['source'].replace('/AI_DNS/','Y:/')))
    roi=np.zeros(source.shape[:2],np.uint8)
    cv2.fillPoly(roi,[np.array(route['polygon'],np.int32)],1)
    normalized=255-source if route['polarity']=='light-on-dark' else source
    p=prepare_scoped(normalized,roi.astype(bool))
    src,y,roi,core,bg,_,_=p
    dist=distance_transform_edt(~(roi&(cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)-y>12))).astype(np.float32)
    target,debug=estimate_background(src,y,roi,dist)
    _,maps,_=normal_correct(p)
    base=src+maps['correction_weight'][...,None]*(bg-src)
    actual=core|(roi&(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)-y>12))
    d=distance_transform_edt(~actual).astype(np.float32)
    rr=smoothstep(distance_transform_edt(roi).astype(np.float32)/12)
    w=smoothstep((d-6)/4)*(1-smoothstep((d-32)/48))*rr
    w[core|(d<=6)|~roi]=0
    excess=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)-cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    h=smoothstep((excess-.5)/2)*(1-smoothstep((d-10)/6))*smoothstep(d)*rr
    h[actual|core|~roi]=0
    w=w+(1-w)*h
    bad=(w>0)&(debug['support96']<1e-5)
    yy,xx=np.nonzero(bad)
    out=Path(f'results/ppt_dehalo_20260924/v14_anchor_failure_v2_{args.index:03d}')
    overlay=source.copy();overlay[bad]=(0,0,255)
    for key,im in [('source',source),('unsupported_overlay',overlay),('unsupported',bad.astype(np.uint8)*255),('anchors',np.rint(debug['anchor_weight']*255).astype(np.uint8)),('roi',roi.astype(np.uint8)*255),('support96',np.clip(debug['support96']*255,0,255).astype(np.uint8)),('core',core.astype(np.uint8)*255)]:write_png(out/(key+'.png'),im)
    n,labels,stats,_=cv2.connectedComponentsWithStats(bad.astype(np.uint8))
    report=dict(index=args.index,count=len(xx),bounds=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None,min_support=float(debug['support96'][bad].min()) if len(xx) else None,core_count=int(core.sum()),anchors=int((debug['anchor_weight']>0).sum()),components=stats[1:].tolist())
    print(json.dumps(report),flush=True)
