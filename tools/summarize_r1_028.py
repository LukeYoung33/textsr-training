"""Mass-weighted R1 conflict and equal-error reference, from immutable diagnostic maps."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from tools.diagnose_r1_028 import OUT,widths,overlay,image8
from tools.run_uniform20_steep_scurve import save


def main():
    """@luna-comment Input:verified28-exposure audit. Output:mass-weighted conflict/equal-error statistics andR1relative overlays. Key Steps:sumactualR1mass,cancelledmass,netreversedmass;actualbatchdenominators foruniform0.06error. Key Params:zero-massnull;referenceatloss-entryY,notRGBchain. State/Dependencies:owned analysis derivatives only."""
    tests=[]
    for n in [1,2,3,4,6,8]:
        mask=np.zeros((41,41),bool);mask[4:37,16:16+n]=True;w=widths(mask);measured=float(np.median(w[15:25,16:16+n]));assert abs(measured-n)<.01,(n,measured);tests.append(dict(nominal=n,measured=measured))
    r=json.loads((OUT/'REPORT.json').read_text());stats=r['statistics']
    for values in stats.values():
        for v in values.values():v.update(opposed_gradient_mass=0.,cancelled_gradient_mass=0.,reversed_gradient_mass=0.,equal_error_gradient_mass=0.)
    for patch in r['patches']:
        d=np.load(Path(patch['folder'])/'maps.npz');bins=np.where(d['width']<=2.00001,0,np.where(d['width']<=4.00001,1,2));batch=r['batches'][patch['batch']]
        for kind,support,normalizer,weight in [('missing','support_m','text_pixels',1.),('false','support_f','ring_pixels',.25)]:
            g=d[kind];rest=d['total']-g;opposed=g*rest<0;reversed_=g*d['total']<0
            for b in range(3):
                sel=(bins==b)&(d[support]>0);v=stats[kind][str(b)]
                v['opposed_gradient_mass']+=float(np.abs(g)[sel&opposed].sum());v['cancelled_gradient_mass']+=float(np.minimum(np.abs(g),np.abs(rest))[sel&opposed].sum());v['reversed_gradient_mass']+=float(np.abs(g)[sel&reversed_].sum());v['equal_error_gradient_mass']+=float(sel.sum())*weight/(1+math.exp(-3))/max(batch[normalizer],1e-6)
    for values in stats.values():
        eq=sum(v['equal_error_gradient_mass'] for v in values.values())
        for v in values.values():
            for key in ['opposed','cancelled','reversed']:v[key+'_gradient_fraction']=v[key+'_gradient_mass']/max(v['gradient_sum'],1e-30)
            v['equal_error_expected_share']=v['equal_error_gradient_mass']/max(eq,1e-30)
    for view in r['views']:
        if view['id']=='000':folder=OUT/'full';d=np.load(OUT/'full_maps.npz')
        else:folder=OUT/'patches'/view['id'];d=np.load(folder/'maps.npz')
        base=cv2.imread(str(folder/'output.png'));g=d['r1'];rest=d['total']-g;strength=np.abs(g)/(np.abs(g)+np.abs(rest)+1e-20)
        value=overlay(base,strength,1.);path=folder/'r1_relative.png';assert cv2.imwrite(str(path),value)
        view['images']['r1_relative']=str(path.relative_to(OUT)).replace('\\','/')
    r['width_tests']=tests;r['equal_error_reference']='Same0.06signedviolation atloss-entryY;exactcohort foreground/ring denominators;excludesdownstreamRGBgamutchain.'
    save(OUT/'REPORT.json',r)
    print(json.dumps({k:{b:{x:v[x] for x in ['gradient_share','cancelled_gradient_fraction','reversed_gradient_fraction','equal_error_expected_share']} for b,v in a.items()} for k,a in stats.items()}))


if __name__=='__main__':main()
