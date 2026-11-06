"""Reanalyse saved frozen-update tensors at the observed three-pixel stroke bands."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

ROIS={'stroke_a':(slice(14,39),slice(171,177)),'stroke_b':(slice(55,80),slice(197,201))}
def profile(z,roi): return np.asarray(z)[0,0,roi[0],roi[1]].mean(1)
def display_y(rgb):
 a=np.clip(np.asarray(rgb),0.,1.); return (.299*a[:,0:1]+.587*a[:,1:2]+.114*a[:,2:3])
def coeff(z):
 n=len(z); q=np.arange(n); A=np.stack([np.ones(n),q,q*q],1); trend=A@np.linalg.lstsq(A,z,rcond=None)[0]
 return np.fft.rfft((z-trend)*np.hanning(n)),np.fft.rfftfreq(n)
def measure(error,response,roi):
 E,f=coeff(profile(error,roi)); R,_=coeff(profile(response,roi)); m=(f>=.25)&(f<=.4)
 before=float((abs(E[m])**2).sum()); resp=float((abs(R[m])**2).sum()); signed=float((2*np.real(np.conj(E[m])*R[m])).sum())
 return {'n_rows':len(profile(error,roi)),'frequencies':f[m].tolist(),'before_energy':before,'response_energy':resp,'signed_dE':signed,'after_energy':before+signed+resp,'absolute_energy_change':before+signed+resp-before}
def main():
 p=argparse.ArgumentParser();p.add_argument('--tensors',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();z=np.load(a.tensors);base=z['baseline_y'];gt=z['gt_y'];c0=z['c0_y'];out={'contract':'row profile = mean over stated x range; quadratic detrend plus Hann; .25-.4 cycles/pixel is observed 3px candidate','rois':{}}
 domains={'raw':(base,gt,c0),'display_clipped_float':(display_y(z['baseline_rgb']),display_y(z['gt_rgb']),display_y(z['c0_rgb']))}
 for rn,roi in ROIS.items():
  domain_rows={}
  for domain,(b,g,c) in domains.items():
   rows={}
   for k in z.files:
    if not k.endswith('_y') or k in ('baseline_y','c0_y','gt_y'):continue
    key=k[:-2]; post=z[k] if domain=='raw' else display_y(z[key+'_rgb']); response=post-b
    rows[key]={'GT':measure(b-g,response,roi),'C0':measure(b-c,response,roi),'RAW_no_reference':measure(b,response,roi)}
   bf,fr=coeff(profile(b,roi)); domain_rows[domain]={'profiles':{'baseline':profile(b,roi).tolist(),'gt':profile(g,roi).tolist(),'c0':profile(c,roi).tolist()},'baseline_profile_fft':{'frequencies':fr.tolist(),'power':(abs(bf)**2).tolist()},'rows':rows}
  out['rois'][rn]=domain_rows
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','out':str(a.out),'cells':len(rows)}))
if __name__=='__main__':main()
