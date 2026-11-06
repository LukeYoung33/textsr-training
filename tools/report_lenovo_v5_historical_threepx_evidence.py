"""Unified clipped-float 3px ROI evidence for preserved historical branch dumps."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path('results/historical_branch_dump/historical_original_v2'); OUT=Path('results/r1_historical_threepx_current_evidence.json')
ROIS={'stroke_a':(slice(14,39),slice(171,177)),'stroke_b':(slice(55,80),slice(197,201))}
def y(a):
 a=np.clip(a,0,1); return .299*a[...,0]+.587*a[...,1]+.114*a[...,2]
def fft(a,roi):
 z=y(a)[roi].mean(1); q=np.arange(len(z));A=np.stack([np.ones(len(z)),q,q*q],1);z=(z-A@np.linalg.lstsq(A,z,rcond=None)[0])*np.hanning(len(z));return np.fft.rfft(z),np.fft.rfftfreq(len(z))
def main():
 out={'contract':'clipped float RGB to luma; row-average ROI; quadratic detrend; Hann; .25-.4 c/p','rois':{}}
 for rn,roi in ROIS.items():
  arms={}
  for arm in ('c0','r1'):
   vals={n:np.load(ROOT/arm/(n+'.npy')) for n in ('input','local_only','context_only','full')}; C={n:fft(v,roi)[0] for n,v in vals.items()};f=fft(vals['full'],roi)[1];m=(f>=.25)&(f<=.4);full=C['full']
   arms[arm]={n:{'band_energy':float((abs(c[m])**2).sum()),'phase_cosine_to_full':float(np.real(np.vdot(full[m],c[m]))/(np.linalg.norm(full[m])*np.linalg.norm(c[m]))) if np.linalg.norm(c[m]) else None} for n,c in C.items()}
  out['rois'][rn]=arms
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(OUT)
if __name__=='__main__':main()
