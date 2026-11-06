"""Decode preserved Y4UV context residuals and enforce historical RGB closure."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
from realesrgan.utils import YUV2RGB

ROOT=Path('results'); BRANCH=ROOT/'historical_branch_dump/historical_original_v2/r1'
FREQ=ROOT/'r1_rca_stage2_frequency_20260918/r1/r1'
def decode(y4):
 t=torch.from_numpy(np.asarray(y4)).unsqueeze(0).float()
 return np.clip(YUV2RGB(y4uv_to_yuv444(t),format='RGB',is_train=False,uv_normalized=True)[0].permute(1,2,0).numpy(),0.,1.)
def main():
 inp=np.load(BRANCH/'input_y4uv.npy'); target=np.load(BRANCH/'context_only.npy'); out={}
 roots=[FREQ,ROOT/'r1_rca_context_depth_20260918/run_1600/r1_context_depth_ablation_20260918_1600/r1',ROOT/'r1_rca_stage2_probe_20260918/run_1700/r1_stage2_probe_20260918_1700/r1',ROOT/'r1_rca_cross_context_20260919/run_0900/r1_cross_context_probe_20260919_0900',ROOT/'r1_rca_stage2_branch_20260919/run_1210/r1_stage2_branch_probe_20260919_1210']
 paths=[p for root in roots if root.is_dir() for p in root.glob('*context_residual_y4uv.npy')]
 rois={'stroke_a':(slice(14,39),slice(171,177)),'stroke_b':(slice(55,80),slice(197,201))}
 def band(a,roi):
  z=(.299*a[...,0]+.587*a[...,1]+.114*a[...,2])[roi].mean(1);q=np.arange(len(z));A=np.stack([np.ones(len(z)),q,q*q],1);c=np.fft.rfft((z-A@np.linalg.lstsq(A,z,rcond=None)[0])*np.hanning(len(z)));f=np.fft.rfftfreq(len(z));return c,(f>=.25)&(f<=.4)
 for path in paths:
  rgb=decode(inp+np.load(path)); diff=rgb-target
  entry={'max_abs_vs_historical_context_only':float(abs(diff).max()),'rms_vs_historical_context_only':float(np.sqrt((diff*diff).mean())),'shape':list(rgb.shape),'threepx':{}}
  for n,roi in rois.items():
   c,m=band(rgb,roi); ref,_=band(target,roi);entry['threepx'][n]={'energy':float((abs(c[m])**2).sum()),'phase_cosine_to_context_only':float(np.real(np.vdot(ref[m],c[m]))/(np.linalg.norm(ref[m])*np.linalg.norm(c[m]))) if np.linalg.norm(c[m]) else None}
  out[str(path.relative_to(ROOT))]=entry
 # Display only: fixed native crop, nearest-neighbour fourfold enlargement.
 picks=[('1 native_context',BRANCH/'context_only.npy'),('2 block0',FREQ/'block0_context_residual_y4uv.npy'),('3 mid',ROOT/'r1_rca_stage2_probe_20260918/run_1700/r1_stage2_probe_20260918_1700/r1/tmid_half_increment_context_residual_y4uv.npy'),('4 full_block1',FREQ/'full_block1_context_residual_y4uv.npy'),('5 cross_r1block1_c0tail',ROOT/'r1_rca_cross_context_20260919/run_0900/r1_cross_context_probe_20260919_0900/h2_r1block1_c0tail_context_residual_y4uv.npy'),('6 cross_r1delta_r1tail',ROOT/'r1_rca_cross_context_20260919/run_0900/r1_cross_context_probe_20260919_0900/h3_r1delta_r1tail_context_residual_y4uv.npy')]
 tiles=[]
 for name,path in picks:
  im=np.load(path) if 'native_context' in name else decode(inp+np.load(path)); crop=np.clip(im[8:88,156:228],0,1); tile=Image.fromarray(np.rint(crop*255).astype('uint8')).resize((288,320),Image.Resampling.NEAREST);ImageDraw.Draw(tile).rectangle((0,0,160,18),fill='black');ImageDraw.Draw(tile).text((3,3),name,fill='white');tiles.append(tile)
 sheet=Image.new('RGB',(288*3,320*2));[sheet.paste(tile,((i%3)*288,(i//3)*320)) for i,tile in enumerate(tiles)];sheet.save('results/r1_residual_closure_contact_sheet.png')
 Path('results/r1_residual_closure_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
