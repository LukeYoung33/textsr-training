"""Frozen J J^T sinusoidal cotangent gain probe (R1/C0, stage2[1])."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from basicsr.archs import build_network
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.losses import basic_text_line_y4uv_v3_losses as _reg
from realesrgan.utils import YUV2RGB

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True); p.add_argument('--r1-checkpoint',type=Path,required=True); p.add_argument('--c0-checkpoint',type=Path,required=True); p.add_argument('--historical-lq',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--eta',type=float,default=1e-5); return p.parse_args()
def rgb(p,d):
 with Image.open(p) as im: a=np.asarray(im.convert('RGB'),np.float64)/255
 return torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1)[None].to(d)
def dec(y): return YUV2RGB(y4uv_to_yuv444(y),format='RGB',is_train=True,uv_normalized=True)
def cotangent(h,w,kind,freq,phase,d):
 y,x=torch.meshgrid(torch.arange(h,device=d,dtype=torch.float64),torch.arange(w,device=d,dtype=torch.float64),indexing='ij')
 if kind=='horizontal': z=torch.sin(2*np.pi*freq*y+phase)
 elif kind=='vertical': z=torch.sin(2*np.pi*freq*x+phase)
 elif kind=='diag45': z=torch.sin(2*np.pi*freq*(x+y)/np.sqrt(2)+phase)
 else: z=torch.sin(2*np.pi*freq*(x-y)/np.sqrt(2)+phase)
 return z/torch.linalg.vector_norm(z)
def main():
 a=args(); root=a.output_root.resolve();
 if root.exists(): raise FileExistsError(root)
 root.mkdir(parents=True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 d=torch.device(a.device); doc=yaml.safe_load(a.config.read_text()); lq=rgb_to_y4uv(rgb(a.historical_lq,d)).double(); report={'status':'COMPLETE','contract':'equal-L2 native-output Y cotangents; eval J J^T through stage2[1] only; no checkpoint writes','eta':a.eta,'runs':{}}
 for arm,ckpt in [('r1',a.r1_checkpoint),('c0',a.c0_checkpoint)]:
  net=build_network(doc['network_g']).to(d).double(); report[arm]=load_strict_checkpoint(net,ckpt,map_location='cpu'); net.eval()
  with torch.no_grad(): base=dec(net(lq))
  pars=list(net.stage2[1].parameters()); h,w=base.shape[-2:]
  for kind in ('horizontal','vertical','diag45','diag135'):
   for freq in (.08,.20,.32):
    responses=[]
    for phase in (0.,np.pi/2):
     out=net(lq); v=cotangent(h,w,kind,freq,phase,d); vy4=F.pixel_unshuffle(v[None,None],2)
     g=torch.autograd.grad((out[:,:4]*vy4).sum(),pars)
     gram=float(sum((q*q).sum() for q in g).detach()) # v^T J J^T v in native-Y coordinates
     clone=copy.deepcopy(net); names=dict(clone.named_parameters()); orig=dict(net.named_parameters())
     with torch.no_grad():
      for p,q in zip(pars,g): names[next(k for k,z in orig.items() if z is p)].add_(-a.eta*q)
      response=F.pixel_shuffle(clone(lq)[:,:4],2)-F.pixel_shuffle(out[:,:4],2)
     r=response[:,0]
     responses.append({'projection':float((r*v).sum()),'projection_over_eta':float((r*v).sum()/a.eta),'jacobian_gram_vJJTv':gram,'rms':float(torch.sqrt((r*r).mean()))})
    report['runs'][f'{arm}/{kind}/{freq}']={'phase_mean_projection':float(np.mean([x['projection'] for x in responses])),'phase_mean_projection_over_eta':float(np.mean([x['projection_over_eta'] for x in responses])),'phase_mean_jacobian_gram':float(np.mean([x['jacobian_gram_vJJTv'] for x in responses])),'phase_mean_rms':float(np.mean([x['rms'] for x in responses])),'phases':responses}
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'status':'COMPLETE','report':str(root/'report.json')}))
if __name__=='__main__': main()
