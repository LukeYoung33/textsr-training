"""Append-only C0/R1 checkpoint scan for the observed 3px historical texture."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from basicsr.archs import build_network
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv,y4uv_to_yuv444
from realesrgan.utils import YUV2RGB
import realesrgan.losses.basic_text_line_y4uv_v3_losses
ROIS={'a':(slice(14,39),slice(171,177)),'b':(slice(55,80),slice(197,201))}
def h(p):
 d=hashlib.sha256();d.update(Path(p).read_bytes());return d.hexdigest()
def metric(rgb):
 y=.299*rgb[...,0]+.587*rgb[...,1]+.114*rgb[...,2];out={}
 for n,r in ROIS.items():
  z=y[r].mean(1);q=np.arange(len(z));A=np.stack([np.ones(len(z)),q,q*q],1);c=np.fft.rfft((z-A@np.linalg.lstsq(A,z,rcond=None)[0])*np.hanning(len(z)));f=np.fft.rfftfreq(len(z));m=(f>=.25)&(f<=.4);out[n]={'mean_luminance':float(z.mean()),'deprecated_mean_darkness_same_value':float(z.mean()),'contrast':float(z.max()-z.min()),'band_energy':float((abs(c[m])**2).sum())}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--lq',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();a.out.mkdir(parents=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 d=torch.device(a.device);doc=yaml.safe_load(a.config.read_text());im=np.asarray(Image.open(a.lq).convert('RGB'),np.float32)/255;lq=rgb_to_y4uv(torch.from_numpy(im).permute(2,0,1)[None]).to(d); rows=[]
 for arm in ('c0','r1'):
  for ck in sorted((a.root/f'lenovo_v5_{arm}_250k_20260912/models').glob('net_g_[0-9]*.pth'),key=lambda x:int(x.stem.split('_')[-1])):
   net=build_network(doc['network_g']).to(d);load_strict_checkpoint(net,ck,map_location='cpu');net.eval()
   with torch.no_grad(): rgb=YUV2RGB(y4uv_to_yuv444(net(lq)),format='RGB',is_train=False,uv_normalized=True)[0].permute(1,2,0).cpu().numpy();rgb=np.clip(rgb,0,1)
   np.save(a.out/f'{arm}_{ck.stem}.npy',rgb.astype('float32'));rows.append({'arm':arm,'iteration':int(ck.stem.split('_')[-1]),'checkpoint':str(ck),'metrics':metric(rgb)})
 report={'status':'COMPLETE','tf32_disabled':True,'config_sha256':h(a.config),'rows':rows};(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','rows':len(rows)}))
if __name__=='__main__':main()
