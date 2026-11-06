"""Image-space loss-landscape probe on one handcrafted stroke, not a model output.

Creates a white canvas with a black H-shaped stroke, then independently injects
gap/fade/width/spur/blur defects. For each defect it measures loss excess,
gradient magnitude, correction-direction alignment, and target-ROI gradient
concentration. clDice is intentionally measured in its real ``p``-map space.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from realesrgan.losses.crnn_content_perceptual_loss import CRNNTextLineTileContentLoss
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses
from realesrgan.losses.stroke_core_overshoot_loss import StrokeCoreLoss, OvershootLoss
from realesrgan.losses.gt_sdf_boundary_loss import GTSDFBoundaryLoss
from realesrgan.losses.stroke_geometry_losses import SoftCLDiceLoss

H,W=96,480
def line(a,p1,p2,t=5,c=0): cv2.line(a,p1,p2,c,t,cv2.LINE_8)
def tensor(a,dev,grad=False): return torch.from_numpy(a).to(dev).permute(2,0,1).unsqueeze(0).float().requires_grad_(grad)
def mask_tensor(a,dev): return torch.from_numpy(a[None,None]).to(dev).float()
def craft():
 g=np.ones((H,W,3),np.float32); m=np.zeros((H,W),np.float32)
 for a,c in ((g,(0,0,0)),(m,1)):
   line(a,(165,22),(165,74),5,c); line(a,(300,22),(300,74),5,c); line(a,(165,48),(300,48),5,c)
   # Dense-stroke target: three 5 px stems separated by only 3 px white gaps.
   line(a,(350,22),(350,74),5,c); line(a,(358,22),(358,74),5,c); line(a,(366,22),(366,74),5,c)
 return g,m
def variants(gt,s=1.0):
 out={}; roi={}
 def add(name,fn,r):
  x=gt.copy(); fn(x); out[name]=x; roi[name]=r
 n=max(2,int(round(18*s))); r=np.zeros((H,W),bool); r[45:52,233-n//2:233-n//2+n]=1
 add('gap',lambda x:x.__setitem__((slice(45,52),slice(233-n//2,233-n//2+n)),1.),r)
 # Fading occurs at a stroke endpoint, not in the middle of a solid stroke.
 r=np.zeros((H,W),bool); r[19:37,161:170]=1
 add('fade_tip',lambda x:x.__setitem__((slice(19,37),slice(161,170)),1-.35*s),r)
 r=np.zeros((H,W),bool); r[39:58,205:265]=1
 add('width_thick',lambda x:line(x,(205,48),(265,48),5+int(round(6*s)),(0,0,0)),r)
 r=np.zeros((H,W),bool); r[45:52,265:285]=1
 def thin(x):
  x[44:53,264:286]=1.; line(x,(265,48),(285,48),max(1,5-int(round(4*s))),(0,0,0))
 add('width_thin',thin,r)
 # A short tapered triangular spur, rather than a secondary long stroke.
 r=np.zeros((H,W),bool); r[39:48,231:240]=1
 def spur(x): cv2.fillConvexPoly(x,np.asarray([[232,47],[238,47],[235,int(47-8*s)]],np.int32),(0,0,0))
 add('spur_tip',spur,r)
 r=np.zeros((H,W),bool); r[18:78,155:310]=1
 add('blur',lambda x:x.__setitem__(slice(None),cv2.GaussianBlur(x,(0,0),2.0*s)),r)
 # Mimic zoom + blur: nearest enlargement locally closes the 2–3 px gaps,
 # then Gaussian filtering creates the joined, rounded dense-stroke artifact.
 r=np.zeros((H,W),bool); r[18:78,345:371]=1
 def dense_merge(x):
  crop=x[18:78,345:371]; zoom=cv2.resize(crop,None,fx=1+.35*s,fy=1.,interpolation=cv2.INTER_NEAREST)
  zoom=cv2.GaussianBlur(zoom,(0,0),1.15*s); x[18:78,345:371]=cv2.resize(zoom,(26,60),interpolation=cv2.INTER_LINEAR)
 add('dense_merge',dense_merge,r)
 return out,roi
def stats(loss,x,gt,roi):
 grad=torch.autograd.grad(loss,x,retain_graph=False)[0].detach(); d=(gt-x.detach())
 flat=lambda z:z.reshape(-1)
 align=float((flat(-grad)*flat(d)).sum()/(flat(grad).norm()*flat(d).norm()+1e-12))
 mag=grad.abs().sum(); region=grad.abs()[0,:,roi].sum()
 return float(loss.detach()),float(grad.norm()),align,float(region/(mag+1e-12))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--crnn',type=Path,required=True); p.add_argument('--alphabet',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 d=torch.device(a.device); gt_np,m_np=craft(); valid=np.ones_like(m_np); gt=tensor(gt_np,d); m=mask_tensor(m_np,d); v=mask_tensor(valid,d)
 cp=CRNNTextLineTileContentLoss(recognizer_path=str(a.crnn),alphabet_path=str(a.alphabet),micro_batch_size=1).to(d).eval()
 core=StrokeCoreLoss(max_skeleton_iterations=96).to(d); over=OvershootLoss(kernel_size=5,tol=2/255,mask_radius=2).to(d); sdf=GTSDFBoundaryLoss(radius=3,sdf_temperature=1.,ink_temperature=.05,anchor_radius=3,min_contrast=.03,mask_threshold=.5,profile_loss='bce',boundary_width=1).to(d); cl=SoftCLDiceLoss(alpha=.4,skeleton_iterations=5,max_skeleton_iterations=96,anchor_radius=3,min_contrast=.01,ink_temperature=.05).to(d)
 meta={'valid_hw':[(H,W)],'orientation':['HORIZONTAL'],'cp_eligible':[True]}
 clean={'l1':0.,'cp':0.,'r1_missing':0.,'r1_false':0.,'core':0.,'overshoot':0.,'sdf':0.,'cldice_p':0.}
 rows=[]
 for severity in (.25,.5,.75,1.):
  cases,rois=variants(gt_np,severity)
  cases={'clean':gt_np,**cases}; rois={'clean':m_np>0.5,**rois}
  for name,img in cases.items():
   x=tensor(img,d,True); roi=torch.from_numpy(rois[name]).to(d)
   funcs={'l1':lambda:torch.abs(x-gt).mean(),'cp':lambda:cp(x,gt,meta),'r1_missing':lambda:recall_aware_structure_losses(x,gt,m,v)[0],'r1_false':lambda:recall_aware_structure_losses(x,gt,m,v)[1],'core':lambda:core(x,gt,m,v),'overshoot':lambda:over(x,gt,m,v),'sdf':lambda:sdf(x,gt,m,v)}
   for key,fn in funcs.items():
    value,norm,align,conc=stats(fn(),x,gt,roi); rows.append({'case':name,'severity':severity,'loss':key,'space':'image_x','value':value,'grad_norm':norm,'repair_alignment':align,'roi_concentration':conc})
  # Actual clDice variable is the auxiliary probability map, not x. p is a deterministic probe surrogate only.
   prob=(1-torch.from_numpy(img.mean(2)).to(d)).clamp(0,1).view(1,1,H,W).requires_grad_(True)
   loss=cl(x.detach(),gt,m,v,pred_probability=prob); grad=torch.autograd.grad(loss,prob)[0]; delta=(m-prob.detach()); al=float(((-grad)*delta).sum()/(grad.norm()*delta.norm()+1e-12)); con=float(grad.abs()[0,0,roi].sum()/(grad.abs().sum()+1e-12)); rows.append({'case':name,'severity':severity,'loss':'cldice_p','space':'probability_p','value':float(loss.detach()),'grad_norm':float(grad.norm()),'repair_alignment':al,'roi_concentration':con})
 base={(r['severity'],r['loss']):r['value'] for r in rows if r['case']=='clean'}; l1={(r['severity'],r['case']):r['value'] for r in rows if r['loss']=='l1'}
 for r in rows:
  r['excess_over_clean']=r['value']-base[(r['severity'],r['loss'])]
  r['excess_per_l1']=r['excess_over_clean']/max(l1[(r['severity'],r['case'])],1e-8)
 with (a.output/'metrics.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'metrics.json').write_text(json.dumps({'contract':'handcrafted H-stroke; values are raw loss values, gradients are with respect to x except cldice_p with respect to p','rows':rows},indent=2),encoding='utf8')
 panel=np.concatenate([gt_np,*variants(gt_np,1.)[0].values()],axis=1); cv2.imwrite(str(a.output/'cases.png'),cv2.cvtColor((panel*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
 for metric in ('repair_alignment','roi_concentration'):
  cases_order=list(variants(gt_np,1.)[0]); fig,axes=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
  for ax,name in zip(axes.flat,cases_order):
   for loss_name in ('l1','cp','r1_missing','r1_false','core','overshoot','sdf'):
    g=sorted([z for z in rows if z['case']==name and z['loss']==loss_name],key=lambda z:z['severity'])
    ax.plot([z['severity'] for z in g],[z[metric] for z in g],marker='o',label=loss_name)
   ax.set(title=name,xlabel='severity',ylabel=metric);ax.grid(alpha=.25)
  axes.flat[-1].axis('off'); axes.flat[0].legend(fontsize=7,ncol=2);fig.savefig(a.output/f'{metric}_curves.png',dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(7,4));
  for name in cases_order:
   g=sorted([z for z in rows if z['case']==name and z['loss']=='cldice_p'],key=lambda z:z['severity'])
   ax.plot([z['severity'] for z in g],[z[metric] for z in g],marker='o',label=name)
  ax.set(xlabel='severity',ylabel=f'cldice_p {metric}');ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2);fig.savefig(a.output/f'cldice_p_{metric}_curves.png',dpi=180);plt.close(fig)
 print(a.output/'metrics.csv')
if __name__=='__main__':main()
