"""Frozen, append-only signed local-SGD mechanism audit for Lenovo V5 R1.

The experiment keeps one real synthesized training batch fixed while obtaining
output covectors and a single global-rate SGD direction.  It separately measures
the response on the historical 076_000446/v2 image; PNG/RGB is never used as a
gradient coordinate system.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.losses import build_loss
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444, yuv420_mask_to_y4uv
from realesrgan.lenovo_degradation import LenovoDegradationPipeline
from realesrgan.losses import basic_text_line_y4uv_v3_losses as _registration
from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses
from realesrgan.utils import YUV2RGB
from tools.audit_lenovo_v5_r1_stage2_gradients import _collect_batch


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--r1-checkpoint', required=True, type=Path)
    p.add_argument('--c0-checkpoint', required=True, type=Path)
    p.add_argument('--historical-lq', required=True, type=Path)
    p.add_argument('--historical-gt', required=True, type=Path)
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--indices', default='0,1')
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--eta', default='1e-3,3e-3', help='two global pure-SGD rates; central +/- check rejects non-linear values')
    p.add_argument('--sources', default='native,missing,false_weighted,extra')
    p.add_argument('--groups', default='stage2_1,b3,b13')
    p.add_argument('--dtype', choices=('float32','float64'), default='float64')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def _rgb(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix.lower()=='.npy':
        a=np.load(path).astype(np.float32)
        if a.ndim==3 and a.shape[0] in (3,6): a=np.moveaxis(a,0,-1)
        if a.ndim!=3 or a.shape[-1] not in (3,6): raise ValueError(f'expected HWC RGB/Y4UV npy: {path} {a.shape}')
        if a.shape[-1]==6: raise ValueError('pass historical RGB npy/PNG here; packed Y4UV has a separate contract')
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).unsqueeze(0).to(device)
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).to(device)


def _decode(y4: torch.Tensor) -> torch.Tensor:
    return YUV2RGB(y4uv_to_yuv444(y4), format='RGB', is_train=True, uv_normalized=True)

def _unpack_y4_luma_covector(g: torch.Tensor) -> torch.Tensor:
    """Expose the four luma phase coordinates at native resolution for spectra."""
    return F.pixel_shuffle(g[:, :4], 2)

def _debug_with_stage2(net: torch.nn.Module, x: torch.Tensor) -> tuple[dict[str,torch.Tensor],torch.Tensor]:
    got=[]; hook=net.stage2[1].register_forward_hook(lambda _m,_i,o: got.append(o))
    try: debug=net(x,return_debug=True)
    finally: hook.remove()
    if len(got)!=1: raise RuntimeError(f'expected one stage2[1] output, got {len(got)}')
    return debug,got[0]


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return .299 * rgb[:, :1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]


def _losses(net: torch.nn.Module, lq: torch.Tensor, gt: torch.Tensor, gt_rgb: torch.Tensor, valid: torch.Tensor, text: torch.Tensor, valid_rgb: torch.Tensor, text_rgb: torch.Tensor, native_loss: torch.nn.Module, opt: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    y4 = net(lq)
    rgb = _decode(y4)
    native = native_loss(y4, gt, valid_mask=valid, text_mask=text)
    missing, false, _ = recall_aware_structure_losses(rgb, gt_rgb, text_rgb, valid_rgb, **opt.get('recall_aware_opt', {}))
    fw = float(opt.get('recall_aware_weight_false', 0.0))
    return {'native': native, 'missing': missing, 'false_weighted': fw * false, 'extra': float(opt.get('recall_aware_weight_missing', 0.0)) * missing + fw * false}, y4, rgb


def _groups(net: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    b = net.stage2[1]
    return {'stage2_1': list(b.parameters()), 'b3': [b.conv3x3.weight, b.conv3x3.bias], 'b13': [b.conv1x1.weight, b.conv1x1_3x3.weight, b.conv1x1_3x3.bias]}


def _detrended_power(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel-first power; direction labels retain fy/fx semantics."""
    a = x.detach().float().cpu().numpy()
    if a.ndim == 3: a = a[None]
    b, ch, h, w = a.shape
    yy, xx = np.mgrid[:h, :w]; design = np.stack([np.ones(h*w), yy.ravel(), xx.ravel()], 1)
    window = np.outer(np.hanning(h), np.hanning(w))
    p = []
    for sample in a:
        row=[]
        for c in sample:
            coef, *_ = np.linalg.lstsq(design, c.ravel(), rcond=None)
            z = (c - (design @ coef).reshape(h,w)) * window
            row.append(np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2)
        p.append(row)
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(h)), np.fft.fftshift(np.fft.fftfreq(w)), indexing='ij')
    return np.asarray(p), fy, fx


def _spectrum(x: torch.Tensor) -> dict[str, float]:
    p, fy, fx = _detrended_power(x) # power is summed only after channel FFT
    q = p.sum(1); r = np.sqrt(fx*fx + fy*fy); nz = r > 0
    masks = {'all_non_dc': nz, 'horizontal_fy': (np.abs(fy)>2*np.abs(fx)) & nz, 'vertical_fx': (np.abs(fx)>2*np.abs(fy)) & nz,
             'diag45': (np.abs(fy-fx)<.04) & nz, 'diag135': (np.abs(fy+fx)<.04) & nz,
             'horizontal_band_005_010': (np.abs(fy)>2*np.abs(fx)) & (np.abs(fy)>=.05) & (np.abs(fy)<=.10)}
    per_sample=[{k: float(one[m].mean()) if m.any() else 0. for k,m in masks.items()} for one in q]
    return {'per_sample':per_sample, 'mean':{k:float(np.mean([z[k] for z in per_sample])) for k in masks}}


def _signed_projection(error: torch.Tensor, response: torch.Tensor, roi: tuple[slice,slice]) -> dict[str, float]:
    # `error` is always baseline-reference.  Detrending and Hann are applied
    # before a shifted FFT, so the fy/fx mask and coefficients share coordinates.
    _, fy, fx = _detrended_power(_luma(error[...,roi[0],roi[1]]))
    mask=((np.abs(fy)>2*np.abs(fx))&(np.abs(fy)>=.05)&(np.abs(fy)<=.10))
    fields=[]
    for z in (_luma(error[...,roi[0],roi[1]]),_luma(response[...,roi[0],roi[1]])):
        a=z.detach().double().cpu().numpy()[:,0]; h,w=a.shape[-2:]; yy,xx=np.mgrid[:h,:w]; d=np.stack([np.ones(h*w),yy.ravel(),xx.ravel()],1); win=np.outer(np.hanning(h),np.hanning(w)); fields.append(np.asarray([np.fft.fftshift(np.fft.fft2((v-(d@np.linalg.lstsq(d,v.ravel(),rcond=None)[0]).reshape(h,w))*win)) for v in a]))
    out=[]
    for E,R in zip(fields[0],fields[1]):
        before=float((np.abs(E)**2*mask).sum()); resp=float((np.abs(R)**2*mask).sum()); de=float((2*np.real(np.conj(E)*R)*mask).sum())
        out.append({'before_h_energy':before,'response_h_energy':resp,'signed_energy_change_dE_h':de,'after_h_energy':before+de+resp})
    return {'per_sample':out,'mean':{k:float(np.mean([q[k] for q in out])) for k in out[0]}}


def _rotation_check(pred: torch.Tensor, gt: torch.Tensor, text: torch.Tensor, valid: torch.Tensor, opt: dict[str,Any]) -> dict[str,float]:
    m,f,_=recall_aware_structure_losses(pred,gt,text,valid,**opt.get('recall_aware_opt',{})); p=pred.detach().clone().requires_grad_(True)
    lm,lf,_=recall_aware_structure_losses(p,gt,text,valid,**opt.get('recall_aware_opt',{})); g=torch.autograd.grad(lm+lf,p)[0]
    pr=torch.rot90(pred,1,(-2,-1)); gr=torch.rot90(gt,1,(-2,-1)); tr=torch.rot90(text,1,(-2,-1)); vr=torch.rot90(valid,1,(-2,-1)); q=pr.detach().clone().requires_grad_(True)
    rm,rf,_=recall_aware_structure_losses(q,gr,tr,vr,**opt.get('recall_aware_opt',{})); rg=torch.autograd.grad(rm+rf,q)[0]
    return {'loss_abs_error':float((m+f-rm-rf).abs()),'gradient_rotateback_max_error':float((g-torch.rot90(rg,-1,(-2,-1))).abs().max())}


def main() -> None:
    a=_args(); paths=(a.config,a.r1_checkpoint,a.c0_checkpoint,a.historical_lq,a.historical_gt)
    if any(not p.is_file() for p in paths): raise FileNotFoundError([str(p) for p in paths if not p.is_file()])
    root=a.output_root.resolve()
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'); dtype=torch.float64 if a.dtype=='float64' else torch.float32
    doc=yaml.safe_load(a.config.read_text(encoding='utf8')); indices=[int(x) for x in a.indices.split(',')]
    batch=_collect_batch(build_dataset(doc['datasets']['train']),indices); gt_rgb=batch['gt_rgb'].to(dev).float().clamp(0,1); valid_rgb=batch['valid_mask_rgb'].to(dev).float(); text_rgb=batch['text_mask_rgb'].to(dev).float()
    gen=torch.Generator(device=dev); gen.manual_seed(a.seed)
    with torch.no_grad(): lq_rgb,meta=LenovoDegradationPipeline(doc['lenovo_degradation'])(gt_rgb,gen)
    lq,gt=rgb_to_y4uv(lq_rgb).to(dtype),rgb_to_y4uv(gt_rgb).to(dtype); valid,text=yuv420_mask_to_y4uv(valid_rgb).to(dtype),yuv420_mask_to_y4uv(text_rgb).to(dtype); gt_rgb,valid_rgb,text_rgb=gt_rgb.to(dtype),valid_rgb.to(dtype),text_rgb.to(dtype)
    net=build_network(doc['network_g']).to(dev); digest=load_strict_checkpoint(net,a.r1_checkpoint,map_location='cpu'); net=net.to(dtype); net.train(); native_loss=build_loss(doc['train']['y4uv_native_opt']).to(dev).to(dtype)
    ls,y4,rgb=_losses(net,lq,gt,gt_rgb,valid,text,valid_rgb,text_rgb,native_loss,doc['train'])
    hist_lq=rgb_to_y4uv(_rgb(a.historical_lq,dev)).to(dtype); hist_gt=_rgb(a.historical_gt,dev).to(dtype)
    c0=build_network(doc['network_g']).to(dev); c0hash=load_strict_checkpoint(c0,a.c0_checkpoint,map_location='cpu'); c0=c0.to(dtype); c0.eval()
    net.eval()
    with torch.no_grad(): base_debug,base_stage2=_debug_with_stage2(net,hist_lq); base_hist=_decode(base_debug['output']); c0_hist=_decode(c0(hist_lq))
    with torch.no_grad(): zero_debug,_=_debug_with_stage2(copy.deepcopy(net).eval(),hist_lq); zero_hist=_decode(zero_debug['output'])
    alpha=float(.5*torch.tanh(net.context_scale.detach()).cpu())
    report={'status':'COMPLETE','contract':'g is sourced with network.train() (V5 context STE); cloned finite differences use network.eval(), so the measured response is J_eval J_train^T g. Frozen checkpoints and in-memory pure global-rate SGD only; RGB gradients are covectors and never image-converted','indices':indices,'seed':a.seed,'dtype':a.dtype,'tf32_disabled':True,'historical_lq':str(a.historical_lq),'historical_gt':str(a.historical_gt),'historical_packed_y4uv_shape':list(hist_lq.shape),'network_expected_input_hw':list(net.expected_input_hw),'zero_step_clone_minus_baseline_max':float((zero_hist-base_hist).abs().max()),'zero_step_clone_minus_baseline_rms':float(torch.sqrt(((zero_hist-base_hist)**2).mean())),'checkpoint_sha256':digest,'c0_sha256':c0hash,'context_scale':float(net.context_scale.detach().cpu()),'eval_alpha':alpha,'degradation_metadata':meta,'source_covectors':{},'updates':{},'rotation_equivariance':_rotation_check(rgb,gt_rgb,text_rgb,valid_rgb,doc['train']),'analytic_boundary':{'T':.01,'margin':.03,'equal_gt_missing':.01*float(F.softplus(torch.tensor(-3.))), 'alternating_pm_002_missing':.01*float((F.softplus(torch.tensor(-1.))+F.softplus(torch.tensor(-5.)))/2)}}
    for name,loss in ls.items():
        g=torch.autograd.grad(loss,y4,retain_graph=True)[0]
        rgb_g=torch.autograd.grad(loss,rgb,retain_graph=True,allow_unused=True)[0]
        report['source_covectors'][name]={'loss':float(loss.detach()),'y4_luma_unpacked_covector_spectrum':_spectrum(_unpack_y4_luma_covector(g)),
            'decoded_rgb_covector_spectrum': None if rgb_g is None else _spectrum(rgb_g),
            'coordinate_note':'Y4 and decoded-RGB entries are distinct autograd covectors; neither is a rendered-image conversion.'}
    # Same tensors/LQ, but an eval-mode forward removes the context STE.  This
    # is a diagnostic comparison only; `ls` above remains the update source.
    eval_ls,_,_=_losses(net,lq,gt,gt_rgb,valid,text,valid_rgb,text_rgb,native_loss,doc['train'])
    b3=_groups(net)['b3']
    for name in ls:
        tg=torch.autograd.grad(ls[name],b3,retain_graph=True,allow_unused=True); eg=torch.autograd.grad(eval_ls[name],b3,retain_graph=True,allow_unused=True)
        tn=float(torch.sqrt(sum((v*v).sum() for v in tg if v is not None)).detach()); en=float(torch.sqrt(sum((v*v).sum() for v in eg if v is not None)).detach())
        report['source_covectors'][name]['b3_gradient_l2_train_ste']=tn
        report['source_covectors'][name]['b3_gradient_l2_eval']=en
        report['source_covectors'][name]['b3_train_over_eval_l2']=tn/max(en,1e-30)
    # Validate analytic RGB recall derivative directly against autograd.  We report missing and weighted false separately.
    for name in ('missing','false_weighted'):
        p=rgb.detach().clone().requires_grad_(True); m,f,d=recall_aware_structure_losses(p,gt_rgb,text_rgb,valid_rgb,**doc['train'].get('recall_aware_opt',{})); raw=m if name=='missing' else float(doc['train'].get('recall_aware_weight_false',0))*f; ag=torch.autograd.grad(raw,p)[0]
        y=_luma(p); e=d['polarity']*(y-_luma(gt_rgb)); T=float(doc['train']['recall_aware_opt']['temperature']); margin=float(doc['train']['recall_aware_opt']['margin']); active=d['active']; support=((text_rgb[:,:1]>.5).float()*valid_rgb[:,:1] if name=='missing' else d['ring']) * active
        weight=1. if name=='missing' else float(doc['train'].get('recall_aware_weight_false',0)); denom=support.sum().clamp_min(1e-6)
        hinge=(e-margin)/T if name=='missing' else (-e-margin)/T
        sign=1. if name=='missing' else -1.
        dy=weight*sign*d['polarity']*torch.sigmoid(hinge)*support/denom
        analytic=dy*torch.tensor([.299,.587,.114],device=dev).view(1,3,1,1)
        report['source_covectors'][name]['analytic_rgb_autograd_max_error']=float((ag-analytic).abs().max())
    saved={'baseline':{'rgb':base_hist.detach().cpu().numpy(),'y':_luma(base_hist).detach().cpu().numpy(),'stage2_output':base_stage2.detach().cpu().numpy(),'context_residual':base_debug['context_residual'].detach().cpu().numpy()},'c0':{'rgb':c0_hist.detach().cpu().numpy(),'y':_luma(c0_hist).detach().cpu().numpy()}}
    for source in [x for x in a.sources.split(',') if x]:
      for group,params in ((k,v) for k,v in _groups(net).items() if k in a.groups.split(',')):
        grads=torch.autograd.grad(ls[source],params,retain_graph=True,allow_unused=True)
        for eta in [float(x) for x in a.eta.split(',')]:
            originals=dict(net.named_parameters())
            def displaced(sign: float):
                clone=copy.deepcopy(net); named=dict(clone.named_parameters())
                with torch.no_grad():
                    for op,g in zip(params,grads):
                        if g is not None: named[next(k for k,v in originals.items() if v is op)].add_(sign*eta*g)
                    train_debug,_=_debug_with_stage2(clone,lq); hist_debug,hist_stage2=_debug_with_stage2(clone,hist_lq)
                return train_debug,hist_debug,hist_stage2
            train_minus,hist_minus,hist_stage2=displaced(-1.0); _,hist_plus,_=displaced(1.0)
            new_train=_decode(train_minus['output']); new_hist=_decode(hist_minus['output']); plus_hist=_decode(hist_plus['output'])
            response=new_hist-base_hist; key=f'{source}/{group}/eta_{eta:g}'
            even=plus_hist+new_hist-2*base_hist; odd=(plus_hist-new_hist)/2
            roi=(slice(12,78),slice(156,228)); report['updates'][key]={'train_rgb_delta_spectrum':_spectrum(new_train-rgb),'historical_rgb_delta_spectrum':_spectrum(response),'historical_loss_delta_l1_to_gt':float((new_hist-hist_gt).abs().mean()-(base_hist-hist_gt).abs().mean()),'signed_GT':_signed_projection(base_hist-hist_gt,response,roi),'signed_C0':_signed_projection(base_hist-c0_hist,response,roi),'central_even_rms':float(torch.sqrt((even*even).mean())),'central_odd_rms':float(torch.sqrt((odd*odd).mean())),'central_even_over_odd_rms':float(torch.sqrt((even*even).mean())/torch.sqrt((odd*odd).mean()).clamp_min(1e-30)),'central_difference_rms':float(torch.sqrt((((plus_hist-new_hist)/(2*eta))**2).mean()))}
            saved[key]={'rgb':new_hist.detach().cpu().numpy(),'y':_luma(new_hist).detach().cpu().numpy(),'stage2_output':hist_stage2.detach().cpu().numpy(),'context_residual':hist_minus['context_residual'].detach().cpu().numpy()}
    saved['gt']={'rgb':hist_gt.detach().cpu().numpy(),'y':_luma(hist_gt).detach().cpu().numpy()}
    np.savez_compressed(root/'tensors.npz',train_y4=y4.detach().cpu().numpy(), train_rgb=rgb.detach().cpu().numpy(), **{f'{key}_{field}':value for key,item in saved.items() for field,value in item.items()})
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8'); print(json.dumps({'status':'COMPLETE','report':str(root/'report.json')}))

if __name__=='__main__': main()
