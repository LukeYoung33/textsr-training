"""Isolated, bounded strong directional pilots; historical training remains unchanged."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'v3pre_v3m_three_20261027'
OUT=DATA/'v3_edge_pilot_20261028'
PILOT=OUT
CONTINUATION=False
REPO='/home/guang.yang/workspace/Real-ESRGAN'


def save(path,value):
    """@luna-comment Input:path/JSON state. Output:atomic receipt. Key Steps:replace owned temp. Key Params:UTF8. State/Dependencies:isolated output only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def digest(path):
    """@luna-comment Input:read-only file. Output:SHA256. Key Steps:stream bytes. Key Params:4MiB. State/Dependencies:filesystem."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):value.update(block)
    return value.hexdigest()


def gaussian(value,sigma):
    """@luna-comment Input:B1HW tensor/sigma pixels. Output:Gaussian lowpass. Key Steps:separable reflect convolution. Key Params:3sigma support. State/Dependencies:torch differentiable prediction."""
    import torch
    import torch.nn.functional as F
    radius=math.ceil(3*sigma)
    axis=torch.arange(-radius,radius+1,device=value.device,dtype=value.dtype)
    kernel=torch.exp(-axis.square()/(2*sigma*sigma));kernel=kernel/kernel.sum()
    value=F.conv2d(F.pad(value,(radius,radius,0,0),mode='reflect'),kernel.view(1,1,1,-1))
    return F.conv2d(F.pad(value,(0,0,radius,radius),mode='reflect'),kernel.view(1,1,-1,1))


def gradients(value):
    """@luna-comment Input:B1HW luma. Output:central dx/dy. Key Steps:symmetric differences. Key Params:replicate padding. State/Dependencies:torch autograd."""
    import torch.nn.functional as F
    return F.pad((value[:,:,:,2:]-value[:,:,:,:-2])*.5,(1,1,0,0)),F.pad((value[:,:,2:,:]-value[:,:,:-2,:])*.5,(0,0,1,1))


def objective(model,batch,arm,native):
    """@luna-comment Input:paired model state/line sidecar/arm/native criterion. Output:extra scalar and support logs. Key Steps:GT identity or detached GT-agreed negative projection. Key Params:sigma.7/1.4/2.8,tau.1,16px valid interior. State/Dependencies:network gradients only; input/GT/masks detached."""
    import torch
    import torch.nn.functional as F
    x=F.pixel_shuffle(model.lq[:,:4].detach(),2)
    target=F.pixel_shuffle(model.gt[:,:4].detach(),2)
    valid=model.valid_mask_rgb.detach()
    valid=(1-F.max_pool2d(1-valid,33,1,16)).clamp(0,1)
    valid[:,:,:16]=0;valid[:,:,-16:]=0;valid[:,:,:,:16]=0;valid[:,:,:,-16:]=0
    region=batch['line_region_mask'].to(x).detach()*valid
    tx,ty=gradients(target);xx,xy=gradients(x)
    target_energy=tx.square()+ty.square();input_energy=xx.square()+xy.square()
    if arm=='identity':
        prediction=model.net_g(model.gt[:1].detach())
        base=native(prediction,model.gt[:1].detach(),valid_mask=model.valid_mask[:1],text_mask=model.text_mask[:1])
        r=F.pixel_shuffle(prediction[:,:4]-model.gt[:1,:4].detach(),2)
        support=region[:1]*(target_energy[:1]>.015**2).to(x)
        terms=[]
        for sigma in (.7,1.4,2.8):
            band=r-gaussian(r,sigma)
            terms.append(((torch.sqrt(band.square()+1e-6)-.001)*support).sum()/support.sum().clamp_min(1))
        extra=base+sum(terms)/3
        return extra,dict(support_fraction=float(support.mean()),identity_native=float(base.detach()))
    prediction=F.pixel_shuffle(model.output[:,:4],2)
    residual=prediction-x
    cosine=(tx*xx+ty*xy)/(target_energy*input_energy).sqrt().clamp_min(1e-8)
    ratio=(target_energy/input_energy.clamp_min(1e-8)).sqrt()
    # GT/input agreement gates are independent of predictions; no noise-copy objective.
    agreement=(cosine>.95)&(ratio>.5)&(ratio<2)&(input_energy>.015**2)&((x-target).abs()<.15)
    coherence=(gaussian(xx,1).square()+gaussian(xy,1).square())/gaussian(input_energy,1).clamp_min(1e-8)
    support=(region*agreement.to(x)*(coherence>.2).to(x)).detach()
    terms=[];coverages=[];projections=[]
    for sigma in (.7,1.4,2.8):
        hx=(x-gaussian(x,sigma)).detach();ht=(target-gaussian(target,sigma)).detach()
        hr=residual-gaussian(residual,sigma)
        energy=gaussian(hx.square(),1.4).detach()
        trusted=(support*(hx*ht>0).to(x)*(energy>.005**2).to(x)).detach()
        a=gaussian(hx*hr,1.4)/energy.clamp_min(1e-6)
        terms.append((trusted*F.relu(-a-.1).square()).sum()/trusted.sum().clamp_min(1))
        coverages.append(float(trusted.mean()))
        projections.append(float((trusted*a.detach()).sum()/trusted.sum().clamp_min(1)))
    return sum(terms)/3,dict(support_fraction=sum(coverages)/3,projection=sum(projections)/3)


class Injection:
    def __init__(self,model,arm):
        """@luna-comment Input:existing model/selected arm. Output:isolated criterion hook owner. Key Steps:retain native forward. Key Params:fixed calibrated coefficient. State/Dependencies:no module source edits."""
        self.model,self.arm=model,arm;self.batch=None;self.weight=0.;self.stats={}
        self.native=model.cri_y4uv_native.forward
        self.handle=None
        if arm!='baseline':self.handle=model.cri_y4uv_native.register_forward_hook(self.hook)

    def hook(self,module,args,result):
        """@luna-comment Input:native criterion result. Output:original plus selected objective. Key Steps:call original native for identity without hook recursion. Key Params:calibration-only gradients retained. State/Dependencies:current paired batch/model."""
        import torch
        extra,stats=objective(self.model,self.batch,self.arm,self.native)
        self.last=(result,extra)
        self.stats=dict(native_original=float(result.detach()),extra_raw=float(extra.detach()),weight=self.weight,**stats)
        return result+self.weight*extra


def gradnorm(loss,parameters):
    """@luna-comment Input:scalar/parameter tuple. Output:L2 parameter-gradient norm. Key Steps:non-updating autograd. Key Params:retain graph. State/Dependencies:calibration only."""
    import torch
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return float(torch.sqrt(sum(v.detach().square().sum() for v in values if v is not None)))


def finite_step(optimizer,args,kwargs):
    """@luna-comment Input:optimizer pre-step. Output:fail before invalid update. Key Steps:finite gradient audit. Key Params:all parameters. State/Dependencies:does not clip/change gradients."""
    import torch
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise RuntimeError('Nonfinite optimizer gradient')


def train(arm,steps):
    """@luna-comment Input:arm/bounded cumulative update count. Output:checkpoints/progress/loss/gradient receipts. Key Steps:strict Full250K or own2K+Adam restore;calibrate or reuse fixed weight;identical paired draws with verified resumed prefix;unchanged auxiliary optimizer path. Key Params:added/native initial gradient ratio.5,terminalLR frozen,2K pilot or10K continuation. State/Dependencies:completed V3 bank/CUDA; isolated output."""
    import numpy as np
    import torch
    import yaml
    from tools.train_v3pre_area import bootstrap
    bootstrap(REPO)
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.models.v3pre_area_adapters import V3PreAreaDataset
    torch.set_num_threads(2)
    output=OUT/arm;output.mkdir(parents=True,exist_ok=True)
    assert not (output/'status.json').exists(),'Refusing to overwrite an existing run'
    cfg=yaml.safe_load((SOURCE/'pretrain/training/configs/full.yml').read_text())
    original=SOURCE/'pretrain/training/experiments/v3pre_pretrain_full_20261027'
    start_step=2000 if CONTINUATION else 0
    checkpoint=PILOT/arm/'net_g_2000.pth' if CONTINUATION else original/'models/net_g_250000.pth'
    statepath=PILOT/arm/'optimizer_2000.state' if CONTINUATION else original/'training_states/250000.state'
    cfg['name']='v3_edge_'+arm+'_20261028';cfg['path'].update(pretrain_network_g=str(checkpoint),resume_state=None,experiments_root=str(output/'experiments'))
    cfg['datasets']['train']['type']='V3PreAreaDataset'
    config=output/'config.yml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    sys.argv=[sys.argv[0],'-opt',str(config)]
    opt,_=parse_options(REPO,is_train=True)
    model=build_model(opt)
    state=torch.load(statepath,map_location='cpu',weights_only=False)
    assert state['iter']==(2000 if CONTINUATION else 250000) and len(state['optimizers'])==1
    model.optimizer_g.load_state_dict(state['optimizers'][0])
    original_step=model.optimizer_g.step

    def checked_step(*args,**kwargs):
        """@luna-comment Input:original Adam step args. Output:finite-checked original step. Key Steps:audit before mutation. Key Params:older torch compatible. State/Dependencies:bound original optimizer method."""
        finite_step(model.optimizer_g,args,kwargs)
        return original_step(*args,**kwargs)

    model.optimizer_g.step=checked_step
    learning_rates=[group['lr'] for group in model.optimizer_g.param_groups]
    dataset=V3PreAreaDataset(opt['datasets']['train'])
    assert (dataset.root/'LINE_REGIONS_COMPLETE.json').exists()
    injection=Injection(model,arm);parameters=tuple(p for p in model.net_g.parameters() if p.requires_grad)
    rng=np.random.default_rng(20261028)
    calibration=[]
    if CONTINUATION:
        prior=json.loads((PILOT/arm/'CALIBRATION.json').read_text())
        calibration=prior['records'];injection.weight=prior['weight']
        previous=json.loads((PILOT/arm/'COMPLETE.json').read_text())
        assert previous['passed'] and previous['step']==2000
    elif arm!='baseline':
        for _ in range(8):
            indices=rng.integers(0,len(dataset),size=4)
            batch=dataset.collate_fn([dataset[int(i)] for i in indices]);injection.batch=batch
            model.feed_data(batch);model.output=model.net_g(model.lq)
            native=injection.native(model.output,model.gt,valid_mask=model.valid_mask,text_mask=model.text_mask)
            extra,stats=objective(model,batch,arm,injection.native)
            base_norm=gradnorm(native,parameters);extra_norm=gradnorm(extra,parameters)
            if extra_norm>1e-12 and stats['support_fraction']>0:
                calibration.append(dict(native_norm=base_norm,extra_norm=extra_norm,weight=.5*base_norm/extra_norm,**stats))
            del native,extra
        assert len(calibration)>=4,'Insufficient trusted support for gradient calibration'
        injection.weight=float(np.median([r['weight'] for r in calibration]))
    save(output/'CALIBRATION.json',dict(target_gradient_ratio=.5,weight=injection.weight,records=calibration))
    save(output/'PROVENANCE.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),optimizer=str(statepath),optimizer_sha256=digest(statepath),config_sha256=digest(config),trainer_sha256=digest(__file__),bank_build_id=dataset.manifest['build_id'],sampling='Independent fixed-seed shuffled index passes; same pair sequence in every arm',learning_rates=learning_rates,aux_iteration_offset=250000,fresh_optimizer=False,start_step=start_step,calibration_reused=CONTINUATION))
    # Calibration has no optimizer steps or mutable network buffers in this architecture.
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)['params']
    model.net_g.load_state_dict(loaded,strict=True)
    model.optimizer_g.zero_grad(set_to_none=True)
    random.seed(20261028);np.random.seed(20261028);torch.manual_seed(20261028);torch.cuda.manual_seed_all(20261028)
    generator=torch.Generator().manual_seed(20261028)
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,generator=generator,num_workers=2,collate_fn=dataset.collate_fn,drop_last=True,pin_memory=True)
    started=time.time();sequence=hashlib.sha256();history=[];iterator=iter(loader)
    checkpoints={start_step,4000,6000,steps} if CONTINUATION else {0,100,500,1000,steps}
    torch.save(dict(params=model.net_g.state_dict(),pilot_step=start_step),output/f'net_g_{start_step}.pth')
    if CONTINUATION:
        for _ in range(start_step):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            sequence.update(json.dumps(batch['line_id']).encode())
        assert sequence.hexdigest()==previous['sequence_sha256'],'Original2K paired prefix differs'
        save(output/'RESUME_GATE.json',dict(passed=True,start_step=start_step,prefix_sha256=sequence.hexdigest(),weight=injection.weight,learning_rates=learning_rates))
        torch.save(dict(iter=start_step,optimizers=[model.optimizer_g.state_dict()]),output/f'optimizer_{start_step}.state')
        started=time.time()
    for step in range(start_step+1,steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        sequence.update(json.dumps(batch['line_id']).encode())
        injection.batch=batch
        model.feed_data(batch);model.optimize_parameters(250000+step)
        logs={key:float(value) for key,value in model.get_current_log().items()}
        assert all(math.isfinite(v) for v in logs.values())
        if arm!='baseline':
            logs.update(injection.stats)
            logs['l_native_augmented']=logs['l_native'];logs['l_native']=logs['native_original']
            logs['extra_weighted']=injection.weight*logs['extra_raw']
        elapsed=time.time()-started
        status=dict(arm=arm,phase='complete' if step==steps else 'training',step=step,total=steps,start_step=start_step,elapsed_seconds=elapsed,seconds_per_step=elapsed/(step-start_step),eta_seconds=elapsed/(step-start_step)*(steps-step),updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),sequence_sha256=sequence.hexdigest(),logs=logs)
        if step<=start_step+3 or step%10==0:
            history.append(dict(step=step,**logs));save(output/'status.json',status);save(output/'history.json',history)
            print(json.dumps(status),flush=True)
        if step in checkpoints:
            torch.save(dict(params=model.net_g.state_dict(),pilot_step=step,source_iteration=250000,arm=arm),output/f'net_g_{step}.pth')
            torch.save(dict(iter=step,optimizers=[model.optimizer_g.state_dict()]),output/f'optimizer_{step}.state')
    save(output/'COMPLETE.json',dict(passed=True,**status))


def main():
    """@luna-comment Input:explicit selected experiment CLI. Output:bounded run or failure receipt. Key Steps:validate args and execute. Key Params:no auto-resume/overwrite. State/Dependencies:authorized Remote58 CUDA."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['baseline','identity','guard'],required=True);parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--continue-2k',action='store_true')
    args=parser.parse_args()
    global OUT,CONTINUATION
    CONTINUATION=args.continue_2k
    assert (2000 if CONTINUATION else 0)<args.steps<=(10000 if CONTINUATION else 2000)
    if CONTINUATION:OUT=DATA/'v3_edge_cont10k_20261028'
    if args.smoke:
        OUT=OUT/'engineering_smoke'
    try:train(args.arm,args.steps)
    except Exception as error:
        save(OUT/args.arm/'FAILED.json',dict(error=repr(error),updated_at=datetime.now().isoformat()));raise


if __name__=='__main__':main()
