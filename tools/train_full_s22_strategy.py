"""Opt-in no-SDF V3M strategies; baseline trainer and public APIs unchanged."""
from pathlib import Path
import argparse,copy,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REPO='/home/guang.yang/workspace/Real-ESRGAN'


def degrade_gt(gt,generator):
    """@luna-comment Input:one BCHW RGB target and private CPU RNG. Output:detached native-size synthetic input/parameters. Key Steps:Gaussian.4..1.6;bilinear down/up.5..1;Gaussian noise0..3gray. Key Params:no warp/USM/GT changes. State/Dependencies:torch and existing Gaussian;local seeded RNG."""
    import torch
    import torch.nn.functional as F
    from tools.train_v3_edge_pilot import gaussian
    sigma=.4+1.2*float(torch.rand((),generator=generator));scale=.5+.5*float(torch.rand((),generator=generator));noise=3*float(torch.rand((),generator=generator))/255
    value=torch.cat([gaussian(gt[:,i:i+1],sigma) for i in range(3)],1)
    h,w=gt.shape[-2:];small=(max(2,round(h*scale)),max(2,round(w*scale)))
    value=F.interpolate(F.interpolate(value,size=small,mode='bilinear',align_corners=False),size=(h,w),mode='bilinear',align_corners=False)
    value=value+torch.randn(value.shape,generator=generator,dtype=torch.float32).to(value)*noise
    return value.clamp(0,1).detach(),dict(sigma=sigma,scale=scale,noise_sigma_255=noise*255)


def bootstrap():
    """@luna-comment Input:existing deployed runtime. Output:process-local S22 model/dataset registrations. Key Steps:accepted bootstrap then isolated subclasses only. Key Params:historical noSDF default;SDF requires explicit contrast experiment flag. State/Dependencies:existing registries,no baseline source edits."""
    from tools.train_v3pre_area import bootstrap as original
    original(REPO)
    import torch
    from basicsr.utils.registry import DATASET_REGISTRY,MODEL_REGISTRY
    from realesrgan.models.v3pre_area_adapters import V3PreAreaDataset,AreaAdaptiveNativeL1
    from realesrgan.models.block256_region_model import Block256RegionModel
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    from tools.train_v3_edge_pilot import objective,gradnorm,finite_step,save

    @DATASET_REGISTRY.register()
    class S22Dataset(V3PreAreaDataset):
        def __getitem__(self,index):
            """@luna-comment Input:immutable bank index. Output:unchanged pair/line mask plus train-phase marker. Key Steps:delegate bank reader;optional per-patch target contrast match;mark training draw. Key Params:contrast/text filter assertions default off;geometry unchanged. State/Dependencies:cached line sidecars."""
            item=super().__getitem__(index);item['s22_training_sample']=self.opt.get('phase',self.opt.get('split'))=='train'
            if self.opt.get('require_effective_text',False) and item['s22_training_sample']:
                support=item['line_region_mask']*item['valid_mask_rgb']
                assert float(support.sum())>0,'Filtered training index contains an empty text patch'
                item['effective_text_fraction']=float(support.mean())
            if self.opt.get('patch_contrast_alignment',False) and item['s22_training_sample']:
                from tools.patch_contrast_alignment import match_patch_contrast
                adjusted,stats=match_patch_contrast(item['gt_rgb'].numpy(),item['lq_rgb'].numpy(),item['valid_mask_rgb'].numpy())
                item['gt_rgb']=torch.from_numpy(adjusted);item['gt']=item['gt_rgb']
                for key,value in stats.items():item['contrast_'+key]=float(value)
            return item

    @MODEL_REGISTRY.register()
    class S22Model(Block256RegionModel):
        def __init__(self,opt):
            """@luna-comment Input:isolated strategy config. Output:historical noSDF model or explicitly opted-in SDF contrast experiment with native hook. Key Steps:original initialization;private mixture RNG;calibration state;finite Adam wrapper. Key Params:ratio.5,8active calibration draws. State/Dependencies:unchanged original network/optimizer."""
            super().__init__(opt)
            self.strategy=opt['s22_strategy']['name'];self.extra_weight=0.;self.calibration=[];self.extra_stats={};self.synthetic_count=0;self.total_draws=0
            self.mix_rng=torch.Generator().manual_seed(int(opt['manual_seed'])+9967)
            self.audit_root=Path(opt['s22_strategy']['audit_root']);self.audit_root.mkdir(parents=True,exist_ok=True)
            if not opt.get('s22_patch_contrast_experiment',False):assert self.sdf_boundary_weight==0 and self.cri_sdf_boundary is None
            self.native_forward=self.cri_y4uv_native.forward
            if self.strategy in ['identity','guard']:self.s22_handle=self.cri_y4uv_native.register_forward_hook(self.extra_hook)
            original_step=self.optimizer_g.step

            def checked_step(*args,**kwargs):
                """@luna-comment Input:original Adam args. Output:finite-checked single Adam call. Key Steps:reject nonfinite gradients before mutation. Key Params:no clipping. State/Dependencies:original bound optimizer method."""
                finite_step(self.optimizer_g,args,kwargs);return original_step(*args,**kwargs)

            self.optimizer_g.step=checked_step

        def identity_native(self,pred,target,**kwargs):
            """@luna-comment Input:one GT identity prediction/target. Output:corresponding original native scalar. Key Steps:slice adaptive bound masks and restore full-pair masks/stats. Key Params:one additional GT sample. State/Dependencies:bound criterion,no recursive hooks."""
            criterion=self.cri_y4uv_native
            if not isinstance(criterion,AreaAdaptiveNativeL1):return self.native_forward(pred,target,**kwargs)
            line,valid=criterion.line_region,criterion.raw_valid;stats=getattr(criterion,'stats',None)
            try:
                criterion.line_region=line[:len(pred)];criterion.raw_valid=valid[:len(pred)]
                return self.native_forward(pred,target,**kwargs)
            finally:
                criterion.line_region,criterion.raw_valid=line,valid
                if stats is not None:criterion.stats=stats

        def extra_hook(self,module,args,result):
            """@luna-comment Input:ordinary native scalar. Output:original plus fixed-weight selected objective. Key Steps:existing detached identity/guard objective;record support and raw term. Key Params:weight0 until8active calibration observations. State/Dependencies:current paired forward and batch."""
            if not self.net_g.training:return result
            extra,stats=objective(self,self.s22_batch,self.strategy,self.identity_native)
            self.extra_stats=dict(native_original=float(result.detach()),extra_raw=float(extra.detach()),extra_weight=self.extra_weight,**stats)
            return result+self.extra_weight*extra

        def feed_data(self,data):
            """@luna-comment Input:paired RGB/mask/line batch. Output:normal model state and selected synthetic inputs. Key Steps:original feed;bind adaptive masks;30%or100%private draws replaceLQ only. Key Params:never degrade validation;GT/geometry unchanged. State/Dependencies:private CPU RNG and detachedGT."""
            super().feed_data(data);self.s22_batch=data
            if isinstance(self.cri_y4uv_native,AreaAdaptiveNativeL1):
                self.cri_y4uv_native.line_region=data['line_region_mask'].to(self.device);self.cri_y4uv_native.raw_valid=data['valid_mask_rgb'].to(self.device)
            training=all(data.get('s22_training_sample',[False]))
            if self.strategy in ['gt_degrade30','gt_degrade100'] and training:
                probability=1. if self.strategy=='gt_degrade100' else .3
                selected=torch.rand(len(self.lq),generator=self.mix_rng)<probability
                for i,yes in enumerate(selected):
                    if yes:self.lq_rgb[i:i+1],_=degrade_gt(self.gt_rgb[i:i+1].detach(),self.mix_rng)
                self.lq=rgb_to_y4uv(self.lq_rgb);self.last_paired_selection=~selected
                self.synthetic_count+=int(selected.sum());self.total_draws+=len(selected)
            elif training:self.total_draws+=len(self.lq)

        def optimize_parameters(self,current_iter):
            """@luna-comment Input:scheduled iteration. Output:unchanged scheduled update plus selected objective logs. Key Steps:non-updating gradient calibration;original update;separate diagnostic logs. Key Params:median.5native/extra over8active draws;inactive scratch guard defers calibration. State/Dependencies:original scheduler/Adam/checkpoint flow."""
            if self.strategy in ['identity','guard'] and len(self.calibration)<8:
                self.output=self.net_g(self.lq)
                native=self.native_forward(self.output,self.gt,valid_mask=self.valid_mask,text_mask=self.text_mask)
                extra,stats=objective(self,self.s22_batch,self.strategy,self.identity_native)
                parameters=tuple(p for p in self.net_g.parameters() if p.requires_grad)
                bn=gradnorm(native,parameters);en=gradnorm(extra,parameters)
                if math.isfinite(en) and en>1e-10 and bn>1e-10 and stats['support_fraction']>0:
                    self.calibration.append(dict(iter=current_iter,native_norm=bn,extra_norm=en,weight=.5*bn/en,**stats))
                if len(self.calibration)==8:
                    values=sorted(r['weight'] for r in self.calibration);self.extra_weight=(values[3]+values[4])/2
                    save(self.audit_root/'CALIBRATION.json',dict(passed=True,weight=self.extra_weight,target_gradient_ratio=.5,records=self.calibration))
                del native,extra;self.output=None
            super().optimize_parameters(current_iter)
            if isinstance(self.cri_y4uv_native,AreaAdaptiveNativeL1):self.log_dict.update({k:float(v) for k,v in self.cri_y4uv_native.stats.items()})
            if self.strategy in ['identity','guard']:
                self.log_dict.update({'s22_'+k:float(v) for k,v in self.extra_stats.items()})
                self.log_dict['s22_calibration_count']=len(self.calibration)
                self.log_dict['l_native_total']=self.log_dict['l_native']
                self.log_dict['l_native']=self.extra_stats['native_original']
                self.log_dict['l_s22_extra']=self.extra_weight*self.extra_stats['extra_raw']
            self.log_dict['s22_synthetic_fraction']=self.synthetic_count/max(self.total_draws,1)
            if not self.opt.get('s22_patch_contrast_experiment',False):assert not any('sdf' in k for k in self.log_dict)
            for key,value in self.s22_batch.items():
                if key.startswith(('contrast_','effective_text_')):self.log_dict[key]=sum(value)/max(len(value),1)
            assert all(math.isfinite(float(v)) for v in self.log_dict.values())
            if current_iter<=3 or current_iter%100==0:
                save(self.audit_root/'STATUS.json',dict(iter=current_iter,strategy=self.strategy,synthetic_count=self.synthetic_count,total_draws=self.total_draws,calibration_count=len(self.calibration),weight=self.extra_weight,logs={k:float(v) for k,v in self.log_dict.items()}))


def smoke(config,output):
    """@luna-comment Input:engineering config/owned JSON. Output:real CUDA updates with exact init/declared SDF/contrast/strategy evidence. Key Steps:select CP-containing draw when enabled or filtered index draw when disabled;8updates plus auxiliary-scheduled step;verify finite gradients and changed params. Key Params:no saved weights,full configs untouched. State/Dependencies:pilot bank and original shared init."""
    import torch,numpy as np
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from basicsr.data import build_dataset
    from tools.train_v3_edge_pilot import save,digest,objective
    sys.argv=[sys.argv[0],'-opt',str(config)];opt,_=parse_options(REPO,is_train=True);torch.set_num_threads(2)
    dataset=build_dataset(opt['datasets']['train']);chosen=None
    cp_enabled=opt['train'].get('cp_weight',0)>0
    if not cp_enabled:chosen=tuple(map(int,dataset.index[0]))
    else:
        for si,shard in enumerate(dataset.shards):
            for ri,meta in enumerate(json.loads((dataset.root/shard['metadata']).read_text())):
                if meta['cp_lines']:chosen=(si,ri);break
            if chosen is not None:break
    assert chosen is not None;dataset.index=np.asarray([chosen]*4,np.uint32)
    batch=dataset.collate_fn([dataset[i] for i in range(4)]);assert all(batch['s22_training_sample'])
    model=build_model(opt);expected_sdf=opt['train'].get('sdf_boundary_weight',0)>0;assert (model.cri_sdf_boundary is not None)==expected_sdf and not model.optimizer_g.state
    initial=torch.load(opt['path']['pretrain_network_g'],map_location='cpu')['params'];assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items())
    before={k:v.detach().clone() for k,v in model.net_g.named_parameters()}
    for iteration in [1,2,3,4,5,6,7,8,8000]:model.feed_data(batch);model.optimize_parameters(iteration)
    logs={k:float(v) for k,v in model.get_current_log().items()}
    assert all(math.isfinite(v) for v in logs.values())
    assert any('sdf' in k for k in logs)==expected_sdf
    if opt.get('s22_patch_contrast_experiment',False):assert 'contrast_applied' in logs and logs['input_paired_fraction']==1.
    if cp_enabled:assert logs['cp_accepted_lines']>0 and logs['cp_scale']>0
    else:assert model.cri_cp is None and logs['cp_scale']==0 and logs.get('l_cp',0)==0
    assert logs['uv_strong_scale']>0
    if opt['datasets']['train'].get('require_effective_text',False):assert min(batch['effective_text_fraction'])>0
    assert any(not torch.equal(v,before[k]) for k,v in model.net_g.named_parameters())
    if model.strategy=='gt_degrade30':assert 0<model.synthetic_count<model.total_draws
    if model.strategy=='gt_degrade100':assert model.synthetic_count==model.total_draws==36 and logs['input_paired_fraction']==0
    probe=None
    if model.strategy=='guard':
        original=model.output;model.output=model.lq+.1*(model.net_g(model.lq)-model.lq)-.5*(model.lq-model.gt)
        # A controlled cancelling output verifies the penalty path without an optimizer step.
        import torch.nn.functional as F
        y=F.pixel_shuffle(model.lq[:,:4],2);from tools.train_v3_edge_pilot import gaussian
        cancellation=F.pixel_unshuffle(y-gaussian(y,1.4),2)
        model.output=torch.cat((model.output[:,:4]-.5*cancellation,model.output[:,4:]),1)
        penalty,stats=objective(model,batch,'guard',model.identity_native)
        assert torch.isfinite(penalty) and float(penalty)>0 and stats['support_fraction']>0
        gradients=torch.autograd.grad(penalty,model.output,retain_graph=False)[0];assert torch.isfinite(gradients).all() and float(gradients.abs().sum())>0
        probe=dict(loss=float(penalty.detach()),**stats);model.output=original
    save(output,dict(passed=True,updates=9,strategy=model.strategy,no_sdf=not expected_sdf,no_cp=not cp_enabled,fresh_optimizer=True,initial_exact=True,config_sha256=digest(config),synthetic_count=model.synthetic_count,total_draws=model.total_draws,calibration_count=len(model.calibration),guard_controlled_probe=probe,losses=logs))


def main():
    """@luna-comment Input:explicit training or smoke CLI. Output:registered original train pipeline or diagnostic receipt. Key Steps:bootstrap once;parse task-only flags;delegate unchanged BasicSR. Key Params:no resume or deployment edits. State/Dependencies:isolated config."""
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args();bootstrap()
    if args.smoke_config:smoke(Path(args.smoke_config),Path(args.smoke_output))
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline
        train_pipeline(REPO)


if __name__=='__main__':main()
