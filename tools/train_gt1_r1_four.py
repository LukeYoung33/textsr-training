"""Four confirmed V3M recipes: train-only mix50, nativeL1 and ori/norm R1."""
from pathlib import Path
from collections import OrderedDict
import argparse,json,math,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest


def mix_input(model,data):
    """@luna-comment Input:parent-fedRGBbatch. Output:Bernoulli.5 mildGTinputs withunchangedGT/masks. Key Steps:privateRNGselect,cloneLQ,mildaccepteddegrade,repackY4UV. Key Params:trainonly;all4probability.5. State/Dependencies:modelprivatecounters/RNG;inputbatchimmutable."""
    from tools.train_latest_vosr_mix50 import mild
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    if not all(data.get('s22_training_sample',[False])):return
    probability=float(model.opt['gt1_mix']['probability']);assert probability==.5
    selected=torch.rand(len(model.lq),generator=model.mix_rng)<probability
    model.lq_rgb=model.lq_rgb.clone();model.mixture_parameters=[]
    for i,yes in enumerate(selected):
        if yes:
            model.lq_rgb[i:i+1],params=mild(model.gt_rgb[i:i+1].detach(),model.mix_rng);model.mixture_parameters.append(params)
    model.lq=rgb_to_y4uv(model.lq_rgb);model.last_paired_selection=~selected;model.synthetic_count+=int(selected.sum())


def record(model,iteration):
    """@luna-comment Input:completedupdate/currentlosses. Output:uniformnew4runtelemetry/history. Key Steps:finitecheck,mixturefraction,currentLR,100stepreceipt. Key Params:distincthistoryfileavoidsparentnormduplicates. State/Dependencies:owned auditroot."""
    model.log_dict.update(gt_mixture_fraction=model.synthetic_count/max(model.total_draws,1),gt_mixture_probability=.5)
    assert all(math.isfinite(float(v)) for v in model.log_dict.values())
    if iteration<=3 or iteration%100==0:
        r=dict(iter=iteration,total=500000,lr=model.get_current_learning_rate()[0],updated=time.time(),logs=dict(model.log_dict),synthetic_count=model.synthetic_count,total_draws=model.total_draws)
        save(model.audit_root/'RUN_STATUS.json',r)
        with (model.audit_root/'history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')


class AtomicMixture:
    """Private checkpoint mixin; preserves optimizer/scheduler and private mixture RNG."""
    def save(self,epoch,current_iter):
        """@luna-comment Input:epoch/iteration. Output:atomicmodel/state/hashreceipt. Key Steps:CPUfiniteparams,Adam/schedulers/privateRNGsnapshot,publishreceiptlast. Key Params:latestseparate;numericpairsdurable. State/Dependencies:ownexperimentonly;doesnotclaimDataLoadercursorresume."""
        from tools.train_r1_width_500k import atomic_torch
        path=Path(self.opt['path']['models'])/('net_g_latest.pth' if current_iter==-1 else f'net_g_{current_iter}.pth')
        params={k:v.detach().cpu() for k,v in self.get_bare_model(self.net_g).state_dict().items()}
        assert all(torch.isfinite(v).all() for v in params.values());atomic_torch(path,dict(params=params))
        if current_iter!=-1:
            target=Path(self.opt['path']['training_states'])/f'{current_iter}.state'
            state=dict(epoch=epoch,iter=current_iter,optimizers=[o.state_dict() for o in self.optimizers],schedulers=[s.state_dict() for s in self.schedulers],mixture_rng_state=self.mix_rng.get_state(),mixture_synthetic_count=self.synthetic_count,mixture_total_draws=self.total_draws)
            assert all(torch.isfinite(v).all() for o in state['optimizers'] for item in o['state'].values() for v in item.values() if torch.is_tensor(v))
            atomic_torch(target,state);save(target.with_suffix('.verified.json'),dict(iter=current_iter,model=str(path),model_sha256=digest(path),state=str(target),state_sha256=digest(target),private_mixture_rng_saved=True))

    def resume_training(self,resume_state):
        """@luna-comment Input:owntrainingstate. Output:restoredoptimizer/schedule/privateGTmixstream. Key Steps:parentrestore thenmixturestate/counters. Key Params:requiredownkeys,nosilentmixturereset. State/Dependencies:modelprivateRNG;DataLoaderorderingseparate."""
        super().resume_training(resume_state)
        self.mix_rng.set_state(resume_state['mixture_rng_state']);self.synthetic_count=resume_state['mixture_synthetic_count'];self.total_draws=resume_state['mixture_total_draws']


def bootstrap():
    """@luna-comment Input:ownedprocess. Output:newpaired/mixandnorm/mixregistrations. Key Steps:acceptedR1bootstraponce,subclassprocesslocalmodels. Key Params:originaloriandwidthnormalizedlossesunchanged. State/Dependencies:BasicSRregistry;existingjobsunaffected."""
    from tools.train_r1_width_500k import bootstrap as accepted
    accepted()
    from basicsr.utils.registry import MODEL_REGISTRY
    Paired=MODEL_REGISTRY.get('S22Model');Norm=MODEL_REGISTRY.get('R1WidthModel')

    @MODEL_REGISTRY.register()
    class GT1MixtureModel(AtomicMixture,Paired):
        def feed_data(self,data):
            """@luna-comment Input:normalS22batch. Output:originalnativeGTandtrainonlymildmixLQ. Key Steps:parentfeedthenprivateacceptedmix. Key Params:.5;contrastoffindataset. State/Dependencies:batchimmutable."""
            super().feed_data(data);mix_input(self,data)

        def optimize_parameters(self,current_iter):
            """@luna-comment Input:updateindex. Output:ordinaryL1orL1+oriR1step anduniformtelemetry. Key Steps:unchangedparentlosses/update,appendmixstats. Key Params:onlyconfigenablednative/R1. State/Dependencies:ownoptimizer/audits."""
            super().optimize_parameters(current_iter);record(self,current_iter)

    @MODEL_REGISTRY.register()
    class GT1NormMixtureModel(AtomicMixture,Norm):
        def feed_data(self,data):
            """@luna-comment Input:acceptedwidth-codedbatch. Output:samewidthlabels/GTandmildmixLQ. Key Steps:parentnormfeedthenprivateGTmix. Key Params:.5;validationunchanged. State/Dependencies:ownmodelonly."""
            super().feed_data(data);mix_input(self,data)

        def optimize_parameters(self,current_iter):
            """@luna-comment Input:updateindex. Output:acceptedwidthR1+L1step anduniformtelemetry. Key Steps:parentnormoptimizationthenmixturelogs. Key Params:missing1/false.25,cap4. State/Dependencies:ownoptimizer/sidecars/audits."""
            super().optimize_parameters(current_iter);record(self,current_iter)


def smoke(config,output):
    """@luna-comment Input:fullfrozenrecipe. Output:actualCUDA9update/loss/data/atomicresumechecks. Key Steps:strictinit,actual4drawbatch,mixandvalinvariants,saturatedR1composition,savestate/mixturecontinuation. Key Params:smokeisolatedonly,noproductionweights. State/Dependencies:readybank/widthlabels/CUDA."""
    from basicsr.utils.options import parse_options
    from basicsr.data import build_dataset
    from basicsr.models import build_model
    sys.argv=[sys.argv[0],'-opt',str(config)];opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    assert not opt['datasets']['train']['patch_contrast_alignment'] and opt['gt1_mix']['probability']==.5
    dataset=build_dataset(opt['datasets']['train']);indices=[0,len(dataset)//4,len(dataset)//2,3*len(dataset)//4]
    batch=dataset.collate_fn([dataset[i] for i in indices]);raw=batch['lq_rgb'].clone();target=batch['gt_rgb'].clone()
    model=build_model(opt);initial=torch.load(opt['path']['pretrain_network_g'],map_location='cpu')['params']
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items())
    for i in range(1,10):
        model.update_learning_rate(i,warmup_iter=-1);model.feed_data(batch);model.optimize_parameters(7000+i)
        assert torch.equal(batch['lq_rgb'],raw) and torch.equal(batch['gt_rgb'],target)
    logs=dict(model.log_dict);kind=opt['gt1_mix']['r1'];assert 0<model.synthetic_count<model.total_draws==36
    assert abs(logs['l_total']-logs['l_native']-logs.get('l_missing',0)-logs.get('l_false',0))<1e-6
    if kind=='none':assert abs(logs['l_total']-logs['l_native'])<1e-7
    else:assert logs['l_missing']>0 and logs['l_false']>0
    if kind=='norm':
        for name in ['missing','false']:
            weights=[logs[f'r1_{name}_w{i}'] for i in range(3)];assert max(weights)<=4.00001 and min(weights)>=0
            if sum(logs[f'r1_{name}_n{i}'] for i in range(3))>0:assert abs(logs[f'r1_{name}_mean_weight']-1)<1e-4
    assert any(not torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items())
    model.save(0,9);statepath=Path(opt['path']['training_states'])/'9.state';state=torch.load(statepath,map_location='cpu')
    resumed=build_model(opt);params=torch.load(Path(opt['path']['models'])/'net_g_9.pth',map_location='cpu')['params'];resumed.net_g.load_state_dict(params,strict=True);resumed.resume_training(state)
    assert torch.equal(resumed.mix_rng.get_state(),model.mix_rng.get_state())
    model.feed_data(batch);resumed.feed_data(batch);assert torch.equal(model.last_paired_selection,resumed.last_paired_selection) and torch.allclose(model.lq_rgb,resumed.lq_rgb,atol=1e-7,rtol=0)
    valid=dict(batch);valid['s22_training_sample']=[False]*4;count=model.synthetic_count;model.feed_data(valid)
    assert model.synthetic_count==count and torch.equal(model.lq_rgb,raw.to(model.device))
    save(output,dict(passed=True,iterations=9,r1=kind,probability=.5,strict_shared_init=True,contrast_off=True,mixture_target_immutable=True,validation_not_degraded=True,loss_composition_verified=True,atomic_state_and_private_rng_restore=True,logs=logs))


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args();torch.set_num_threads(2);bootstrap()
    if args.smoke_config:smoke(args.smoke_config,args.smoke_output)
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline;train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
