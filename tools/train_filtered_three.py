"""Real-paired,no-contrast native L1/original-R1/normalized-R1 controls."""
from pathlib import Path
import sys,json,time,math,argparse,copy
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_r1_width_bank import save


def bootstrap():
    """@luna-comment Input:accepted R1 runtime. Output:private paired model subclasses with durable state/telemetry. Key Steps:reuse original loss implementations;no mixture invocation;atomic checkpoint mixin. Key Params:original R1 1/.25,accepted norm cap4. State/Dependencies:process-local registry,existing jobs unaffected."""
    from tools.train_r1_width_500k import bootstrap as accepted
    accepted()
    from basicsr.utils.registry import MODEL_REGISTRY
    from tools.train_gt1_r1_four import AtomicMixture
    Paired=MODEL_REGISTRY.get('S22Model');Norm=MODEL_REGISTRY.get('R1WidthModel')

    class PairedState(AtomicMixture):
        def optimize_parameters(self,current_iter):
            """@luna-comment Input:update index. Output:accepted loss step and common paired-only telemetry. Key Steps:delegate exact parent;assert zero synthetic inputs;write history. Key Params:100-step heartbeat,500K target. State/Dependencies:owned optimizer/audit root."""
            super().optimize_parameters(current_iter);assert self.synthetic_count==0 and self.strategy=='paired'
            self.log_dict.update(gt_mixture_probability=0.,gt_mixture_fraction=0.,input_paired_fraction=1.)
            assert all(math.isfinite(float(v)) for v in self.log_dict.values())
            if current_iter<=3 or current_iter%100==0:
                record=dict(iter=current_iter,lr=self.get_current_learning_rate()[0],updated=time.time(),logs=dict(self.log_dict),synthetic_count=0,total_draws=self.total_draws)
                save(self.audit_root/'RUN_STATUS.json',record)
                with (self.audit_root/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')

        def resume_training(self,state):
            """@luna-comment Input:own numeric checkpoint state. Output:optimizer/scheduler and unused private RNG restored safely. Key Steps:normalize RNG state to CPU before accepted mixin restore. Key Params:no synthetic-input continuation introduced. State/Dependencies:owned model only."""
            adjusted=dict(state);adjusted['mixture_rng_state']=state['mixture_rng_state'].cpu();super().resume_training(adjusted)
            assert self.synthetic_count==0

    @MODEL_REGISTRY.register()
    class FilteredPairedModel(PairedState,Paired):
        """Task-local native L1 or original Missing/False objective."""

    @MODEL_REGISTRY.register()
    class FilteredNormModel(PairedState,Norm):
        """Task-local accepted current-batch width-normalized R1 objective."""


def smoke(config,output):
    """@luna-comment Input:full actual-data recipe. Output:CUDA9update,data/loss/paired/finite/resume evidence. Key Steps:strict shared init;real four-draw batch;zero input/GT modification;objective switches;restore same next update. Key Params:no contrast/no synthetic,no extra aux. State/Dependencies:isolated smoke artifacts only."""
    from basicsr.utils.options import parse_options
    from basicsr.data import build_dataset
    from basicsr.models import build_model
    sys.argv=[sys.argv[0],'-opt',str(config)];opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    assert opt['s22_strategy']['name']=='paired' and not opt['datasets']['train']['patch_contrast_alignment']
    dataset=build_dataset(opt['datasets']['train']);batch=dataset.collate_fn([dataset[i] for i in [0,len(dataset)//4,len(dataset)//2,3*len(dataset)//4]])
    raw=batch['lq_rgb'].clone();target=batch['gt_rgb'].clone();model=build_model(opt)
    initial=torch.load(opt['path']['pretrain_network_g'],map_location='cpu')['params'];assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items()) and not model.optimizer_g.state
    t=opt['train'];kind=opt['filtered_control']['kind']
    if kind=='norm':
        ordinary=copy.deepcopy(opt['datasets']['train']);ordinary['type']='S22Dataset';reference=build_dataset(ordinary)
        for i in [0,len(dataset)//4,len(dataset)//2,3*len(dataset)//4]:
            a,b=dataset[i],reference[i]
            for key in ['lq_rgb','gt_rgb','text_mask_rgb','valid_mask_rgb','line_region_mask']:assert torch.equal(a[key],b[key]),key
    for key in ['cp_weight','sdf_boundary_weight','uv_strong_weight','bg_lf_weight']:assert t.get(key,0)==0
    for i in range(1,10):
        model.update_learning_rate(i,warmup_iter=-1);model.feed_data(batch)
        assert torch.equal(model.lq_rgb,raw.to(model.device)) and torch.equal(model.gt_rgb,target.to(model.device))
        model.optimize_parameters(7000+i)
    logs=dict(model.log_dict);assert model.synthetic_count==0 and model.total_draws==36 and not hasattr(model,'s22_handle')
    assert torch.equal(batch['lq_rgb'],raw) and torch.equal(batch['gt_rgb'],target)
    assert abs(logs['l_total']-logs['l_native']-logs.get('l_missing',0)-logs.get('l_false',0))<1e-6
    if kind=='l1':assert abs(logs['l_total']-logs['l_native'])<1e-7
    else:assert logs['l_missing']>0 and logs['l_false']>0 and t['recall_aware_weight_missing']==1 and t['recall_aware_weight_false']==.25
    if kind=='norm':
        for name in ['missing','false']:
            assert max(logs[f'r1_{name}_w{i}'] for i in range(3))<=4.00001
            if sum(logs[f'r1_{name}_n{i}'] for i in range(3))>0:assert abs(logs[f'r1_{name}_mean_weight']-1)<1e-4
    assert any(not torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items())
    model.save(0,9);state=torch.load(Path(opt['path']['training_states'])/'9.state',map_location=model.device);restored=build_model(copy.deepcopy(opt));restored.net_g.load_state_dict(model.net_g.state_dict());restored.resume_training(state)
    model.feed_data(batch);restored.feed_data(batch);model.optimize_parameters(7010);restored.optimize_parameters(7010)
    assert all(torch.allclose(a,b,rtol=1e-6,atol=1e-8) for a,b in zip(model.net_g.parameters(),restored.net_g.parameters()))
    save(output,dict(passed=True,kind=kind,updates=9,strict_initial=True,fresh_optimizer=True,synthetic_inputs=0,contrast_off=True,lq_gt_unchanged=True,loss_composition_verified=True,next_update_resume=True,logs=logs))


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--smoke-config');p.add_argument('--smoke-output');a,rest=p.parse_known_args();torch.set_num_threads(2);bootstrap()
    if a.smoke_config:smoke(a.smoke_config,a.smoke_output)
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline
        train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
