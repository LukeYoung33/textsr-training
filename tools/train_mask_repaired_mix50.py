"""Matched repaired-target mix50 training with original R1 and resumable private RNG."""
from pathlib import Path
import sys, os, json, time, math, argparse, copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def bootstrap():
    """@luna-comment Input:existing runtime. Output:task-only mix50 model registration. Key Steps:reuse original objectives;add durable checkpoint pairs/RNG and monitoring. Key Params:no width reweighting or public API changes. State/Dependencies:process-local subclass."""
    from tools.train_latest_vosr_mix50 import bootstrap as original
    original()
    import torch
    from basicsr.utils.registry import MODEL_REGISTRY
    from tools.train_r1_width_500k import atomic_torch
    from tools.prepare_r1_width_bank import save,digest
    parent=MODEL_REGISTRY.get('LatestVOSRMix50')

    @MODEL_REGISTRY.register()
    class MaskRepairedMix50(parent):
        def optimize_parameters(self,current_iter):
            """@luna-comment Input:iteration. Output:unchanged original loss update and heartbeat. Key Steps:delegate accepted optimizer;record finite logs/LR/counts. Key Params:100-step telemetry. State/Dependencies:owned model/audit."""
            super().optimize_parameters(current_iter)
            if current_iter<=3 or current_iter%100==0:
                record=dict(iter=current_iter,lr=self.get_current_learning_rate()[0],updated=time.time(),pid=os.getpid(),synthetic_count=self.synthetic_count,total_draws=self.total_draws,logs={k:float(v) for k,v in self.log_dict.items()})
                save(self.audit_root/'TRAIN_STATUS.json',record)
                with (self.audit_root/'loss.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')

        def save(self,epoch,current_iter):
            """@luna-comment Input:epoch/step. Output:atomic weights/optimizer/scheduler/private-RNG pair. Key Steps:save finite weights and state;publish hashes last. Key Params:numeric checkpoints resumable;latest separate. State/Dependencies:owned experiment only."""
            modelpath=Path(self.opt['path']['models'])/('net_g_latest.pth' if current_iter==-1 else f'net_g_{current_iter}.pth')
            params={k:v.detach().cpu() for k,v in self.get_bare_model(self.net_g).state_dict().items()}
            assert all(torch.isfinite(v).all() for v in params.values());atomic_torch(modelpath,{'params':params})
            if current_iter!=-1:
                statepath=Path(self.opt['path']['training_states'])/f'{current_iter}.state'
                state=dict(epoch=epoch,iter=current_iter,optimizers=[o.state_dict() for o in self.optimizers],schedulers=[s.state_dict() for s in self.schedulers],mix_rng=self.mix_rng.get_state(),synthetic_count=self.synthetic_count,total_draws=self.total_draws)
                atomic_torch(statepath,state)
                save(statepath.with_suffix('.verified.json'),dict(iter=current_iter,model=str(modelpath),model_sha256=digest(modelpath),state=str(statepath),state_sha256=digest(statepath)))

        def resume_training(self,resume_state):
            """@luna-comment Input:verified optimizer state. Output:restored optimizer and exact next degradation RNG. Key Steps:parent restore,private RNG/counters restore. Key Params:refuse legacy state without RNG. State/Dependencies:owned model only."""
            super().resume_training(resume_state)
            self.mix_rng.set_state(resume_state['mix_rng'].cpu());self.synthetic_count=resume_state['synthetic_count'];self.total_draws=resume_state['total_draws']


def smoke(config,output):
    """@luna-comment Input:full owned config. Output:real nine-update/validation/mixture/save-resume evidence. Key Steps:strict seeded init;verify exact objective switches;restore next RNG/Adam update. Key Params:L1 or original MF only,mix50 both. State/Dependencies:isolated smoke model/checkpoints."""
    import torch,numpy as np
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from basicsr.data import build_dataset
    from tools.prepare_r1_width_bank import save
    sys.argv=[sys.argv[0],'-opt',str(config)];opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    dataset=build_dataset(opt['datasets']['train']);dataset.index=np.repeat(dataset.index[:1],4,axis=0)
    batch=dataset.collate_fn([dataset[i] for i in range(4)]);model=build_model(opt)
    initial=torch.load(opt['path']['pretrain_network_g'],map_location='cpu')['params']
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items()) and not model.optimizer_g.state
    t=opt['train'];mf=t['recall_aware_weight_missing']>0
    assert t['recall_aware_weight_false']==(.25 if mf else 0)
    for key in ['cp_weight','sdf_boundary_weight','uv_strong_weight','bg_lf_weight']:assert t.get(key,0)==0
    before={k:v.clone() for k,v in model.net_g.state_dict().items()}
    for i in [1,2,3,4,5,6,7,8,8000]:model.feed_data(batch);model.optimize_parameters(i)
    logs={k:float(v) for k,v in model.get_current_log().items()};assert all(math.isfinite(v) for v in logs.values())
    assert any(not torch.equal(v,before[k]) for k,v in model.net_g.state_dict().items())
    if mf:assert logs['l_missing']>0 and logs['l_false']>0
    else:assert abs(logs['l_total']-logs['l_native'])<1e-7
    assert 0<model.synthetic_count<model.total_draws==36 and not hasattr(model,'s22_handle')
    valid=dict(batch);valid['s22_training_sample']=[False]*4;count=model.synthetic_count;model.feed_data(valid)
    assert model.synthetic_count==count and torch.equal(model.lq_rgb,batch['lq_rgb'].to(model.device))
    model.save(0,8000);state=torch.load(Path(opt['path']['training_states'])/'8000.state',map_location=model.device)
    restored=build_model(copy.deepcopy(opt));restored.net_g.load_state_dict(model.net_g.state_dict());restored.resume_training(state)
    model.feed_data(batch);restored.feed_data(batch);assert torch.equal(model.lq_rgb,restored.lq_rgb)
    model.optimize_parameters(8001);restored.optimize_parameters(8001)
    assert all(torch.allclose(a,b,rtol=1e-6,atol=1e-8) for a,b in zip(model.net_g.parameters(),restored.net_g.parameters()))
    save(output,dict(passed=True,updates=9,original_r1=mf,width_reweighting=False,exact_init=True,mixture_train_only=True,resume_next_step=True,logs=logs))


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args()
    import torch
    torch.set_num_threads(2);bootstrap()
    if args.smoke_config:smoke(args.smoke_config,args.smoke_output)
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline
        train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
