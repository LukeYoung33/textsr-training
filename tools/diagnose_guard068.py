"""No-update objective and gradient measurements on actual068 training-bank patches."""
from pathlib import Path
import sys,json,copy,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'guard068_diagnostic_20261103'
from tools.run_uniform20_steep_scurve import save,digest


def similarity(a,b):
    """@luna-comment Input:two detached gradient vectors. Output:norm/cosine. Key Steps:double precision dot. Key Params:zero gradients produce null cosine. State/Dependencies:torch read-only."""
    a=a.double();b=b.double();na=float(a.norm());nb=float(b.norm())
    return dict(left_norm=na,right_norm=nb,cosine=float((a*b).sum()/(na*nb)) if na>1e-15 and nb>1e-15 else None)


def derivatives(loss,output,params):
    """@luna-comment Input:one scalar andshared prediction/parameters. Output:output/parameter gradient vectors. Key Steps:retain graph,zero-fill unused parameters. Key Params:no backward update. State/Dependencies:owned CUDA graph."""
    import torch
    values=torch.autograd.grad(loss,[output,*params],retain_graph=True,allow_unused=True)
    return values[0].detach().flatten(),torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(params,values[1:])])


def support_stats(model,batch,prediction):
    """@luna-comment Input:paired state andY4UV prediction. Output:exact guard support/active fractions. Key Steps:replay original detached gates andthree projections. Key Params:original sigma/tau/masks. State/Dependencies:original Gaussian/gradient helpers;no mutation."""
    import torch
    import torch.nn.functional as F
    from tools.train_v3_edge_pilot import gaussian,gradients
    x=F.pixel_shuffle(model.lq[:,:4].detach(),2);t=F.pixel_shuffle(model.gt[:,:4].detach(),2);p=F.pixel_shuffle(prediction[:,:4].detach(),2)
    v=(1-F.max_pool2d(1-model.valid_mask_rgb.detach(),33,1,16)).clamp(0,1);v[:,:,:16]=0;v[:,:,-16:]=0;v[:,:,:,:16]=0;v[:,:,:,-16:]=0
    region=batch['line_region_mask'].to(x)*v;tx,ty=gradients(t);xx,xy=gradients(x);te=tx.square()+ty.square();xe=xx.square()+xy.square()
    cosine=(tx*xx+ty*xy)/(te*xe).sqrt().clamp_min(1e-8);ratio=(te/xe.clamp_min(1e-8)).sqrt()
    agree=(cosine>.95)&(ratio>.5)&(ratio<2)&(xe>.015**2)&((x-t).abs()<.15)
    coherence=(gaussian(xx,1).square()+gaussian(xy,1).square())/gaussian(xe,1).clamp_min(1e-8);support=region*agree*(coherence>.2)
    stats=[]
    for sigma in [.7,1.4,2.8]:
        hx=x-gaussian(x,sigma);ht=t-gaussian(t,sigma);r=p-x;hr=r-gaussian(r,sigma);energy=gaussian(hx.square(),1.4)
        trusted=support*(hx*ht>0)*(energy>.005**2);a=gaussian(hx*hr,1.4)/energy.clamp_min(1e-6);loss=(trusted*F.relu(-a-.1).square()).sum()/trusted.sum().clamp_min(1)
        stats.append(dict(sigma=sigma,loss=float(loss),trusted_pixels=float(trusted.sum()),active_pixels=float((trusted*(a<-.1)).sum()),trusted_fraction=float(trusted.mean())))
    return dict(loss=sum(r['loss'] for r in stats)/3,bands=stats,active_among_trusted=sum(r['active_pixels'] for r in stats)/max(sum(r['trusted_pixels'] for r in stats),1))


def main():
    """@luna-comment Input:frozen6current/1historical checkpoints and068bank. Output:32patch counterfactual costs andgradient conflict report. Key Steps:strict init,original targetalignment,no-update original Full backward,guard/native autograd,GT-equality probe,state equality. Key Params:paired input for everymodel,500K saturated auxiliary schedule. State/Dependencies:isolated CUDA process,original files immutable."""
    import numpy as np,torch,yaml,cv2
    from tools.train_latest_vosr_500k import bootstrap
    bootstrap()
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from basicsr.data import build_dataset
    from tools.train_v3_edge_pilot import objective
    torch.set_num_threads(2);cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    source=DATA/'latest_vosr_four_500k_20261102';bindings=json.loads((DATA/'latest_six_500k_mojito280_20261103/MODEL_BINDINGS.json').read_text())
    previous=json.loads((DATA/'s22_guard_contrast_four_mojito280_20261030/MODEL_BINDINGS.json').read_text())['nocp_nosdf'];bindings['previous250']=previous
    coeff={a:json.loads((source/'audits'/(a+'_guard')/'CALIBRATION.json').read_text())['weight'] for a in ['v3m','v5']}
    oldcal=json.loads((DATA/'s22_nocp_text_guard_20261029/audits/full_nosdf/CALIBRATION.json').read_text());coeff['previous250']=oldcal['weight']
    config=yaml.safe_load((source/'configs/v3m_guard.yml').read_text());manifest=json.loads(Path(config['datasets']['train']['manifest']).read_text());bank=Path(config['datasets']['train']['root'])
    si,shard=next((i,s) for i,s in enumerate(manifest['shards']) if s['sample_id']=='mojito_0143_IMG_20260605_040125766');meta=json.loads((bank/shard['metadata']).read_text());index=np.load(bank/manifest['unique_index']);allowed=set(map(int,index[index[:,0]==si,1]))
    pool=[i for i,r in enumerate(meta) if r['category']=='baseline' and i in allowed]
    roi=[i for i in pool if 768<=meta[i]['xy'][0]<=1792 and 2304<=meta[i]['xy'][1]<=2560]
    chosen=roi.copy()
    for i in np.linspace(0,len(pool)-1,32,dtype=int):
        if pool[i] not in chosen:chosen.append(pool[i])
        if len(chosen)==32:break
    chosen=chosen[:32];assert len(chosen)==32
    save(OUT/'PATCH_BINDING.json',dict(sample_id=shard['sample_id'],shard=shard,shard_id=si,baseline_available=len(pool),rows=[dict(row=i,xy=meta[i]['xy'],screenshot_near=i in roi) for i in chosen],sampling='screenshot-near baseline patches plus evenly spaced effective-text baseline patches;no augmentation',coefficients=coeff))
    allrecords=[]
    for key,binding in bindings.items():
        assert digest(binding['config'])==binding['config_sha256'] and digest(binding['checkpoint'])==binding['checkpoint_sha256']
        cfg=yaml.safe_load(Path(binding['config']).read_text());cfg['model_type']='S22Model';cfg['train']['model_type']='S22Model';cfg['name']='diagnose068_'+key
        cfg['s22_strategy'].update(name='paired',audit_root=str(OUT/'runtime'/key));cfg['path'].update(pretrain_network_g=binding['checkpoint'],resume_state=None)
        cfg['datasets']=copy.deepcopy(config['datasets']);cfg['train']['scheduler']={'type':'LatestVOSR500K'};cfg['train']['optim_g']['lr']=1e-4
        path=OUT/(key+'.yml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));sys.argv=[sys.argv[0],'-opt',str(path)];opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
        model=build_model(opt);model.net_g.eval();dataset=build_dataset(opt['datasets']['train']);dataset.index=np.asarray([(si,i) for i in chosen],np.uint32)
        state={k:v.detach().clone() for k,v in model.net_g.state_dict().items()};adam=copy.deepcopy(model.optimizer_g.state_dict());params=tuple(p for p in model.net_g.parameters() if p.requires_grad)
        weight=coeff['previous250'] if key=='previous250' else coeff['v5' if key.startswith('v5') else 'v3m']
        records=[]
        for start in range(0,32,4):
            batch=dataset.collate_fn([dataset[i] for i in range(start,start+4)]);model.feed_data(batch);model.output=model.net_g(model.lq)
            expected=model.output.detach().clone();native=model.native_forward(model.output,model.gt,valid_mask=model.valid_mask,text_mask=model.text_mask);guard,stats=objective(model,batch,'guard',model.identity_native)
            gn_out,gn_param=derivatives(native,model.output,params);gg_out,gg_param=derivatives(guard,model.output,params)
            telemetry=support_stats(model,batch,model.output);assert abs(telemetry['loss']-float(guard.detach()))<1e-7
            native_value=float(native.detach());guard_value=float(guard.detach());actual=model.output
            with torch.no_grad():
                model.output=model.gt;gtguard,gtstats=objective(model,batch,'guard',model.identity_native);gtmaps=support_stats(model,batch,model.gt)
                model.output=model.lq;zero,zstats=objective(model,batch,'guard',model.identity_native);assert float(zero)==0
            model.output=actual
            del native,guard,actual
            def retain_output(module,args,value):
                """@luna-comment Input:owned forward result. Output:retain prediction gradient. Key Steps:retain_grad only. Key Params:diagnostic hook. State/Dependencies:temporary network hook."""
                value.retain_grad()
            handle=model.net_g.register_forward_hook(retain_output);oldstep=model.optimizer_g.step
            def no_update(*args,**kwargs):
                """@luna-comment Input:Adam arguments. Output:no mutation. Key Steps:return. Key Params:no-update diagnostic only. State/Dependencies:owned optimizer."""
                return None
            model.optimizer_g.step=no_update;model.feed_data(batch);model.optimize_parameters(500000);handle.remove();model.optimizer_g.step=oldstep
            assert torch.allclose(model.output.detach(),expected,atol=1e-6,rtol=0)
            gf_param=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in params]);gf_out=model.output.grad.detach().flatten()
            assert torch.isfinite(gf_param).all() and torch.isfinite(gg_param).all()
            metrics={}
            for domain,gn,gg,gf in [('parameter',gn_param,gg_param,gf_param),('output',gn_out,gg_out,gf_out)]:
                metrics[domain]=dict(guard_vs_native=similarity(weight*gg,gn),guard_vs_full_without_guard=similarity(weight*gg,gf),native_vs_full_without_guard=similarity(gn,gf),native_vs_full_plus_guard=similarity(gn,gf+weight*gg),guard_to_native_norm=float(weight*gg.norm()/gn.norm().clamp_min(1e-15)),guard_to_full_norm=float(weight*gg.norm()/gf.norm().clamp_min(1e-15)))
            logs={k:float(v) for k,v in model.get_current_log().items()};record=dict(model=key,batch=start//4,patch_rows=chosen[start:start+4],native=native_value,guard_raw=guard_value,guard_weight=weight,weighted_guard=weight*guard_value,gt_guard_raw=float(gtguard),gt_guard_weighted=weight*float(gtguard),guard_telemetry=telemetry,gt_guard_telemetry=gtmaps,gradients=metrics,full_without_guard_logs=logs)
            records.append(record);allrecords.append(record)
            if start==0:
                np.savez_compressed(OUT/(key+'_patches.npz'),input=batch['lq_rgb'].numpy(),target=batch['gt_rgb'].numpy(),output=model.output.detach().cpu().numpy())
            model.optimizer_g.zero_grad(set_to_none=True)
        assert all(torch.equal(v,state[k]) for k,v in model.net_g.state_dict().items());assert not model.optimizer_g.state and not adam['state']
        save(OUT/(key+'_RESULT.json'),dict(passed=True,network_unchanged=True,optimizer_unchanged=True,checkpoint_sha256=binding['checkpoint_sha256'],records=records))
        save(OUT/'STATUS.json',dict(models_completed=key,records=len(allrecords),total_models=7));print(key,'complete',flush=True)
        del model,dataset,state;torch.cuda.empty_cache()
    save(OUT/'RESULTS.json',dict(passed=True,no_training_updates=True,records=allrecords,coefficients=coeff,patch_count=32,model_count=7))


if __name__=='__main__':main()
