"""Opt-in synchronized-mask V3M native L1 plus capped width-balanced R1 trainer."""
from pathlib import Path
import sys,json,os,time,math,copy,argparse
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def group_weights(counts,cap=4.,eps=1e-4):
    """@luna-comment Input:three nonnegative GPU support counts. Output:detached mean-one capped weights. Key Steps:inverse sqrt current fractions;3-pass active-set solve. Key Params:cap4,epsilon1e-4,absentzero. State/Dependencies:GPU tensors,no CPU synchronization or EMA."""
    counts=counts.float();p=counts/counts.sum().clamp_min(1);score=(p+eps).rsqrt();free=counts>0;w=torch.zeros_like(p)
    for _ in range(3):
        scale=(1-(p*w*(~free)).sum())/(p*score*free).sum().clamp_min(1e-12)
        candidate=score*scale;w=torch.where(free,candidate.clamp(max=cap),w);free=free&(candidate<cap)
    return w.detach()


def width_r1(pred,gt,text,valid,groups):
    """@luna-comment Input:RGB pair,B1HW masks and0..2 labels. Output:raw missing/false plus audit scalars. Key Steps:original polarity/ring/active gates;independent current-batch group weights;smooth hinge. Key Params:radius3,margin.03,temp.01,contrast.03. State/Dependencies:float32 loss accumulations,pred differentiable."""
    pred=pred.float();gt=gt.float();text=(text[:,:1]>.5).float();valid=(valid[:,:1]>.5).float()
    ring=(F.max_pool2d(text,7,1,3)>.5).float()*(1-text)*valid;text=text*valid
    coeff=pred.new_tensor([.299,.587,.114])[None,:,None,None];py=(pred*coeff).sum(1,keepdim=True);gy=(gt*coeff).sum(1,keepdim=True)
    contrast=(gy*ring).sum((2,3),keepdim=True)/ring.sum((2,3),keepdim=True).clamp_min(1e-6)-(gy*text).sum((2,3),keepdim=True)/text.sum((2,3),keepdim=True).clamp_min(1e-6)
    polarity=torch.where(contrast>=0,1.,-1.);active=(contrast.abs()>=.03).float();error=polarity*(py-gy)
    onehot=torch.cat([(groups==i).float() for i in range(3)],dim=1);losses=[];stats={}
    for name,support,e in [('missing',text*active,error),('false',ring*active,-error)]:
        counts=(onehot*support).sum((0,2,3));weights=group_weights(counts);mapw=(onehot*weights[None,:,None,None]).sum(1,keepdim=True)
        loss=(.01*F.softplus((e-.03)/.01)*support*mapw).sum()/support.sum().clamp_min(1e-6);losses.append(loss)
        for i in range(3):stats[f'r1_{name}_w{i}']=weights[i];stats[f'r1_{name}_n{i}']=counts[i]
        stats[f'r1_{name}_mean_weight']=(counts*weights).sum()/counts.sum().clamp_min(1)
    stats['recall_active_ratio']=active.mean();return *losses,stats


def atomic_torch(path,value):
    """@luna-comment Input:owned checkpoint/value. Output:atomic torch file. Key Steps:save/fsync/rename. Key Params:same-filesystem temp. State/Dependencies:checkpoint destination only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def bootstrap():
    """@luna-comment Input:existing runtime. Output:experiment-only dataset/model registrations. Key Steps:accepted bootstrap,sidecar masks,width-R1 optimizer andatomic save. Key Params:no public/base source changes. State/Dependencies:process-localregistry."""
    from tools.train_latest_vosr_500k import bootstrap as original
    original()
    from basicsr.utils.registry import MODEL_REGISTRY,DATASET_REGISTRY
    from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    from tools.prepare_r1_width_bank import save,digest
    ParentDataset=DATASET_REGISTRY.get('S22Dataset');ParentModel=MODEL_REGISTRY.get('S22Model')

    @DATASET_REGISTRY.register()
    class R1WidthDataset(ParentDataset):
        def __getitem__(self,index):
            """@luna-comment Input:immutable draw. Output:originalRGB/contrast withupdated mask/groups. Key Steps:parent reader;bounded sidecar mmap;replace both mask aliases. Key Params:trainonly sidecars;val unchanged andnotusedforR1. State/Dependencies:ownworkerLRU."""
            item=super().__getitem__(index)
            if not item['s22_training_sample']:
                item['width_groups']=torch.zeros_like(item['text_mask_rgb'],dtype=torch.uint8);return item
            si,ri=map(int,self.index[index])
            if not hasattr(self,'width_cache'):self.width_cache=OrderedDict()
            if si not in self.width_cache:
                path=self.root/'width_codes'/(self.shards[si]['array']+'.codes.npy');values=np.load(path,mmap_mode='r');assert values.shape==(self.shards[si]['count'],256,256) and values.dtype==np.uint8;self.width_cache[si]=values
            self.width_cache.move_to_end(si);value=self.width_cache[si][ri].copy()
            while len(self.width_cache)>self.cache_shards:self.width_cache.popitem(last=False)
            item['width_groups']=torch.from_numpy((value&3)[None]);item['text_mask_rgb']=torch.from_numpy(((value&4)>0)[None]).float();item['text_mask']=item['text_mask_rgb']
            return item

        def __getstate__(self):
            """@luna-comment Input:worker spawn. Output:state without mmaphandles. Key Steps:parentclear thenwidthclear. Key Params:boundedcache. State/Dependencies:no input mutation."""
            return {**super().__getstate__(),'width_cache':OrderedDict()}

    @MODEL_REGISTRY.register()
    class R1WidthModel(ParentModel):
        def feed_data(self,data):
            """@luna-comment Input:pairedbatchwithgroups. Output:ordinary native data plusGPU groupmap. Key Steps:parent conversion thenlabels upload. Key Params:no synthetic replacement. State/Dependencies:currentbatch overwritten."""
            super().feed_data(data);self.width_groups=data['width_groups'].to(self.device,non_blocking=True)

        def optimize_parameters(self,current_iter):
            """@luna-comment Input:iteration. Output:one finite L1/R1 Adam update andheartbeat. Key Steps:network/nativeL1;decoded weightedR1;original ramp;finite backward/step;scalar logs. Key Params:missing1/false.25;nootherloss. State/Dependencies:ownedmodel/optimizer/audit."""
            self.optimizer_g.zero_grad(set_to_none=True);self.output=self._forward_v5(self.lq)
            native=self.cri_y4uv_native(self.output,self.gt,valid_mask=self.valid_mask,text_mask=self.text_mask)
            rgb=YUV2RGB(y4uv_to_yuv444(self.output),format='RGB',is_train=True,uv_normalized=True)
            text,valid=self._structure_masks();missing,false,stats=width_r1(rgb,self.gt_rgb,text,valid,self.width_groups)
            ms=self._schedule(1.,current_iter);fs=self._schedule(.25,current_iter);total=native+ms*missing+fs*false
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite total before backward')
            total.backward();self.optimizer_g.step()
            logs=dict(l_native=native.detach(),l_missing=(ms*missing).detach(),l_false=(fs*false).detach(),l_total=total.detach(),l_missing_raw=missing.detach(),l_false_raw=false.detach(),missing_scale=ms,false_scale=fs,**stats)
            self.log_dict=OrderedDict((k,float(v)) for k,v in logs.items());assert all(math.isfinite(v) for v in self.log_dict.values())
            if current_iter<=3 or current_iter%100==0:
                record=dict(iter=current_iter,total=500000,lr=self.get_current_learning_rate()[0],updated=time.time(),pid=os.getpid(),logs=self.log_dict)
                save(self.audit_root/'TRAIN_STATUS.json',record)
                with (self.audit_root/'loss.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')

        def save(self,epoch,current_iter):
            """@luna-comment Input:epoch/update. Output:atomic model,state andpairedhashreceipt. Key Steps:CPUweights;optimizer/schedulers;publishreceiptlast. Key Params:latestweights separate;numericpairs resumable. State/Dependencies:owned experiment paths."""
            modelpath=Path(self.opt['path']['models'])/('net_g_latest.pth' if current_iter==-1 else f'net_g_{current_iter}.pth')
            params={k:v.detach().cpu() for k,v in self.get_bare_model(self.net_g).state_dict().items()};assert all(torch.isfinite(v).all() for v in params.values());atomic_torch(modelpath,{'params':params})
            if current_iter!=-1:
                statepath=Path(self.opt['path']['training_states'])/f'{current_iter}.state'
                state=dict(epoch=epoch,iter=current_iter,optimizers=[o.state_dict() for o in self.optimizers],schedulers=[s.state_dict() for s in self.schedulers])
                assert all(torch.isfinite(v).all() for o in state['optimizers'] for item in o['state'].values() for v in item.values() if torch.is_tensor(v))
                atomic_torch(statepath,state)
                save(statepath.with_suffix('.verified.json'),dict(iter=current_iter,model=str(modelpath),model_sha256=digest(modelpath),state=str(statepath),state_sha256=digest(statepath)))


def smoke(config,output):
    """@luna-comment Input:owned smoke config. Output:CUDA updates/losscomposition/save-resume evidence. Key Steps:strictinit,realupdatedmaskbatch,9steps,snapshotrestore,schedulerboundarycheck. Key Params:independent smoke artifacts only. State/Dependencies:realruntimeandGPU."""
    from basicsr.utils.options import parse_options
    from basicsr.data import build_dataset
    from basicsr.models import build_model
    from tools.prepare_r1_width_bank import save
    from tools.train_latest_vosr_500k import rate
    sys.argv=[sys.argv[0],'-opt',str(config)];opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    dataset=build_dataset(opt['datasets']['train']);batch=dataset.collate_fn([dataset[i] for i in range(4)]);model=build_model(opt)
    initial=torch.load(opt['path']['pretrain_network_g'],map_location='cpu')['params'];assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.net_g.state_dict().items())
    before={k:v.clone() for k,v in model.net_g.state_dict().items()}
    for i in [1,2,3,4,5,6,7,8,8000]:model.feed_data(batch);model.optimize_parameters(i)
    logs=model.get_current_log();assert abs(logs['l_total']-logs['l_native']-logs['l_missing']-logs['l_false'])<1e-6
    assert all(logs['r1_'+t+'_w'+str(i)]<=4.00001 for t in ['missing','false'] for i in range(3))
    assert any(not torch.equal(v,before[k]) for k,v in model.net_g.state_dict().items());model.save(0,8000)
    state=torch.load(Path(opt['path']['training_states'])/'8000.state',map_location=model.device);restored=build_model(copy.deepcopy(opt));restored.net_g.load_state_dict(model.net_g.state_dict());restored.resume_training(state)
    assert all(torch.equal(a,b) for a,b in zip(model.net_g.parameters(),restored.net_g.parameters()))
    # Same next input must produce identical parameters after the restored Adam step.
    model.feed_data(batch);restored.feed_data(batch);model.optimize_parameters(8001);restored.optimize_parameters(8001)
    assert all(torch.allclose(a,b,rtol=1e-6,atol=1e-8) for a,b in zip(model.net_g.parameters(),restored.net_g.parameters()))
    boundaries={str(i):rate(i) for i in [1,199999,200000,300000,400000,450000,500000]};assert boundaries['500000']==5e-6
    save(output,dict(passed=True,logs=logs,boundaries=boundaries,exact_init=True,resume_next_step=True,updated_mask=True))


def finalize(config):
    """@luna-comment Input:frozen500K recipe. Output:strict terminal model/state validation andfull620-image patch metrics. Key Steps:verifycheckpointpair andfinite state;loadexactweights;no-update validation;atomicreceipt. Key Params:LR5e-6,total500K,originalvalidationdomain. State/Dependencies:ownedreport only."""
    import yaml
    from basicsr.utils.options import parse_options
    from basicsr.data import build_dataset
    from basicsr.models import build_model
    from tools.prepare_r1_width_bank import OUT,save,digest
    cfg=yaml.safe_load(Path(config).read_text());exp=Path(cfg['path']['experiments_root'])/cfg['name'];receipt=json.loads((exp/'training_states/500000.verified.json').read_text())
    assert digest(receipt['model'])==receipt['model_sha256'] and digest(receipt['state'])==receipt['state_sha256']
    state=torch.load(receipt['state'],map_location='cpu');assert state['iter']==500000 and abs(state['optimizers'][0]['param_groups'][0]['lr']-5e-6)<1e-12
    params=torch.load(receipt['model'],map_location='cpu')['params'];assert all(torch.isfinite(v).all() for v in params.values())
    cfg['path']['pretrain_network_g']=receipt['model'];cfg['path']['resume_state']=None;cfg['name']+='_final_validation';cfg['s22_strategy']['audit_root']=str(OUT/'final_audit')
    owned=OUT/'final_validation.yml';owned.write_text(yaml.safe_dump(cfg,sort_keys=False));sys.argv=[sys.argv[0],'-opt',str(owned)];opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    model=build_model(opt);assert all(torch.equal(v.cpu(),params[k]) for k,v in model.net_g.state_dict().items());dataset=build_dataset(opt['datasets']['val']);loader=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2,collate_fn=dataset.collate_fn)
    sums=dict(native=0.,reference=0.);count=0;model.net_g.eval()
    with torch.no_grad():
        for batch in loader:
            model.feed_data(batch);model.test();n=len(model.output);a=model.cri_y4uv_native(model.output,model.gt,valid_mask=model.valid_mask,text_mask=model.text_mask);b=model.cri_yuv444_reference(model.output,model.gt,valid_mask=model.valid_mask,text_mask=model.text_mask)
            assert torch.isfinite(a) and torch.isfinite(b);sums['native']+=float(a)*n;sums['reference']+=float(b)*n;count+=n
            if count%400==0:print('FINAL_VALIDATION',count,len(dataset),flush=True)
    assert count==len(dataset) and count>0 and len(dataset.shards)==620
    save(OUT/'FINAL_VALIDATION.json',dict(passed=True,iter=500000,lr=5e-6,parents=620,patches=count,metrics={k:v/count for k,v in sums.items()},checkpoint_sha256=receipt['model_sha256'],validation_domain='unchanged historical620, not newly dilatedGT',strict_load=True,updated=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');parser.add_argument('--final-config');args,rest=parser.parse_known_args();torch.set_num_threads(2);bootstrap()
    if args.smoke_config:smoke(args.smoke_config,args.smoke_output)
    elif args.final_config:finalize(args.final_config)
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline;train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
