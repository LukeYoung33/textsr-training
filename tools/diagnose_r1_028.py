"""Frozen real-input R1 spatial/parameter gradient audit on training sample028."""
from pathlib import Path
import sys,json,inspect,textwrap,types,copy,time,hashlib,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'r1_gradient_028_250k_20261103'
WIDTH2=os.environ.get('R1_WIDTH2')=='1'
BASELINE_OUT=OUT
if WIDTH2:OUT=DATA/'r1_gradient_028_width2_250k_20261103'
SAMPLE='capture_261011_003171'
TERMS=['native','missing','false','r1','other_aux','total']


def widths(mask):
    """@luna-comment Input:binary foreground HW. Output:nearest-skeleton local diameter estimate everywhere. Key Steps:2xnearest mask EDT,max ineach2x2cell,skeleton propagation. Key Params:exact axial integer bars;curved/junction estimates approximate. State/Dependencies:scipy/skimage CPU,immutable mask."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    if not mask.any():return np.zeros(mask.shape,np.float32)
    high=np.repeat(np.repeat(mask,2,0),2,1)
    distance=ndi.distance_transform_edt(high).reshape(mask.shape[0],2,mask.shape[1],2).max((1,3))
    skeleton=skeletonize(mask);nearest=ndi.distance_transform_edt(~skeleton,return_distances=False,return_indices=True)
    return distance[tuple(nearest)].astype(np.float32)


def state_hash(net):
    """@luna-comment Input:network. Output:state SHA256. Key Steps:ordered key/tensor bytes. Key Params:includes buffers. State/Dependencies:read-only CPU copies."""
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def instrument(model):
    """@luna-comment Input:effective trained model. Output:bound lossgraph-only evaluator. Key Steps:locate real optimize implementation;captureR1 inputs;return beforebackward/Adam. Key Params:unique anchored replacements;same formula/order. State/Dependencies:private compiled function,production code untouched."""
    for cls in type(model).__mro__:
        f=cls.__dict__.get('optimize_parameters')
        if f is not None and 'total.backward()' in inspect.getsource(f):break
    else:raise RuntimeError('Effective loss implementation not found')
    source=textwrap.dedent(inspect.getsource(f));save(OUT/'IMPLEMENTATION.json',dict(file=inspect.getsourcefile(f),sha256=digest(inspect.getsourcefile(f)),class_name=cls.__name__))
    anchor='missing, false, details = recall_aware_structure_losses(output_rgb, self.gt_rgb, text_rgb, valid_rgb, **train_opt.get("recall_aware_opt", {}))'
    assert source.count(anchor)==1 and source.count('total.backward()')==1
    source=source.replace(anchor,anchor+'\n        audit_text, audit_valid, audit_details = text_rgb, valid_rgb, details')
    source=source.replace('total.backward()', 'return dict(native=native_loss, missing=missing_scale*missing, false=false_scale*false, total=total, rgb=output_rgb, text=audit_text, valid=audit_valid, details=audit_details, logs=logs)')
    namespace=dict(f.__globals__);exec(compile(source,'<frozen_r1_loss_graph>','exec'),namespace)
    return types.MethodType(namespace['optimize_parameters'],model)


def vector_metrics(left,right):
    """@luna-comment Input:two flat detached gradients. Output:norms,ratio andcosine. Key Steps:double precision dot. Key Params:zero cosine nullable. State/Dependencies:CPU torch."""
    import torch
    a=left.double();b=right.double();na=float(a.norm());nb=float(b.norm())
    return dict(left_norm=na,right_norm=nb,norm_ratio=na/nb if nb>0 else None,cosine=float(torch.dot(a,b)/(na*nb)) if na*nb>0 else None)


def image8(rgb):
    """@luna-comment Input:CHW RGB float. Output:uint8 BGR. Key Steps:clip solely fordisplay. Key Params:no analysis clipping. State/Dependencies:numpy."""
    return np.rint(np.clip(rgb.transpose(1,2,0),0,1)*255).astype(np.uint8)[...,::-1]


def overlay(base,values,limit,signed=False):
    """@luna-comment Input:BGR base andgradient HW. Output:fixed-scale color overlay. Key Steps:sharedlinear normalization withalpha proportionaltomagnitude. Key Params:cyanpositiveYdescent/orangenegative;unsignedturbo. State/Dependencies:displayonly,rawmaps retained."""
    amount=np.clip(np.abs(values)/max(limit,1e-20),0,1)
    if signed:
        colors=np.empty_like(base);colors[:]=(40,75,240);colors[values>0]=(240,190,20)
    else:colors=cv2.applyColorMap(np.rint(amount*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    alpha=(amount*.88)[...,None]
    return np.rint(base*(1-alpha)+colors*alpha).astype(np.uint8)


def main():
    """@luna-comment Input:completedmix50 config,250K checkpoint andsample028 immutablebank. Output:no-update rawgradient maps/stats/PNGs. Key Steps:actualdataset/losspath,25baselinepatches inB4fixedcohorts,autograd,analyticchecks,widthbins,spatialaverage. Key Params:realinputbranch,exacttrainingtargetcontrast,no parameters updated. State/Dependencies:owned outputs;CUDAselectedexternally."""
    import torch,yaml
    import torch.nn.functional as F
    from tools.train_latest_vosr_mix50 import bootstrap
    bootstrap()
    from basicsr.models import build_model
    from basicsr.data import build_dataset
    from basicsr.utils.options import parse_options
    from basicsr.utils.registry import MODEL_REGISTRY
    torch.set_num_threads(2);cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    base=DATA/'latest_vosr_mix50_500k_20261102';config=base/'configs/v3m_mix50.yml';cfg=yaml.safe_load(config.read_text())
    ckpt=base/'experiments'/cfg['name']/'models/net_g_250000.pth';assert ckpt.exists()
    cfg['path']['pretrain_network_g']=str(ckpt);cfg['path']['resume_state']=None;cfg['s22_strategy']['audit_root']=str(OUT/'unused_audit');cfg['name']='r1_028_frozen'
    owned=OUT/'diagnostic.yml';owned.write_text(yaml.safe_dump(cfg,sort_keys=False));sys.argv=[sys.argv[0],'-opt',str(owned)]
    opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    dataset=build_dataset(opt['datasets']['train']);model=build_model(opt)
    expected=torch.load(ckpt,map_location='cpu')['params'];assert all(torch.equal(v.cpu(),expected[k]) for k,v in model.net_g.state_dict().items())
    assert not hasattr(model,'s22_handle') and opt['train']['recall_aware_weight_missing']==1 and opt['train']['recall_aware_weight_false']==.25
    assert opt['datasets']['train']['patch_contrast_alignment'] and not model.optimizer_g.state
    model.net_g.eval();before=state_hash(model.net_g);evaluate=instrument(model)
    si,shard=next((i,s) for i,s in enumerate(dataset.shards) if s['sample_id']==SAMPLE)
    metas=json.loads((dataset.root/shard['metadata']).read_text());eligible={tuple(map(int,r)) for r in dataset.index if int(r[0])==si}
    ids=[i for i,m in enumerate(metas) if m['category']=='baseline' and (si,i) in eligible];assert len(ids)==25
    dataset.index=np.asarray([(si,i) for i in ids],np.uint32)
    source_manifest=DATA/'full_s22_training_matrix_20261028/source_view/manifests/train.jsonl'
    source=next(json.loads(l) for l in source_manifest.read_text().splitlines() if SAMPLE in l)
    for k in ['gt','lq','text_mask','valid_mask']:assert digest(source['files'][k])==source['output_sha256'][k]
    original=cv2.imread(source['original_gt_lineage']['path']);H,W=original.shape[:2]
    if WIDTH2:
        target_root=DATA/'original_gt_dilation30_20261102'
        target_record=next(r for r in json.loads((target_root/'RESULTS.json').read_text())['rows'] if r['id']=='028')
        target_path=target_root/target_record['images']['width2']
        assert digest(target_record['source']['original'])==digest(source['original_gt_lineage']['path'])
        assert digest(target_path)==target_record['output_sha256']['width2']
        target=cv2.imread(str(target_path));assert target.shape==original.shape
        save(OUT/'TARGET_BINDING.json',dict(path=str(target_path),sha256=digest(target_path),mask='unchanged baseline mask; GT-only counterfactual',contrast='same per-patch alignment',nominal_width_increase=2))
    original_mask=cv2.imread(source['files']['text_mask'],0);width=widths(original_mask>127.5)
    full={k:np.zeros((H,W),np.float64) for k in TERMS+['support_m','support_f','activation_m','activation_f','weak_soft_mask']};images={k:np.zeros((H,W,3),np.float64) for k in ['lq','gt','output']};coverage=np.zeros((H,W),np.float64)
    stats={kind:{str(b):dict(pixels=0,gradient_sum=0.,loss_sum=0.,activation_sum=0.,strong_pixels=0,native_gradient=0.,total_gradient=0.,opposed_pixels=0) for b in range(3)} for kind in ['missing','false']}
    params=tuple(p for p in model.net_g.parameters() if p.requires_grad);patches=[];batches=[];all_maps=[];analytical_max=0.;start=time.time()
    for offset in range(0,len(ids),4):
        positions=[(offset+j)%len(ids) for j in range(4)];batch=dataset.collate_fn([dataset[i] for i in positions])
        if WIDTH2:
            from tools.patch_contrast_alignment import match_patch_contrast
            for j,pos in enumerate(positions):
                meta=metas[ids[pos]];x,y=map(int,meta['xy']);h,w=map(int,meta['real_hw'])
                crop=np.pad(target[y:y+h,x:x+w],((0,256-h),(0,256-w),(0,0)),mode='edge')
                rgb=crop[...,::-1].transpose(2,0,1).astype(np.float32)/255.
                adjusted,contrast=match_patch_contrast(rgb,batch['lq_rgb'][j].numpy(),batch['valid_mask_rgb'][j].numpy())
                batch['gt_rgb'][j]=torch.from_numpy(adjusted);batch['gt'][j]=batch['gt_rgb'][j]
                for key,value in contrast.items():batch['contrast_'+key][j]=float(value)
        MODEL_REGISTRY.get('S22Model').feed_data(model,batch)
        assert torch.equal(model.lq_rgb,batch['lq_rgb'].to(model.device))
        result=evaluate(250000);result['r1']=result['missing']+result['false'];result['other_aux']=result['total']-result['native']-result['r1']
        grads={};vectors={}
        for term in TERMS:
            got=torch.autograd.grad(result[term],(model.output,)+params,retain_graph=True,allow_unused=True)
            grads[term]=F.pixel_shuffle(got[0][:,:4],2).detach().cpu().numpy()[:,0]
            vectors[term]=torch.cat([(g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu')).reshape(-1) for g,p in zip(got[1:],params)])
        assert np.allclose(grads['missing']+grads['false'],grads['r1'],rtol=2e-5,atol=1e-10)
        assert np.allclose(grads['native']+grads['r1']+grads['other_aux'],grads['total'],rtol=2e-4,atol=1e-9)
        tm=(result['text'][:,:1]>.5).float();vm=(result['valid'][:,:1]>.5).float();active=result['details']['active'];sm=tm*vm*active;sf=result['details']['ring']*active
        from realesrgan.losses.recall_aware_structure_loss import _luma,_soft_hinge
        e=result['details']['polarity']*(_luma(result['rgb'])-_luma(model.gt_rgb));am=torch.sigmoid((e-.03)/.01);af=torch.sigmoid((-e-.03)/.01)
        # Verify the exact loss-entry RGB derivative before any RGB gamut conversion chain.
        rgb_grad=torch.autograd.grad(result['r1'],result['rgb'],retain_graph=True)[0]
        analytical=result['details']['polarity']*(sm*am/sm.sum().clamp_min(1e-6)-.25*sf*af/sf.sum().clamp_min(1e-6))
        expected_rgb=analytical*torch.tensor([.299,.587,.114],device=model.device)[None,:,None,None]
        err=float((rgb_grad-expected_rgb).abs().max());analytical_max=max(analytical_max,err);assert torch.allclose(rgb_grad,expected_rgb,rtol=3e-5,atol=1e-10)
        support={'missing':sm.detach().cpu().numpy()[:,0],'false':sf.detach().cpu().numpy()[:,0]};activation={'missing':am.detach().cpu().numpy()[:,0],'false':af.detach().cpu().numpy()[:,0]}
        lm={'missing':(_soft_hinge(e,.03,.01)*sm/sm.sum().clamp_min(1e-6)).detach().cpu().numpy()[:,0],'false':(.25*_soft_hinge(-e,.03,.01)*sf/sf.sum().clamp_min(1e-6)).detach().cpu().numpy()[:,0]}
        record=dict(offset=offset,positions=positions,losses={k:float(result[k].detach()) for k in TERMS},active=active.flatten().tolist(),text_pixels=float(sm.sum()),ring_pixels=float(sf.sum()),gradient_relations={k:vector_metrics(vectors['r1'],vectors[k]) for k in ['native','other_aux','total']},r1_vs_rest=vector_metrics(vectors['r1'],vectors['total']-vectors['r1']))
        batches.append(record)
        lq=model.lq_rgb.detach().cpu().numpy();gt=model.gt_rgb.detach().cpu().numpy();pred=result['rgb'].detach().cpu().numpy()
        for j,pos in enumerate(positions):
            if offset+j>=len(ids):continue
            meta=metas[ids[pos]];x,y=map(int,meta['xy']);h,w=map(int,meta['real_hw']);assert np.array_equal(meta['matrix'],np.eye(3))
            pg={k:grads[k][j,:h,:w] for k in TERMS};ps={k:support[k][j,:h,:w] for k in support};pa={k:activation[k][j,:h,:w] for k in activation}
            region=width[y:y+h,x:x+w];bins=np.where(region<=2.00001,0,np.where(region<=4.00001,1,2))
            soft=batch['text_mask_rgb'][j,0,:h,:w].numpy();weak=((soft>0)&(soft<=.5)).astype(np.float32)
            data=dict(pg);data.update(support_m=ps['missing'],support_f=ps['false'],activation_m=pa['missing']*ps['missing'],activation_f=pa['false']*ps['false'],weak_soft_mask=weak,width=region)
            folder=OUT/'patches'/f'{pos+1:03d}';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'maps.npz',**data,lq=lq[j,:,:h,:w],gt=gt[j,:,:h,:w],output=pred[j,:,:h,:w])
            for k in full:full[k][y:y+h,x:x+w]+=data[k]
            for k,value in [('lq',lq[j]),('gt',gt[j]),('output',pred[j])]:images[k][y:y+h,x:x+w]+=image8(value)[:h,:w]
            coverage[y:y+h,x:x+w]+=1
            for kind in ['missing','false']:
                for b in range(3):
                    select=(bins==b)&(ps[kind]>0);v=stats[kind][str(b)];g=np.abs(pg[kind]);count=int(select.sum())
                    v['pixels']+=count;v['gradient_sum']+=float(g[select].sum());v['loss_sum']+=float(lm[kind][j,:h,:w][select].sum());v['activation_sum']+=float(pa[kind][select].sum());v['strong_pixels']+=int((pa[kind][select]>.5).sum());v['native_gradient']+=float(np.abs(pg['native'][select]).sum());v['total_gradient']+=float(np.abs(pg['total'][select]).sum());v['opposed_pixels']+=int(((pg[kind]*(pg['total']-pg[kind])<0)&select).sum())
            patches.append(dict(id=f'{pos+1:03d}',xy=[x,y],hw=[h,w],batch=offset//4,metadata=meta,contrast={k:float(v[j]) for k,v in batch.items() if k.startswith('contrast_')},folder=str(folder)))
            all_maps.extend(np.abs(data[k]).reshape(-1) for k in TERMS)
        save(OUT/'STATUS.json',dict(completed=min(offset+4,len(ids)),total=len(ids),seconds=time.time()-start));print(offset,flush=True)
        del result
    assert state_hash(model.net_g)==before and not model.optimizer_g.state
    assert np.all(coverage>0)
    for k in full:full[k]=(full[k]/coverage).astype(np.float32)
    for k in images:images[k]=np.rint(images[k]/coverage[...,None]).astype(np.uint8)
    nonzero=np.concatenate(all_maps);nonzero=nonzero[nonzero>0];limit=float(np.quantile(nonzero,.995))
    if WIDTH2:limit=json.loads((BASELINE_OUT/'REPORT.json').read_text())['shared_gradient_limit']
    np.savez_compressed(OUT/'full_maps.npz',**full,width=width,coverage=coverage)
    for kind,buckets in stats.items():
        count=sum(v['pixels'] for v in buckets.values());mass=sum(v['gradient_sum'] for v in buckets.values());loss=sum(v['loss_sum'] for v in buckets.values())
        for v in buckets.values():
            n=v['pixels'];v.update(area_share=n/max(count,1),gradient_share=v['gradient_sum']/max(mass,1e-30),loss_share=v['loss_sum']/max(loss,1e-30),gradient_per_pixel=v['gradient_sum']/n if n else None,activation_mean=v['activation_sum']/n if n else None,strong_fraction=v['strong_pixels']/n if n else None,opposed_fraction=v['opposed_pixels']/n if n else None,equal_error_expected_share=n/max(count,1))
    views=[]
    for record in [dict(id='000',xy=[0,0],folder=str(OUT/'full'))]+patches:
        folder=Path(record['folder']);folder.mkdir(exist_ok=True)
        if record['id']=='000':maps=full;rgb=images;ref=original;wid=width
        else:
            z=np.load(folder/'maps.npz');maps=z;rgb={k:image8(z[k]) for k in ['lq','gt','output']};x,y=record['xy'];h,w=record['hw'];ref=original[y:y+h,x:x+w];wid=z['width']
        outputs=dict(rgb,original_gt=ref)
        for k in TERMS:outputs[k+'_gradient']=overlay(rgb['output'],maps[k],limit)
        outputs['r1_direction']=overlay(rgb['output'],-maps['r1'],limit,signed=True)
        outputs['missing_activation']=overlay(rgb['output'],maps['activation_m'],1.)
        outputs['false_activation']=overlay(rgb['output'],maps['activation_f'],1.)
        outputs['support']=np.stack([maps['support_f']*255,maps['support_m']*255,maps['weak_soft_mask']*180],2).clip(0,255).astype(np.uint8)
        label=np.where(wid<=2.00001,0,np.where(wid<=4.00001,1,2));palette=np.array([[240,165,30],[50,180,60],[60,60,235]],np.uint8);wb=palette[label];valid=maps['support_m']>0;outputs['width_bins']=np.where(valid[...,None],wb,rgb['output'])
        paths={}
        for k,value in outputs.items():
            path=folder/(k+'.png');assert cv2.imwrite(str(path),value,[cv2.IMWRITE_PNG_COMPRESSION,3]);paths[k]=str(path.relative_to(OUT)).replace('\\','/')
        views.append(dict(id=record['id'],xy=record['xy'],width=ref.shape[1],height=ref.shape[0],images=paths))
    save(OUT/'REPORT.json',dict(passed=True,checkpoint=str(ckpt),checkpoint_sha256=digest(ckpt),config=str(config),config_sha256=digest(config),source=source,bank_shard=shard,model_state_unchanged=True,optimizer_empty=True,gradient_analytic_max_error=analytical_max,shared_gradient_limit=limit,statistics=stats,batches=batches,patches=patches,views=views,protocol=dict(unique_patches=25,batch_size=4,padded_exposures=3,statistics_exclude_padding=True,full_view='overlap mean fordisplay;quantification uses unique patch exposures',width='nearest skeleton,2xgrid diameter approximation',input_branch='realpaired V3LQ;syntheticreplacement bypassed',gt='actual S2.2 trainingtarget withperpatch contrastalignment',scope='outputY partial derivatives andparameter gradient norms;not Adam update orcausaltraining ablation')))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
