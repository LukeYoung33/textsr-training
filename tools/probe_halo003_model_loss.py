"""No-update tile-shift and effective R1 training-gradient probes for halo003."""
from pathlib import Path
import sys
import json
import hashlib
import copy
import numpy as np
import cv2
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
from tools.diagnose_halo003_images import OUT,ROIS,luma,mask_of
DATA=OUT.parent


def vector(a,b):
    """@luna-comment Input:two detachedgradientvectors. Output:normratio/cosine. Key Steps:doubleprecisionnorm/dot. Key Params:zeronormnull. State/Dependencies:readonlytorchCPU."""
    a=a.double();b=b.double();na=float(a.norm());nb=float(b.norm())
    return dict(left_norm=na,right_norm=nb,ratio=na/max(nb,1e-30),cosine=float(torch.dot(a,b)/(na*nb)) if na*nb else None)


def net_hash(net):
    """@luna-comment Input:network. Output:stabletensorstatehash. Key Steps:orderednamesandCPUbytes. Key Params:buffersincluded. State/Dependencies:readonlystate."""
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tile_probe():
    """@luna-comment Input:frozen370KR1and003cachedinput/raw. Output:exactreplay/even112gridshiftmetricsandROIcrops. Key Steps:strictsameweights,acceptedrestore,shiftthenunpad,comparecommoninteriorresiduals. Key Params:diagnosticgridoriginonly;nooptimizernorweightupdates. State/Dependencies:CUDA/unchangedstitcher."""
    import yaml
    import realesrgan
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    folder=DATA/'v3m_r1_width_370k_mojito32_20261104';arm='v3m_dilate2_r1_width_370k'
    binding=json.loads((folder/'MODEL_BINDINGS.json').read_text())[arm]
    for k in ['config','checkpoint']:assert digest(binding[k])==binding[k+'_sha256']
    assert digest(ROOT/'tools/restore_crop8_block256.py')==binding['stitcher_sha256']
    e=next(s for s in json.loads((folder/'selection.json').read_text())['samples'] if s['id']=='003')
    source=cv2.cvtColor(cv2.imread(e['input']),cv2.COLOR_BGR2RGB);assert e['orientation']==1 and digest(e['input'])==e['input_sha256']
    network=build_network(yaml.safe_load(Path(binding['config']).read_text())['network_g']).cuda().eval()
    network.load_state_dict(torch.load(binding['checkpoint'],map_location='cpu')['params'],strict=True);before=net_hash(network)
    baseline,_=restore_block(source,network,'cuda');old=cv2.cvtColor(cv2.imread(str(folder/'results'/arm/'final/003.png')),cv2.COLOR_BGR2RGB)
    error=np.abs(baseline.astype(np.int16)-old.astype(np.int16));assert error.max()<=1
    padded=np.pad(source,((112,0),(112,0),(0,0)),mode='edge');shift,_=restore_block(padded,network,'cuda');shift=shift[112:,112:]
    records={}
    for name,(x0,y0,x1,y1) in ROIS.items():
        a=luma(baseline[y0:y1,x0:x1])-luma(source[y0:y1,x0:x1]);b=luma(shift[y0:y1,x0:x1])-luma(source[y0:y1,x0:x1])
        mask,threshold,eligible,skel,dist=mask_of(luma(source[y0:y1,x0:x1]));background=dist>4
        aa=a[background];bb=b[background]
        records[name]=dict(background_pixels=int(background.sum()),residual_corr=float(np.corrcoef(aa,bb)[0,1]),
            mean_abs_grid_difference_dn=float(np.abs(aa-bb).mean()),baseline_abs_residual_dn=float(np.abs(aa).mean()),
            shifted_abs_residual_dn=float(np.abs(bb).mean()))
        cv2.imwrite(str(OUT/'crops'/name/'r1_shift112.png'),cv2.cvtColor(shift[y0:y1,x0:x1],cv2.COLOR_RGB2BGR))
    assert net_hash(network)==before
    save(OUT/'TILE_PROBE.json',dict(passed=True,baseline_max_dn=int(error.max()),baseline_exact_fraction=float((error==0).mean()),
        shift_px=112,haar_phase_preserved=True,weights_unchanged=True,checkpoint_sha256=binding['checkpoint_sha256'],rois=records))
    del network;torch.cuda.empty_cache()


def gradient_probe():
    """@luna-comment Input:actualR1dataset/model/frozen370K andtraining028. Output:16actualpatchlossgradients,GTidentitydirection andmaskarea telemetry. Key Steps:realbootstrap/feed/contrast/native/R1;autogradonly,neverAdamstep. Key Params:B4x4,actualgroups/cap4,GTidentityisgradientprobe notproofcombinedoptimum. State/Dependencies:owneddiagnosticconfig/audit;sourceimmutable."""
    import yaml
    import torch.nn.functional as F
    from tools.train_r1_width_500k import width_r1
    from basicsr.utils.options import parse_options
    from basicsr.data import build_dataset
    from basicsr.models import build_model
    from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    folder=DATA/'v3m_r1_width_370k_mojito32_20261104';binding=next(iter(json.loads((folder/'MODEL_BINDINGS.json').read_text()).values()))
    cfg=yaml.safe_load(Path(binding['config']).read_text());cfg['name']='halo003_frozen_r1_probe'
    cfg['path'].update(pretrain_network_g=binding['checkpoint'],resume_state=None,experiments_root=str(OUT/'unused_experiments'))
    cfg['s22_strategy']['audit_root']=str(OUT/'unused_audit')
    path=OUT/'training_probe.yml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));sys.argv=[sys.argv[0],'-opt',str(path)]
    opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    dataset=build_dataset(opt['datasets']['train']);model=build_model(opt)
    model.net_g.eval();before=net_hash(model.net_g);assert not model.optimizer_g.state
    expected=torch.load(binding['checkpoint'],map_location='cpu')['params']
    assert set(expected)==set(model.net_g.state_dict()) and all(torch.equal(v.detach().cpu(),expected[k]) for k,v in model.net_g.state_dict().items())
    import inspect
    implementation=inspect.getsourcefile(width_r1)
    save(OUT/'IMPLEMENTATION_BINDING.json',dict(width_r1_file=implementation,width_r1_sha256=digest(implementation),
        strict_checkpoint_match=True,checkpoint_sha256=binding['checkpoint_sha256'],
        active_recipe={k:v for k,v in cfg['train'].items() if k.endswith('_weight') or k.startswith('recall_aware_weight')},
        loss_parameters=dict(ring_radius=3,margin=.03,temperature=.01,width_cap=4)))
    si,shard=next((i,s) for i,s in enumerate(dataset.shards) if s['sample_id']=='capture_261011_003171')
    metas=json.loads((dataset.root/shard['metadata']).read_text());eligible={int(r[1]) for r in dataset.index if int(r[0])==si}
    ids=[i for i,m in enumerate(metas) if m['category']=='baseline' and i in eligible]
    ids=[ids[int(i)] for i in np.linspace(0,len(ids)-1,16,dtype=int)];assert len(set(ids))==16
    dataset.index=np.asarray([(si,i) for i in ids],np.uint32)
    array=np.load(dataset.root/shard['array'],mmap_mode='r')
    params=tuple(p for p in model.net_g.parameters() if p.requires_grad);batches=[]
    for start in range(0,16,4):
        batch=dataset.collate_fn([dataset[i] for i in range(start,start+4)]);model.feed_data(batch)
        output=model._forward_v5(model.lq)
        native=model.cri_y4uv_native(output,model.gt,valid_mask=model.valid_mask,text_mask=model.text_mask)
        rgb=YUV2RGB(y4uv_to_yuv444(output),format='RGB',is_train=True,uv_normalized=True)
        text,valid=model._structure_masks();missing,false,stats=width_r1(rgb,model.gt_rgb,text,valid,model.width_groups)
        r1=missing+.25*false
        vectors={};pixel={}
        for name,loss in [('native',native),('r1',r1)]:
            gradients=torch.autograd.grad(loss,(output,)+params,retain_graph=True,allow_unused=True)
            pixel[name]=gradients[0].detach().cpu().flatten()
            vectors[name]=torch.cat([(g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu')).flatten() for g,p in zip(gradients[1:],params)])
        target=model.gt_rgb.detach().clone().requires_grad_(True)
        tm,tf,ts=width_r1(target,model.gt_rgb,text,valid,model.width_groups)
        equality_grad=torch.autograd.grad(tm+.25*tf,target)[0].sum(1,keepdim=True)
        mask=(text[:,:1]>.5)&(valid[:,:1]>.5)
        ring=(F.max_pool2d(mask.float(),7,1,3)>.5)&~mask&(valid[:,:1]>.5)
        gy=(model.gt_rgb*target.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        polarity=torch.where((gy*ring).sum((2,3),keepdim=True)/ring.sum((2,3),keepdim=True).clamp_min(1)-(gy*mask).sum((2,3),keepdim=True)/mask.sum((2,3),keepdim=True).clamp_min(1)>=0,1.,-1.)
        # Positive aligned descent means stroke enhancement inside, background brightening outside for dark text.
        descent=-equality_grad
        active_inside=(polarity*descent)[mask].detach().cpu().numpy()
        active_outside=(polarity*descent)[ring].detach().cpu().numpy()
        old_count=sum(int((array[ids[i],...,6]>127).sum()) for i in range(start,start+4))
        row=dict(patch_ids=ids[start:start+4],native=float(native),missing=float(missing),false=float(false),
            r1=float(r1),parameter_gradient=vector(vectors['r1'],vectors['native']),output_gradient=vector(pixel['r1'],pixel['native']),
            old_text_pixels=old_count,new_text_pixels=int(mask.sum()),
            gt_identity_r1=float(tm+.25*tf),gt_identity_gradient_norm=float(equality_grad.norm()),
            identity_stroke_enhancing_fraction=float((active_inside<0).mean()),
            identity_ring_opposite_polarity_fraction=float((active_outside>0).mean()),
            group_stats={k:float(v) for k,v in stats.items()})
        batches.append(row)
        if start==0:
            for j in range(4):
                for name,value in [('lq',model.lq_rgb),('gt',model.gt_rgb),('pred',rgb)]:
                    im=np.rint(np.clip(value[j].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
                    cv2.imwrite(str(OUT/f'train028_p{j}_{name}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        print('GRADIENT_BATCH',start//4,flush=True)
    assert not model.optimizer_g.state and net_hash(model.net_g)==before
    save(OUT/'R1_GRADIENT_PROBE.json',dict(passed=True,training_sample='capture_261011_003171',patches=16,
        checkpoint_iteration=370000,optimizer_steps=0,weights_unchanged=True,gt_identity_is_not_combined_optimum_proof=True,batches=batches))


def target_probe():
    """@luna-comment Input:training028 originalGT andactualwidth2target. Output:inputanchoredwidth/areachangesperinitialwidthclass. Key Steps:sourcehashgates,fixedhalfcontrastmask/EDT,comparetargetatoldskeleton. Key Params:literalpixelwidthdiagnosticnot003GT. State/Dependencies:immutabletrainingmanifest."""
    manifest=DATA/'original_aligned_gt_dilate2_mask_sync_20261103/manifests/train.jsonl'
    row=next(json.loads(line) for line in manifest.read_text().splitlines() if '"sample_id": "capture_261011_003171"' in line)
    original=Path(row['original_gt_lineage']['path']);target=Path(row['files']['gt'])
    assert digest(original)==row['original_gt_lineage']['sha256'] and digest(target)==row['output_sha256']['gt']
    a=cv2.cvtColor(cv2.imread(str(original)),cv2.COLOR_BGR2RGB);b=cv2.cvtColor(cv2.imread(str(target)),cv2.COLOR_BGR2RGB)
    mask,threshold,eligible,skel,distance=mask_of(luma(a));new=(luma(b)<threshold)&eligible
    before=2*__import__('scipy').ndimage.distance_transform_edt(mask)[skel]
    after=2*__import__('scipy').ndimage.distance_transform_edt(new)[skel]
    groups={}
    for name,lo,hi in [('thin',0,4),('medium',4,8),('thick',8,1000)]:
        valid=(before>lo)&(before<=hi)
        if valid.any():groups[name]=dict(points=int(valid.sum()),before_mean_px=float(before[valid].mean()),after_mean_px=float(after[valid].mean()),width_gain_mean_px=float((after-before)[valid].mean()),relative_gain=float(((after-before)/np.maximum(before,1e-6))[valid].mean()))
    save(OUT/'TARGET_WIDTH_PROBE.json',dict(passed=True,sample=row['sample_id'],original=str(original),target=str(target),groups=groups))


def matched_iteration_probe():
    """@luna-comment Input:existing350Kcheckpointsforbothdilate2L1/R1. Output:003no-updatewidth/ringcomparisonatmatchediteration. Key Steps:strictloadsharednetworkconfig,sameacceptedrestore,fixedinputmaskmetrics. Key Params:350Kboth;mask/lossstilljointlydifferent. State/Dependencies:readonlysourcecheckpoints;ownednewcrops."""
    import yaml
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    selection=json.loads((DATA/'v3m_r1_width_370k_mojito32_20261104/selection.json').read_text())['samples']
    entry=next(s for s in selection if s['id']=='003');source=cv2.cvtColor(cv2.imread(entry['input']),cv2.COLOR_BGR2RGB)
    records={}
    for label,folder,config in [('l1_350k','v3m_original_dilate2_l1_500k_20261103','configs/v3m_dilate2_l1.yml'),('r1_350k','v3m_dilate2_l1_r1_width_500k_20261103','launch.yml')]:
        root=DATA/folder;cfg=yaml.safe_load((root/config).read_text());checkpoint=root/'experiments'/cfg['name']/'models/net_g_350000.pth';before=digest(checkpoint)
        net=build_network(cfg['network_g']).cuda().eval();net.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
        result,_=restore_block(source,net,'cuda');record=dict(checkpoint=str(checkpoint),sha256=before,rois={})
        for name,(x0,y0,x1,y1) in ROIS.items():
            a=luma(source[y0:y1,x0:x1]);b=luma(result[y0:y1,x0:x1]);mask,threshold,eligible,skel,distance=mask_of(a)
            if name=='blank':continue
            new=(b<threshold)&eligible;wa=2*__import__('scipy').ndimage.distance_transform_edt(mask)[skel];wb=2*__import__('scipy').ndimage.distance_transform_edt(new)[skel]
            record['rois'][name]=dict(area_ratio=float(new.sum()/max(mask.sum(),1)),width_change_mean_px=float((wb-wa).mean()),
                rings=[dict(lo=lo,hi=hi,mean_dn=float((b-a)[(distance>lo)&(distance<=hi)].mean())) for lo,hi in [(0,2),(2,4),(4,8),(8,16),(16,32)]])
            cv2.imwrite(str(OUT/'crops'/name/(label+'.png')),cv2.cvtColor(result[y0:y1,x0:x1],cv2.COLOR_RGB2BGR))
        assert digest(checkpoint)==before;records[label]=record
        del net;torch.cuda.empty_cache()
    save(OUT/'MATCHED_350K.json',dict(passed=True,iteration=350000,sample='003',no_optimizer_steps=True,records=records))


if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(2)
    from tools.train_r1_width_500k import bootstrap
    bootstrap()
    tile_probe();gradient_probe();target_probe()
    save(OUT/'PROBE_COMPLETE.json',dict(passed=True,no_optimizer_steps=True))
