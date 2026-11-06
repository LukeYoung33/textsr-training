"""Matched heldout validation and separated missing/false no-update gradients."""
from pathlib import Path
import sys,json,copy
import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
from tools.diagnose_halo003_images import OUT
DATA=OUT.parent


def main():
    """@luna-comment Input:fourmatched500Kmodels,originalheldout583bank. Output:8parents/32patchvalidationmetrics,separatemissing/falsegradientprojections. Key Steps:actualdataset/nativecriterion,sharedGTpolarity/ring,inference4arms,firstorderhalo-directionaudit. Key Params:2parents/source,4baselinepatches/parent,zerooptimizersteps;teacherGTreference. State/Dependencies:immutablecheckpoints/validationdata;diagnosticoutputs only."""
    import yaml
    from tools.train_latest_vosr_500k import bootstrap
    bootstrap()
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from basicsr.data import build_dataset
    from basicsr.archs import build_network
    from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    from realesrgan.losses.recall_aware_structure_loss import recall_aware_structure_losses,_luma
    from tools.probe_halo003_model_loss import net_hash
    torch.set_num_threads(2);cv2.setNumThreads(2)
    frozen=DATA/'v3m_five_controls_mojito32_20261104';old=DATA/'latest_six_500k_mojito280_20261103'
    arms={}
    for name,folder in [('v3m_l1',old),('v3m_mf_only',frozen),('v3m_uv_only',frozen),('v3m_mf_uv',frozen)]:
        arms[name]=json.loads((folder/'MODEL_BINDINGS.json').read_text())[name]
    cfg=yaml.safe_load(Path(arms['v3m_mf_only']['config']).read_text());cfg['name']='halo_false_validation_probe'
    cfg['path'].update(pretrain_network_g=arms['v3m_mf_only']['checkpoint'],resume_state=None,experiments_root=str(OUT/'unused_validation_experiments'))
    cfg['s22_strategy']['audit_root']=str(OUT/'unused_validation_audit')
    path=OUT/'validation_probe.yml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));sys.argv=[sys.argv[0],'-opt',str(path)]
    opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True);dataset=build_dataset(opt['datasets']['val']);model=build_model(opt)
    assert not opt['datasets']['val'].get('patch_contrast_alignment') and not model.optimizer_g.state
    networks={};hashes={}
    for arm,b in arms.items():
        assert digest(b['checkpoint'])==b['checkpoint_sha256'] and digest(b['config'])==b['config_sha256']
        conf=yaml.safe_load(Path(b['config']).read_text());net=build_network(conf['network_g']).cuda().eval();net.load_state_dict(torch.load(b['checkpoint'],map_location='cpu')['params'],strict=True)
        networks[arm]=net;hashes[arm]=net_hash(net)
    parents=[]
    for group in ['HXA','HXB','lenovo0926','lenovo1011']:
        members=sorted([(i,s) for i,s in enumerate(dataset.shards) if s['dataset']==group],key=lambda x:x[1]['sample_id'])
        for fraction in [.25,.75]:parents.append(members[round((len(members)-1)*fraction)])
    results=[]
    for parent_index,(si,shard) in enumerate(parents):
        meta=json.loads((dataset.root/shard['metadata']).read_text());eligible=[i for i,m in enumerate(meta) if m['category']=='baseline']
        array=np.load(dataset.root/shard['array'],mmap_mode='r')
        candidates=[]
        for i in eligible:
            ratio=float((array[i,...,6]>127).mean())
            if .015<ratio<.6:candidates.append(i)
        assert len(candidates)>=4,(shard['sample_id'],len(candidates))
        ids=[candidates[int(i)] for i in np.linspace(0,len(candidates)-1,4,dtype=int)]
        dataset.index=np.asarray([(si,i) for i in ids],np.uint32);batch=dataset.collate_fn([dataset[i] for i in range(4)]);model.feed_data(batch)
        text,valid=model._structure_masks();gt=model.gt_rgb;record=dict(sample=shard['sample_id'],dataset=shard['dataset'],patch_ids=ids,models={})
        for arm,net in networks.items():
            packed=net(model.lq);rgb=YUV2RGB(y4uv_to_yuv444(packed),format='RGB',is_train=True,uv_normalized=True)
            missing,false,details=recall_aware_structure_losses(rgb,gt,text,valid,**cfg['train']['recall_aware_opt'])
            ring=details['ring']*details['active'];inside=(text[:,:1]>.5).float()*(valid[:,:1]>.5).float()*details['active']
            e=details['polarity']*(_luma(rgb.clamp(0,1))-_luma(gt));mass=ring.sum().clamp_min(1)
            metric=dict(ring_pixels=int(ring.sum()),halo_gt_above2dn_fraction=float(((e>2/255)*ring).sum()/mass),
                halo_gt_excess_dn=float((F.relu(e-2/255)*ring).sum()/mass*255),
                ring_signed_error_dn=float((e*ring).sum()/mass*255),ring_absolute_error_dn=float((e.abs()*ring).sum()/mass*255),
                foreground_opposite_error_dn=float((-e*inside).sum()/inside.sum().clamp_min(1)*255))
            record['models'][arm]=metric
            if arm=='v3m_mf_only':
                native=model.cri_y4uv_native(packed,model.gt,valid_mask=model.valid_mask,text_mask=model.text_mask)
                halo=(F.relu(details['polarity']*(_luma(rgb)-_luma(gt))-2/255)*ring).sum()/mass
                parameters=tuple(p for p in net.parameters() if p.requires_grad)
                gh=torch.autograd.grad(halo,parameters,retain_graph=True,allow_unused=True)
                h=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(gh,parameters)]).double()
                effects={}
                for name,term in [('native',native),('missing',missing),('false',.25*false)]:
                    g=torch.autograd.grad(term,(rgb,)+parameters,retain_graph=True,allow_unused=True)
                    v=torch.cat([(a if a is not None else torch.zeros_like(p)).detach().flatten() for a,p in zip(g[1:],parameters)]).double()
                    # Positive projection means ordinary gradient descent increases the halo score.
                    projection=float(-torch.dot(h,v));den=float(h.norm()*v.norm())
                    gy=g[0].sum(1,keepdim=True) if g[0] is not None else torch.zeros_like(ring)
                    effects[name]=dict(loss=float(term),parameter_norm=float(v.norm()),halo_direction_projection=projection,
                        halo_direction_cosine=projection/den if den>0 else None,
                        direct_ring_gradient_abs=float((gy.abs()*ring).sum()) if g[0] is not None else None)
                record['gradient_effects']=effects
                record['halo_score_norm']=float(h.norm())
            if parent_index<8:
                folder=OUT/'validation_crops'/str(parent_index+1);folder.mkdir(parents=True,exist_ok=True)
                for j in range(4):
                    im=np.rint(rgb[j].detach().clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
                    cv2.imwrite(str(folder/(arm+f'_p{j}.png')),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        if parent_index<8:
            for label,value in [('input',model.lq_rgb),('target',gt)]:
                for j in range(4):
                    im=np.rint(value[j].detach().clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
                    cv2.imwrite(str(OUT/'validation_crops'/str(parent_index+1)/(label+f'_p{j}.png')),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        results.append(record);print('VALIDATION_PARENT',parent_index+1,flush=True)
    assert not model.optimizer_g.state and all(net_hash(net)==hashes[arm] for arm,net in networks.items())
    summary={}
    for base,added in [('v3m_l1','v3m_mf_only'),('v3m_uv_only','v3m_mf_uv')]:
        diff=[r['models'][added]['halo_gt_excess_dn']-r['models'][base]['halo_gt_excess_dn'] for r in results]
        summary[base+'__'+added]=dict(parents=8,increased_parents=sum(x>0 for x in diff),mean_halo_gt_excess_change_dn=float(np.mean(diff)),per_parent=diff)
    save(OUT/'VALIDATION_FALSE_PROBE.json',dict(passed=True,parents=8,patches=32,heldout=True,optimizer_steps=0,
        reference='historical heldout validation GT;not newly dilated targets',summary=summary,records=results,
        gradient_contract='positive projection: ordinary gradient descent increases current GT-referenced halo score;not actual Adam trajectory'))


if __name__=='__main__':main()
