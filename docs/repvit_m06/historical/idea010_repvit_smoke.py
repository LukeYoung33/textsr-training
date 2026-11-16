"""IDEA-010?paired???????????????????CUDA?????"""
from pathlib import Path
import copy
import sys
import torch
from tools.prepare_r1_width_bank import save

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
    errors = {name: float((value-restored.net_g.state_dict()[name]).abs().max()) for name,value in model.net_g.state_dict().items() if value.is_floating_point()}
    loss_error = abs(float(model.log_dict['l_total'])-float(restored.log_dict['l_total']))
    matched = all(torch.allclose(value, restored.net_g.state_dict()[name], rtol=1e-6, atol=1e-7) for name,value in model.net_g.state_dict().items() if value.is_floating_point())
    save(str(output)+'.resume.json',dict(max_abs=max(errors.values()),loss_abs=loss_error,by_tensor=errors,all_state_tensors_matched=matched,criterion='FP32 all state tensors rtol1e-6/atol1e-7; loss <=1e-7; CUDA bilinear backward is nondeterministic'))
    assert matched and loss_error <= 1e-7
    save(output,dict(passed=True,kind=kind,updates=9,strict_initial=True,fresh_optimizer=True,synthetic_inputs=0,contrast_off=True,lq_gt_unchanged=True,loss_composition_verified=True,next_update_resume=True,logs=logs))
