"""Accepted paired/mix50 trainer with MF-only,UV-only andnative-L1 mixture startup checks."""
from pathlib import Path
import sys,argparse,inspect,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def smoke(config,output):
    """@luna-comment Input:factorial full recipe. Output:9CUDAupdates andexactactive/inactive loss/path receipts. Key Steps:reuseacceptedupdatecheck;verifydeclaredMF/UV,allotherauxoff andtrain-onlymix. Key Params:frozenoldinit/freshAdam. State/Dependencies:ownedstartupmodelonly."""
    import tools.train_full_s22_strategy as original
    from tools.run_uniform20_steep_scurve import save
    source=inspect.getsource(original.smoke);old="assert 'contrast_applied' in logs and logs['input_paired_fraction']==1.";assert source.count(old)==1
    source=source.replace(old,"assert 'contrast_applied' in logs and 0 <= logs['input_paired_fraction'] <= 1.")
    assert source.count("    assert logs['uv_strong_scale']>0")==1
    source=source.replace("    assert logs['uv_strong_scale']>0","    assert logs['uv_strong_scale']==opt['train']['uv_strong_weight']")
    checks="""
    assert model.strategy in ['paired','mixed50'] and not hasattr(model,'s22_handle') and not model.calibration
    t=opt['train'];mf=t['recall_aware_weight_missing']>0;uv=t['uv_strong_weight']>0
    assert not(mf and uv) and t['bg_lf_weight']==0 and t['bg_lf_opt'] is None and logs['bg_lf_scale']==0 and logs.get('l_bg_lf',0)==0
    if mf:assert t['recall_aware_weight_missing']==1. and t['recall_aware_weight_false']==.25 and logs['l_missing']>0 and logs['l_false']>0
    else:assert t['recall_aware_weight_false']==0 and t['recall_aware_opt'] is None and logs.get('l_missing',0)==logs.get('l_false',0)==0
    if uv:assert t['uv_strong_weight']==1. and logs['l_uv_strong']>0
    else:assert t['uv_strong_opt'] is None and logs.get('l_uv_strong',0)==0
    if not mf and not uv:
        assert abs(logs['l_total']-logs['l_native'])<1e-7
        assert all(v==0 for k,v in logs.items() if k.endswith('_scale'))
    if model.strategy=='mixed50':
        assert not mf and not uv and 0<model.synthetic_count<model.total_draws==36
        target=batch['gt_rgb'].clone();raw=batch['lq_rgb'].clone();model.feed_data(batch);selected=~model.last_paired_selection
        assert torch.equal(batch['gt_rgb'],target) and torch.equal(batch['lq_rgb'],raw) and torch.equal(model.lq_rgb[~selected],raw.to(model.device)[~selected])
        valid=dict(batch);valid['s22_training_sample']=[False]*4;count=model.synthetic_count;model.feed_data(valid)
        assert model.synthetic_count==count and torch.equal(model.lq_rgb,raw.to(model.device))
    else:assert model.synthetic_count==0 and logs['input_paired_fraction']==1.
"""
    source=source.replace('    probe=None',checks+'\n    probe=None');namespace=dict(original.__dict__);exec(compile(source,'<aux_factorial_smoke>','exec'),namespace);namespace['smoke'](config,output)
    result=json.loads(Path(output).read_text());result.update(guard_absent=True,loss_switches_verified=True,mixture_validation_verified=True);save(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args()
    from tools.train_latest_vosr_mix50 import bootstrap
    bootstrap()
    if args.smoke_config:smoke(Path(args.smoke_config),Path(args.smoke_output))
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline;train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
