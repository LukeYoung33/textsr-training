"""Reuse accepted paired/mix50 training with isolated three-arm startup assertions."""
from pathlib import Path
import argparse,inspect,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def smoke(config,output):
    """@luna-comment Input:full orselective paired/mixed recipe. Output:9realCUDAupdates withobjective/path checks. Key Steps:reuse accepted smoke andaddconfig-bound BG/MF/UV/guard/mixture assertions. Key Params:allauxweights matched priorruns. State/Dependencies:ownedsmoke only,productiontraining unchanged."""
    import tools.train_full_s22_strategy as original
    from tools.run_uniform20_steep_scurve import save
    source=inspect.getsource(original.smoke)
    old="assert 'contrast_applied' in logs and logs['input_paired_fraction']==1."
    assert source.count(old)==1
    source=source.replace(old,"assert 'contrast_applied' in logs and 0 <= logs['input_paired_fraction'] <= 1.")
    checks="""
    assert model.strategy in ['paired','mixed50'] and not hasattr(model,'s22_handle') and not model.calibration
    assert opt['train']['recall_aware_weight_missing']==1. and opt['train']['recall_aware_weight_false']==.25
    assert logs['l_missing']>0 and logs['l_false']>0 and logs['uv_strong_scale']==1.
    expected_bg=opt['train']['bg_lf_weight'];assert expected_bg in [0.,2.]
    assert logs['bg_lf_scale']==expected_bg
    if expected_bg==0:assert opt['train']['bg_lf_opt'] is None and logs.get('l_bg_lf',0)==0
    else:assert logs['l_bg_lf']>0
    if model.strategy=='mixed50':
        assert 0 < model.synthetic_count < model.total_draws == 36
        target=batch['gt_rgb'].clone();raw=batch['lq_rgb'].clone();model.feed_data(batch)
        selected=~model.last_paired_selection
        assert torch.equal(batch['gt_rgb'],target) and torch.equal(batch['lq_rgb'],raw)
        assert torch.equal(model.lq_rgb[~selected],raw.to(model.device)[~selected])
        validation=dict(batch);validation['s22_training_sample']=[False]*4;count=model.synthetic_count;model.feed_data(validation)
        assert model.synthetic_count==count and torch.equal(model.lq_rgb,raw.to(model.device))
    else:assert model.synthetic_count==0 and logs['input_paired_fraction']==1.
"""
    source=source.replace('    probe=None',checks+'\n    probe=None')
    namespace=dict(original.__dict__);exec(compile(source,'<v3m_ablation_smoke>','exec'),namespace);namespace['smoke'](config,output)
    result=json.loads(Path(output).read_text());result.update(guard_absent=True,missing_false_uv_active=True,bg_config_verified=True,mixture_or_paired_path_verified=True);save(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args()
    from tools.train_latest_vosr_mix50 import bootstrap
    bootstrap()
    if args.smoke_config:smoke(Path(args.smoke_config),Path(args.smoke_output))
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline;train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
