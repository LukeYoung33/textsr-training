"""Isolated mild-GT mixture; unchanged Full auxiliary losses without edge guard."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def mild(gt,rng):
    """@luna-comment Input:detached BCHW RGB target and CPU RNG. Output:synthetic RGB and parameters. Key Steps:Gaussian,small down/up,noise. Key Params:sigma0.3..0.9,scale0.8..1,noise0..2/255. State/Dependencies:private RNG;target immutable."""
    import torch
    import torch.nn.functional as F
    from tools.train_v3_edge_pilot import gaussian
    sigma=.3+.6*float(torch.rand((),generator=rng));scale=.8+.2*float(torch.rand((),generator=rng));noise=2*float(torch.rand((),generator=rng))/255
    value=torch.cat([gaussian(gt[:,i:i+1],sigma) for i in range(3)],1)
    h,w=gt.shape[-2:];small=(max(2,round(h*scale)),max(2,round(w*scale)))
    value=F.interpolate(F.interpolate(value,size=small,mode='bilinear',align_corners=False),size=(h,w),mode='bilinear',align_corners=False)
    value=value+torch.randn(value.shape,generator=rng).to(value)*noise
    return value.clamp(0,1).detach(),dict(sigma=sigma,scale=scale,noise_sigma_255=noise*255)


def bootstrap():
    """@luna-comment Input:owned runtime. Output:mix50 registered model. Key Steps:reuse scheduler and paired Full;replace selected LQ after target alignment. Key Params:train-only Bernoulli0.5. State/Dependencies:original classes/processes unchanged."""
    from tools.train_latest_vosr_500k import bootstrap as base
    base()
    import torch
    from basicsr.utils.registry import MODEL_REGISTRY
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    parent=MODEL_REGISTRY.get('S22Model')

    @MODEL_REGISTRY.register()
    class LatestVOSRMix50(parent):
        def feed_data(self,data):
            """@luna-comment Input:paired batch. Output:half-probability synthetic LQ,unchanged targets/masks. Key Steps:parent feed then private draw and RGB repack. Key Params:validation untouched. State/Dependencies:model counters/RNG only."""
            super().feed_data(data)
            assert self.strategy=='mixed50' and not hasattr(self,'s22_handle')
            if not all(data.get('s22_training_sample',[False])):return
            selected=torch.rand(len(self.lq),generator=self.mix_rng)<.5
            self.lq_rgb=self.lq_rgb.clone();self.mixture_parameters=[]
            for i,yes in enumerate(selected):
                if yes:
                    self.lq_rgb[i:i+1],params=mild(self.gt_rgb[i:i+1].detach(),self.mix_rng);self.mixture_parameters.append(params)
            self.lq=rgb_to_y4uv(self.lq_rgb);self.last_paired_selection=~selected
            self.synthetic_count+=int(selected.sum())


def smoke(config,output):
    """@luna-comment Input:complete training recipe. Output:CUDA9updates plus exact path/validation checks. Key Steps:reuse full smoke;assert mixture/aux/no guard;verify feed invariants separately. Key Params:no optimizer checkpoint export. State/Dependencies:owned smoke artifacts."""
    import inspect,torch,numpy as np
    import tools.train_full_s22_strategy as original
    from tools.run_uniform20_steep_scurve import save
    source=inspect.getsource(original.smoke)
    old="assert 'contrast_applied' in logs and logs['input_paired_fraction']==1."
    assert source.count(old)==1
    source=source.replace(old,"assert 'contrast_applied' in logs and 0 <= logs['input_paired_fraction'] <= 1.")
    insert="""
    assert model.strategy=='mixed50' and not hasattr(model,'s22_handle') and not model.calibration
    assert 0 < model.synthetic_count < model.total_draws == 36
    assert logs['bg_lf_scale']>0 and logs['uv_strong_scale']>0
    gt=model.gt.clone();raw=batch['lq_rgb'].clone();target=batch['gt_rgb'].clone()
    model.feed_data(batch)
    selected=~model.last_paired_selection
    assert torch.equal(model.gt,gt) and torch.equal(batch['gt_rgb'],target) and torch.equal(batch['lq_rgb'],raw)
    assert torch.equal(model.lq_rgb[~selected],raw.to(model.device)[~selected])
    valid=dict(batch);valid['s22_training_sample']=[False]*4
    count=model.synthetic_count;model.feed_data(valid)
    assert model.synthetic_count==count and torch.equal(model.lq_rgb,raw.to(model.device))
"""
    source=source.replace('    probe=None',insert+'\n    probe=None')
    namespace=dict(original.__dict__);exec(compile(source,'<mix50_smoke>','exec'),namespace);namespace['smoke'](config,output)
    result=json.loads(Path(output).read_text());result.update(mixture_path_verified=True,validation_unchanged=True,guard_absent=True,auxiliary_losses_active=True);save(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--smoke-config');parser.add_argument('--smoke-output');args,rest=parser.parse_known_args();bootstrap()
    if args.smoke_config:smoke(Path(args.smoke_config),Path(args.smoke_output))
    else:
        sys.argv=[sys.argv[0],*rest];from basicsr.train import train_pipeline;train_pipeline('/home/guang.yang/workspace/Real-ESRGAN')
