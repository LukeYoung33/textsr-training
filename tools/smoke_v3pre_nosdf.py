"""Actual CUDA scratch-model updates proving SDF is absent and other losses execute."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    """@luna-comment Input:isolated YAML/receipt CLI. Output:finite actual CUDA update and no-SDF evidence. Key Steps:registered dataset/model;eligible CP/border/background batch;updates0/8000;verify no SDF criterion/logs and fresh Adam. Key Params:smoke saves no weights. State/Dependencies:immutable V3 bank and existing bootstrap."""
    import numpy as np
    import torch
    from tools.train_v3pre_area import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.models.v3pre_area_adapters import V3PreAreaDataset
    from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
    from tools.train_v3_edge_pilot import save,digest
    config,output=map(Path,sys.argv[1:3]);sys.argv=[sys.argv[0],'-opt',str(config)]
    opt,_=parse_options('/home/guang.yang/workspace/Real-ESRGAN',is_train=True)
    torch.set_num_threads(2)
    adaptive=opt['model_type']=='V3PreAreaModel'
    dataset=(V3PreAreaDataset if adaptive else OfflineAugmentedRegionDataset)(opt['datasets']['train'])
    refs={}
    for si,shard in enumerate(dataset.shards):
        for ri,meta in enumerate(json.loads((dataset.root/shard['metadata']).read_text())):
            for key,yes in [('cp',bool(meta['cp_lines'])),('background',meta['background_only']),('border',meta['real_hw']!=[256,256])]:
                if yes:refs.setdefault(key,(si,ri))
        if len(refs)==3:break
    assert 'cp' in refs
    chosen=list(refs.values())
    while len(chosen)<4:chosen.append(refs['cp'])
    dataset.index=np.asarray(chosen[:4],np.uint32)
    batch=dataset.collate_fn([dataset[i] for i in range(4)])
    model=build_model(opt)
    assert model.sdf_boundary_weight==0 and model.cri_sdf_boundary is None
    assert not model.optimizer_g.state and opt['path']['resume_state'] is None
    loaded=torch.load(opt['path']['pretrain_network_g'],map_location='cpu')['params']
    assert all(torch.equal(value.detach().cpu(),loaded[key]) for key,value in model.net_g.state_dict().items())
    before={key:value.detach().clone() for key,value in model.net_g.named_parameters()}
    for iteration in [0,8000]:
        model.feed_data(batch);model.optimize_parameters(iteration)
    logs={key:float(value) for key,value in model.get_current_log().items()}
    assert all(np.isfinite(value) for value in logs.values())
    assert not any('sdf' in key for key in logs)
    assert logs['cp_accepted_lines']>0 and logs['cp_scale']>0 and logs['uv_strong_scale']>0
    assert any(not torch.equal(value,before[key]) for key,value in model.net_g.named_parameters())
    details=dict(passed=True,updates=2,sdf_module_absent=True,sdf_terms_absent=True,fresh_optimizer=True,initial_weights_exact=True,config_sha256=digest(config),losses=logs,sample_classes=list(refs))
    if adaptive:
        from tools.smoke_v3pre_training import check_loss
        details['adaptive_numerical_checks']=check_loss()
        assert logs['area_a']>=1
    save(output,details);print(json.dumps(details),flush=True)


if __name__=='__main__':main()
