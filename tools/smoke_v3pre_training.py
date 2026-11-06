"""Correctness, gradients and actual CUDA full-loss checks before scheduled training."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_loss():
    """@luna-comment Input:registered adaptive class. Output:numerical/gradient evidence.
    Key Steps:coverage/empty/padding/cap1 reference and gradient-ratio cases plus
    CUDA cap1/cap4 throughput check. Key Params:fixed masks/residuals; no network.
    State/Dependencies:torch CUDA; full training path checked separately.
    """
    import torch
    from realesrgan.models.v3pre_area_adapters import AreaAdaptiveNativeL1
    from realesrgan.data.basic_text_line_y4uv_v3 import yuv420_mask_to_y4uv
    checks = []
    for fraction in [0, .1, .2, .3, .5, 1]:
        region = torch.zeros(1, 1, 20, 20, device='cuda')
        region.flatten()[:int(400*fraction)] = 1
        valid = torch.ones_like(region)
        pred = torch.full((1, 6, 10, 10), .6, device='cuda', requires_grad=True)
        target = torch.full_like(pred, .2)
        loss = AreaAdaptiveNativeL1(cap=4)
        loss.raw_valid, loss.line_region = valid, region
        value = loss(pred, target)
        torch.testing.assert_close(value, value.new_tensor(.4))
        value.backward()
        if 0 < fraction < 1:
            packed = yuv420_mask_to_y4uv(region)
            inside = pred.grad[:, :4][packed[:, :4] == 1].mean()
            outside = pred.grad[:, :4][packed[:, :4] == 0].mean()
            torch.testing.assert_close(inside/outside, loss.stats['area_a'])
        control = AreaAdaptiveNativeL1(cap=1)
        control.raw_valid, control.line_region = valid, region
        pred2 = pred.detach().clone().requires_grad_()
        actual = control(pred2, target)
        c = pred.new_tensor([1,1,1,1,.25,.25])[None,:,None,None]
        reference = (((pred2-target).abs()*c).sum((1,2,3))/(c.sum()*100)).mean()
        torch.testing.assert_close(actual, reference)
        g1 = torch.autograd.grad(actual, pred2, retain_graph=True)[0]
        g2 = torch.autograd.grad(reference, pred2)[0]
        torch.testing.assert_close(g1, g2)
        loss.raw_valid = torch.zeros_like(valid)
        assert float(loss(pred, target)) == 0
        checks.append(fraction)
    # Mixed valid sizes and explicit padding invariance are independent of region coverage.
    pred = torch.rand(2, 6, 8, 8, device='cuda', requires_grad=True)
    valid = torch.ones(2, 1, 16, 16, device='cuda')
    valid[1, :, 8:] = 0
    region = torch.zeros_like(valid)
    region[:, :, :4] = 1
    module = AreaAdaptiveNativeL1()
    module.raw_valid, module.line_region = valid, region
    value = module(pred, torch.zeros_like(pred))
    padded_pred = torch.nn.functional.pad(pred, (0, 4, 0, 4))
    module.raw_valid = torch.nn.functional.pad(valid, (0, 8, 0, 8))
    module.line_region = torch.nn.functional.pad(region, (0, 8, 0, 8))
    torch.testing.assert_close(module(padded_pred, torch.zeros_like(padded_pred)), value)
    return dict(passed=True, coverages=checks, cap1_value_gradient_equal=True,
                adaptive_gradient_ratio=True, empty_valid_zero=True, mixed_padding_invariant=True)


def smoke(config, output):
    """@luna-comment Input:isolated config/pilot bank. Output:strict CUDA/auxiliary/update/throughput receipt.
    Key Steps:actual registered dataset/model, selected CP/border/background/category
    samples, iter0/8000 updates, compare auxiliary losses for identical tensors,
    benchmark cap1/cap4 same full model. Key Params:no saved pilot weights.
    State/Dependencies:complete pilot bank and actual CRNN; scheduler fails closed.
    """
    import copy
    import numpy as np
    import torch
    from tools.prepare_v3pre_training import sha, save
    from tools.train_v3pre_area import bootstrap
    repository = '/home/guang.yang/workspace/Real-ESRGAN'
    bootstrap(repository)
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.models.v3pre_area_adapters import V3PreAreaDataset
    from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
    sys.argv = [sys.argv[0], '-opt', str(config)]
    opt, _ = parse_options(repository, is_train=True)
    opt.update(dist=False, rank=0, world_size=1)
    adaptive = opt['model_type'] == 'V3PreAreaModel'
    dataset = (V3PreAreaDataset if adaptive else OfflineAugmentedRegionDataset)(opt['datasets']['train'])
    refs = {}
    for si, shard in enumerate(dataset.shards):
        for ri, meta in enumerate(json.loads((dataset.root / shard['metadata']).read_text())):
            for key, yes in [('cp', bool(meta['cp_lines'])), ('background', meta['background_only']),
                             ('border', meta['real_hw'] != [256, 256]), ('oblique', meta['orientation'] == 'OBLIQUE')]:
                if yes:
                    refs.setdefault(key, (si, ri))
        if len(refs) == 4:
            break
    assert 'cp' in refs
    chosen = list(refs.values())
    while len(chosen) < 4:
        chosen.append(chosen[0])
    dataset.index = np.asarray(chosen[:4], np.uint32)
    batch = dataset.collate_fn([dataset[i] for i in range(4)])
    model = build_model(opt)
    before = {k: v.detach().clone() for k, v in model.net_g.named_parameters()}
    for iteration in [0, 8000]:
        model.feed_data(batch)
        model.optimize_parameters(iteration)
    logs = {k: float(v) for k, v in model.get_current_log().items()}
    assert all(np.isfinite(v) for v in logs.values())
    assert any(not torch.equal(v, before[k]) for k, v in model.net_g.named_parameters())
    assert logs['cp_accepted_lines'] > 0 and logs['cp_scale'] > 0 and 'l_sdf_boundary' in logs
    details = dict(passed=True, config_sha256=sha(config), updates=2, losses=logs,
                   strict_load=True, selected_sample_classes=list(refs), real_model_branch=opt['model_type'])
    if adaptive:
        details['numerical_gradient_checks'] = check_loss()
        assert 'area_a' in logs and logs['area_a'] >= 1
        # Alternate strength on the same model and batch. No data-loader difference.
        timings = {1: [], 4: []}
        frozen_net = copy.deepcopy(model.net_g.state_dict())
        frozen_optimizer = copy.deepcopy(model.optimizer_g.state_dict())
        matched_logs = {}
        for cycle in range(6):
            for cap in ((1, 4) if cycle % 2 == 0 else (4, 1)):
                model.cri_y4uv_native.cap = cap
                for step in range(6):
                    # Restore outside timing: both arms see identical predictions,
                    # auxiliary branches and optimizer state on every measured step.
                    model.net_g.load_state_dict(frozen_net)
                    model.optimizer_g.load_state_dict(frozen_optimizer)
                    model.feed_data(batch)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    model.optimize_parameters(8000)
                    torch.cuda.synchronize()
                    if step > 1:
                        timings[cap].append(time.perf_counter()-started)
                matched_logs[cap] = {k:float(v) for k,v in model.get_current_log().items()}
        ignored = {'l_native', 'l_selected', 'l_total', 'area_a', 'area_r', 'area_unweighted_l1', 'area_weighted_l1'}
        for key in matched_logs[1]:
            if key not in ignored:
                assert abs(matched_logs[1][key]-matched_logs[4][key]) < 1e-6, key
        details['auxiliary_losses_equal_for_identical_prediction'] = True
        ratio = float(np.median(timings[4])/np.median(timings[1]))
        details['matched_cap_throughput'] = dict(ratio=ratio, median_seconds={k:float(np.median(v)) for k,v in timings.items()})
        assert ratio <= 1.05, details['matched_cap_throughput']
        model.cri_y4uv_native.cap = 4
    save(output, details)
    print(json.dumps(details), flush=True)


if __name__ == '__main__':
    smoke(Path(sys.argv[1]), Path(sys.argv[2]))
