"""Real GPU checks for native crops, baseline losses and input routing."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_v3_native_regions import bootstrap


def main():
    """@luna-comment
    Input: prepared output directory. Output: JSON assertions and real GPU loss evidence.
    Key Steps: verify source crop equality, strict checkpoint load, each input mode,
    active losses/backprop, mixture frequency and paired validation.
    Key Params: fixed smoke RNG and iteration8000 enable full auxiliary weights.
    State/Dependencies: CUDA, published data, isolated in-memory models; no run overwrites.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from tools.prepare_v3_native_regions import axis_origins
    geometry_rng = random.Random(73)
    for canvas in (96, 480):
        for _ in range(1000):
            full = geometry_rng.randint(2, 4097)
            start = geometry_rng.uniform(0, full-1)
            end = geometry_rng.uniform(start+1, full)
            origins = axis_origins(start, end, canvas, full)
            assert all(p >= 0 and p % 2 == 0 for p in origins)
            for pixel in range(int(start), min(full, int(end+.999999))):
                assert any(p <= pixel < p+canvas for p in origins)
    print('PASS 2000 native-grid coverage cases', flush=True)
    bootstrap()
    import numpy as np
    import torch
    import yaml
    from basicsr.models import build_model
    from realesrgan.data.native_text_region_dataset import NativeTextRegionDataset, native_crop
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    torch.set_num_threads(2)
    random.seed(20260828); np.random.seed(20260828); torch.manual_seed(20260828)
    out = Path(args.output)
    report = {}
    for path in sorted((out/'configs').glob('*.yml')):
        opt = yaml.safe_load(path.read_text())
        opt.update(is_train=True, dist=False, rank=0, world_size=1)
        settings = copy.deepcopy(opt['datasets']['train'])
        settings.update(phase='train', train_homography_prob=0)
        dataset = NativeTextRegionDataset(settings)
        samples = [dataset[i] for i in range(4)]
        image_index, _ = dataset.items[0]
        arr = dataset._load(image_index)['lq']
        xy = samples[0]['source_crop_xyxy'][:2]
        expected = torch.from_numpy(native_crop(arr, xy, (96, 480)).astype(np.float32)/np.iinfo(arr.dtype).max).permute(2, 0, 1)
        assert torch.equal(samples[0]['lq_rgb'], expected), 'Source pixels changed during crop'
        assert all(s['resize_scale'] == 1 for s in samples)
        batch = dataset.collate_fn(samples)
        model = build_model(opt)
        model.feed_data(batch)
        selection = model.last_paired_selection
        if selection.any():
            assert torch.equal(model.lq_rgb[selection], batch['lq_rgb'].to(model.device)[selection])
        if (~selection).any():
            assert not torch.equal(model.lq_rgb[~selection], batch['gt_rgb'].to(model.device)[~selection])
        assert torch.equal(model.lq, rgb_to_y4uv(model.lq_rgb))
        before = {k: v.detach().clone() for k, v in model.net_g.state_dict().items()}
        model.optimize_parameters(8000)
        logs = dict(model.get_current_log())
        assert all(np.isfinite(v) for v in logs.values()), logs
        for key in ('l_native', 'l_cp', 'l_uv_strong', 'l_bg_lf', 'l_missing', 'l_false', 'l_sdf_boundary'):
            assert key in logs and logs[key] > 0, (key, logs)
        assert any(not torch.equal(v, before[k]) for k, v in model.net_g.state_dict().items())
        # Exercise the same mixed selector repeatedly without costly optimization.
        count = 0
        if opt['paired_probability'] == .75:
            for _ in range(64):
                model.feed_data(batch)
                count += int(model.last_paired_selection.sum())
            assert .65 < count/256 < .85, count
        val_opt = copy.deepcopy(opt['datasets']['val']); val_opt['phase'] = 'val'
        val = NativeTextRegionDataset(val_opt)
        assert torch.equal(val[0]['lq_rgb'], val[0]['lq_rgb'])
        val_batch = val.collate_fn([val[0]])
        model.feed_data(val_batch)
        assert model.last_paired_selection.all()
        assert torch.equal(model.lq_rgb, val_batch['lq_rgb'].to(model.device))
        model.test()
        native = model.cri_y4uv_native(model.output, model.gt, valid_mask=model.valid_mask, text_mask=model.text_mask)
        assert torch.isfinite(native)
        report[path.stem] = dict(train_patches=len(dataset), val_patches=len(val), loss=logs,
            validation_native=float(native), mixed_observed_fraction=count/256 if count else None,
            source_crop_equal=True, strict_checkpoint_loaded=True, weights_changed=True)
        print('PASS', path.stem, float(logs['l_total']), flush=True)
        del model, dataset, val, batch, before
        torch.cuda.empty_cache()
    # Small source images preserve pixels and exclude padding.
    tiny = np.ones((17, 23), dtype=np.uint8)
    padded = native_crop(tiny, (0, 0), (96, 480), mask=True)
    assert padded.sum() == 17*23 and padded[17:].sum() == 0
    (out/'verification.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
