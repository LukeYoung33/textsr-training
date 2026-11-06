"""Frozen latest complete R1-width370K checkpoint on the established Mojito32."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_five_controls_mojito32 as core
from tools.run_uniform20_steep_scurve import save, digest

SOURCE = core.DATA/'v3m_dilate2_l1_r1_width_500k_20261103'
ARM = 'v3m_dilate2_r1_width_370k'
core.OUT = core.DATA/'v3m_r1_width_370k_mojito32_20261104'
core.ARMS = {ARM: 'V3M · 原始GT+2px/新mask · L1＋粗细R1 · 370K'}
core.__file__ = __file__


def prepare():
    """@luna-comment Input:explicitrunningR1experiment frozen370K/config. Output:immutableevaluationbinding/32selection. Key Steps:copyhashstrictcheckpoint/config,comparebaselinearchitecture,bindactualiteration. Key Params:370000only;traininguntouched. State/Dependencies:acceptedgenericrunner;sourcefilesreadonly."""
    import yaml
    out = core.OUT
    config = SOURCE/'launch.yml'
    cfg = yaml.safe_load(config.read_text())
    checkpoint = SOURCE/'experiments'/cfg['name']/'models/net_g_370000.pth'
    before = digest(checkpoint)
    config_hash = digest(config)
    (out/'models').mkdir(exist_ok=True)
    target = out/'models'/(ARM+'.pth')
    conf = out/'models'/(ARM+'.yml')
    shutil.copy2(checkpoint, target)
    shutil.copy2(config, conf)
    assert digest(target) == before == digest(checkpoint)
    assert digest(conf) == config_hash == digest(config)
    binding = dict(config=str(conf), config_sha256=config_hash, checkpoint=str(target),
        checkpoint_sha256=before, source_checkpoint=str(checkpoint), iteration=370000,
        label=core.ARMS[ARM], stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    shutil.copy2(core.DATA/'mojito280_subset32_20261104/selection.json', out/'selection.json')
    baseline = core.DATA/'v3m_original_dilate2_l1_500k_20261103/configs/v3m_dilate2_l1.yml'
    old = yaml.safe_load(baseline.read_text())
    architecture_differences = {k: dict(baseline=old['network_g'].get(k), r1=cfg['network_g'].get(k))
        for k in set(old['network_g']) | set(cfg['network_g']) if old['network_g'].get(k) != cfg['network_g'].get(k)}
    save(out/'MODEL_BINDINGS.json', {ARM: binding})
    save(out/'RUN_BINDING.json', dict(selection_sha256=digest(out/'selection.json'),
        post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'), outputs=64,
        iteration=370000, baseline_iteration=500000, architecture_differences=architecture_differences,
        training_differences=['synchronized text mask', 'width-aware R1 missing/false loss', 'checkpoint iteration']))


core.prepare = prepare
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--arm', default=ARM)
    args = parser.parse_args()
    core.worker(args.arm) if args.mode == 'worker' else core.run()
