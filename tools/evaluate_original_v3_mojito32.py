"""Unthickened original-aligned-GT V3 Full250K on identical V3 cached fixed32."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_five_controls_mojito32 as core
from tools.run_uniform20_steep_scurve import save, digest

SOURCE = core.DATA/'original_gt_vosr2u_l1_six_20261023/training_20261023'
ARM = 'v3_original_aligned_gt_full_250k'
core.OUT = core.DATA/'v3_original_aligned_gt_250k_mojito32_20261104'
core.ARMS = {ARM: 'V3 · 原始对齐GT无增粗 · Full · 250K'}
core.__file__ = __file__


def prepare():
    """@luna-comment Input:confirmedoriginalGT V3Full250K. Output:hashcopiedmodel/config andsame32V3selection. Key Steps:sourceGT/config/network/iterationassertions,hashcopy,binding. Key Params:V3notV3M,no GT dilation. State/Dependencies:read-onlyhistoricaltraining;acceptedgenericrunner."""
    import yaml
    out = core.OUT
    config = SOURCE/'configs/original_full_v3.yml'
    cfg = yaml.safe_load(config.read_text())
    assert cfg['network_g']['type'] == 'BasicTextLineR4StudentY4UVV3'
    assert cfg['train']['total_iter'] == 250000
    assert 'bank_original_gt_v1' in cfg['datasets']['train']['root']
    checkpoint = SOURCE/'experiments'/cfg['name']/'models/net_g_250000.pth'
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
        checkpoint_sha256=before, source_checkpoint=str(checkpoint), iteration=250000,
        label=core.ARMS[ARM], stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    shutil.copy2(core.DATA/'mojito280_subset32_20261104/selection.json', out/'selection.json')
    save(out/'MODEL_BINDINGS.json', {ARM: binding})
    save(out/'RUN_BINDING.json', dict(selection_sha256=digest(out/'selection.json'),
        post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'), outputs=64,
        iteration=250000, architecture='V3', training_target='original aligned GT; no dilation',
        training_parents=4873, inference_input='same cached V3 preprocessing as all current controls',
        comparison_limits=['V3 vs V3M', 'Full vs L1/R1', '250K vs500K/370K', 'different training populations']))


core.prepare = prepare
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--arm', default=ARM)
    args = parser.parse_args()
    core.worker(args.arm) if args.mode == 'worker' else core.run()
