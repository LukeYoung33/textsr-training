"""Run the historical0921 V5 original with the exact current V3 HX fusion path."""
import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """@luna-comment
    Input: frozen0921 checkpoint/YAML and current52 HX photos/traces.
    Output: matched full-frame0921 PNGs, previews and hashed summary.
    Key Steps: verify exact model, register V5, point shared evaluator to isolated
    config/output root, call unchanged native inference and full-box feather.
    Key Params: final250K-equivalent historical latest;96x480 stride64x448/feather16.
    State/Dependencies: shared evaluator globals in this isolated process only.
    """
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch')
    import tools.auto_hx_v3_native as shared
    original = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_v5_p17_effect_iteration_20260920/lenovo_v5_p17_original_r1_bglf_uv_250k_20260920')
    checkpoint = original/'models/net_g_latest.pth'
    assert shared.sha(checkpoint) == '617dac8daa0e608b5df91300fa56646883f7dafb749d36ece4b01cd74ef6b575'
    output = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/v3_newdeg_vs_0921_original_20261016')
    (output/'configs').mkdir(parents=True, exist_ok=True)
    config = ROOT/'.artifacts/v3_native_four_training_20261015/baseline.yml'
    shutil.copy2(config, output/'configs/0921_original.yml')
    shared.TRAIN = output
    shared.OUT = output
    shared.evaluate('0921_original', checkpoint)


if __name__ == '__main__':
    main()
