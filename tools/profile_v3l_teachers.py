"""Measure width-only teachers using materialized deployment convolutions."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profile(network):
    """@luna-comment
    Input: CPU V3M/V3L model at its declared canvas. Output: parameter/MAC/tap-byte report.
    Key Steps: fuse a copy; count executed Conv2d MACs by output shape; run all stage taps.
    Key Params: batch1 FP32, one multiply-accumulate=one MAC; excludes Haar, fusion setup,
    interpolation, bias and activation arithmetic. State/Dependencies: existing exact ECB exporter.
    """
    from tools.export_basic_text_line_y4uv_v3_checkpoint_deploy_onnx import fuse_reparameterized_modules
    from realesrgan.archs.basic_text_line_r4_teacher_y4uv_v3l_arch import FEATURE_TAPS
    deployed = copy.deepcopy(network).eval()
    blocks = fuse_reparameterized_modules(deployed)
    total, feature_bytes, handles = [], [], []

    def count(module, inputs, output):
        """@luna-comment
        Input: executed Conv2d hook. Output: append exact dense/grouped convolution MAC count.
        Key Steps: multiply output elements by per-output kernel size.
        Key Params: actual groups/kernel dimensions. State/Dependencies: local accumulation list.
        """
        total.append(output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1])

    def feature_size(module, inputs, output):
        """@luna-comment
        Input: stage output hook. Output: append output storage bytes (not peak training memory).
        Key Steps: use tensor element count and size. Key Params: batch1 FP32.
        State/Dependencies: local list; no retained tensors.
        """
        feature_bytes.append(output.numel() * output.element_size())

    try:
        for module in deployed.modules():
            if isinstance(module, torch.nn.Conv2d):
                handles.append(module.register_forward_hook(count))
        for name in FEATURE_TAPS:
            handles.append(getattr(deployed.backbone, name).register_forward_hook(feature_size))
        data = torch.zeros(1, 6, *network.expected_input_hw)
        with torch.no_grad():
            deployed(data)
    finally:
        for handle in handles:
            handle.remove()
    return dict(widths=list(network.feature_widths), canvas_hw=list(network.canvas_hw),
                packed_shape=list(data.shape), convolution_macs=sum(total),
                training_parameters=sum(p.numel() for p in network.parameters()),
                deployed_parameters=sum(p.numel() for p in deployed.parameters()),
                stage_output_bytes=sum(feature_bytes), fused_blocks=blocks)


def main():
    """@luna-comment
    Input: CLI canvas and report path. Output: measured JSON for baseline and four teachers.
    Key Steps: explicit accepted bootstrap; profile actual fused graphs; assert target tolerance.
    Key Params: square canvas256 default and 5% target tolerance, CPU thread1.
    State/Dependencies: local repository only; creates report parent and JSON, no checkpoints.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canvas', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/v3l_feature_distillation_20261106/profiles.json')
    args = parser.parse_args()
    from tools.train_v3m_block256 import bootstrap
    bootstrap(str(ROOT))
    from realesrgan.archs.basic_text_line_r4_student_y4uv_v3m_arch import BasicTextLineR4StudentY4UVV3M
    from realesrgan.archs.basic_text_line_r4_teacher_y4uv_v3l_arch import BasicTextLineR4TeacherY4UVV3L, TEACHER_WIDTHS
    torch.set_num_threads(1)
    baseline = profile(BasicTextLineR4StudentY4UVV3M(canvas_hw=(args.canvas, args.canvas)))
    teachers = {}
    for target in TEACHER_WIDTHS:
        row = profile(BasicTextLineR4TeacherY4UVV3L(compute_ratio=target, canvas_hw=(args.canvas, args.canvas)))
        row['actual_ratio'] = row['convolution_macs'] / baseline['convolution_macs']
        row['relative_target_error'] = row['actual_ratio'] / target - 1
        if abs(row['relative_target_error']) > .05:
            raise RuntimeError(f'Teacher {target}x misses the 5% MAC tolerance')
        teachers[str(target)] = row
    report = dict(convention='Executed deployed Conv2d MACs; excludes fixed Haar, interpolation, bias, activation and fusion setup; not training MACs or peak memory.',
                  baseline=baseline, teachers=teachers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
