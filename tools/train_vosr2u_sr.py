"""Explicit isolated registration for true128-to256 Compact SR training."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap(repository):
    """@luna-comment
    Input: approved existing training repository. Output: original losses and dedicated SR registrations.
    Key Steps: reuse block bootstrap; explicitly load staged Compact/SR architecture and LR/HR adapters.
    Key Params: do not auto-discover unrelated modules. State/Dependencies: process-local Python registries.
    """
    from tools.train_vosr_block256 import bootstrap as block_bootstrap
    block_bootstrap(repository)
    for name in ('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_compact_arch',
                 'realesrgan.archs.basic_text_line_r4_student_y4uv_v5_compact_sr_arch',
                 'realesrgan.archs.basic_text_line_r4_student_y4uv_v5_compact_sr_hrskip_arch',
                 'realesrgan.data.offline_sr2u_dataset', 'realesrgan.models.sr2u_block_model'):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, ROOT.joinpath(*name.split('.')).with_suffix('.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)


def main():
    """@luna-comment
    Input: repo-root and BasicSR options. Output: isolated SR train/validation/checkpoints.
    Key Steps: register explicit modules and delegate the existing training pipeline.
    Key Params: SR128 input/HR256 config; no legacy restoration weight substituted.
    State/Dependencies: CUDA, immutable bank and owned output path.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repo-root', required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(args.repo_root)


if __name__ == '__main__':
    main()
