"""Explicit V5Lite registration on the accepted VOSR2u block/loss pipeline."""
import argparse
import importlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def bootstrap(repository):
    """@luna-comment
    Input: existing training repository. Output: accepted block/textboost plus V5Lite registrations.
    Key Steps: reuse validated V3M extension bootstrap and import only the new architecture.
    Key Params: no existing class replacement. State/Dependencies: process-local registries and source paths.
    """
    from tools.train_v3m_block256 import bootstrap as original
    result=original(repository)
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5lite_arch')
    return result


def main():
    """@luna-comment
    Input: repo-root and BasicSR options. Output: isolated configured training/checkpoints.
    Key Steps: explicit bootstrap then unchanged train_pipeline. Key Params: YAML owns recipe and loss weights.
    State/Dependencies: CUDA and complete paired bank; no benchmark weights loaded.
    """
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--repo-root',required=True)
    args,rest=parser.parse_known_args();sys.argv=[sys.argv[0],*rest]
    bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(args.repo_root)


if __name__=='__main__':main()
