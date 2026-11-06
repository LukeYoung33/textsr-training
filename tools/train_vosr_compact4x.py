"""Register the isolated compact architecture and reuse the approved block256 training pipeline."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def bootstrap(repository):
    """@luna-comment
    Input: Existing training repository root. Output: Block adapter plus compact network registration.
    Key Steps: Reuse original bootstrap then explicitly import the new architecture.
    Key Params: Process-local registration; source tree must contain the compact module.
    State/Dependencies: Existing datasets, losses and Block256RegionModel remain unchanged.
    """
    from tools.train_vosr_block256 import bootstrap as original
    result=original(repository)
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_compact_arch')
    return result


def main():
    """@luna-comment
    Input: --repo-root and standard BasicSR -opt config arguments. Output: Independent compact training.
    Key Steps: Register exact existing block trainer and compact model; invoke standard train pipeline.
    Key Params: New config has no pretrained original-width V3/V5 checkpoint or resume state.
    State/Dependencies: CUDA/data/CRNN assets required only when explicitly launching training.
    """
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--repo-root',required=True)
    args,rest=parser.parse_known_args();sys.argv=[sys.argv[0],*rest]
    repository=bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(str(repository))


if __name__=='__main__':main()
