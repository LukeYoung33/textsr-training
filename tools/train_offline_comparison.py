"""Versioned isolated entry for the approved V3/V5 offline matrix."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap(repo_root):
    """@luna-comment
    Input: legacy repository root. Output: native offline and V5 registrations.
    Key Steps: register the staged reader then explicit V5 module.
    Key Params: repo_root. State/Dependencies: process-local registry only.
    """
    from tools.train_offline_augmented_regions import bootstrap as offline_bootstrap
    repository = offline_bootstrap(repo_root)
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch')
    return repository


def main():
    """@luna-comment
    Input: staging root and BasicSR CLI. Output: independent training run.
    Key Steps: register both architectures and invoke existing pipeline.
    Key Params: -opt selects one arm. State/Dependencies: CUDA and stored bank.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repo-root', required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    repository = bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(str(repository))


if __name__ == '__main__':
    main()
