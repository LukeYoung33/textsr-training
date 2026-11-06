"""Explicit opt-in bootstrap for V3-preprocessed V3M/area-adaptive experiments."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap(repository):
    """@luna-comment Input:existing complete runtime repo. Output:historical+opt-in registries.
    Key Steps:accepted V3M bootstrap; expose only new model adapter path and import.
    Key Params:YAML controls activation. State/Dependencies:process-local registries.
    """
    from tools.train_v3m_block256 import bootstrap as original
    original(repository)
    package = sys.modules['realesrgan.models']
    path = str(ROOT / 'realesrgan/models')
    if path not in package.__path__:
        package.__path__.insert(0, path)
    importlib.import_module('realesrgan.models.v3pre_area_adapters')


def main():
    """@luna-comment Input:repo-root/BasicSR CLI. Output:authorized independent training.
    Key Steps:register opt-in extensions then unchanged train pipeline. Key Params:YAML.
    State/Dependencies:CUDA and immutable verified V3-preprocessed bank.
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
