"""Explicit V3/V3M block training registration with optional text-region boost."""
import argparse
import importlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def bootstrap(repository):
    """@luna-comment
    Input: existing BasicSR repository path. Output: block model plus V3M/textboost registry entries.
    Key Steps: reuse accepted block bootstrap; import only new extension modules.
    Key Params: architecture/loss activation controlled by YAML. State/Dependencies: process-local registries.
    """
    from tools.train_vosr_block256 import bootstrap as block
    result=block(repository)
    # The accepted bootstrap can bind legacy package roots. Expose this
    # experiment's explicit extensions without replacing registered baselines.
    for name in ('realesrgan.archs','realesrgan.losses'):
        package=sys.modules[name]
        extension=str(ROOT.joinpath(*name.split('.')))
        if extension not in package.__path__:package.__path__.insert(0,extension)
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v3m_arch')
    importlib.import_module('realesrgan.losses.text_region_boost_loss')
    return result


def main():
    """@luna-comment
    Input: repo-root and BasicSR CLI. Output: isolated user-requested training/checkpoints.
    Key Steps: register extensions then delegate unchanged train_pipeline.
    Key Params: YAML owns all data/loss/optimizer settings. State/Dependencies: CUDA and immutable paired bank.
    """
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--repo-root',required=True)
    args,rest=parser.parse_known_args();sys.argv=[sys.argv[0],*rest]
    bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(args.repo_root)


if __name__=='__main__':main()
