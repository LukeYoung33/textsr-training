"""Register the isolated full-block training adapter and run the existing pipeline."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def bootstrap(repository):
    """@luna-comment
    Input: legacy training repository. Output: existing V3/V5/reader plus block-model registration.
    Key Steps: reuse offline bootstrap, load only isolated block adapter.
    Key Params:staged source path. State/Dependencies:process-local registry; legacy code unchanged.
    """
    from tools.train_offline_comparison import bootstrap as original
    result=original(repository)
    name='realesrgan.models.block256_region_model'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'realesrgan/models/block256_region_model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return result


def main():
    """@luna-comment
    Input: --repo-root and standard BasicSR options. Output: independent full-block training.
    Key Steps: register adapters and delegate exact existing training pipeline.
    Key Params:explicit256 configs. State/Dependencies:CUDA and complete paired bank.
    """
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--repo-root',required=True)
    args,rest=parser.parse_known_args();sys.argv=[sys.argv[0],*rest]
    repository=bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(str(repository))


if __name__=='__main__':main()
