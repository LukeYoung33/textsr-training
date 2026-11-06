"""Isolated native training entry for the offline-preprocessed paired shard bank."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bootstrap(repo_root=None):
    """@luna-comment
    Input: optional existing training repository for a staged dataset package.
    Output: historical registrations plus the isolated offline reader.
    Key Steps: use native bootstrap, then import dataset adapter at its own package path.
    Key Params: repo_root defaults current repository. State/Dependencies: process registries.
    """
    repository = Path(repo_root) if repo_root else ROOT
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from tools.train_v3_native_regions import bootstrap as native_bootstrap
    native_bootstrap()
    name = 'realesrgan.data.offline_augmented_region_dataset'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'realesrgan/data/offline_augmented_region_dataset.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return repository


def main():
    """@luna-comment
    Input: BasicSR CLI options and optional --repo-root for remote staging.
    Output: training artifacts only when explicitly invoked by the user.
    Key Steps: remove staging option, register reader, delegate existing train pipeline.
    Key Params: -opt uses isolated paired-only configuration. State/Dependencies: BasicSR.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repo-root')
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    repository = bootstrap(args.repo_root)
    from basicsr.train import train_pipeline
    train_pipeline(str(repository))


if __name__ == '__main__':
    main()
