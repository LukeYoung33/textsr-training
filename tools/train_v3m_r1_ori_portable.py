"""Portable entry for the executed paired-only V3M L1+original-R1 recipe."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap():
    """@luna-comment Input:none. Output:accepted model/data/scheduler registrations.
    Key Steps:bind historical bootstrap repository to this checkout, reuse exact registrations.
    Key Params:no loss overrides. State/Dependencies:process-local historical module only.
    """
    import tools.train_full_s22_strategy as strategy
    strategy.REPO = str(ROOT)
    from tools.train_filtered_three import bootstrap as accepted
    accepted()


def main():
    """@luna-comment Input:bound YAML and optional check-only flag. Output:CPU check or training.
    Key Steps:validate explicit recipe,register dependencies,delegate unchanged BasicSR pipeline.
    Key Params:paired-only original R1; check never accesses data or starts training.
    State/Dependencies:explicit user config; formal training requires CUDA and prepared bank.
    """
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-opt', required=True)
    parser.add_argument('--check-only', action='store_true')
    args, rest = parser.parse_known_args()
    text = Path(args.opt).read_text(encoding='utf-8-sig')
    if '${' in text:
        raise ValueError('Bind all template paths before running this entry')
    options = yaml.safe_load(text)
    train = options['train']
    if (options['model_type'] != 'FilteredPairedModel'
            or options['network_g']['type'] != 'BasicTextLineR4StudentY4UVV3M'
            or options['s22_strategy']['name'] != 'paired'
            or train['recall_aware_weight_missing'] != 1
            or train['recall_aware_weight_false'] != 0.25
            or options['filtered_control']['kind'] != 'ori'
            or options['filtered_control']['synthetic_gt_probability'] != 0
            or options['filtered_control']['contrast_alignment']
            or options['datasets']['train'].get('patch_contrast_alignment', False)):
        raise ValueError('This entry requires the paired-only original-R1 recipe')
    bootstrap()
    if args.check_only:
        import torch
        from basicsr.archs import build_network
        from basicsr.utils.registry import DATASET_REGISTRY, MODEL_REGISTRY
        from tools.train_latest_vosr_500k import rate
        torch.set_num_threads(1)
        network = build_network(options['network_g'])
        DATASET_REGISTRY.get('S22Dataset')
        MODEL_REGISTRY.get('FilteredPairedModel')
        print(json.dumps(dict(checked=True, trained=False,
                             parameters=sum(p.numel() for p in network.parameters()),
                             lr=[rate(i) for i in [1, 200000, 300000, 400000, 450000, 500000]])))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Formal training requires CUDA; use --check-only for CPU validation')
    sys.argv = [sys.argv[0], '-opt', str(Path(args.opt).resolve()), *rest]
    from basicsr.train import train_pipeline
    train_pipeline(str(ROOT))


if __name__ == '__main__':
    main()
