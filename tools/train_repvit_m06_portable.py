"""RepViT-M0.6 两组已执行配方的便携入口，不自动下载数据或权重。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def main():
    """@luna-comment Input:绑定配置或初始化路径；Output:初始化、校验或训练。
    Key Steps:检查配方、复用配对训练注册、注册独立架构；Key Params:原始精确GELU。
    State/Dependencies:仅初始化显式写权重；check-only不读取数据、不启动训练。
    """
    import torch
    import yaml
    from idea010_repvit_arch import RepViTM06NativeRestore
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-opt')
    parser.add_argument('--initialize')
    parser.add_argument('--seed', type=int, default=20260828)
    parser.add_argument('--check-only', action='store_true')
    args, rest = parser.parse_known_args()
    torch.set_num_threads(2)
    if args.initialize:
        if args.opt or args.check_only or rest:
            parser.error('--initialize 不能与训练或检查参数混用')
        target = Path(args.initialize)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        model = RepViTM06NativeRestore()
        with target.open('xb') as stream:
            torch.save({'params': model.state_dict()}, stream)
        print('Fresh random initialization:', target)
        return
    if not args.opt:
        parser.error('需要 -opt 或 --initialize')
    text = Path(args.opt).read_text(encoding='utf-8-sig')
    if '${' in text:
        raise ValueError('请先绑定配置中所有路径')
    cfg = yaml.safe_load(text)
    arm = cfg['filtered_control']['kind']
    train = cfg['train']
    expected = (1., .25) if arm == 'ori' else (0., 0.)
    if (arm not in ('ori', 'l1') or cfg['model_type'] != 'FilteredPairedModel'
            or cfg['network_g'] != {'type': 'RepViTM06NativeRestore'}
            or cfg['s22_strategy']['name'] != 'paired'
            or cfg['filtered_control']['synthetic_gt_probability'] != 0
            or cfg['filtered_control']['contrast_alignment']
            or (train['recall_aware_weight_missing'], train['recall_aware_weight_false']) != expected
            or any(d.get('patch_contrast_alignment', False) for d in cfg['datasets'].values())):
        raise ValueError('配置不符合 RepViT 原始配对 L1/R1 配方')
    if args.check_only and sys.platform == 'win32':
        model = RepViTM06NativeRestore().eval()
        with torch.no_grad():
            shape = list(model(torch.zeros(1, 6, 128, 128)).shape)
        print(json.dumps(dict(checked=True, shape=shape, trained=False,
                              runtime_registration='not checked: requires Linux fcntl')))
        return
    from train_v3m_r1_ori_portable import bootstrap
    bootstrap()
    from basicsr.utils.registry import ARCH_REGISTRY
    ARCH_REGISTRY.register()(RepViTM06NativeRestore)
    if args.check_only:
        from basicsr.archs import build_network
        from basicsr.utils.registry import DATASET_REGISTRY, MODEL_REGISTRY
        model = build_network(cfg['network_g'])
        DATASET_REGISTRY.get('S22Dataset')
        MODEL_REGISTRY.get('FilteredPairedModel')
        print(json.dumps(dict(checked=True, trained=False,
                              parameters=sum(p.numel() for p in model.parameters()))))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练需要已验证的 CUDA 环境')
    sys.argv = [sys.argv[0], '-opt', str(Path(args.opt).resolve()), *rest]
    from basicsr.train import train_pipeline
    train_pipeline(str(ROOT))


if __name__ == '__main__':
    main()
