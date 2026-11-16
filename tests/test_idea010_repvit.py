"""RepViT私有恢复结构与非平凡融合验证。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import torch
from torch import nn
from idea010_repvit_arch import RepViTM06NativeRestore, RepDW, RepBlock, SE, fuse, profile


def test_official_depth_and_shape():
    """@luna-comment Input:固定随机模型；Output:形状/块数断言；Key Steps:前向和枚举；Key Params:17block/7SE；State/Dependencies:CPU。"""
    torch.set_num_threads(2)
    model = RepViTM06NativeRestore().eval()
    assert [len(stage) for stage in model.stages] == [2, 3, 10, 2]
    assert sum(isinstance(m, RepBlock) for m in model.modules()) == 17
    assert sum(isinstance(m, SE) for m in model.modules()) == 7
    x = torch.rand(1, 6, 128, 128)
    with torch.no_grad():
        assert model(x).shape == x.shape


def test_nontrivial_bn_and_branch_fusion():
    """@luna-comment Input:随机多分支及非默认BN；Output:融合误差断言；Key Steps:非零head/BN统计后比较batch1/2；Key Params:1e-4；State/Dependencies:仅CPU模型。"""
    torch.manual_seed(610)
    torch.set_num_threads(2)
    model = RepViTM06NativeRestore().eval()
    with torch.no_grad():
        model.head.weight.normal_(0, .02)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-.1, .1)
                m.running_var.uniform_(.8, 1.2)
                m.weight.uniform_(.8, 1.2)
                m.bias.uniform_(-.05, .05)
    deployed = fuse(model)
    assert not any(isinstance(m, (RepDW, nn.BatchNorm2d)) for m in deployed.modules())
    with torch.no_grad():
        for batch in (1, 2):
            x = torch.rand(batch, 6, 128, 128)
            assert (model(x) - deployed(x)).abs().max().item() < 1e-4


def test_fused_profile_includes_restore_and_se():
    """@luna-comment Input:完整模型；Output:真实执行MAC断言；Key Steps:融合并计数；Key Params:不计训练额外DW1；State/Dependencies:CPU副本。"""
    torch.set_num_threads(2)
    result = profile(RepViTM06NativeRestore().eval())
    assert result['repconv_fused']
    assert result['backbone_major_macs'] == 127914400
    assert result['major_macs'] > result['backbone_major_macs']
    assert any('reduce' in name for name in result['convolution_by_module'])
