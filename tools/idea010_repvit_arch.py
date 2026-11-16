"""IDEA-010：官方M0.6宽度/深度的独立恢复模型，支持精确推理融合。"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from idea009_evit_arch import Conv, UpFuse


class ConvBN(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, zero=False):
        """@luna-comment Input:卷积形状；Output:Conv/BN；Key Steps:初始化及末BN零值；Key Params:zero用于channel residual；State/Dependencies:私有参数。"""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        nn.init.constant_(self.bn.weight, 0 if zero else 1)

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:卷积BN；Key Steps:顺序执行；Key Params:构造固定；State/Dependencies:train更新BN。"""
        return self.bn(self.conv(value))


class RepDW(nn.Module):
    def __init__(self, channels):
        """@luna-comment Input:通道；Output:官方RepVGGDW形状；Key Steps:DW3BN/DW1/identity及外BN；Key Params:stride1；State/Dependencies:私有参数。"""
        super().__init__()
        self.conv = ConvBN(channels, channels, 3, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:同形；Key Steps:三分支相加后BN；Key Params:无激活；State/Dependencies:私有BN。"""
        return self.bn(self.conv(value) + self.conv1(value) + value)

    def fused(self):
        """@luna-comment Input:eval模块；Output:新DW3卷积；Key Steps:内BN融合/补齐1x1和identity/外BN融合；Key Params:保留bias；State/Dependencies:不改原模块。"""
        assert not self.training
        result = nn.utils.fuse_conv_bn_eval(self.conv.conv, self.conv.bn)
        identity = torch.zeros_like(result.weight)
        identity[:, 0, 1, 1] = 1
        result.weight.data.add_(F.pad(self.conv1.weight, (1, 1, 1, 1)) + identity)
        result.bias.data.add_(self.conv1.bias)
        return nn.utils.fuse_conv_bn_eval(result, self.bn)


class SE(nn.Module):
    def __init__(self, channels):
        """@luna-comment Input:C；Output:SE比例.25；Key Steps:池化/降维ReLU/升维Sigmoid；Key Params:各宽度可被4整除；State/Dependencies:私有参数。"""
        super().__init__()
        self.reduce = nn.Conv2d(channels, channels // 4, 1)
        self.expand = nn.Conv2d(channels // 4, channels, 1)

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:通道门控同形；Key Steps:空间均值后SE；Key Params:Sigmoid；State/Dependencies:私有权重。"""
        gate = value.mean((2, 3), keepdim=True)
        return value * torch.sigmoid(self.expand(F.relu(self.reduce(gate))))


class RepBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_se):
        """@luna-comment Input:C/stride/SE；Output:M0.6 block；Key Steps:token mixer及残差channel mixer；Key Params:MLP2/GELU精确；State/Dependencies:私有权重。"""
        super().__init__()
        squeeze = SE(cin) if use_se else nn.Identity()
        if stride == 2:
            self.token = nn.Sequential(ConvBN(cin, cin, 3, 2, cin), squeeze, ConvBN(cin, cout))
        else:
            assert cin == cout
            self.token = nn.Sequential(RepDW(cin), squeeze)
        self.channel = nn.Sequential(ConvBN(cout, 2 * cout), nn.GELU(), ConvBN(2 * cout, cout, zero=True))

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:token混合后channel残差；Key Steps:两mixer；Key Params:仅channel外残差；State/Dependencies:不跨下采样加输入。"""
        value = self.token(value)
        return value + self.channel(value)


class RepViTM06NativeRestore(nn.Module):
    def __init__(self):
        """@luna-comment Input:固定native合同；Output:M0.6恢复网；Key Steps:双stem/17blocks/skip恢复；Key Params:原宽40/80/160/320；State/Dependencies:从头训练，不下载权重。"""
        super().__init__()
        self.canvas_hw, self.haar_switch = (256, 256), False
        self.stem_half = nn.Sequential(ConvBN(6, 20, 3, 2), nn.GELU())
        self.stem_quarter = ConvBN(20, 40, 3, 2)
        # Each tuple is (output width, stride, SE); exact official M0.6 sequence.
        configs = [[(40, 1, True), (40, 1, False)],
                   [(80, 2, False), (80, 1, True), (80, 1, False)],
                   [(160, 2, False)] + [(160, 1, se) for se in (True, False, True, False, True, False, True, False, False)],
                   [(320, 2, False), (320, 1, True)]]
        stages, cin = [], 40
        for config in configs:
            blocks = []
            for cout, stride, use_se in config:
                blocks.append(RepBlock(cin, cout, stride, use_se))
                cin = cout
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.shallow = Conv(6, 8)
        self.decoder = nn.ModuleList([UpFuse(320, 160), UpFuse(160, 80), UpFuse(80, 40), UpFuse(40, 20), UpFuse(20, 8)])
        self.head = nn.Conv2d(8, 6, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-4)
        nn.init.zeros_(self.head.bias)

    def forward(self, value):
        """@luna-comment Input:B6x128x128；Output:同形残差恢复；Key Steps:编码/五级skip解码/加输入；Key Params:固定网格；State/Dependencies:不修改输入。"""
        if value.ndim != 4 or tuple(value.shape[1:]) != (6, 128, 128):
            raise ValueError('IDEA-010 requires packed Bx6x128x128 input')
        half = self.stem_half(value)
        result = self.stem_quarter(half)
        features = []
        for stage in self.stages:
            result = stage(result)
            features.append(result)
        skips = list(reversed(features[:-1])) + [half, self.shallow(value)]
        for decoder, skip in zip(self.decoder, skips):
            result = decoder(result, skip)
        return value + self.head(result)


def fuse(model):
    """@luna-comment Input:eval模型；Output:独立融合副本；Key Steps:先RepDW后通用ConvBN；Key Params:不越过GELU/SE；State/Dependencies:不改训练权重。"""
    result = copy.deepcopy(model).eval()

    def replace(parent):
        """@luna-comment Input:父模块；Output:替换融合子模块；Key Steps:递归并识别RepDW/ConvBN/恢复Conv；Key Params:eval BN统计；State/Dependencies:仅副本。"""
        for name, child in list(parent.named_children()):
            if isinstance(child, RepDW):
                setattr(parent, name, child.fused())
            elif isinstance(child, ConvBN):
                setattr(parent, name, nn.utils.fuse_conv_bn_eval(child.conv, child.bn))
            elif isinstance(child, Conv):
                if isinstance(child.norm, nn.BatchNorm2d):
                    child.conv = nn.utils.fuse_conv_bn_eval(child.conv, child.norm)
                    child.norm = nn.Identity()
                replace(child)
            else:
                replace(child)
    replace(result)
    return result


def profile(model):
    """@luna-comment Input:训练模型；Output:实际融合Conv MAC统计；Key Steps:独立融合/逐Conv hook前向；Key Params:含SE/decoder，不含激活池化插值乘加；State/Dependencies:CPU副本。"""
    deployed = fuse(model).cpu()
    counts, handles = {}, []

    def hook(name):
        """@luna-comment Input:模块名；Output:局部计数闭包；Key Steps:绑定名字；Key Params:真实输出尺寸；State/Dependencies:counts。"""
        def count(module, inputs, output):
            """@luna-comment Input:Conv输出；Output:记录MAC；Key Steps:输出元素乘分组核；Key Params:batch1；State/Dependencies:本次字典。"""
            counts[name] = output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        return count
    try:
        for name, module in deployed.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            deployed(torch.zeros(1, 6, 128, 128))
    finally:
        for handle in handles:
            handle.remove()
    encoder = sum(v for k, v in counts.items() if k.startswith(('stem_', 'stages.')))
    total = sum(counts.values())
    return dict(parameters=sum(p.numel() for p in model.parameters()), deployed_parameters=sum(p.numel() for p in deployed.parameters()),
                backbone_major_macs=encoder, restoration_head_macs=total - encoder, major_macs=total,
                ratio=total / 34258944, convolution_by_module=counts, repconv_fused=True,
                convention='Executed fused Conv2d including SE at batch1; excludes activation/pooling/interpolation/elementwise/data packing')
