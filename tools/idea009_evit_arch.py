"""IDEA-009：MIT B0形状配置的独立原生恢复实验，无第三方运行依赖。"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, norm=True, act=True, bias=False):
        """@luna-comment Input:卷积尺寸/归一化开关；Output:私有卷积层；Key Steps:Conv/可选BN/Hardswish；Key Params:same padding；State/Dependencies:私有权重与BN状态。"""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(cout) if norm else nn.Identity()
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:卷积特征；Key Steps:卷积/BN/激活；Key Params:构造器固定；State/Dependencies:训练模式更新BN。"""
        return self.act(self.norm(self.conv(value)))


class DSConv(nn.Module):
    def __init__(self, channels, residual=False):
        """@luna-comment Input:通道数；Output:同通道深度可分离卷积；Key Steps:DW3x3/point1x1；Key Params:末层无激活；State/Dependencies:BN及参数。"""
        super().__init__()
        self.depth = Conv(channels, channels, 3, groups=channels)
        self.point = Conv(channels, channels, act=False)
        self.residual = residual

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:同形；Key Steps:深度/逐点卷积；Key Params:stride1；State/Dependencies:私有参数。"""
        result = self.point(self.depth(value))
        return result + value if self.residual else result


class MBConv(nn.Module):
    def __init__(self, cin, cout, stride=1, fewer_norm=False, residual=True):
        """@luna-comment Input:输入输出宽度/步长；Output:MBConv扩展4；Key Steps:升维/DW/投影；Key Params:fewer_norm采用偏置替代前两BN；State/Dependencies:仅stride1同宽启用残差。"""
        super().__init__()
        mid = 4 * cin
        self.expand = Conv(cin, mid, norm=not fewer_norm, bias=fewer_norm)
        self.depth = Conv(mid, mid, 3, stride, groups=mid, norm=not fewer_norm, bias=fewer_norm)
        self.project = Conv(mid, cout, act=False)
        self.residual = residual and stride == 1 and cin == cout

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:投影特征；Key Steps:三卷积并按条件加skip；Key Params:固定扩展4；State/Dependencies:私有权重及BN。"""
        result = self.project(self.depth(self.expand(value)))
        return result + value if self.residual else result


class LiteMLA(nn.Module):
    def __init__(self, channels):
        """@luna-comment Input:B0通道64或128；Output:两尺度ReLU注意力；Key Steps:QKV/5x5聚合/输出投影；Key Params:head_dim16/eps1e-15；State/Dependencies:私有参数及内部MAC遥测。"""
        super().__init__()
        self.dim = 16
        self.heads = channels // self.dim
        self.qkv = Conv(channels, 3 * channels, norm=False, act=False)
        self.aggregate = nn.Sequential(
            nn.Conv2d(3 * channels, 3 * channels, 5, padding=2, groups=3 * channels, bias=False),
            nn.Conv2d(3 * channels, 3 * channels, 1, groups=3 * self.heads, bias=False))
        self.project = Conv(2 * channels, channels, act=False)
        self.last_attention_macs = 0

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:同形全局注意力；Key Steps:双尺度QKV/ReLU核/归一化矩阵乘；Key Params:N>16线性路径，否则二次等价路径；State/Dependencies:无输入修改，MAC诊断内部记录。"""
        batch, _, height, width = value.shape
        base = self.qkv(value)
        qkv = torch.cat((base, self.aggregate(base)), dim=1)
        heads, tokens = 2 * self.heads, height * width
        qkv = qkv.reshape(batch, heads, 3 * self.dim, tokens)
        query, key, val = qkv.split(self.dim, dim=2)
        query, key = F.relu(query), F.relu(key)
        if tokens > self.dim:
            padded = F.pad(val, (0, 0, 0, 1), value=1)
            result = (padded @ key.transpose(-1, -2)) @ query
            result = result[:, :, :-1] / (result[:, :, -1:] + 1e-15)
            self.last_attention_macs = batch * 2 * heads * (self.dim + 1) * self.dim * tokens
        else:
            weights = key.transpose(-1, -2) @ query
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-15)
            result = val @ weights
            self.last_attention_macs = batch * 2 * heads * self.dim * tokens * tokens
        return self.project(result.reshape(batch, -1, height, width))


class EfficientBlock(nn.Module):
    def __init__(self, channels):
        """@luna-comment Input:通道；Output:MIT B0注意力与局部模块；Key Steps:LiteMLA残差/MBConv残差；Key Params:dim16/expand4；State/Dependencies:本模块参数。"""
        super().__init__()
        self.attention = LiteMLA(channels)
        self.local = MBConv(channels, channels, fewer_norm=True)

    def forward(self, value):
        """@luna-comment Input:BCHW；Output:同形；Key Steps:全局注意力残差后局部卷积；Key Params:无dropout；State/Dependencies:私有参数。"""
        value = value + self.attention(value)
        return self.local(value)


class UpFuse(nn.Module):
    def __init__(self, cin, cout):
        """@luna-comment Input:低清/skip通道；Output:单级恢复模块；Key Steps:低清1x1投影/双线性/skip融合/DS残差；Key Params:align_corners=False；State/Dependencies:不改变skip。"""
        super().__init__()
        self.project = Conv(cin, cout, norm=False, act=False)
        self.refine = DSConv(cout)

    def forward(self, value, skip):
        """@luna-comment Input:低分辨特征与同坐标skip；Output:skip尺寸特征；Key Steps:先投影再插值并相加、局部细化；Key Params:以skip尺寸为准；State/Dependencies:私有参数。"""
        value = F.interpolate(self.project(value), size=skip.shape[-2:], mode='bilinear', align_corners=False) + skip
        return value + self.refine(value)


class EfficientViTB0NativeRestore(nn.Module):
    def __init__(self):
        """@luna-comment Input:固定native256合同；Output:B0骨干及私有恢复模型；Key Steps:五级B0/浅层skip/五级decoder/残差头；Key Params:width8/16/32/64/128；State/Dependencies:从头随机初始化，BN缓冲属于本模型。"""
        super().__init__()
        self.canvas_hw = (256, 256)
        self.haar_switch = False
        self.stem = nn.Sequential(Conv(6, 8, 3, 2), DSConv(8, residual=True))
        self.stages = nn.ModuleList([
            nn.Sequential(MBConv(8, 16, 2), MBConv(16, 16)),
            nn.Sequential(MBConv(16, 32, 2), MBConv(32, 32)),
            nn.Sequential(MBConv(32, 64, 2, fewer_norm=True), EfficientBlock(64), EfficientBlock(64)),
            nn.Sequential(MBConv(64, 128, 2, fewer_norm=True), EfficientBlock(128), EfficientBlock(128))])
        self.shallow = Conv(6, 8)
        self.decoder = nn.ModuleList([UpFuse(128, 64), UpFuse(64, 32), UpFuse(32, 16), UpFuse(16, 8), UpFuse(8, 8)])
        self.head = nn.Conv2d(8, 6, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-4)
        nn.init.zeros_(self.head.bias)

    def encode(self, value):
        """@luna-comment Input:B6x128x128；Output:五级特征列表；Key Steps:stem后四stage；Key Params:网格64/32/16/8/4；State/Dependencies:BN遵循train/eval。"""
        value = self.stem(value)
        features = [value]
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        return features

    def forward(self, value):
        """@luna-comment Input:B6x128x128 native；Output:同形未裁剪恢复值；Key Steps:编码/逆序skip解码/残差加输入；Key Params:拒绝其他尺寸；State/Dependencies:原输入不变。"""
        if value.ndim != 4 or tuple(value.shape[1:]) != (6, 128, 128):
            raise ValueError('IDEA-009 requires packed Bx6x128x128 input')
        features = self.encode(value)
        result = features[-1]
        skips = list(reversed(features[:-1])) + [self.shallow(value)]
        for decoder, skip in zip(self.decoder, skips):
            result = decoder(result, skip)
        return value + self.head(result)


def profile(model):
    """@luna-comment Input:完整恢复模型；Output:实际执行Conv和注意力MAC/参数；Key Steps:复制eval模型/hook前向/按encoder和恢复头汇总；Key Params:batch1，不含BN/激活/插值/逐元素运算；State/Dependencies:不改变原模型或BN状态。"""
    deployed = copy.deepcopy(model).cpu().eval()
    counts, handles = {}, []

    def hook(name):
        """@luna-comment Input:模块名；Output:计数hook；Key Steps:捕获私有名字；Key Params:每MAC一次乘加；State/Dependencies:本次计数字典。"""
        def count(module, inputs, output):
            """@luna-comment Input:实际卷积输出；Output:累积该层MAC；Key Steps:输出元素乘每输出输入核大小；Key Params:groups正确折算；State/Dependencies:本地字典。"""
            counts[name] = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        return count

    try:
        for name, module in deployed.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            result = deployed(torch.zeros(1, 6, 128, 128))
        assert result.shape == (1, 6, 128, 128)
    finally:
        for handle in handles:
            handle.remove()
    attention = sum(module.last_attention_macs for module in deployed.modules() if isinstance(module, LiteMLA))
    encoder_conv = sum(value for name, value in counts.items() if name.startswith(('stem.', 'stages.')))
    restoring = sum(value for name, value in counts.items() if name.startswith(('shallow.', 'decoder.', 'head')))
    total = encoder_conv + restoring + attention
    return dict(parameters=sum(p.numel() for p in model.parameters()), packed_shape=[1, 6, 128, 128],
        full_canvas=[256, 256], backbone_conv_macs=encoder_conv, attention_matmul_macs=attention,
        backbone_major_macs=encoder_conv + attention, restoration_head_macs=restoring,
        major_macs=total, approximate_flops=2 * total, ratio=total / 34258944,
        v3m_deployed_conv_macs=34258944, convolution_by_module=counts,
        excludes=['BatchNorm', 'Hardswish', 'ReLU', 'interpolation', 'bias', 'normalization division', 'residual adds', 'data packing'],
        convention='Executed Conv2d plus attention matmuls; 1 MAC = 1 multiply-accumulate; complete restoration model')
