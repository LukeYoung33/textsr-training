import torch
import torch.nn as nn
import torch.nn.functional as F


NET_CONFIG_rec = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
    "blocks5":
    [[3, 128, 256, (1, 2), False], [5, 256, 256, 1, False],
     [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, (2, 1), True], [5, 512, 512, 1, True],
                [5, 512, 512, (2, 1), False], [5, 512, 512, 1, False]]
}

def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = torch.as_tensor(1 - drop_prob)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


#################### Backbone: PPLCNetV3 ####################

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # torch: F.relu6(x + 3., inplace=self.inplace) / 6.
        # paddle: F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.

class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)

class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            raise NotImplementedError
        elif act_type == 'hard_sigmoid':
            self.act = Hsigmoid(inplace)#nn.Hardsigmoid(inplace=inplace)#Hsigmoid(inplace)#
        elif act_type == 'hard_swish' or act_type == 'hswish':
            self.act = Hswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)

class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([scale_value]))
        self.bias = nn.Parameter(torch.Tensor([bias_value]))

    def forward(self, x):
        return self.scale * x + self.bias

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 lr_mult=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            out_channels,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ConvBNLayer_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act='gelu'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = Activation(act_type=act, inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

class Act(nn.Module):
    def __init__(self, act="hswish", lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish(inplace=True)
        else:
            assert act == "relu"
            self.act = Activation(act)
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        return self.lab(self.act(x))

class LearnableRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 num_conv_branches=1,
                 lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = nn.BatchNorm2d(
            num_features=in_channels,
        ) if out_channels == in_channels and stride == 1 else None

        self.conv_kxk = nn.ModuleList([
            ConvBNLayer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                groups=groups,
                lr_mult=lr_mult) for _ in range(self.num_conv_branches)
        ])

        self.conv_1x1 = ConvBNLayer(
            in_channels,
            out_channels,
            1,
            stride,
            groups=groups,
            lr_mult=lr_mult) if kernel_size > 1 else None

        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = Act(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        # for export
        if self.is_repped:
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2:
                out = self.act(out)
            return out

        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out

    def rep(self):
        if self.is_repped:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        self.is_repped = True

    def _pad_kernel_1x1_to_kxk(self, kernel1x1, pad):
        if not isinstance(kernel1x1, torch.Tensor):
            return 0
        else:
            return nn.functional.pad(kernel1x1, [pad, pad, pad, pad])

    def _get_kernel_bias(self):
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(kernel_conv_1x1,
                                                      self.kernel_size // 2)

        kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel_conv_kxk += kernel
            bias_conv_kxk += bias

        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        return kernel_reparam, bias_reparam

    def _fuse_bn_tensor(self, branch):
        if not branch:
            return 0, 0
        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size,
                     self.kernel_size),
                    dtype=branch.weight.dtype)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = identity * x
        return x

class LCNetV3Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dw_size,
                 use_se=False,
                 conv_kxk_num=4,
                 lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr)
        if use_se:
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x

class PPLCNetV3(nn.Module):
    def __init__(self,
                 scale=1.0,
                 conv_kxk_num=4,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 lab_lr=0.1,
                 det=False,
                 **kwargs):
        super().__init__()
        self.scale = scale
        self.lr_mult_list = lr_mult_list
        self.det = det

        self.net_config = NET_CONFIG_rec

        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        assert len(self.lr_mult_list
                   ) == 6, "lr_mult_list length should be 6 but got {}".format(
                       len(self.lr_mult_list))

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=make_divisible(16 * scale),
            kernel_size=3,
            stride=2,
            lr_mult=self.lr_mult_list[0])

        self.blocks2 = nn.Sequential(*[
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[1],
                lab_lr=lab_lr)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[2],
                lab_lr=lab_lr)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[3],
                lab_lr=lab_lr)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[4],
                lab_lr=lab_lr)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[5],
                lab_lr=lab_lr)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks6"])
        ])
        self.out_channels = make_divisible(512 * scale)

        if self.det:
            mv_c = [16, 24, 56, 480]
            self.out_channels = [
                make_divisible(self.net_config["blocks3"][-1][2] * scale),
                make_divisible(self.net_config["blocks4"][-1][2] * scale),
                make_divisible(self.net_config["blocks5"][-1][2] * scale),
                make_divisible(self.net_config["blocks6"][-1][2] * scale),
            ]

            self.layer_list = nn.ModuleList([
                nn.Conv2d(self.out_channels[0], int(mv_c[0] * scale), 1, 1, 0),
                nn.Conv2d(self.out_channels[1], int(mv_c[1] * scale), 1, 1, 0),
                nn.Conv2d(self.out_channels[2], int(mv_c[2] * scale), 1, 1, 0),
                nn.Conv2d(self.out_channels[3], int(mv_c[3] * scale), 1, 1, 0)
            ])
            self.out_channels = [
                int(mv_c[0] * scale), int(mv_c[1] * scale),
                int(mv_c[2] * scale), int(mv_c[3] * scale)
            ]

    def forward(self, x):
        out_list = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        if self.det:
            out_list[0] = self.layer_list[0](out_list[0])
            out_list[1] = self.layer_list[1](out_list[1])
            out_list[2] = self.layer_list[2](out_list[2])
            out_list[3] = self.layer_list[3](out_list[3])
            return out_list

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x

######################## Neck: SequenceEncoder ####################

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='gelu',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Activation(act_type=act_layer, inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvMixer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1, [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            )

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=[8, 25],
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1, dtype=torch.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            mask_inf = torch.full([H * W, H * W], fill_value=float("-Inf"), dtype=torch.float32)
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(1)
            # self.mask = mask[None, None, :]
        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.permute(0, 1, 3, 2)))
        if self.mixer == 'Local':
            attn += self.mask
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer='gelu',
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == 1
        x = x.squeeze(dim=2)
        # x = x.transpose([0, 2, 1])  # paddle (NTC)(batch, width, channels)
        x = x.permute(0,2,1)
        return x

class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'svtr': EncoderWithSVTR,
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x

class EncoderWithRNN_(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn1 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
        self.rnn2 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)

class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True) # batch_first:=True

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
            )

    def forward(self, x):
        x = self.fc(x)
        return x

class EncoderWithSVTR(nn.Module):
    def __init__(
            self,
            in_channels,
            dims=64,  # XS
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            kernel_size=[3,3],
            attn_drop_rate=0.1,
            drop_path=0.,
            qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer_Head(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act='swish')
        self.conv2 = ConvBNLayer_Head(
            in_channels // 8, hidden_dims, kernel_size=1, act='swish')

        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer='swish',
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                epsilon=1e-05,
                prenorm=False) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer_Head(
            hidden_dims, in_channels, kernel_size=1, act='swish')
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer_Head(
            2 * in_channels, in_channels // 8, padding=1, act='swish')

        self.conv1x1 = ConvBNLayer_Head(
            in_channels // 8, dims, kernel_size=1, act='swish')
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)

        for blk in self.svtr_block:
            z = blk(z)

        z = self.norm(z)
        # last stage
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))

        return z

########################### Head: CTCHead ##########################

class CTCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,)
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            # predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result

