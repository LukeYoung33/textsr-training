import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from ecb import ECB
# from Jieer ecb.py + ecbsr.py
import sys
sys.path.append('.')
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import ResidualDenseBlock, RRDB, RRDBNet, RRDBNetNV21

def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = nn.ReLU()

    return nn.Sequential(modules)


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier, phase):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.phase = phase

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            if phase == 'Encoder':
                conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=2, padding=1)
            elif phase == 'Decoder':
                #conv1 = nn.ConvTranspose2d(self.mid_planes, self.out_planes, kernel_size=3, stride=2, padding=1, output_padding=1)
                conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            else:
                conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)

            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)

            # conv-3x3
            if self.phase == 'Encoder':
                # explicitly padding with bias
                y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
                y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=2, padding=0)
            elif self.phase == 'Decoder':
                # y1 = F.conv_transpose2d(input=y0, weight=self.k1, bias=self.b1, stride=2, padding=1, output_padding=1)
                # y0 = self.upsample(y0)
                # explicitly padding with bias
                y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
                y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
            else:
                # explicitly padding with bias
                y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
                y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
            # print(f'{self.type} finished')

        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)

            # conv-3x3
            if self.phase == 'Encoder':
                # explicitly padding with bias
                y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
                y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=2, padding=0, groups=self.out_planes)
            elif self.phase == 'Decoder':
                # y0 = self.upsample(y0)
                y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
                # conv-3x3
                y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
                # y1 = F.conv_transpose2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=2, padding=1, output_padding=1, groups=self.out_planes)
            else:
                # explicitly padding with bias
                y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
                # conv-3x3
                y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
            #print(f'{self.type} finished')

        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            if self.phase == 'Decoder':
                # re-param conv kernel
                RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
                # re-param conv bias
                RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
                RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
            else:
                # re-param conv kernel
                RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
                # re-param conv bias
                RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
                RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1


        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB



class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False, train_repParam=False, phase='Encoder'):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.train_repParam = train_repParam
        self.phase = phase

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if self.phase == 'Encoder':
            self.conv3x3 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1, stride=2)
        else:
            self.conv3x3 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)

        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier, phase)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1, phase)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1, phase)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1, phase)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')
        # self.RK, self.RB = self.rep_params()

    def forward(self, x):
        if self.train_repParam:
            y = self.conv3x3(x)
        elif self.training:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            if self.phase =='Encoder':
                y = F.conv2d(input=x, weight=RK, bias=RB, stride=2, padding=1)
            else:
                y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)

        if self.act_type != 'linear':
            y = self.act(y)
        #print(f'{self.phase} finished')
        #print(f'{x.shape}')

        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        # NOTE: no permute needed — conv3x3 is always nn.Conv2d regardless of
        # phase, so its weight shape (out, in, 3, 3) is already correct.
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

class ECB2(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False, train_repParam=False, phase='Encoder', with_proj=False):
        super(ECB2, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.train_repParam = train_repParam
        self.phase = phase
        self.with_proj = with_proj

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False


        #self.conv3x3 = nn.ConvTranspose2d(self.inp_planes, self.out_planes, 3, stride=2, padding=1, output_padding=1)
        self.conv3x3 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)

        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier, phase)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1, phase)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1, phase)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1, phase)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

        self.proj_conv = Conv2D(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1, is_seperable=True, has_relu=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # self.RK, self.RB = self.rep_params()

    def forward(self, inputs):
        x, skip = inputs
        if self.with_proj:
            skip = self.proj_conv(skip)
        if self.train_repParam:
            y = self.conv3x3(x)
        elif self.training:
            x_up = self.upsample(x)
            y = self.conv3x3(x_up)     + \
                self.conv1x1_3x3(x_up) + \
                self.conv1x1_sbx(x_up) + \
                self.conv1x1_sby(x_up) + \
                self.conv1x1_lpl(x_up)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()

            #RK = RK.permute(1, 0, 2, 3)
            x = self.upsample(x)
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
            # y = F.conv_transpose2d(input=x, weight=RK, bias=RB, stride=2, padding=1, output_padding=1)

        y = y + skip
        if self.act_type != 'linear':
            y = self.act(y)
        #print(f'{self.phase} finished')

        return y# + skip

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        #if self.phase == 'Decoder':
        #    K0 = K0.permute(1, 0, 2, 3)
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


class ECB_Module(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, scale_multiplier, act_type='prelu', with_idt=False, train_repParam=False, phase='Encoder', with_proj=False):
        super(ECB_Module, self).__init__()
        assert phase in ("Encoder", "Decoder", "Conv"), "phase should be one of ('Encoder', 'Decoder', 'Conv')"

        module_list = []

        if depth_multiplier > 1:
            for i in range(depth_multiplier - 1):
                module_list.append(
                    ECB((i+1)*inp_planes, inp_planes, scale_multiplier, act_type=act_type, with_idt=with_idt, train_repParam=train_repParam, phase='Conv')
                )
        if phase in ("Encoder", "Conv"):
            module_list.append(
                ECB(depth_multiplier*inp_planes, out_planes, scale_multiplier, act_type=act_type, with_idt=with_idt, train_repParam=train_repParam, phase=phase)
            )
        else:
            module_list.append(
                ECB2(depth_multiplier*inp_planes, out_planes, scale_multiplier, act_type=act_type, with_idt=with_idt, train_repParam=train_repParam, phase=phase, with_proj=with_proj)
            )

        self.module_list = nn.Sequential(*module_list)

    def forward(self, x):
        temp = x.clone() if isinstance(x, torch.Tensor) else x[0].clone()
        out = {}
        for i in range(len(self.module_list)-1):
            out[str(i)] = self.module_list[i](temp)
            temp = torch.cat((temp, out[str(i)]), dim=1)
        y = self.module_list[-1](temp) if isinstance(x, torch.Tensor) else self.module_list[-1]((temp, x[1]))
        del temp
        return y


class _ECBSRUnetBranch(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False, use_upsampler=True, input_conv=False):
        super(_ECBSRUnetBranch, self).__init__()
        expand_channel = [expand_channel] * (layer_num+1) if type(expand_channel) in (int, float) else expand_channel
        assert len(expand_channel)==(layer_num+1)
        depth_factor = [depth_factor] * layer_num if type(depth_factor) in (int, float) else depth_factor
        assert len(depth_factor)==layer_num
        scale_factor = [scale_factor] * (layer_num+1) if type(scale_factor) in (int, float) else scale_factor
        assert len(scale_factor)==(layer_num+1)
        self.layer_num = layer_num
        self.use_upsampler = use_upsampler
        self.input_conv = input_conv
        if input_conv:
            self.input_conv = Conv2D(in_channel, in_channel, kernel_size=1, stride=1, padding=0, has_relu=True)

        self.conv0 = ECB(in_channel, int(in_channel*expand_channel[0]), depth_multiplier=scale_factor[0], act_type=act_type,  with_idt = with_idt, phase = 'Conv')
        self.encdec = ECB(int(in_channel*expand_channel[-1]), int(in_channel*expand_channel[-1]), depth_multiplier=scale_factor[-1], act_type=act_type, with_idt = with_idt, phase = 'Conv')
        self.out = ECB(int(in_channel*expand_channel[0]), out_channel, depth_multiplier=1.0, act_type='linear', with_idt = with_idt, phase = 'Conv')

        if self.use_upsampler:
            self.upsampler = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        for i in range(layer_num):
            enc = ECB_Module(int(in_channel*expand_channel[i]), int(in_channel*expand_channel[i+1]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Encoder')
            dec = ECB_Module(int(in_channel*expand_channel[-(i+1)]), int(in_channel*expand_channel[-(i+2)]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Decoder', with_proj=with_proj)
            setattr(self, f'enc{i+1}', enc)
            setattr(self, f'dec{i+1}', dec)

    def forward(self, input):
        x = self.input_conv(input) if self.input_conv else input
        enc_out = {
            '0': self.conv0(x)
        }
        for i in range(self.layer_num):
            enc_out[f'{i+1}'] = getattr(self, f'enc{i+1}')(enc_out[f'{i}'])

        dec_out = {'0': self.encdec(enc_out[f'{self.layer_num}'])}
        for i in range(self.layer_num):
            dec_out[f'{i+1}'] = getattr(self, f'dec{i+1}')((dec_out[f'{i}'], enc_out[f'{self.layer_num-(i+1)}']))

        y = dec_out[f'{self.layer_num}']
        if self.use_upsampler:
            y = self.upsampler(y)
        y = self.out(y)
        return y


def _to_stage_list(value, length):
    return [value] * length if isinstance(value, (int, float)) else list(value)


def _derive_freq_split_expand(base_expand, high_ratio=0.625):
    base_expand = list(base_expand)
    total_expand = [4.0 * float(v) for v in base_expand]
    high_expand = [v * high_ratio for v in total_expand]
    low_expand = [v * (1.0 - high_ratio) for v in total_expand]
    return high_expand, low_expand


class LiteY4UVEarlyStage(nn.Module):
    def __init__(self, in_ch=6, feat=32):
        super(LiteY4UVEarlyStage, self).__init__()

        hidden = feat // 2

        # 主特征
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.ReLU(),
        )

        # cheap feature (几乎无MAC)
        self.cheap = nn.Conv2d(hidden, hidden, 1, groups=hidden, bias=False)

        self.shortcut = nn.Conv2d(in_ch, feat, 1, bias=False)

    def forward(self, x):

        primary = self.primary(x)
        cheap = self.cheap(primary)

        out = torch.cat([primary, cheap], dim=1)

        return out + self.shortcut(x)


class NV21ToUnshuffled_spit(nn.Module):
    def __init__(self, with_conv=False):
        super(NV21ToUnshuffled_spit, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.with_conv = with_conv
        if with_conv:
            self.conv = Conv2D(4, 4, kernel_size=1, padding=0, stride=1, has_relu=True)

    def forward(self, input_tensor):
        B, C, H, W = input_tensor.shape
        assert C == 1, "Input tensor must have 1 channel for NV21 format."
        assert H % 3 == 0 and W % 2 == 0, "Height must be multiple of 3 and Width must be multiple of 2."
        H = H * 2 // 3  # Original height

        y_plane = input_tensor[:, :, :H, :]
        vu_plane = input_tensor[:, :, H:, :]
        v = vu_plane[:, :, :, 0::2]
        u = vu_plane[:, :, :, 1::2]
        y_unshuffled = self.unshuffle(y_plane)
        if self.with_conv:
            y_unshuffled = self.conv(y_unshuffled)
        return y_unshuffled, y_plane, u, v


class UnshuffledToYUV(nn.Module):
    def __init__(self):
        super(UnshuffledToYUV, self).__init__()
        self.shuffle = nn.PixelShuffle(2)
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, y4uv):
        return self.forward_train(y4uv) if self.training else self.forward_test(y4uv)

    def forward_train(self, y4uv):
        if y4uv.shape[1] == 6:
            y_unshuffled, u, v = y4uv[:, :4, ...], y4uv[:, 4:5, ...], y4uv[:, 5:, ...]
        elif y4uv.shape[1] == 12:
            y_unshuffled, u, v = y4uv[:, :4, ...], y4uv[:, 4:5, ...], y4uv[:, 8:9, ...]
        else:
            raise ValueError("Input tensor must have 6 or 12 channels.")

        y = self.shuffle(y_unshuffled)
        return y, u, v

    def forward_test(self, y4uv):
        y, u, v = self.forward_train(y4uv)
        uv = self.upsampler(torch.cat([u, v], dim=1))
        return torch.cat([y, uv], dim=1)


class Y4ToHaarLikeWavelet(nn.Module):
    """
    将 PixelUnshuffle(r=2) 后的 Y4 [B, 4, H, W] 转换为类 2D Haar 小波的 4 个分量。

    输出形状: [B, 4, H, W]
    通道顺序：
        0: LL        - 低频近似 (四个像素均值)
        1: Horizontal - 水平梯度 (垂直边缘)
        2: Vertical   - 竖直梯度 (水平边缘)
        3: Diagonal   - 对角梯度 (checkerboard / 对角线特征)

    归一化因子统一使用 /4，与均值保持一致的数值范围。
    支持可学习缩放（可选）。
    """
    def __init__(self, learnable_scale: bool=False):
        super(Y4ToHaarLikeWavelet, self).__init__()
        self.learnable_scale = learnable_scale

        if learnable_scale:
            # 可选：让每个分量有独立的缩放因子（初始化为1）
            self.scale = nn.Parameter(torch.ones(4), requires_grad=True)
        else:
            self.scale = None

    def forward(self, Y4: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y4: [B, 4, H, W]，来自 PixelUnshuffle(downscale_factor=2) 的结果

        Returns:
            wavelet: [B, 4, H, W]，四个小波-like 分量
        """
        # 最好提前检查输入通道数（调试时很有用）
        if Y4.shape[1] != 4:
            raise ValueError(f"Expected 4 channels from PixelUnshuffle, got {Y4.shape(1)}")

        ch0 = Y4[:, 0:1]   # TL
        ch1 = Y4[:, 1:2]   # TR
        ch2 = Y4[:, 2:3]   # BL
        ch3 = Y4[:, 3:4]   # BR

        LL = (ch0 + ch1 + ch2 + ch3) / 4.0
        LH = (ch0 - ch1 + ch2 - ch3) / 4.0
        HL = (ch0 + ch1 - ch2 - ch3) / 4.0
        HH = (ch0 - ch1 - ch2 + ch3) / 4.0

        bands = torch.cat([LL, LH, HL, HH], dim=1)

        if self.learnable_scale and self.scale is not None:
            # [B,4,H,W] * [4] → broadcasting
            bands = bands * self.scale.view(1, -1, 1, 1)

        return bands


class HaarLikeToY4(nn.Module):
    """
    将 [LL, LH, HL, HH] 逆变换回原始 Y4 [B,4,H,W]
    与 Y4ToHaarLikeWavelet 正向完全匹配（正向 /4 → 逆向不乘额外因子）
    """
    def __init__(self, learnable_scale: bool = False):
        super().__init__()
        self.learnable_scale = learnable_scale
        if learnable_scale:
            # 如果开启可学习缩放，建议初始化接近 1.0（因为正向已经 /4）
            self.scale = nn.Parameter(torch.ones(4), requires_grad=True)
        else:
            self.scale = None

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        if bands.shape[1] != 4:
            raise ValueError(f"Expected 4 bands, got {bands.shape[1]}")

        LL, LH, HL, HH = bands[:, 0:1], bands[:, 1:2], bands[:, 2:3], bands[:, 3:4]

        ch0 = LL + LH + HL + HH      # TL
        ch1 = LL - LH + HL - HH      # TR
        ch2 = LL + LH - HL - HH      # BL
        ch3 = LL - LH - HL + HH      # BR

        Y4 = torch.cat([ch0, ch1, ch2, ch3], dim=1)

        if self.learnable_scale and self.scale is not None:
            Y4 = Y4 * self.scale.view(1, -1, 1, 1)

        return Y4


class ECBSR(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors):
        super(ECBSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None


        self.conv0 = ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.enc1 = ECB(self.channel_nums, self.channel_nums * 2, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc2 = ECB(self.channel_nums * 2, self.channel_nums * 4, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc3 = ECB(self.channel_nums * 4, self.channel_nums * 8, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')

        self.encdec = ECB(self.channel_nums * 8, self.channel_nums * 8, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')

        self.dec1 = ECB2(self.channel_nums * 8, self.channel_nums * 4, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')
        self.dec2 = ECB2(self.channel_nums * 4, self.channel_nums * 2, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')
        self.dec3 = ECB2(self.channel_nums * 2, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')

        self.out0 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.out1 = ECB(self.channel_nums, 1, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt, phase = 'Conv')

        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        #self.upsample2 = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)


    def forward(self, deblur, blur=None):


        x_in = torch.concat([deblur, blur], dim=1)

        h,w = x_in.shape[-2:]
        ph = (8 - h%8) % 8
        pw = (8 - w%8) % 8
        x_in = F.pad(x_in, (0, pw, 0, ph), mode='reflect')


        conv0 = self.conv0(x_in) #16
        conv1 = self.enc1(conv0) #32
        conv2 = self.enc2(conv1) #64
        conv3 = self.enc3(conv2) #128

        conv5 = self.encdec(conv3) #128

        up2 = self.dec1((conv5, conv2)) #64
        up1 = self.dec2((up2, conv1))
        x0 = self.dec3((up1, conv0))


        y = self.upsample(x0)
        y = self.out0(y)
        y = self.out1(y)

        y = y[..., :h*self.scale, :w*self.scale]
        #blur_up = self.upsample2(blur)
        blur_up = self.upsample(blur)
        return y + blur_up

#
class ECBSRUnetV2(nn.Module):
    def __init__(self, in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale, with_proj=False):
        super(ECBSRUnetV2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel_nums = expand_channel
        self.module_nums = module_nums
        self.scale = scale
        self.with_idt = with_idt
        self.act_type = act_type

        self.input_layer = NV21ToUnshuffled_spit()
        self.conv0 = ECB(self.in_channel, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.enc1 = ECB(self.channel_nums, self.channel_nums * 2, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc2 = ECB(self.channel_nums * 2, self.channel_nums * 4, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc3 = ECB(self.channel_nums * 4, self.channel_nums * 8, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')

        self.encdec = ECB(self.channel_nums * 8, self.channel_nums * 8, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')

        self.dec1 = ECB2(self.channel_nums * 8, self.channel_nums * 4, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)
        self.dec2 = ECB2(self.channel_nums * 4, self.channel_nums * 2, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)
        self.dec3 = ECB2(self.channel_nums * 2, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)

        self.out0 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.out1 = ECB(self.channel_nums, 1, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt, phase = 'Conv')

        self.upsampler = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        #self.upsample2 = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)


    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)
        x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)

        conv0 = self.conv0(x) #16
        conv1 = self.enc1(conv0) #32
        conv2 = self.enc2(conv1) #64
        conv3 = self.enc3(conv2) #128

        conv5 = self.encdec(conv3) #128

        up2 = self.dec1((conv5, conv2)) #64
        up1 = self.dec2((up2, conv1))
        x0 = self.dec3((up1, conv0))


        y = self.upsampler(x0)
        y = self.out0(y)
        y = self.out1(y)
        y_skip = torch.cat([y_plane, self.upsampler(torch.cat([u_plane, v_plane], dim=1))], dim=1)
        y = y + y_skip
        return y


class ECBSRUnetV3(ECBSRUnetV2):
    def __init__(self, in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale, with_proj=False):
        super(ECBSRUnetV3, self).__init__(in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale)
        self.conv0 = ECB(self.in_channel, int(self.in_channel*1.5), depth_multiplier=2.0, act_type=self.act_type,  with_idt = self.with_idt, phase = 'Conv')
        self.enc1 = ECB(int(self.in_channel*1.5), self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc2 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc3 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')

        self.encdec = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')

        self.dec1 = ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)
        self.dec2 = ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)
        self.dec3 = ECB2(self.channel_nums, int(self.in_channel*1.5), depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)

        # self.out0 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.out1 = ECB(int(self.in_channel*1.5), self.out_channel, depth_multiplier=1.0, act_type='linear', with_idt = self.with_idt, phase = 'Conv')
        # self.out1 = Conv2D(self.channel_nums, self.out_channel, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)
        x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)

        conv0 = self.conv0(x) #16
        conv1 = self.enc1(conv0) #32
        conv2 = self.enc2(conv1) #64
        conv3 = self.enc3(conv2) #128

        conv5 = self.encdec(conv3) #128

        up2 = self.dec1((conv5, conv2)) #64
        up1 = self.dec2((up2, conv1))
        x0 = self.dec3((up1, conv0))


        y = self.upsampler(x0)
        # y = self.out0(y)
        y = self.out1(y)
        y_skip = torch.cat([y_plane, self.upsampler(torch.cat([u_plane, v_plane], dim=1))], dim=1)
        y = y + y_skip
        return y



class ECBSRUnetV3L(ECBSRUnetV3):
    def __init__(self, in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale, with_proj=False):
        super(ECBSRUnetV3, self).__init__(in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale)
        self.conv0 = ECB(self.in_channel, int(self.in_channel*1.5), depth_multiplier=2.0, act_type=self.act_type,  with_idt = self.with_idt, phase = 'Conv')
        self.enc1 = nn.Sequential(*[
            ECB(int(self.in_channel*1.5), self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder'),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])
        self.enc2 = nn.Sequential(*[
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder'),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])
        self.enc3 = nn.Sequential(*[
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder'),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])
        self.enc4 = nn.Sequential(*[
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder'),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])

        self.encdec = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')

        self.dec1 = nn.Sequential(*[
            ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])
        self.dec2 = nn.Sequential(*[
            ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])
        self.dec3 = nn.Sequential(*[
            ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj),
            ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        ])
        self.dec4 = ECB2(self.channel_nums, int(self.in_channel*1.5), depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder', with_proj=with_proj)

        # self.out0 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.out1 = ECB(int(self.in_channel*1.5), self.out_channel, depth_multiplier=1.0, act_type='linear', with_idt = self.with_idt, phase = 'Conv')
        # self.out1 = Conv2D(self.channel_nums, self.out_channel, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)
        x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)

        conv0 = self.conv0(x) #16
        conv1 = self.enc1(conv0) #32
        conv2 = self.enc2(conv1) #64
        conv3 = self.enc3(conv2) #128
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4) #128

        up3 = self.dec1((conv5, conv3)) #64
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        x0 = self.dec4((up1, conv0))


        y = self.upsampler(x0)
        # y = self.out0(y)
        y = self.out1(y)
        y_skip = torch.cat([y_plane, self.upsampler(torch.cat([u_plane, v_plane], dim=1))], dim=1)
        y = y + y_skip
        return y



class ECBSRUnet_Std(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False):
        super(ECBSRUnet_Std, self).__init__()
        expand_channel = [expand_channel] * (layer_num+1) if type(expand_channel) in (int, float) else expand_channel
        assert len(expand_channel)==(layer_num+1)
        depth_factor = [depth_factor] * layer_num if type(depth_factor) in (int, float) else depth_factor
        assert len(depth_factor)==layer_num
        scale_factor = [scale_factor] * (layer_num+1) if type(scale_factor) in (int, float) else scale_factor
        assert len(scale_factor)==(layer_num+1)
        self.layer_num = layer_num


        self.input_layer = NV21ToUnshuffled_spit()
        self.conv0 = nn.Sequential(
            Conv2D(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0, has_relu=True),
            ECB(in_channel, int(in_channel*expand_channel[0]), depth_multiplier=scale_factor[0], act_type=act_type,  with_idt = with_idt, phase = 'Conv')
        )

        self.encdec = ECB(int(in_channel*expand_channel[-1]), int(in_channel*expand_channel[-1]), depth_multiplier=scale_factor[-1], act_type=act_type, with_idt = with_idt, phase = 'Conv')
        self.out = ECB(int(in_channel*expand_channel[0]), out_channel, depth_multiplier=1.0, act_type='linear', with_idt = with_idt, phase = 'Conv')
        self.upsampler = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        for i in range(layer_num):
            enc = ECB_Module(int(in_channel*expand_channel[i]), int(in_channel*expand_channel[i+1]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Encoder')
            dec = ECB_Module(int(in_channel*expand_channel[-(i+1)]), int(in_channel*expand_channel[-(i+2)]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Decoder', with_proj=with_proj)
            setattr(self, f'enc{i+1}', enc)
            setattr(self, f'dec{i+1}', dec)

    def forward(self, input):
        # y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)
        # x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)

        x, y_plane, u_plane, v_plane = input[:]

        enc_out = {
            '0': self.conv0(x)
        }
        for i in range(self.layer_num):
            enc_out[f'{i+1}'] = getattr(self, f'enc{i+1}')(enc_out[f'{i}'])

        dec_out = {'0': self.encdec(enc_out[f'{self.layer_num}'])}
        for i in range(self.layer_num):
            dec_out[f'{i+1}'] = getattr(self, f'dec{i+1}')((dec_out[f'{i}'], enc_out[f'{self.layer_num-(i+1)}']))

        y = self.upsampler(dec_out[f'{self.layer_num}'])
        y = self.out(y)
        y_skip = torch.cat([y_plane, self.upsampler(torch.cat([u_plane, v_plane], dim=1))], dim=1)
        y = y + y_skip
        return y



class ECBSRUnet_StdV2(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False):
        super(ECBSRUnet_StdV2, self).__init__()
        expand_channel = [expand_channel] * (layer_num+1) if type(expand_channel) in (int, float) else expand_channel
        assert len(expand_channel)==(layer_num+1)
        depth_factor = [depth_factor] * layer_num if type(depth_factor) in (int, float) else depth_factor
        assert len(depth_factor)==layer_num
        scale_factor = [scale_factor] * (layer_num+1) if type(scale_factor) in (int, float) else scale_factor
        assert len(scale_factor)==(layer_num+1)
        self.layer_num = layer_num

        self.input_layer = NV21ToUnshuffled_spit()
        self.conv0 = nn.Sequential(
            Conv2D(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0, has_relu=True),
            ECB(in_channel, int(in_channel*expand_channel[0]), depth_multiplier=scale_factor[0], act_type=act_type,  with_idt = with_idt, phase = 'Conv'))
        # self.conv0 = LiteY4UVEarlyStage(in_ch=in_channel, feat=int(in_channel*expand_channel[0]))
        self.encdec = ECB(int(in_channel*expand_channel[-1]), int(in_channel*expand_channel[-1]), depth_multiplier=scale_factor[-1], act_type=act_type, with_idt = with_idt, phase = 'Conv')
        self.out = ECB(int(in_channel*expand_channel[0]), out_channel, depth_multiplier=1.0, act_type='linear', with_idt = with_idt, phase = 'Conv')
        # self.upsampler = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        for i in range(layer_num):
            enc = ECB_Module(int(in_channel*expand_channel[i]), int(in_channel*expand_channel[i+1]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Encoder')
            dec = ECB_Module(int(in_channel*expand_channel[-(i+1)]), int(in_channel*expand_channel[-(i+2)]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Decoder', with_proj=with_proj)
            setattr(self, f'enc{i+1}', enc)
            setattr(self, f'dec{i+1}', dec)

        self.out_layer = UnshuffledToYUV()

    def forward(self, input):
        # y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)

        x, y_plane, u_plane, v_plane = input[:]

        # u4 = u_plane.repeat(1, 4, 1, 1)
        # v4 = v_plane.repeat(1, 4, 1, 1)

        # x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)
        # x = torch.cat([y_unshuffled, u4, v4], dim=1)

        enc_out = {
            '0': self.conv0(x)
        }
        for i in range(self.layer_num):
            enc_out[f'{i+1}'] = getattr(self, f'enc{i+1}')(enc_out[f'{i}'])

        dec_out = {'0': self.encdec(enc_out[f'{self.layer_num}'])}
        for i in range(self.layer_num):
            dec_out[f'{i+1}'] = getattr(self, f'dec{i+1}')((dec_out[f'{i}'], enc_out[f'{self.layer_num-(i+1)}']))

        y = self.out(dec_out[f'{self.layer_num}'])
        y = y + x
        # return y
        return self.out_layer(y)



class ECBSRUnet_StdV3(ECBSRUnet_StdV2):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False):
        super(ECBSRUnet_StdV3, self).__init__(in_channel, out_channel, layer_num, expand_channel, depth_factor, scale_factor, with_idt, act_type, scale, with_proj)
        self.out_layer = nn.PixelShuffle(2)



class ECBSRUnet_StdV4(ECBSRUnet_StdV2):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False):
        super(ECBSRUnet_StdV4, self).__init__(in_channel, out_channel, layer_num, expand_channel, depth_factor, scale_factor, with_idt, act_type, scale, with_proj)
        expand_channel = [expand_channel] * (layer_num+1) if type(expand_channel) in (int, float) else expand_channel
        assert len(expand_channel)==(layer_num+1)
        depth_factor = [depth_factor] * layer_num if type(depth_factor) in (int, float) else depth_factor
        assert len(depth_factor)==layer_num
        scale_factor = [scale_factor] * (layer_num+1) if type(scale_factor) in (int, float) else scale_factor
        assert len(scale_factor)==(layer_num+1)
        self.layer_num = layer_num

        self.conv0 = nn.Sequential(
            Conv2D(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0, has_relu=True),
            ECB(in_channel, int(in_channel*expand_channel[0]), depth_multiplier=scale_factor[0], act_type=act_type,  with_idt = with_idt, phase = 'Conv')
        )
        self.out_layer = nn.PixelShuffle(2)



class ECBSRUnet_StdV5(ECBSRUnet_StdV4):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False, learnable_haar=False):
        super(ECBSRUnet_StdV5, self).__init__(in_channel, out_channel, layer_num, expand_channel, depth_factor, scale_factor, with_idt, act_type, scale, with_proj)

        self.y4_2_haar = Y4ToHaarLikeWavelet(learnable_scale=learnable_haar)
        self.haar_2_y4 = HaarLikeToY4(learnable_scale=learnable_haar)

    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)
        # y_unshuffled, y_plane, u_plane, v_plane = input[:]

        u4 = u_plane.repeat(1, 4, 1, 1)
        v4 = v_plane.repeat(1, 4, 1, 1)

        y_haar = self.y4_2_haar(y_unshuffled)

        x = torch.cat([y_haar, u4, v4], dim=1)
        # x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)

        enc_out = {
            '0': self.conv0(x)
        }
        for i in range(self.layer_num):
            enc_out[f'{i+1}'] = getattr(self, f'enc{i+1}')(enc_out[f'{i}'])

        dec_out = {'0': self.encdec(enc_out[f'{self.layer_num}'])}
        for i in range(self.layer_num):
            dec_out[f'{i+1}'] = getattr(self, f'dec{i+1}')((dec_out[f'{i}'], enc_out[f'{self.layer_num-(i+1)}']))

        y = self.out(dec_out[f'{self.layer_num}'])
        y = y + x

        y = torch.cat([self.haar_2_y4(y[:, :4]), y[:, 4:]], dim=1)

        return self.out_layer(y)



class ECBSRUnet_StdV5_FreqSplit(nn.Module):
    """
    Frequency-split variant derived from ECBSRUnet_StdV5.

    Pipeline:
    1. NV21 -> half-resolution Y4UV representation: [Y1, Y2, Y3, Y4, U, V]
    2. Y4 -> Haar-like subbands: [LL, LH, HL, HH]
    3. High-frequency branch processes [LH, HL, HH] with a heavier branch
    4. Low-frequency branch processes [LL, U, V] with a lighter branch
    5. Merge back to Y4UV and convert to YUV444 in eval mode

    The default branch widths are chosen such that:
        3 * high_expand + 3 * low_expand ~= 12 * base_expand
    which keeps the first-stage feature width close to StdV5 while biasing more
    capacity to the high-frequency path.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        layer_num=4,
        expand_channel=1.5,
        depth_factor=1,
        scale_factor=2,
        with_idt=False,
        act_type='prelu',
        scale=2,
        with_proj=False,
        learnable_haar=False,
        high_expand_channel=None,
        low_expand_channel=None,
        high_depth_factor=None,
        low_depth_factor=None,
        high_scale_factor=None,
        low_scale_factor=None,
        high_ratio=0.625,
        branch_input_conv=False,
    ):
        super(ECBSRUnet_StdV5_FreqSplit, self).__init__()
        self.input_layer = NV21ToUnshuffled_spit()
        self.out_layer = UnshuffledToYUV()
        self.y4_2_haar = Y4ToHaarLikeWavelet(learnable_scale=learnable_haar)
        self.haar_2_y4 = HaarLikeToY4(learnable_scale=learnable_haar)

        base_expand = _to_stage_list(expand_channel, layer_num + 1)
        base_depth = _to_stage_list(depth_factor, layer_num)
        base_scale = _to_stage_list(scale_factor, layer_num + 1)

        default_high_expand, default_low_expand = _derive_freq_split_expand(base_expand, high_ratio=high_ratio)
        high_expand = default_high_expand if high_expand_channel is None else _to_stage_list(high_expand_channel, layer_num + 1)
        low_expand = default_low_expand if low_expand_channel is None else _to_stage_list(low_expand_channel, layer_num + 1)
        high_depth = base_depth if high_depth_factor is None else _to_stage_list(high_depth_factor, layer_num)
        low_depth = base_depth if low_depth_factor is None else _to_stage_list(low_depth_factor, layer_num)
        high_scale = base_scale if high_scale_factor is None else _to_stage_list(high_scale_factor, layer_num + 1)
        low_scale = base_scale if low_scale_factor is None else _to_stage_list(low_scale_factor, layer_num + 1)

        self.high_branch = _ECBSRUnetBranch(
            in_channel=3,
            out_channel=3,
            layer_num=layer_num,
            expand_channel=high_expand,
            depth_factor=high_depth,
            scale_factor=high_scale,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
            use_upsampler=False,
            input_conv=branch_input_conv,
        )
        self.low_branch = _ECBSRUnetBranch(
            in_channel=3,
            out_channel=3,
            layer_num=layer_num,
            expand_channel=low_expand,
            depth_factor=low_depth,
            scale_factor=low_scale,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
            use_upsampler=False,
            input_conv=branch_input_conv,
        )

    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)

        y_haar = self.y4_2_haar(y_unshuffled)
        low_in = torch.cat([y_haar[:, :1], u_plane, v_plane], dim=1)
        high_in = y_haar[:, 1:]

        low_res = self.low_branch(low_in)
        high_res = self.high_branch(high_in)

        low_out = low_in + low_res
        high_out = high_in + high_res

        y_haar_out = torch.cat([low_out[:, :1], high_out], dim=1)
        y4_out = self.haar_2_y4(y_haar_out)
        y4uv_out = torch.cat([y4_out, low_out[:, 1:]], dim=1)

        return self.out_layer(y4uv_out)



class ECBSRUnet_StdV5_up(ECBSRUnet_StdV5):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False, learnable_haar=False):
        super(ECBSRUnet_StdV5_up, self).__init__(in_channel, out_channel, layer_num, expand_channel, depth_factor, scale_factor, with_idt, act_type, scale, with_proj)

        self.y4_2_haar = Y4ToHaarLikeWavelet(learnable_scale=learnable_haar)
        self.haar_2_y4 = HaarLikeToY4(learnable_scale=learnable_haar)
        self.up_layer = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        self.skip_layer = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)

    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)
        # y_unshuffled, yuv444, u_plane, v_plane = input[:]

        u4 = u_plane.repeat(1, 4, 1, 1)
        v4 = v_plane.repeat(1, 4, 1, 1)

        y_haar = self.y4_2_haar(y_unshuffled)

        x = torch.cat([y_haar, u4, v4], dim=1)
        # x = torch.cat([y_unshuffled, u_plane, v_plane], dim=1)

        enc_out = {
            '0': self.conv0(x)
        }
        for i in range(self.layer_num):
            enc_out[f'{i+1}'] = getattr(self, f'enc{i+1}')(enc_out[f'{i}'])

        dec_out = {'0': self.encdec(enc_out[f'{self.layer_num}'])}
        for i in range(self.layer_num):
            dec_out[f'{i+1}'] = getattr(self, f'dec{i+1}')((dec_out[f'{i}'], enc_out[f'{self.layer_num-(i+1)}']))

        y = self.up_layer(dec_out[f'{self.layer_num}'] + x)
        y = self.out(y)

        y = torch.cat([self.haar_2_y4(y[:, :4]), y[:, 4:]], dim=1)
        skip = self.skip_layer(torch.cat([y_plane, self.skip_layer(torch.cat([u_plane, v_plane], dim=1))], dim=1))

        return self.out_layer(y) + skip



class ECBSRUnet_RGB_Std(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num=4, expand_channel=1.5, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False, pixel_shuffle=False):
        super(ECBSRUnet_RGB_Std, self).__init__()
        expand_channel = [expand_channel] * (layer_num+1) if type(expand_channel) in (int, float) else expand_channel
        assert len(expand_channel)==(layer_num+1)
        depth_factor = [depth_factor] * layer_num if type(depth_factor) in (int, float) else depth_factor
        assert len(depth_factor)==layer_num
        scale_factor = [scale_factor] * (layer_num+1) if type(scale_factor) in (int, float) else scale_factor
        assert len(scale_factor)==(layer_num+1)
        self.layer_num = layer_num

        if pixel_shuffle:
            in_channel *= 4
            out_channel *= 4
            expand_channel[0] /= 2
            expand_channel[-1] /= 2
            self.input_layer = nn.PixelUnshuffle(2)
            self.output_layer = nn.PixelShuffle(2)
        else:
            self.input_layer = nn.Identity()
            self.output_layer = nn.Identity()

        self.conv0 = ECB(in_channel, int(in_channel*expand_channel[0]), depth_multiplier=scale_factor[0], act_type=act_type,  with_idt = with_idt, phase = 'Conv')
        self.encdec = ECB(int(in_channel*expand_channel[-1]), int(in_channel*expand_channel[-1]), depth_multiplier=scale_factor[-1], act_type=act_type, with_idt = with_idt, phase = 'Conv')
        self.out = ECB(int(in_channel*expand_channel[0]), out_channel, depth_multiplier=1.0, act_type='linear', with_idt = with_idt, phase = 'Conv')

        for i in range(layer_num):
            enc = ECB_Module(int(in_channel*expand_channel[i]), int(in_channel*expand_channel[i+1]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Encoder')
            dec = ECB_Module(int(in_channel*expand_channel[-(i+1)]), int(in_channel*expand_channel[-(i+2)]), depth_multiplier=depth_factor[i], scale_multiplier=scale_factor[i+1], act_type=act_type, with_idt=with_idt, phase='Decoder', with_proj=with_proj)
            setattr(self, f'enc{i+1}', enc)
            setattr(self, f'dec{i+1}', dec)

    def forward(self, input):
        x = self.input_layer(input)
        enc_out = {
            '0': self.conv0(x)
        }
        for i in range(self.layer_num):
            enc_out[f'{i+1}'] = getattr(self, f'enc{i+1}')(enc_out[f'{i}'])

        dec_out = {'0': self.encdec(enc_out[f'{self.layer_num}'])}
        for i in range(self.layer_num):
            dec_out[f'{i+1}'] = getattr(self, f'dec{i+1}')((dec_out[f'{i}'], enc_out[f'{self.layer_num-(i+1)}']))

        y = dec_out[f'{self.layer_num}']
        y = self.out(y)
        y = self.output_layer(y)
        y_skip = input
        y = y + y_skip
        return y



class ECBSRUnet_Std_YUVSplit(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num=4, y_expand_channel=2.0, uv_expand_channel=1.0, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=False, uv_in=2):
        super(ECBSRUnet_Std_YUVSplit, self).__init__()
        self.input_layer = NV21ToUnshuffled_spit()

        y_in = in_channel - uv_in
        self.y_branch = _ECBSRUnetBranch(
            in_channel=y_in,
            out_channel=1,
            layer_num=layer_num,
            expand_channel=y_expand_channel,
            depth_factor=depth_factor,
            scale_factor=scale_factor,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
            use_upsampler=False,
        )
        self.uv_branch = _ECBSRUnetBranch(
            in_channel=uv_in,
            out_channel=2,
            layer_num=layer_num,
            expand_channel=uv_expand_channel,
            depth_factor=depth_factor,
            scale_factor=scale_factor,
            with_idt=with_idt,
            act_type=act_type,
            scale=scale,
            with_proj=with_proj,
            use_upsampler=True,
        )
        self.upsampler = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, input):
        y_unshuffled, y_plane, u_plane, v_plane = self.input_layer(input)

        y_pred = self.y_branch(y_plane)
        uv_pred = self.uv_branch(torch.cat([u_plane, v_plane], dim=1))

        y_out = y_pred + y_plane
        uv_skip = self.upsampler(torch.cat([u_plane, v_plane], dim=1))
        uv_out = uv_pred + uv_skip
        y = torch.cat([y_out, uv_out], dim=1)
        return y


class RRDB_ECBUnet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        num_rrdb=3,
        rrdb_nf=64,
        rrdb_gc=32,
        num_layer=4,
        ecb_depth=3,
        ecb_scale=2,
        with_idt=False,
        act_type='prelu',
        with_proj=False,
        scale=2,
    ):
        super(RRDB_ECBUnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rrdb = num_rrdb
        self.rrdb_nf = rrdb_nf
        self.rrdb_gc = rrdb_gc
        self.num_layer = num_layer
        self.scale = scale

        self.unshuffle = nn.PixelUnshuffle(2)
        in_feat = in_channels * 4

        # Pre RRDB: conv_first + RRDB trunk + conv_body (follows RRDBNet pattern)
        self.pre_conv_first = nn.Conv2d(in_feat, rrdb_nf, 3, 1, 1)
        self.pre_rrdb = nn.Sequential(*[ResidualDenseBlock(rrdb_nf, rrdb_gc) for _ in range(num_rrdb)])
        self.pre_conv_body = nn.Conv2d(rrdb_nf, rrdb_nf, 3, 1, 1)

        # Build a lightweight Unet using ECB_Module blocks; keep feature channels = rrdb_nf
        self.enc_modules = nn.ModuleList()
        self.dec_modules = nn.ModuleList()
        for i in range(num_layer):
            enc = ECB_Module(rrdb_nf, rrdb_nf, depth_multiplier=ecb_depth, scale_multiplier=ecb_scale, act_type=act_type, with_idt=with_idt, phase='Encoder')
            dec = ECB_Module(rrdb_nf, rrdb_nf, depth_multiplier=ecb_depth, scale_multiplier=ecb_scale, act_type=act_type, with_idt=with_idt, phase='Decoder', with_proj=with_proj)
            self.enc_modules.append(enc)
            self.dec_modules.append(dec)

        self.encdec = ECB(rrdb_nf, rrdb_nf, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt, phase='Conv')

        # Post RRDB (same layout as pre)
        self.post_conv_first = nn.Conv2d(rrdb_nf, rrdb_nf, 3, 1, 1)
        self.post_rrdb = nn.Sequential(*[ResidualDenseBlock(rrdb_nf, rrdb_gc) for _ in range(num_rrdb)])
        self.post_conv_body = nn.Conv2d(rrdb_nf, rrdb_nf, 3, 1, 1)

        # Final conv to produce out_channels * (scale^2) and PixelShuffle to upsample
        self.conv_up = Conv2D(rrdb_nf, rrdb_nf, 3, 1, 1, has_relu=True)
        # if scale == 2:
        #     self.conv_up2 = Conv2D(rrdb_nf, rrdb_nf, 3, 1, 1, has_relu=True)

        self.conv_hr = Conv2D(rrdb_nf, rrdb_nf, 3, 1, 1, has_relu=True)
        self.conv_last = nn.Conv2d(rrdb_nf, out_channels, 3, 1, 1)
        # self.pixel_shuffle = nn.PixelShuffle(scale)

        # small upsampler used inside decoders if needed
        self.upsampler = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        returns: [B, out_channels, H, W]
        """
        # downsample via pixel unshuffle
        feat = self.unshuffle(x)

        # pre rrdb trunk
        feat0 = self.pre_conv_first(feat)
        body = self.pre_rrdb(feat0)
        pre_feat = feat0 + self.pre_conv_body(body)          # Total MAC: 569M

        # UNet encoder
        enc_out = {'0': pre_feat}
        for i in range(self.num_layer):
            enc_out[f'{i+1}'] = self.enc_modules[i](enc_out[f'{i}'])

        # bottleneck
        dec_in = self.encdec(enc_out[f'{self.num_layer}'])

        # UNet decoder
        dec_out = {'0': dec_in}
        for i in range(self.num_layer):
            # pairing with corresponding encoder output
            skip = enc_out[f'{self.num_layer-(i+1)}']
            dec_out[f'{i+1}'] = self.dec_modules[i]((dec_out[f'{i}'], skip))

        unet_out = dec_out[f'{self.num_layer}']                 # Total MAC: 592M

        # post rrdb trunk
        post_first = self.post_conv_first(unet_out)
        post_body = self.post_rrdb(post_first)
        post_feat = post_first + self.post_conv_body(post_body)

        # skip connection between pre and post rrdb outputs
        merged = post_feat + pre_feat           # Total MAC: 1.13G

        # final mapping and pixelshuffle upsampling

        if self.scale == 1:
            out = self.conv_last(self.conv_hr(self.upsampler(merged)))
        elif self.scale == 2:
            out = self.conv_up(self.upsampler(merged))
            out = self.conv_last(self.conv_hr(self.upsampler(out)))

        return out
if __name__ == '__main__':

    device = torch.device('cpu')
    # model = ECBSRUnet_RGB_Std(in_channel=3, out_channel=3, layer_num=3, expand_channel=2, depth_factor=1, scale_factor=2, with_idt=False, act_type='prelu', scale=1, with_proj=True, pixel_shuffle=True).to(device)
    """
    type: ECBSRUnet_StdV5_FreqSplit
    in_channel: 12
    out_channel: 12
    layer_num: 4
    expand_channel: 1.5
    depth_factor: 1
    scale_factor: 2
    with_idt: false
    act_type: prelu
    scale: 2
    with_proj: true
    learnable_haar: false
    """
    # model = ECBSRUnet_StdV5(in_channel=12, out_channel=12, layer_num=4, expand_channel=[2, 4, 8, 16, 32], depth_factor=4, scale_factor=2, with_idt=False, act_type='prelu', scale=2, with_proj=True).to(device)
    # model = RRDBNetNV21(num_in_ch=12, num_out_ch=12, num_feat=12, num_block=3, num_grow_ch=4, scale=1).to(device)
    # model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=16, num_block=3, num_grow_ch=4, scale=1).to(device)
    # model = RRDBNet(num_in_ch=12, num_out_ch=12, num_feat=64, num_block=23, num_grow_ch=32, scale=1).to(device)
    model = RRDB_ECBUnet(in_channels=4, out_channels=4, num_rrdb=1, rrdb_nf=12, rrdb_gc=4, num_layer=4, ecb_depth=3, ecb_scale=2, with_idt=False, act_type='prelu', with_proj=True, scale=1)
    # model.eval()
    # x1 = torch.rand(1, 1, 768, 512).to(device)
    x1 = torch.rand(1, 4, 512, 512).to(device)
    # x1 = torch.rand(1, 12, 256, 256).to(device)
    # x1 = [torch.rand(1, 4, 256, 256).to(device), torch.rand(1, 3, 512, 512).to(device), torch.rand(1, 1, 256, 256).to(device), torch.rand(1, 1, 256, 256).to(device)]
    y = model(x1)
    print(y.shape)


    from torchinfo import summary
    summary(model, input_data=[x1], device="cpu")

    from thop import profile

    # model = ECBSRUnetV3L(in_channel=6, out_channel=3, expand_channel=32, module_nums=4, with_idt=False, act_type='prelu', scale=2, with_proj=True).to(device)
    # model.eval()
    # x1 = torch.rand(1, 1, 768, 512)
    flops, params = profile(model, inputs=[x1])
    print("FLOPs:", flops / 1e6, "M")
    print("Params:", params)

    pass


    # y = torch.tensor([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.], [11., 12., 15., 16.]]]])  # [B, C, H, W] = [1, 1, 2, 2]
    # print("Input:\n", y)
    # haar = Y4ToHaarLikeWavelet(learnable_scale=False)
    # y4 = F.pixel_unshuffle(y, downscale_factor=2)  # [1, 4, 1, 1]
    # print("Y4:\n", y4)
    # haar_out = haar(y4)
    # print("Haar-like Wavelet Output:\n", haar_out)

    # def haar_dwt_2d(y: torch.Tensor) -> torch.Tensor:
    #     """
    #     输入: y [B,1,H,W] 或 [B,H,W]
    #     输出: y4_like [B,4, H//2, W//2]  (LL, LH, HL, HH)
    #     """
    #     if y.dim() == 3:
    #         y = y.unsqueeze(1)  # [B,1,H,W]

    #     # 行方向低/高通 (Haar)
    #     # low = (y[:,:,0::2,:] + y[:,:,1::2,:]) / √2
    #     # high = (y[:,:,0::2,:] - y[:,:,1::2,:]) / √2
    #     # 但为了数值稳定，常省略√2或只除2
    #     y_even = y[:,:,0::2,:]   # [B,1,H/2,W]
    #     y_odd  = y[:,:,1::2,:]

    #     L = (y_even + y_odd) * 0.5
    #     H = (y_even - y_odd) * 0.5

    #     # 列方向
    #     L_even = L[:,:,:,0::2]
    #     L_odd  = L[:,:,:,1::2]
    #     H_even = H[:,:,:,0::2]
    #     H_odd  = H[:,:,:,1::2]

    #     LL = (L_even + L_odd) * 0.5
    #     LH = (L_even - L_odd) * 0.5
    #     HL = (H_even + H_odd) * 0.5
    #     HH = (H_even - H_odd) * 0.5

    #     # 堆叠成 [B,4, H/2, W/2]
    #     return torch.cat([LL, LH, HL, HH], dim=1)

    # print("Haar DWT Output:\n", haar_dwt_2d(y))
