import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from ecb import ECB
# from Jieer ecb.py + ecbsr.py
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
            self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1, stride=2)
        else:
            self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)

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
        self.RK, self.RB = self.rep_params()

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
            # RK, RB = self.rep_params()
            if self.phase =='Encoder':
                y = F.conv2d(input=x, weight=self.RK, bias=self.RB, stride=2, padding=1)
            else:
                y = F.conv2d(input=x, weight=self.RK, bias=self.RB, stride=1, padding=1)

        if self.act_type != 'linear':
            y = self.act(y)
        #print(f'{self.phase} finished')
        #print(f'{x.shape}')

        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        if self.phase == 'Decoder':
            K0 = K0.permute(1, 0, 2, 3)
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
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False, train_repParam=False, phase='Encoder'):
        super(ECB2, self).__init__()

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


        #self.conv3x3 = nn.ConvTranspose2d(self.inp_planes, self.out_planes, 3, stride=2, padding=1, output_padding=1)
        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)

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

        self.RK, self.RB = self.rep_params()

    def forward(self, inputs):
        x, skip = inputs
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
            # RK, RB = self.rep_params()

            #RK = RK.permute(1, 0, 2, 3)
            x = self.upsample(x)
            y = F.conv2d(input=x, weight=self.RK, bias=self.RB, stride=1, padding=1)
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


class NV21ToUnshuffled_spit(nn.Module):
    def __init__(self):
        super(NV21ToUnshuffled_spit, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(2)

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
        return y_unshuffled, y_plane, u, v


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


class ECBSRUnetV2(nn.Module):
    def __init__(self, in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale):
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

        self.dec1 = ECB2(self.channel_nums * 8, self.channel_nums * 4, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')
        self.dec2 = ECB2(self.channel_nums * 4, self.channel_nums * 2, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')
        self.dec3 = ECB2(self.channel_nums * 2, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')

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
    def __init__(self, in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale):
        super(ECBSRUnetV3, self).__init__(in_channel, out_channel, expand_channel, module_nums, with_idt, act_type, scale)
        self.conv0 = ECB(self.in_channel, int(self.channel_nums//2), depth_multiplier=2.0, act_type=self.act_type,  with_idt = self.with_idt, phase = 'Conv')
        self.enc1 = ECB(int(self.channel_nums//2), self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc2 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')
        self.enc3 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Encoder')

        self.encdec = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')

        self.dec1 = ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')
        self.dec2 = ECB2(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')
        self.dec3 = ECB2(self.channel_nums, int(self.channel_nums//2), depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Decoder')

        # self.out0 = ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt, phase = 'Conv')
        self.out1 = ECB(int(self.channel_nums//2), self.out_channel, depth_multiplier=1.0, act_type='linear', with_idt = self.with_idt, phase = 'Conv')
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

if __name__ == '__main__':

    device = torch.device('cpu')
    model = ECBSRUnetV3(in_channel=6, out_channel=3, expand_channel=16, module_nums=4, with_idt=False, act_type='prelu', scale=2).to(device)
    model.eval()
    x1 = torch.rand(1, 1, 768, 512)
    y = model(x1.to(device))
    print(y.shape)


    from torchinfo import summary
    summary(model, input_data=[x1], device="cpu")

    from thop import profile

    model = ECBSRUnetV3(in_channel=6, out_channel=3, expand_channel=16, module_nums=4, with_idt=False, act_type='prelu', scale=2).to(device)
    model.eval()
    x1 = torch.rand(1, 1, 768, 512)
    flops, params = profile(model, inputs=[x1])
    print("FLOPs:", flops / 1e9, "G")
    print("Params:", params)

    pass

