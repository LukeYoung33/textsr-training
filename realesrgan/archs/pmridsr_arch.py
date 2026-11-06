import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import sys
sys.path.append('.')
from basicsr.utils.registry import ARCH_REGISTRY
import pdb
from realesrgan.utils import YUV2RGB, RGB2YUV, YUV2RGBConv, nv21_to_yuv444, tensor2img_feat
from basicsr.utils import DiffJPEG, USMSharp, imwrite, tensor2img

import cv2

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


class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1,kernel_size: int =5,same_channel: bool=False, is_seperable=False,):
        super().__init__()

        if kernel_size ==5:
            padding = 2
        elif kernel_size == 3:
            padding = 1
        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, is_seperable=is_seperable, has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, is_seperable=is_seperable, has_relu=False)

        if same_channel:
            self.proj = (
                nn.Identity()
                if stride == 1 else
                Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=is_seperable, has_relu=False)
            )
        else:
            self.proj = (
                nn.Identity()
                if stride == 1 and in_channels == out_channels else
                Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=is_seperable,
                       has_relu=False)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int,kernel_size: int,same_channel: bool, is_seperable: bool=False):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//4,
            out_channels=out_channels,
            stride=2,
            kernel_size = kernel_size,
            same_channel=same_channel,
            is_seperable=is_seperable,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=1,
                kernel_size=kernel_size,
                same_channel=same_channel,
                is_seperable=is_seperable,
            )
        )

    return nn.Sequential(*blocks)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, decoder_skip: bool= True, is_seperable=False):
        super().__init__()

        padding = kernel_size // 2
        self.decoder_skip = decoder_skip
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=is_seperable, has_relu=True,
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=is_seperable, has_relu=False,
        )
        if in_channels != out_channels:
            self.proj = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=False, has_relu=False,
        )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        if self.decoder_skip:
            x = x + self.proj(inp)
        return x


class DecoderStage(nn.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int,fast_upsample: bool,decoder_skip:bool, is_seperable: bool = False):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3,decoder_skip=decoder_skip, is_seperable=is_seperable)
        if fast_upsample:
            blocks = []
            if in_channels != out_channels:
                blocks.append(Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False,
                                     has_relu=False))

            blocks.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.upsample = nn.Sequential(*blocks)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=is_seperable, has_relu=True)
        # M.init.msra_normal_(self.upsample.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y


class DecoderStageV2(DecoderStage):
    def __init__(self, in_channels, skip_in_channels, out_channels, fast_upsample, decoder_skip, is_seperable: bool = False):
        super(DecoderStageV2, self).__init__(in_channels, skip_in_channels, out_channels, fast_upsample, decoder_skip, is_seperable)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_seperable=is_seperable, has_relu=False)

class DecoderStageV3(DecoderStage):
    def __init__(self, in_channels, skip_in_channels, out_channels, fast_upsample, decoder_skip):
        super(DecoderStageV3, self).__init__(in_channels, skip_in_channels, out_channels, fast_upsample, decoder_skip)
        self.proj_conv = nn.Identity()


class NV212Unshuffled(nn.Module):
    def __init__(self) -> None:
        super(NV212Unshuffled, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
    
    def forward(self, input_tensor):
        B, C, H_total, W = input_tensor.shape
        assert C == 1, "Input tensor must have shape (B, 1, H*3/2, W)"
        assert H_total % 3 == 0, "Height of input tensor must be divisible by 3"
        assert W % 2 == 0, "Width of input tensor must be even"
        H = H_total * 2 // 3

        # Extract Y plane
        Y = input_tensor[:, :, :H, :]
        # Extrace UV plane
        UV = input_tensor[:, :, H:, :]      # .reshape(B, 2, H // 2, W // 2)
        v = UV[:, :, :, 0::2]
        u = UV[:, :, :, 1::2]

        y_unshuffled = self.unshuffle(Y)  # (B, 4, H/2, W/2)
        output = torch.cat([y_unshuffled, u, v], dim=1)  # (B, 6, H/2, W/2)
        return output


class NV212Unshuffled_split(nn.Module):
    def __init__(self) -> None:
        super(NV212Unshuffled_split, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
    
    def forward(self, input_tensor):
        B, C, H_total, W = input_tensor.shape
        assert C == 1, "Input tensor must have shape (B, 1, H*3/2, W)"
        assert H_total % 3 == 0, "Height of input tensor must be divisible by 3"
        assert W % 2 == 0, "Width of input tensor must be even"
        H = H_total * 2 // 3

        # Extract Y plane
        Y = input_tensor[:, :, :H, :]
        # Extrace UV plane
        UV = input_tensor[:, :, H:, :]      # .reshape(B, 2, H // 2, W // 2)
        v = UV[:, :, :, 0::2]
        u = UV[:, :, :, 1::2]

        y_unshuffled = self.unshuffle(Y)  # (B, 4, H/2, W/2)
        return y_unshuffled, Y, u, v  # (B, 6, H/2, W/2)


class YUV2RGBConv(nn.Module):
    def __init__(self):
        super(YUV2RGBConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        weight1 = torch.tensor([
            [[[1.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
            [[[0.0]], [[0.0]], [[1.0]]]
        ])
        weight2 = torch.tensor([
            [[[1.0]], [[0.0]], [[1.13983]]],
            [[[1.0]], [[-0.39465]], [[-0.580607]]],
            [[[1.0]], [[2.03211]], [[0.0]]]
        ])
        self.conv1.weight.data = nn.Parameter(weight1)
        bias = torch.tensor([0.0, -0.5, -0.5])
        self.conv1.bias.data = bias
        self.conv2.weight.data = nn.Parameter(weight2)
        self.conv1.requires_grad_(False)
        self.conv2.requires_grad_(False)

    def forward(self, yuv: torch.Tensor):
        return self.conv2(self.conv1(yuv))

@ARCH_REGISTRY.register()
class PMRIDSRArch(nn.Module):
    def __init__(self, in_channels, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super().__init__()
        self.conv0 = Conv2D(in_channels=in_channels, out_channels=feat_base, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base, out_channels=feat_base, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=feat_base, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
        if fast_upsample:
            self.uplayer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners='False')
        else:
            self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, inp):
        # inp = self._check_and_padding(inp)

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        # print("up1", up1.shape)
        # print("conv0", conv0.shape)
        # print("inp", inp.shape)
        x0 = self.dec4((up1, conv0))

        x1 = self.out0(x0)
        x = self.out1(x1)
        pred = x + self.uplayer(inp)

        return pred


@ARCH_REGISTRY.register()
class PMRIDSRV2Arch(nn.Module):
    def __init__(self, in_channels, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super().__init__()
        self.conv0 = Conv2D(in_channels=in_channels, out_channels=feat_base, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0], is_seperable=True)
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1], is_seperable=True)
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2], is_seperable=True)
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3], is_seperable=True)

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip, is_seperable=True)
        self.dec2 = DecoderStage(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip, is_seperable=True)
        self.dec3 = DecoderStage(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip, is_seperable=True)
        self.dec4 = DecoderStage(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip, is_seperable=True)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3, is_seperable=True)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        # self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
        # if fast_upsample:
        #     self.uplayer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # else:
        #     self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, inp, label_mask=None):
        # inp = self._check_and_padding(inp)

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        # print("up1", up1.shape)
        # print("conv0", conv0.shape)
        # print("inp", inp.shape)
        x0 = self.dec4((up1, conv0))

        x1 = self.out0(x0)
        pred = x1 + inp
        # x = self.out1(x1)
        # pred = x + self.uplayer(inp)

        return pred


@ARCH_REGISTRY.register()
class PMRIDSRV3Arch(nn.Module):
    def __init__(self, in_channels, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super().__init__()
        self.conv0 = Conv2D(in_channels=in_channels, out_channels=feat_base, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
        # if fast_upsample:
            # self.out1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
            # self.uplayer = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        # else:
        #     self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
        #     self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        self.out = nn.Sequential(
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

    def forward(self, inp, label_mask=None):
        # inp = self._check_and_padding(inp)

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        conv2 = self.dec2((conv3, conv2))
        conv1 = self.dec3((conv2, conv1))
        conv0 = self.dec4((conv1, conv0))

        # conv0 = self.out1(self.out0(conv0))
        # pred = self.out(conv0+ self.uplayer(inp))
        conv0 = self.out0(conv0)
        pred = self.out(conv0 + inp)
        cv2.imwrite(f'debug/out1_V3RGB.png', tensor2img_feat(conv0+ inp))
        cv2.imwrite(f'debug/pred_V3RGB.png', tensor2img_feat(pred))

        return pred
    

@ARCH_REGISTRY.register()
class PMRIDSRV3NV21Arch(nn.Module):
    def __init__(self, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super(PMRIDSRV3NV21Arch, self).__init__()
        self.input_layer = NV212Unshuffled()
        self.conv0 = Conv2D(in_channels=6, out_channels=feat_base, kernel_size=1, padding=0, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
        if fast_upsample:
            self.out1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # self.uplayer = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        else:
            self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
            # self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        self.out = nn.Sequential(
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

    def forward(self, inp, label_mask=None):
        inp = self.input_layer(inp)
        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        conv2 = self.dec2((conv3, conv2))
        conv1 = self.dec3((conv2, conv1))
        conv0 = self.dec4((conv1, conv0))

        conv0 = self.out1(self.out0(conv0))
        pred = self.out(conv0)

        return pred


@ARCH_REGISTRY.register()
class PMRIDSRV3NV21skipArch(nn.Module):
    def __init__(self, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super(PMRIDSRV3NV21skipArch, self).__init__()
        self.input_layer = NV212Unshuffled_split()
        self.conv0 = Conv2D(in_channels=6, out_channels=feat_base, kernel_size=1, padding=0, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
        if fast_upsample:
            self.out1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.uplayer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
            # self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        self.out = nn.Sequential(
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

    def forward(self, inp, label_mask=None):
        cvtColor = YUV2RGBConv().to(inp.device)
        y_unshuffled, y, u, v = self.input_layer(inp)
        conv0 = self.conv0(torch.cat([y_unshuffled,u,v], dim=1))
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        for c in range(192):
            cv2.imwrite(f'debug/dec1_{c}.png', tensor2img_feat(conv3[:, c:c+1, ...]))
        conv2 = self.dec2((conv3, conv2))
        for c in range(128):
            cv2.imwrite(f'debug/dec2_{c}.png', tensor2img_feat(conv2[:, c:c+1, ...]))
        conv1 = self.dec3((conv2, conv1))
        for c in range(64):
            cv2.imwrite(f'debug/dec3_{c}.png', tensor2img_feat(conv1[:, c:c+1, ...]))
        conv0 = self.dec4((conv1, conv0))
        for c in range(32):
            cv2.imwrite(f'debug/dec4_{c}.png', tensor2img_feat(conv0[:, c:c+1, ...]))
        yuv_skip = torch.cat([y, self.uplayer(torch.cat([u, v], dim=1))], dim=1)
        # pdb.set_trace()
        
        out0 = self.out0(conv0)
        for c in range(3):
            cv2.imwrite(f'debug/out0_{c}.png', tensor2img_feat(out0[:, c:c+1, ...]))
        cv2.imwrite(f'debug/out0_show.png', tensor2img_feat(cvtColor(out0)))
        out1 = self.out1(out0)
        cv2.imwrite(f'debug/out0_up_show.png', tensor2img_feat(cvtColor(out1)))
        pred = self.out(yuv_skip + out1)

        return pred


@ARCH_REGISTRY.register()
class PMRIDSRV4NV21skipArch(nn.Module):
    def __init__(self, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super(PMRIDSRV4NV21skipArch, self).__init__()
        self.input_layer = NV212Unshuffled_split()
        self.conv0 = Conv2D(in_channels=6, out_channels=feat_base, kernel_size=1, padding=0, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.dec1 = DecoderStageV2(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStageV2(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStageV2(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStageV2(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
        if fast_upsample:
            self.out1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.uplayer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
            # self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        # self.out = nn.Sequential(
        #     DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
        #     DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        # )

    def forward(self, inp, label_mask=None):
        cvtColor = YUV2RGBConv().to(inp.device)
        y_unshuffled, y, u, v = self.input_layer(inp)
        conv0 = self.conv0(torch.cat([y_unshuffled,u,v], dim=1))
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        for c in range(192):
            cv2.imwrite(f'debug/dec1_{c}.png', tensor2img_feat(conv3[:, c:c+1, ...]))
        conv2 = self.dec2((conv3, conv2))
        for c in range(128):
            cv2.imwrite(f'debug/dec2_{c}.png', tensor2img_feat(conv2[:, c:c+1, ...]))
        conv1 = self.dec3((conv2, conv1))
        for c in range(64):
            cv2.imwrite(f'debug/dec3_{c}.png', tensor2img_feat(conv1[:, c:c+1, ...]))
        conv0 = self.dec4((conv1, conv0))
        for c in range(32):
            cv2.imwrite(f'debug/dec4_{c}.png', tensor2img_feat(conv0[:, c:c+1, ...]))
        yuv_skip = torch.cat([y, self.uplayer(torch.cat([u, v], dim=1))], dim=1)
        # pdb.set_trace()
        
        out0 = self.out0(conv0)
        # for c in range(3):
        #     cv2.imwrite(f'debug/out0_{c}.png', tensor2img_feat(out0[:, c:c+1, ...]))
        cv2.imwrite(f'debug/out0_show.png', tensor2img_feat(cvtColor(out0)))
        out1 = self.out1(out0)
        cv2.imwrite(f'debug/out0_up_show.png', tensor2img_feat(cvtColor(out1)))
        pred = yuv_skip + out1

        return pred


@ARCH_REGISTRY.register()
class PMRIDSRV5NV21skipArch(nn.Module):
    def __init__(self, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super(PMRIDSRV5NV21skipArch, self).__init__()
        self.input_layer = NV212Unshuffled_split()
        self.conv0 = Conv2D(in_channels=6, out_channels=feat_base, kernel_size=1, padding=0, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
        if fast_upsample:
            self.out1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.uplayer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
            # self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        # self.out = nn.Sequential(
        #     DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
        #     DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        # )

    def forward(self, inp, label_mask=None):
        cvtColor = YUV2RGBConv().to(inp.device)
        y_unshuffled, y, u, v = self.input_layer(inp)
        conv0 = self.conv0(torch.cat([y_unshuffled,u,v], dim=1))
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        # for c in range(conv3.shape[1]):
        #     cv2.imwrite(f'debug/dec1_{c}.png', tensor2img_feat(conv3[:, c:c+1, ...]))
        conv2 = self.dec2((conv3, conv2))
        # for c in range(conv2.shape[1]):
        #     cv2.imwrite(f'debug/dec2_{c}.png', tensor2img_feat(conv2[:, c:c+1, ...]))
        conv1 = self.dec3((conv2, conv1))
        # for c in range(conv1.shape[1]):
        #     cv2.imwrite(f'debug/dec3_{c}.png', tensor2img_feat(conv1[:, c:c+1, ...]))
        conv0 = self.dec4((conv1, conv0))
        # for c in range(conv0.shape[1]):
        #     cv2.imwrite(f'debug/dec4_{c}.png', tensor2img_feat(conv0[:, c:c+1, ...]))
        yuv_skip = torch.cat([y, self.uplayer(torch.cat([u, v], dim=1))], dim=1)
        # pdb.set_trace()
        
        out0 = self.out0(conv0)
        # for c in range(3):
        #     cv2.imwrite(f'debug/out0_{c}.png', tensor2img_feat(out0[:, c:c+1, ...]))
        cv2.imwrite(f'debug/out0_show.png', tensor2img_feat(cvtColor(out0)))
        out1 = self.out1(out0)
        cv2.imwrite(f'debug/out0_up_show.png', tensor2img_feat(cvtColor(out1)))
        pred = yuv_skip + out1
        cv2.imwrite(f'debug/pred.png', tensor2img_feat(cvtColor(pred)))

        return pred
    

@ARCH_REGISTRY.register()
class PMRIDSRV6NV21skipArch(PMRIDSRV5NV21skipArch):
    def __init__(self, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super(PMRIDSRV6NV21skipArch, self).__init__(feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip)
    
    def forward(self, inp, label_mask=None):
        cvtColor = YUV2RGBConv().to(inp.device)
        y_unshuffled, y, u, v = self.input_layer(inp)
        conv0 = self.conv0(torch.cat([y_unshuffled,u,v], dim=1))
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        # for c in range(conv3.shape[1]):
        #     cv2.imwrite(f'debug/dec1_{c}.png', tensor2img_feat(conv3[:, c:c+1, ...]))
        conv2 = self.dec2((conv3, conv2))
        # for c in range(conv2.shape[1]):
        #     cv2.imwrite(f'debug/dec2_{c}.png', tensor2img_feat(conv2[:, c:c+1, ...]))
        conv1 = self.dec3((conv2, conv1))
        # for c in range(conv1.shape[1]):
        #     cv2.imwrite(f'debug/dec3_{c}.png', tensor2img_feat(conv1[:, c:c+1, ...]))
        conv0 = self.dec4((conv1, conv0))
        # for c in range(conv0.shape[1]):
        #     cv2.imwrite(f'debug/dec4_{c}.png', tensor2img_feat(conv0[:, c:c+1, ...]))
        yuv_skip = torch.cat([y, self.uplayer(torch.cat([u, v], dim=1))], dim=1)
        # pdb.set_trace()
        
        out0 = self.out0(conv0)
        # for c in range(3):
        #     cv2.imwrite(f'debug/out0_{c}.png', tensor2img_feat(out0[:, c:c+1, ...]))
        pred = self.out1(out0)
        cv2.imwrite(f'debug/pred_noskip.png', tensor2img_feat(cvtColor(pred)))

        return pred

@ARCH_REGISTRY.register()
class PMRIDSRV3CvtRGBArch(nn.Module):
    def __init__(self, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super(PMRIDSRV3CvtRGBArch, self).__init__()
        self.input_layer = NV212Unshuffled()
        self.conv0 = Conv2D(in_channels=6, out_channels=feat_base, kernel_size=1, padding=0, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base*2, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base*2, out_channels=feat_base*4, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base*4, out_channels=feat_base*6, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base*6, out_channels=feat_base*8, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base*8, out_channels=feat_base*8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base*8, skip_in_channels=feat_base*6, out_channels=feat_base*6,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base*6, skip_in_channels=feat_base*4, out_channels=feat_base*4,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base*4, skip_in_channels=feat_base*2, out_channels=feat_base*2,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base*2, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=out_channels, kernel_size=3)
        # self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
        if fast_upsample:
            self.out1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # self.uplayer = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        else:
            self.out1 = nn.ConvTranspose2d(feat_base, out_channels, kernel_size=2, stride=2, padding=0)
            # self.uplayer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        self.out = nn.Sequential(
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            DecoderBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

    def forward(self, inp, label_mask=None):
        inp = self.input_layer(inp)
        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        conv3 = self.dec1((conv5, conv3))
        conv2 = self.dec2((conv3, conv2))
        conv1 = self.dec3((conv2, conv1))
        conv0 = self.dec4((conv1, conv0))

        conv0 = self.out1(self.out0(conv0))
        pred = self.out(conv0)

        return pred


@ARCH_REGISTRY.register()
class PMRIDEnhanceArch(nn.Module):
    def __init__(self, in_channels, feat_base, out_channels, kernel_size, same_channel, fast_upsample, num_block, decoder_skip=True):
        super().__init__()
        self.conv0 = Conv2D(in_channels=in_channels, out_channels=feat_base, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[0])
        self.enc2 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[1])
        self.enc3 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[2])
        self.enc4 = EncoderStage(in_channels=feat_base, out_channels=feat_base, kernel_size=kernel_size,same_channel=same_channel,num_blocks=num_block[3])

        self.encdec = Conv2D(in_channels=feat_base, out_channels=feat_base, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec2 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec3 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)
        self.dec4 = DecoderStage(in_channels=feat_base, skip_in_channels=feat_base, out_channels=feat_base,fast_upsample=fast_upsample,decoder_skip=decoder_skip)

        self.out0 = DecoderBlock(in_channels=feat_base, out_channels=feat_base, kernel_size=3)
        self.out1 = Conv2D(in_channels=feat_base, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def forward(self, inp):
        # inp = self._check_and_padding(inp)

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        # print("up1", up1.shape)
        # print("conv0", conv0.shape)
        # print("inp", inp.shape)
        x0 = self.dec4((up1, conv0))

        x1 = self.out0(x0)
        x = self.out1(x1)
        pred = x + inp

        return pred


@ARCH_REGISTRY.register()
class IdentityArch(nn.Module):
    def __init__(self, upscale=1) -> None:
        super().__init__()
        if upscale == 1:
            self.m = nn.Identity()
        elif upscale > 0:
            self.m = nn.Upsample(scale_factor=upscale, mode='bicubic')
    
    def forward(self, x):
        return self.m(x)

@ARCH_REGISTRY.register()
class IdentityNV21ToYUV444(nn.Module):
    def __init__(self):
        super(IdentityNV21ToYUV444, self).__init__()
        self.input_layer = NV212Unshuffled_split()
        self.uplayer = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=False)
        
    def forward(self, x):
        _, y, u, v = self.input_layer(x)
        x = torch.cat([y, self.uplayer(torch.cat([u, v], dim=1))], dim=1)
        return x

if __name__=="__main__":
    model = PMRIDSRV5NV21skipArch(
        feat_base=16,
        out_channels=3,
        kernel_size=3,
        same_channel=False,
        fast_upsample=True,
        num_block=[2, ],
        decoder_skip=True
    )
    
    x = torch.randn([1, 1, 384, 256], dtype=torch.float32)
    y=model(x)
    print(y.shape)

    from torchinfo import summary
    col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds")
    summary(model, input_size=(1, 1, 384, 256), col_names=col_names, verbose=1)