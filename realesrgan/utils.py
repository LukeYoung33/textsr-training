import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
import sys
sys.path.append('./')
from basicsr.utils.download_util import load_file_from_url
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None,
                 canny=False):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        self.canny = canny

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()
        self.yuv_cvt = YUV2RGBConv().to(self.device)

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a
    
    def generate_canny_map(self, img):
        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        if self.canny:
            canny_map = self.generate_canny_map(img)
            canny_map = torch.tensor(canny_map).unsqueeze(0)
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img = torch.cat([img, canny_map], dim=0)
        else:
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()
        del img

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        # if self.scale == 2:
        #     self.mod_scale = 2
        # elif self.scale == 1:
        #     self.mod_scale = 4
        self.mod_scale = 1         #1      #16
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.output = self.model(self.img)
        del self.img

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        if isinstance(img, np.ndarray):
            h_input, w_input = img.shape[0:2]
            # img: numpy
            img = img.astype(np.float32)
            if np.max(img) > 256:  # 16-bit image
                max_range = 65535
                print('\tInput is a 16-bit image')
            else:
                max_range = 255
            img = img / max_range
            if len(img.shape) == 2:  # gray image
                # img_mode = 'L'
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img_mode = 'YUV'
                img = np.expand_dims(img, axis=-1)
            elif img.shape[2] == 4:  # RGBA image with alpha channel
                img_mode = 'RGBA'
                alpha = img[:, :, 3]
                img = img[:, :, 0:3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if alpha_upsampler == 'realesrgan':
                    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
            else:
                img_mode = 'RGB'
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ------------------- process image (without the alpha channel) ------------------- #
            self.pre_process(img)
        else:
            h_input, w_input = img.shape[2:]
            img_mode = 'YUV'
            max_range = 255
            self.img = img.to(self.device)
            del img
            if self.half:
                self.img = self.img.half()
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        del self.output
        if img_mode == 'YUV':
            output_img = self.yuv_cvt(output_img)
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output_img = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output_img = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output_img = cv2.resize(
                output_img, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output_img, img_mode


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')


def RGB2YUV420(rgb:torch.Tensor, format='NV21'):
    assert rgb.dim() == 4
    assert rgb.size(1) == 3
    assert format in ('NV12', 'NV21')
    
    B, C, H, W = rgb.shape
    assert H % 2 == 0 and W % 2 == 0
    
    R = rgb[:, 0:1, :, :]
    G = rgb[:, 1:2, :, :]
    B = rgb[:, 2:3, :, :]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.147 * R - 0.289 * G + 0.436 * B + 0.5
    V = 0.615 * R - 0.515 * G - 0.100 * B + 0.5
    
    U_down = F.avg_pool2d(U, kernel_size=2, stride=2)
    V_down = F.avg_pool2d(V, kernel_size=2, stride=2)
    
    if format == 'NV12':
        uv_plane = torch.cat([U_down, V_down], dim=2)
    else:
        uv_plane = torch.cat([V_down, U_down], dim=2)
    
    yuv420 = torch.cat([Y, uv_plane], dim=2)
    return yuv420
    


def RGB2YUV(rgb: torch.Tensor, format='RGB', out_format='NV21', norm: bool=True):
    """
    Convert RGB image tensor to YUV-444 or YUV-420.

    args:
        rgb (torch.Tensor): Input RGB image tensor, shape of (N, 3, H, W) or (3, H, W) or (H, W, 3), value range(0, 1).
        format (string): Color format of input image, either RGB or BGR.
        norm (bool): Whether to adjust output UV channels' value range into (0, 1), default is True.
    return:
        yuv (torch.Tensor): Output YUV image tensor, shape of (N, 3, H, W) or (3, H, W) or (H, W, 3),
                            value range Y (0, 1), UV(-0.5, 0.5) if norm is False else(0, 1)
    """
    assert format.lower() in ('rgb', 'bgr'), f"Input argument 'format' only support RGB and BGR, but got {format} instead."

    # Adjust input shape into (N, 3, H, W)
    if len(rgb.shape) == 3:
        if rgb.shape[0] == 3:
            rgb = rgb.unsqueeze(0)
        elif rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Input shape should be one of (N, 3, H, W) or (3, H, W) or (H, W, 3), but got {rgb.shape}.")
    elif len(rgb.shape) == 4:
        assert rgb.shape[1] == 3, f"Input shape should be one of (N, 3, H, W) or (3, H, W) or (H, W, 3), but got {rgb.shape}."
    else:
        raise ValueError(f"Input shape should be one of (N, 3, H, W) or (3, H, W) or (H, W, 3), but got {rgb.shape}.")
    rgb = rgb if format.lower()=='rgb' else rgb[:, [2, 1, 0], ...]

    # Check value range
    # assert rgb.min() >= 0 and rgb.max() <= 1, f"Input value out of range [0, 1] with min {rgb.min()}, max {rgb.max()} , please check."

    transform_matrix = torch.tensor([
        [ 0.299,    0.587,    0.114  ],
        [-0.14713, -0.28886,  0.436  ],
        [ 0.615,   -0.51499, -0.10001]
    ], dtype=rgb.dtype, device=rgb.device)

    yuv = torch.einsum('ij, ...jhw -> ...ihw', transform_matrix, rgb)
    if norm:
        yuv[..., 1:, :, :] += 0.5

    u_down = torch.nn.functional.avg_pool2d(yuv[:, 1:2, :, :], kernel_size=2, stride=2)
    v_down = torch.nn.functional.avg_pool2d(yuv[:, 2:3, :, :], kernel_size=2, stride=2)
    if out_format.upper() == 'NV12':
        uv_plane = torch.zeros(yuv.shape[0], 1, yuv.shape[2]//2, yuv.shape[3], dtype=yuv.dtype, device=yuv.device)
        uv_plane[..., 0::2] = u_down
        uv_plane[..., 1::2] = v_down
        yuv = torch.cat([yuv[:, 0:1, :, :], uv_plane], dim=2)
    elif out_format.upper() == 'NV21':
        uv_plane = torch.zeros(yuv.shape[0], 1, yuv.shape[2]//2, yuv.shape[3], dtype=yuv.dtype, device=yuv.device)
        uv_plane[..., 1::2] = u_down
        uv_plane[..., 0::2] = v_down
        yuv = torch.cat([yuv[:, 0:1, :, :], uv_plane], dim=2)
    elif out_format.upper() == 'YUV':
        pass
    else:
        raise ValueError("Only YUV, NV12 and NV21 are supported for out_format.")
    return yuv


def nv21_to_yuv444(nv21_tensor):
    """
    将NV21 tensor转换为YUV444 tensor
    
    NV21格式: 
    - Y平面: 全分辨率 [H, W]
    - UV平面: 半分辨率交织 [H/2, W/2] 存储为 VUVUVU...
    
    参数:
        nv21_tensor: 输入NV21 tensor, 形状为 [B, 1, H*3/2, W]
    
    返回:
        yuv444_tensor: 输出YUV444 tensor, 形状为 [B, 3, H, W]
    """
    assert nv21_tensor.dim() == 4, "输入tensor必须是4维的 [B, 1, H*3/2, W]"
    assert nv21_tensor.size(1) == 1, "输入tensor必须有1个通道"
    
    B, _, total_H, W = nv21_tensor.shape
    
    # 计算原始高度H
    H = total_H * 2 // 3
    assert H % 2 == 0 and W % 2 == 0, "高度和宽度必须是偶数"
    assert total_H == H * 3 // 2, f"输入高度 {total_H} 不符合 H*3/2 的格式"
    
    # 分离Y平面和UV平面
    Y_plane = nv21_tensor[:, :, :H, :]                    # [B, 1, H, W]
    UV_plane = nv21_tensor[:, :, H:, :]                   # [B, 1, H/2, W]
    
    # 处理UV平面 - NV21格式是VU交替存储
    UV_processed = _process_nv21_uv_plane(UV_plane)       # [B, 2, H/2, W/2]
    
    # 分离U和V分量
    V_down = UV_processed[:, 0:1, :, :]                   # [B, 1, H/2, W/2]
    U_down = UV_processed[:, 1:2, :, :]                   # [B, 1, H/2, W/2]
    
    # 上采样UV分量到全分辨率
    U_plane = F.interpolate(U_down, size=(H, W), mode='bilinear', align_corners=False)
    V_plane = F.interpolate(V_down, size=(H, W), mode='bilinear', align_corners=False)
    
    # 拼接YUV三个平面
    yuv444_tensor = torch.cat([Y_plane, U_plane, V_plane], dim=1)  # [B, 3, H, W]
    
    return yuv444_tensor

def _process_nv21_uv_plane(uv_plane):
    """
    处理NV21的UV平面，从交织格式分离出V和U分量
    
    NV21 UV平面格式: [V0, U0, V1, U1, V2, U2, ...]
    """
    B, _, H_half, W = uv_plane.shape
    
    # 将UV平面在宽度维度拆分为奇偶列
    # 奇数列是V分量，偶数列是U分量
    uv_reshaped = uv_plane.reshape(B, 1, H_half, W//2, 2)  # [B, 1, H/2, W/2, 2]
    
    # 分离V和U分量
    # NV21: 第一个元素是V，第二个是U
    V_components = uv_reshaped[:, :, :, :, 0]              # [B, 1, H/2, W/2]
    U_components = uv_reshaped[:, :, :, :, 1]              # [B, 1, H/2, W/2]
    
    # 拼接为 [V, U] 格式
    uv_separated = torch.cat([V_components, U_components], dim=1)  # [B, 2, H/2, W/2]
    
    return uv_separated

def YUV2RGB(yuv: torch.Tensor, format='RGB', is_train=False, uv_normalized=True):
    """
    Convert YUV-444 image tensor to RGB.

    args:
        yuv (torch.Tensor): Input YUV image tensor, shape of (N, 3, H, W), value range Y (0, 1), UV (-0.5, 0.5) or (0, 1).
        format (string): Color format of output image, either RGB or BGR.
        uv_normalized (bool): Whether U/V use the stored [0, 1] representation
            with 0.5 as the neutral chroma value. BASIC YUV tensors use this
            representation; no value-based range inference is performed.
    return:
        rgb (torch.Tensor): Output RGB image tensor, shape of (N, 3, H, W), value range (0, 1).
    """
    assert format.lower() in ('rgb', 'bgr'), f"Input argument 'format' only support RGB and BGR, but got {format} instead."
    assert len(yuv.shape)==4 and yuv.shape[1]==3, f"Input YUV shape should be (N, 3, H, W), but got {yuv.shape}."

    # BASIC stores Y/U/V as normalized [0, 1] tensors.  U/V are centered at
    # 0.5, so the offset must be decoded unconditionally.  Inferring the
    # representation from a batch min/max is incorrect when a network output
    # overshoots below 0 or above 1: one bad pixel would change the conversion
    # of every pixel in the batch.  Build a new tensor instead of mutating the
    # caller's tensor in place.
    if uv_normalized:
        yuv_decoded = torch.cat([yuv[:, 0:1, ...], yuv[:, 1:, ...] - 0.5], dim=1)
    else:
        yuv_decoded = yuv

    transform_matrix = torch.tensor([
        [1.0,  0.0,      1.13983],
        [1.0, -0.39465, -0.58060],
        [1.0,  2.03211,  0.0    ]
    ], dtype=yuv_decoded.dtype, device=yuv_decoded.device)

    rgb = torch.einsum('ij, ...jhw -> ...ihw', transform_matrix, yuv_decoded)
    if not is_train:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = rgb if format.lower()=='rgb' else rgb[:, [2, 1, 0], ...]
    return rgb


class YUV2RGBConv(nn.Module):
    def __init__(self):
        """@luna-comment Input:no constructor arguments. Output:fixed YUV-to-RGB convolution modules.
        Key Steps:initialize identity/offset layer and inverse-color matrix; freeze both layers.
        Key Params:UV offset0.5 and existing full-range coefficients. State/Dependencies:PyTorch modules.
        """
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
        """@luna-comment Input:BCHW YUV tensor with normalized offset UV. Output:unclamped BCHW RGB.
        Key Steps:subtract chroma offsets then apply fixed inverse-color convolution.
        Key Params:no resize or quantization. State/Dependencies:frozen constructor layers.
        """
        return self.conv2(self.conv1(yuv))


def tensor2img_feat(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu()     # .clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result



if __name__=="__main__":
    img_path = r'E:\workspace\data\M196_front_1_calibration\front2_vis_check\bstsll_0000001211_input_0_6528x2448_3264x2448_ev[0]_iso-3200_expo-99996000_orientation180.png'
    img = cv2.imread(img_path)
    img_tensor = torch.from_numpy((img / 255.0).astype(np.float32))

    yuv = RGB2YUV(rgb=img_tensor, format="BGR", norm=True)
    yuv_show = (yuv[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    yuv_show = cv2.cvtColor(yuv_show, cv2.COLOR_YUV2BGR)
    cv2.imwrite('./yuv_show.png', yuv_show)
    rgb = YUV2RGB(yuv, format='BGR')
    rgb_show = (rgb[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite('./rgb_show.png', rgb_show)
