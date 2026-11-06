import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

def nv21_to_rgb(nv21_packed, H, W):
        y_plane = nv21_packed[:H, :]
        uv_plane = nv21_packed[H:, :].reshape((H // 2, W // 2, 2))
        v_plane = uv_plane[:, :, 0]
        u_plane = uv_plane[:, :, 1]
        u_upsampled = cv2.resize(u_plane, (W, H), interpolation=cv2.INTER_LINEAR)
        v_upsampled = cv2.resize(v_plane, (W, H), interpolation=cv2.INTER_LINEAR)
        yuv_image = np.stack([y_plane, u_upsampled, v_upsampled], axis=-1)
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        return rgb_image

@DATASET_REGISTRY.register()
class YUVPairedDataset(Dataset):
    """
    Dataset that reads NV21 (.yuv) images (default 4096x3072) and returns a pair in NV21 packed format:
      - img1, img2: tensors with shape (1, H*1.5, W) dtype=torch.uint8 (NV21 packed: Y rows then interleaved VU rows)
      - mask: tensor with shape (1, H, W) dtype=torch.uint8 (grayscale)
    Crops are synchronized and ensured to be chroma-aligned (top and left even).

    Args:
      paths: a folder path or a path to a txt file containing one yuv path per line.
      width, height: resolution of the NV21 frames (default 4096x3072).
      main_keyword: substring in filename identifying the primary image (e.g. 'input_0').
      alt_keyword: substring to replace main_keyword to obtain the paired image (e.g. 'input_1').
      mask_keyword: substring to replace main_keyword to obtain the mask path (e.g. 'mask').
      crop_size: int or (h,w) for random crop size.
      recursive: if paths is a folder, whether to search recursively for .yuv files.
    """

    def __init__(self, opt):
        self.opt = opt
        if isinstance(self.opt['crop_size'], int):
            self.crop_h = self.crop_w = self.opt['crop_size']
        else:
            self.crop_h, self.crop_w = self.opt['crop_size']
        self.width = int(self.opt['width'])
        self.height = int(self.opt['height'])
        self.keywords = self.opt['keyword_list']
        self.mask_keyword = self.opt['mask_keyword']
        paths = self.opt['paths']

        # build list of primary paths
        if os.path.isdir(paths):
            pattern = '**/*.yuv' if self.opt['recursive'] else '*.yuv'
            self.paths = sorted(glob.glob(os.path.join(paths, pattern), recursive=self.opt['recursive']))
        elif os.path.isfile(paths) and paths.lower().endswith('.txt'):
            with open(paths, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            self.paths = lines
        else:
            raise ValueError("paths must be a folder or a .txt listing yuv files")

        if len(self.paths) == 0:
            raise RuntimeError("No .yuv files found")

    def __len__(self):
        return len(self.paths)

    def _read_nv21_packed(self, path):
        """
        Read NV21 (.yuv) file and return packed NV21 as uint8 array shape (H + H//2, W)
        Packed layout: first H rows = Y plane (HxW), next H//2 rows = interleaved VU plane (H/2 x W)
        """
        expected = self.width * self.height * 3 // 2
        with open(path, 'rb') as f:
            data = f.read()
        if len(data) != expected:
            raise ValueError(f"YUV file size mismatch {path}: expected {expected}, got {len(data)}")

        arr = np.frombuffer(data, dtype=np.uint8)
        y_size = self.width * self.height
        uv_size = y_size // 2
        y = arr[:y_size].reshape((self.height, self.width))
        uv = arr[y_size:y_size + uv_size].reshape((self.height // 2, self.width))
        packed = np.vstack([y, uv])  # shape (H + H//2, W)
        return packed

    def _read_mask(self, path):
        """
        Read mask as grayscale uint8 HxW. Supports .yuv (NV21) masks or image files.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.yuv':
            packed = self._read_nv21_packed(path)  # packed NV21
            # convert NV21 packed to RGB then to gray OR extract Y plane directly
            y_plane = packed[:self.height, :]
            return y_plane.astype(np.uint8)
        else:
            m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(f"Mask exists but cannot be read: {path}")
            if m.ndim == 3:
                gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            else:
                gray = m
            return gray.astype(np.uint8)

    def _random_crop_coords_aligned(self, H, W):
        """
        Return top,left such that top and left are even (chroma aligned), and crop fits.
        Returns (top, left, bottom, right).
        """
        if H < self.crop_h or W < self.crop_w:
            raise ValueError(f"Image too small for crop ({H}x{W}) vs crop {self.crop_h}x{self.crop_w}")
        # ensure even coordinates for chroma alignment
        max_top = H - self.crop_h
        max_left = W - self.crop_w
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        if top % 2 != 0:
            top = top - 1 if top > 0 else top + 1
        if left % 2 != 0:
            left = left - 1 if left > 0 else left + 1
        # clamp
        top = max(0, min(top, max_top))
        left = max(0, min(left, max_left))
        return top, left, top + self.crop_h, left + self.crop_w

    def __getitem__(self, index):
        primary_path = self.paths[index]

        # find which keyword appears in the path
        found_keyword = None
        for k in self.keywords:
            if k in primary_path:
                found_keyword = k
                break
        if found_keyword is None:
            raise ValueError(f"None of keywords {self.keywords} found in path '{primary_path}'")

        # choose secondary keyword randomly from keywords excluding the found one
        other_keys = [k for k in self.keywords if k != found_keyword]
        if not other_keys:
            raise ValueError("No alternative keywords available (keyword_list contains only the found keyword).")
        alt_keyword = random.choice(other_keys)
        secondary_path = primary_path.replace(found_keyword, alt_keyword)
        mask_path = primary_path.replace(found_keyword+'.yuv', found_keyword+self.mask_keyword+'.png')

        if not os.path.exists(primary_path):
            raise FileNotFoundError(f"Primary file not found: {primary_path}")
        if not os.path.exists(secondary_path):
            raise FileNotFoundError(f"Secondary file not found: {secondary_path}")

        img1_packed = self._read_nv21_packed(primary_path)   # (H + H//2, W) uint8
        img2_packed = self._read_nv21_packed(secondary_path) # (H + H//2, W) uint8

        if not os.path.exists(mask_path):
            mask_full = np.ones((self.height, self.width), dtype=np.uint8) * 255
        else:
            mask_full = self._read_mask(mask_path)  # H x W uint8

        # get crop coords aligned even
        top, left, bottom, right = self._random_crop_coords_aligned(self.height, self.width)

        # For packed NV21 we need to crop combined rows: Y rows = top:bottom, UV rows = (top//2):(bottom//2)
        uv_top = top // 2
        uv_bottom = (bottom) // 2  # integer division
        # packed indices: rows 0..H-1 -> Y; rows H..H+H//2-1 -> UV
        img1_crop_y = img1_packed[top:bottom, left:right]
        img1_crop_uv = img1_packed[self.height + uv_top : self.height + uv_bottom, left:right]
        img1_crop_packed = np.vstack([img1_crop_y, img1_crop_uv])

        img2_crop_y = img2_packed[top:bottom, left:right]
        img2_crop_uv = img2_packed[self.height + uv_top : self.height + uv_bottom, left:right]
        img2_crop_packed = np.vstack([img2_crop_y, img2_crop_uv])

        mask_crop = mask_full[top:bottom, left:right]
        
        img1_rgb = nv21_to_rgb(img1_crop_packed, self.crop_h, self.crop_w)
        img2_rgb = nv21_to_rgb(img2_crop_packed, self.crop_h, self.crop_w)

        # convert to tensors: keep uint8 NV21 packed layout, shape (1, H*1.5, W)
        img1_t = torch.from_numpy(img1_crop_packed[None, ...].copy() / 255.0).to(torch.float32)
        img2_t = torch.from_numpy(img2_crop_packed[None, ...].copy() / 255.0).to(torch.float32)
        mask_t = torch.from_numpy(mask_crop[None, ...].copy() / 255.0).to(torch.float32)  # shape (1, H, W)
        img1_rgb = torch.from_numpy(img1_rgb.copy() / 255.0).to(torch.float32).permute(2, 0, 1)
        img2_rgb = torch.from_numpy(img2_rgb.copy() / 255.0).to(torch.float32).permute(2, 0, 1)

        return {
            'lq': img1_t,               # (1, crop_h + crop_h//2, crop_w)
            'gt': img2_t,               # (1, crop_h + crop_h//2, crop_w)
            'mask': mask_t,               # (1, crop_h, crop_w)
            'lq_rgb': img1_rgb,
            'gt_rgb': img2_rgb,
            'primary_path': primary_path,
            'secondary_path': secondary_path,
            'mask_path': mask_path
        }

if __name__ == '__main__':

    

    # from basicsr.utils import tensor2img

    root_dir = r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data'

    dataset = YUVPairedDataset(paths=root_dir,
                               width=4096,
                               height=3072,
                               crop_size=256,
                               recursive=False)
    for i in range(len(dataset)):
        sample = dataset[i]
        img1_show = (sample['img1'][0].numpy() * 255).astype(np.uint8)
        img2_show = (sample['img2'][0].numpy() * 255).astype(np.uint8)
        mask_show = (sample['mask'][0].numpy() * 255).astype(np.uint8)
        # Convert NV21 packed to RGB for visualization
        img1_rgb = nv21_to_rgb(img1_show, dataset.crop_h, dataset.crop_w)
        img2_rgb = nv21_to_rgb(img2_show, dataset.crop_h, dataset.crop_w)
        cv2.imshow('img1_rgb', img1_rgb)
        cv2.imshow('img2_rgb', img2_rgb)
        cv2.imshow('mask', mask_show)
        cv2.waitKey(0)

