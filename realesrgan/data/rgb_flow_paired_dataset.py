import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .build_blur_with_2_frame import optical_flow_motion_blur

class RGBFlowPairedDataset(Dataset):
    """
    Read pairs of RGB images + mask (searching paths like YUVPairedDataset).
    For each pair:
      - read two RGB images (cv2 BGR), read mask (grayscale)
      - compute optical_flow_motion_blur(frame1, frame2) -> blurred_bgr
      - perform a synchronized random crop (top/left even for chroma alignment)
      - convert both cropped BGR images to packed NV21 (shape (H + H//2, W) uint8)
      - return tensors:
          img1: blurred NV21 packed as float32 normalized [0,1], shape (1, H*1.5, W)
          img2: second original frame NV21 packed as float32 normalized [0,1], shape (1, H*1.5, W)
          mask: grayscale crop as float32 normalized [0,1], shape (1, H, W)
    """
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, paths, crop_size=256, keyword_list=['input_00', 'input_01', 'input_02', 'input_03', 'input_04', 'input_05'], mask_keyword='_mask', recursive=True):
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        else:
            self.crop_h, self.crop_w = crop_size
        self.keywords = keyword_list
        self.mask_keyword = mask_keyword

        # build list of image files
        if os.path.isdir(paths):
            pattern = '**/*' if recursive else '*'
            all_files = sorted(glob.glob(os.path.join(paths, pattern), recursive=recursive))
            self.paths = [p for p in all_files if os.path.splitext(p)[1].lower() in self.IMG_EXTS]
        elif os.path.isfile(paths) and paths.lower().endswith('.txt'):
            with open(paths, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            self.paths = [l for l in lines if os.path.splitext(l)[1].lower() in self.IMG_EXTS]
        else:
            raise ValueError("paths must be a folder or a .txt listing image files")

        if len(self.paths) == 0:
            raise RuntimeError("No image files found")

    def __len__(self):
        return len(self.paths)

    def _random_crop_coords_aligned(self, H, W):
        if H < self.crop_h or W < self.crop_w:
            raise ValueError(f"Image too small for crop ({H}x{W}) vs crop {self.crop_h}x{self.crop_w}")
        max_top = H - self.crop_h
        max_left = W - self.crop_w
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        if top % 2 != 0:
            top = top - 1 if top > 0 else top + 1
        if left % 2 != 0:
            left = left - 1 if left > 0 else left + 1
        top = max(0, min(top, max_top))
        left = max(0, min(left, max_left))
        return top, left, top + self.crop_h, left + self.crop_w

    def _derive_mask_candidates(self, primary_path, found_keyword):
        root, ext = os.path.splitext(primary_path)
        candidates = []
        # candidate 1: replace "keyword_rgb" -> "keyword_mask"
        candidates.append(primary_path.replace(found_keyword + "_rgb", found_keyword + self.mask_keyword))
        # candidate 2: replace keyword -> keyword_mask (keep ext)
        candidates.append(primary_path.replace(found_keyword, found_keyword + self.mask_keyword))
        # candidate 3: same basename + mask.png
        candidates.append(root + self.mask_keyword + '.png')
        # candidate 4: same dirname + found_keyword_mask + original suffix after keyword (best-effort)
        basename = os.path.basename(root)
        dirname = os.path.dirname(primary_path)
        cand = os.path.join(dirname, basename.replace(found_keyword, found_keyword + self.mask_keyword) + '.png')
        candidates.append(cand)
        # ensure unique and keep only existing
        seen = set()
        out = []
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    def _bgr_to_nv21_packed(self, bgr_img):
        """Convert BGR uint8 HxW x3 -> NV21 packed uint8 (H + H//2, W)"""
        H, W = bgr_img.shape[:2]
        # convert to I420 (Y U V planes)
        yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
        # cv2 returns a 2D array with H + H//2 rows, but treat as flat to extract planes
        flat = yuv_i420.flatten()
        y_size = W * H
        uv_size = y_size // 4
        y_plane = flat[:y_size]
        u_plane = flat[y_size:y_size + uv_size]
        v_plane = flat[y_size + uv_size:y_size + 2 * uv_size]
        # interleave VU for NV21
        uv_interleaved = np.empty(2 * uv_size, dtype=np.uint8)
        uv_interleaved[0::2] = v_plane
        uv_interleaved[1::2] = u_plane
        nv21 = np.concatenate([y_plane, uv_interleaved])
        packed = nv21.reshape((H + H // 2, W))
        return packed.astype(np.uint8)

    def __getitem__(self, index):
        primary_path = self.paths[index]

        # find keyword in path
        found_keyword = None
        for k in self.keywords:
            if k in primary_path:
                found_keyword = k
                break
        if found_keyword is None:
            raise ValueError(f"None of keywords {self.keywords} found in path '{primary_path}'")

        # choose secondary keyword
        other_keys = [k for k in self.keywords if k != found_keyword]
        if not other_keys:
            raise ValueError("No alternative keywords available in keyword_list")
        alt_keyword = random.choice(other_keys)
        secondary_path = primary_path.replace(found_keyword, alt_keyword)

        if not os.path.exists(primary_path):
            raise FileNotFoundError(f"Primary file not found: {primary_path}")
        if not os.path.exists(secondary_path):
            raise FileNotFoundError(f"Secondary file not found: {secondary_path}")

        # read BGR images
        frame1 = cv2.imread(primary_path, cv2.IMREAD_COLOR)
        frame2 = cv2.imread(secondary_path, cv2.IMREAD_COLOR)
        if frame1 is None or frame2 is None:
            raise FileNotFoundError(f"Failed to read images: {primary_path}, {secondary_path}")

        H, W = frame1.shape[:2]
        # compute motion-blurred frame (BGR)
        blurred = optical_flow_motion_blur(frame1, frame2, num_samples=7)

        # derive mask path candidates and pick existing
        mask_candidates = self._derive_mask_candidates(primary_path, found_keyword)
        mask_path = None
        for c in mask_candidates:
            if os.path.exists(c):
                mask_path = c
                break

        if mask_path is None:
            # fallback: full white mask
            mask_full = np.ones((H, W), dtype=np.uint8) * 255
        else:
            m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if m is None:
                mask_full = np.ones((H, W), dtype=np.uint8) * 255
            else:
                if m.ndim == 3:
                    mask_full = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                else:
                    mask_full = m
                # ensure same size as images (resize if needed)
                if mask_full.shape[:2] != (H, W):
                    mask_full = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

        # random crop aligned even
        top, left, bottom, right = self._random_crop_coords_aligned(H, W)

        # crop blurred and second original (we return blurred as img1 and frame2 as img2)
        blurred_crop = blurred[top:bottom, left:right]
        frame2_crop = frame2[top:bottom, left:right]
        mask_crop = mask_full[top:bottom, left:right]

        # convert to NV21 packed
        nv21_1 = self._bgr_to_nv21_packed(blurred_crop)  # (H + H//2_crop, W)
        nv21_2 = self._bgr_to_nv21_packed(frame2_crop)

        # to tensors normalized to [0,1], shape (1, H*1.5, W)
        img1_t = torch.from_numpy(nv21_1[None, ...].copy() / 255.0).to(torch.float32)
        img2_t = torch.from_numpy(nv21_2[None, ...].copy() / 255.0).to(torch.float32)
        mask_t = torch.from_numpy(mask_crop[None, ...].copy() / 255.0).to(torch.float32)

        return {
            'img1': img1_t,
            'img2': img2_t,
            'mask': mask_t,
            'primary_path': primary_path,
            'secondary_path': secondary_path,
            'mask_path': mask_path
        }


if __name__ == '__main__':
    # quick smoke test (adjust path)
    root_dir = r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data'
    ds = RGBFlowPairedDataset(paths=root_dir, crop_size=256, recursive=False)
    sample = ds[0]
    print('img1', sample['img1'].shape, 'img2', sample['img2'].shape, 'mask', sample['mask'].shape)