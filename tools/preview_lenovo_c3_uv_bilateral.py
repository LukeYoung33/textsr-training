"""Experimental C3 -> joint UV bilateral sweep; does not alter training atoms."""
from pathlib import Path
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realesrgan import lenovo_atomic_torch as atoms
from realesrgan.lenovo_degradation import LenovoDegradationPipeline


def bilateral_uv(uv, spatial_sigma, color_sigma_255):
    """Joint Cr/Cb range weights, no Y guidance, reflect padding, no quantization."""
    radius = math.ceil(2 * spatial_sigma)
    h, w = uv.shape[-2:]
    padded = F.pad(uv, (radius, radius, radius, radius), mode='reflect')
    numerator = torch.zeros_like(uv)
    denominator = torch.zeros_like(uv[:, :1])
    color_sigma = color_sigma_255 / 255
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[:, :, radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            distance = (neighbor-uv).square().sum(dim=1, keepdim=True)
            weight = torch.exp(-distance/(2*color_sigma**2) - (dx*dx+dy*dy)/(2*spatial_sigma**2))
            numerator += neighbor * weight
            denominator += weight
    return numerator / denominator.clamp_min(1e-12)


def to_image(x):
    array = x[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def main():
    torch.set_num_threads(4)
    out = ROOT/'datasets/lenovo_c3_uv_bilateral_preview_v1'
    out.mkdir(parents=True, exist_ok=True)
    presets = [('mild', 1.4, 6.0), ('medium', 2.8, 12.0), ('wide', 4.0, 24.0)]
    manifest = {'status': 'experimental_not_training', 'seed': 17,
                'operation': 'C3 then joint UV bilateral; Y unchanged; no Y guidance; no gain compensation',
                'records': []}
    pipeline = LenovoDegradationPipeline()
    for index in range(3):
        source = ROOT/f'datasets/lenovo_atomic_equivalence_v1/gt_{index}.png'
        clean = torch.from_numpy(np.asarray(Image.open(source).convert('RGB')).copy()).permute(2, 0, 1)[None].float()/255
        h, w = clean.shape[-2:]
        rng = np.random.default_rng(17)
        fields = {key: torch.from_numpy(rng.normal(0, 1, (h, w)).astype(np.float32))[None, None] for key in ('cr', 'cb')}
        noisy, _ = pipeline.torch_atomic(clean, 'c3', torch.Generator().manual_seed(17), random_inputs=fields)
        ycc = atoms.ycc(noisy)
        entries = [('clean', clean), ('C3 current', noisy)]
        diagnostics = []
        for label, spatial, color in presets:
            filtered = ycc.clone()
            filtered[:, 1:] = bilateral_uv(ycc[:, 1:], spatial, color)
            assert torch.equal(filtered[:, :1], ycc[:, :1])
            result = atoms.rgb(filtered).clamp(0, 1)
            filename = f'{index}_{label}.png'
            to_image(result).save(out/filename)
            entries.append((f'{label}: space={spatial}, UV={color}', result))
            residual = filtered[:, 1:] - atoms.ycc(clean)[:, 1:]
            diagnostics.append({'preset': label, 'spatial_sigma_px': spatial,
                                'color_sigma_255': color, 'radius_px': math.ceil(2*spatial),
                                'uv_residual_rms_255': float(residual.square().mean().sqrt()*255),
                                'output': filename})
        # Two-row layout preserves a legible pixel scale in the viewer.
        width, height = w*2, h*2
        sheet = Image.new('RGB', (width*3, (height+30)*2), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, (label, value) in enumerate(entries):
            x, y = (j % 3)*width, (j // 3)*(height+30)
            sheet.paste(to_image(value).resize((width, height), Image.Resampling.NEAREST), (x, y+30))
            draw.text((x+8, y+8), label+' | 2x nearest', fill='black')
        sheet.save(out/f'{index}_comparison_2x.png')
        # UV visualizer is explicitly diagnostic; 4x chroma residual around gray.
        uv_panels = []
        for label, value in entries[1:]:
            residual = atoms.ycc(value)[:, 1:] - atoms.ycc(clean)[:, 1:]
            diagnostic = torch.cat([torch.full_like(clean[:, :1], .5), residual*4+128/255], 1)
            uv_panels.append(to_image(atoms.rgb(diagnostic)))
        uv_sheet = Image.new('RGB', (w*4, h+30), 'white')
        uv_draw = ImageDraw.Draw(uv_sheet)
        for j, panel in enumerate(uv_panels):
            uv_sheet.paste(panel, (j*w, 30))
            uv_draw.text((j*w+4, 6), entries[j+1][0]+' | UV residual x4', fill='black')
        uv_sheet.save(out/f'{index}_uv_residual_x4.png')
        manifest['records'].append({'source': str(source), 'presets': diagnostics})
    # Meaningful invariants: flat UV remains flat; wide range approaches Gaussian.
    flat = torch.full((1, 2, 16, 24), 128/255)
    assert torch.allclose(bilateral_uv(flat, 2, 12), flat, atol=1e-6)
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
