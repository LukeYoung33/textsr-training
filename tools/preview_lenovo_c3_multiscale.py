"""Candidate multiscale chroma blotches formed before enlargement.

Independent experiment only: does not replace the accepted C3 or certify an ISP.
"""
from pathlib import Path
import sys
import json
import math

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realesrgan import lenovo_atomic_torch as ops


from realesrgan.lenovo_c3 import standardized, guided_smooth, generate


def picture(x):
    return Image.fromarray(x[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())


def main():
    torch.set_num_threads(4)
    out = ROOT/'datasets/lenovo_c3_multiscale_preview_v1'
    out.mkdir(parents=True, exist_ok=True)
    presets = [
        ('A medium patches', {}),
        ('B larger patches', {'sigmas':(3.,7.,14.)}),
        ('C stronger patches', {'amplitude':10.}),
        ('D different UV direction', {'rho':.75}),
        ('E luma guided', {'guided':True}),
        ('F more zoom', {'scale':.35}),
    ]
    records = []
    for index in range(3):
        source = ROOT/f'datasets/lenovo_atomic_equivalence_v1/gt_{index}.png'
        x = torch.from_numpy(np.asarray(Image.open(source).convert('RGB')).copy()).permute(2,0,1)[None].float()/255
        entries, diagnostics = [('GT', picture(x))], []
        for label, settings in presets:
            result, base, residual, meta = generate(x, **settings)
            if len(entries)==1:
                entries.append(('blur + downsample + enlarge, no chroma noise', picture(base)))
            filename = f'{index}_{label[0]}.png'
            picture(result).save(out/filename)
            entries.append((label, picture(result)))
            # Use constant gray Y to expose chroma alone, without changing noise.
            diagnostic = torch.cat((torch.full_like(x[:,:1],.5), residual+128/255),1)
            diagnostics.append((label+' | UV on gray, gain=1', picture(ops.rgb(diagnostic))))
            records.append({'source':str(source), 'output':filename, **meta})
        def sheet(items, filename, columns, magnification):
            w,h=items[0][1].size; w*=magnification; h*=magnification
            canvas=Image.new('RGB',(columns*w,math.ceil(len(items)/columns)*(h+26)),'white')
            draw=ImageDraw.Draw(canvas)
            for i,(label,img) in enumerate(items):
                xx,yy=(i%columns)*w,(i//columns)*(h+26)
                canvas.paste(img.resize((w,h),Image.Resampling.NEAREST),(xx,yy+26))
                draw.text((xx+5,yy+6),label,fill='black')
            canvas.save(out/filename)
        sheet(entries,f'{index}_comparison_2x.png',2,2)
        sheet(diagnostics,f'{index}_chroma_only.png',2,2)
    (out/'manifest.json').write_text(json.dumps({'status':'candidate_not_training','records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__': main()
