"""Experimental luma-guided containment of multiscale chroma blotches."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realesrgan import lenovo_atomic_torch as ops
from preview_lenovo_c3_multiscale import generate, guided_smooth, picture


from realesrgan.lenovo_c3 import protect


def main():
    torch.set_num_threads(4)
    out = ROOT/'datasets/lenovo_c3_edge_protection_preview_v1'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(3):
        source = ROOT/f'datasets/lenovo_atomic_equivalence_v1/gt_{index}.png'
        clean = torch.from_numpy(np.asarray(Image.open(source).convert('RGB')).copy()).permute(2,0,1)[None].float()/255
        noisy, base, residual, meta = generate(clean, amplitude=6.)
        protected, delta, confidence = protect(clean, base, residual)
        stronger, base2, residual2, _ = generate(clean, amplitude=10.)
        protected2, delta2, _ = protect(clean, base2, residual2)
        panels = [('GT',clean), ('capture, no chroma noise',base),
                  ('before: background floor=0.18',noisy),('after: edge protected, floor=0',protected),
                  ('before: amplitude=10',stronger),('after: protected amplitude=10',protected2)]
        w,h = clean.shape[-1]*2, clean.shape[-2]*2
        sheet = Image.new('RGB',(w*2,(h+26)*3),'white')
        draw = ImageDraw.Draw(sheet)
        for i,(label,value) in enumerate(panels):
            x,y=(i%2)*w,(i//2)*(h+26)
            sheet.paste(picture(value).resize((w,h),Image.Resampling.NEAREST),(x,y+26))
            draw.text((x+5,y+5),label+' | 2x nearest',fill='black')
        sheet.save(out/f'{index}_comparison_2x.png')
        picture(protected).save(out/f'{index}_protected.png')
        picture(protected2).save(out/f'{index}_protected_stronger.png')
        picture(confidence.expand(-1,3,-1,-1)).save(out/f'{index}_luma_confidence.png')
        diagnostics=[]
        for label,d in [('before',residual),('protected',delta)]:
            z=torch.cat((torch.full_like(clean[:,:1],.5),d+128/255),1)
            diagnostics.append(picture(ops.rgb(z)))
        uv = Image.new('RGB',(clean.shape[-1]*2,clean.shape[-2]+26),'white')
        uv_draw=ImageDraw.Draw(uv)
        for i,panel in enumerate(diagnostics):
            uv.paste(panel,(i*clean.shape[-1],26))
            uv_draw.text((i*clean.shape[-1]+5,5),('before' if i==0 else 'edge protected')+' | UV residual gain=1',fill='black')
        uv.save(out/f'{index}_uv_comparison.png')
        records.append({'source':str(source), 'generation':meta,
                        'edge_protection':{'guide':'clean input luminance', 'range_sigma':.035,
                                           'relative_threshold_255':5.,'background_floor':0.,
                                           'uses_ocr_mask':False},
                        'uv_rms_before_255':float(residual.square().mean().sqrt()*255),
                        'uv_rms_after_255':float(delta.square().mean().sqrt()*255),
                        'zero_confidence_fraction':float((confidence==0).float().mean())})
    (out/'manifest.json').write_text(json.dumps({'status':'experimental_not_training','records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__': main()
