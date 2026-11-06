"""Corrected V5 camera inference: normalize each complete text line to height96."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CAPTURE = None


def restore_height96(rgb, boxes, network, device, blend_source=None):
    """@luna-comment
    Input: frozen-preprocessed uint8 RGB, original OCR boxes, V5 and original RGB base.
    Output: native full frame and line/tile geometry; optional actual012 diagnostics.
    Key Steps: resize complete lines to96, horizontal overlap inference, normalized
    tile blend, native box resize, average box overlap, full-union16px feather.
    Key Params: bicubic; even rounded proportional width;480 tiles/64 overlap;
    edge pad for short lines, positive32px tile ramps, unchanged original outside boxes.
    State/Dependencies: strict loaded network, torch color transforms; CAPTURE only
    writes diagnostics and does not change model inputs or fusion calculations.
    """
    import cv2
    import numpy as np
    import torch
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv,y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    from tools.auto_hx_v3_native import save_json
    original = rgb if blend_source is None else blend_source
    h,w = rgb.shape[:2]
    assert original.shape == rgb.shape
    totals = np.zeros((h,w,3),np.float32)
    counts = np.zeros((h,w),np.float32)
    rows,lines = [],[]
    patch_count=0
    ramp = np.minimum(np.minimum(np.arange(480)+1,480-np.arange(480))/32,1).astype(np.float32)
    for line_index,(x0,y0,x1,y1) in enumerate(boxes,1):
        bh,bw=y1-y0,x1-x0
        nw=max(2,2*round((bw*96/bh)/2))
        line=cv2.resize(rgb[y0:y1,x0:x1],(nw,96),interpolation=cv2.INTER_CUBIC)
        if CAPTURE is not None:
            source_line=cv2.resize(original[y0:y1,x0:x1],(nw,96),interpolation=cv2.INTER_CUBIC)
        last=max(0,nw-480)
        xs=sorted(set(list(range(0,last,416))+[last]))
        sums=np.zeros((96,nw,3),np.float32)
        weights=np.zeros(nw,np.float32)
        for offset in range(0,len(xs),8):
            origins=xs[offset:offset+8]
            patches=[np.pad(line[:,x:x+480],((0,0),(0,max(0,480-(nw-x))),(0,0)),mode='edge') for x in origins]
            batch=torch.from_numpy(np.stack(patches).transpose(0,3,1,2).copy()).to(device).float()/255
            assert tuple(batch.shape[1:])==(3,96,480)
            with torch.inference_mode():
                y4uv=network(rgb_to_y4uv(batch))
                raw=YUV2RGB(y4uv_to_yuv444(y4uv),is_train=True)
                decoded=raw.clamp(0,1)
            assert torch.isfinite(decoded).all() and decoded.shape==batch.shape
            values=decoded.permute(0,2,3,1).cpu().numpy()
            for j,(x,value) in enumerate(zip(origins,values)):
                valid=min(480,nw-x)
                sums[:,x:x+valid]+=value[:,:valid]*ramp[None,:valid,None]
                weights[x:x+valid]+=ramp[:valid]
                patch_count+=1
                if CAPTURE is not None:
                    tag=f'{patch_count:04d}'
                    source_patch=np.pad(source_line[:,x:x+480],((0,0),(0,480-valid),(0,0)),mode='edge')
                    output=np.uint8(np.clip(np.rint(value*255),0,255))
                    difference=np.uint8(np.clip(128+5*(output.astype(np.float32)-patches[j]),0,255))
                    for key,pixels in [('source',source_patch),('input',patches[j]),('output',output),('diff',difference)]:
                        assert cv2.imwrite(str(CAPTURE/'patches'/(tag+'_'+key+'.png')),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
                    raw_rgb=raw[j].permute(1,2,0).cpu().numpy()
                    np.save(CAPTURE/'raw_rgb'/(tag+'.npy'),raw_rgb)
                    np.save(CAPTURE/'raw_y4uv'/(tag+'.npy'),y4uv[j].cpu().numpy())
                    rows.append(dict(id=tag,line_index=line_index,x=x0+x*bw/nw,y=y0,
                        width=valid*bw/nw,height=bh,model_x=x,model_valid_width=valid,
                        model_width=480,model_height=96,
                        mae_vs_input=float(np.abs(output.astype(float)-patches[j]).mean()),
                        rgb_min=float(raw_rgb.min()),rgb_max=float(raw_rgb.max()),
                        clipped_fraction=float(((raw_rgb<0)|(raw_rgb>1)).mean())))
        assert (weights>0).all()
        restored=cv2.resize(sums/weights[None,:,None],(bw,bh),interpolation=cv2.INTER_CUBIC).clip(0,1)
        totals[y0:y1,x0:x1]+=restored
        counts[y0:y1,x0:x1]+=1
        lines.append(dict(line=line_index,bbox=[x0,y0,x1,y1],native_hw=[bh,bw],
                          normalized_hw=[96,nw],scale_y=96/bh,scale_x=nw/bw,patches=len(xs)))
    mask=(counts>0).astype(np.uint8)
    prediction=totals/np.maximum(counts[...,None],1)
    distance=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    t=np.clip((distance-1)/16,0,1);alpha=t*t*(3-2*t)
    result=np.uint8(np.clip(np.rint(original.astype(np.float32)*(1-alpha[...,None])+prediction*255*alpha[...,None]),0,255))
    assert np.array_equal(result[mask==0],original[mask==0])
    details=dict(patches=patch_count,boxes=len(boxes),outside_unchanged=True,
        canvas_hw=[96,480],stride_hw=[96,416],box_feather=16,
        geometry='text_line_height96_bicubic',tile_overlap=64,lines=lines)
    if CAPTURE is not None:
        save_json(CAPTURE/'index.json',dict(status='passed',image_id='012',
            name='IMG_20260605_014719726.jpg',height_normalized=True,
            patches=len(rows),width=w,height=h,text_boxes=boxes,details=details,rows=rows))
    return result,details


def main():
    """@luna-comment
    Input: worker/run/pilot CLI. Output: isolated corrected-height results.
    Key Steps: bind existing strict V5 worker and supervisor to this entry/output,
    substitute only restoration geometry, optionally export image012 model patches.
    Key Params: pilot rank11/shards280 selects012; standard run remains four workers.
    State/Dependencies: process-local bindings; original adapter source is unmodified.
    """
    global CAPTURE
    import tools.auto_hx_v3_native as shared
    import tools.run_mojito_camera_v5_strict_h96_20261016 as adapter
    adapter.OUT=adapter.OUT.with_name('mojito_camera_v5_height96_20261016')
    adapter.__file__=__file__
    shared.restore=restore_height96
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['run','worker','pilot'],default='run')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=280)
    args=parser.parse_args()
    if args.mode=='run':adapter.supervise()
    elif args.mode=='pilot':
        CAPTURE=adapter.OUT/'diagnostic_012'
        for folder in ('patches','raw_rgb','raw_y4uv'):(CAPTURE/folder).mkdir(parents=True,exist_ok=True)
        adapter.worker(11,280,280)
        meta=json.loads((CAPTURE/'index.json').read_text())
        receipt=json.loads((adapter.OUT/'receipts/IMG_20260605_014719726.json').read_text())
        for key in ('source_sha256','checkpoint_sha256','model_input_pixel_sha256','output_sha256'):meta[key]=receipt[key]
        meta['pilot_output_sha256']=receipt['output_sha256']
        shared.save_json(CAPTURE/'index.json',meta)
        print('PASS normalized012',meta['patches'])
    else:adapter.worker(args.rank,args.shards,args.limit)


if __name__=='__main__':main()
