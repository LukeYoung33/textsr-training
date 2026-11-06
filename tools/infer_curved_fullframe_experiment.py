"""Run glyph-normalized curved or straight-strip inference over every DB OCR candidate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from infer_curved_text_line_experiment import _fit_curve_and_glyph_height, _tile_restore
from infer_basic_text_line_full_image import (_load_network, _read_rgb, _write_rgb, _write_gray,
    NativeY4UVTileRunner, load_lq_preprocess_config, rgb_to_yuv444, yuv444_to_rgb,
    standard_polygon_paste_alpha, standard_pasteback)


def restore_one(rgb, probability, polygon, score, runner, target_height, rectify, output_yuv, owner):
    curve, glyph, meta = _fit_curve_and_glyph_height(rgb, probability, polygon)
    x0,y0,x1,y1=meta['crop_xyxy']; scale=float(target_height/max(glyph,1e-6)); width=x1-x0; out_w=max(8,int(round(width*scale)))
    uu,vv=np.meshgrid(np.arange(out_w,dtype=np.float32),np.arange(96,dtype=np.float32)); sx=(x0+uu/scale).astype(np.float32)
    source_curve=curve+y0; center=np.interp(sx[0],np.arange(width,dtype=np.float32)+x0,source_curve) if rectify else np.full(out_w,float(np.median(source_curve)),np.float32)
    sy=(center[None]+(vv-48.0)/scale).astype(np.float32)
    strip=cv2.remap(rgb,sx,sy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    prediction=_tile_restore(strip,runner)
    xx,yy=np.meshgrid(np.arange(x0,x1,dtype=np.float32),np.arange(y0,y1,dtype=np.float32))
    inverse_curve=np.interp(xx[0],np.arange(width,dtype=np.float32)+x0,source_curve) if rectify else np.full(width,float(np.median(source_curve)),np.float32)
    mx=((xx-x0)*scale).astype(np.float32); my=(48.0+(yy-inverse_curve[None])*scale).astype(np.float32)
    restored=cv2.remap(prediction.transpose(1,2,0),mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    valid=(mx>=0)&(mx<=out_w-1)&(my>=0)&(my<=95)
    restored[~valid]=rgb[y0:y1,x0:x1][~valid]
    alpha=standard_polygon_paste_alpha(rgb,polygon,(x0,y0,x1,y1),4)
    _, update, _=standard_pasteback(rgb_to_yuv444(rgb),output_yuv,owner,(x0,y0,x1,y1),rgb_to_yuv444(restored),alpha,score)
    meta.update({'scale':scale,'rectification':rectify,'rectified_hw':[96,out_w],'updated_pixels':int(update.sum())})
    return meta


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True);p.add_argument('--ocr-json',type=Path,required=True);p.add_argument('--probability-map',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--network-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--target-glyph-height',type=float,default=64.);p.add_argument('--core-to-glyph',type=float,default=2.5);p.add_argument('--rectify',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    rgb=_read_rgb(a.input); data=json.loads(a.ocr_json.read_text());prob=np.load(a.probability_map);device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');net,_=_load_network(a.checkpoint,device,a.network_config);runner=NativeY4UVTileRunner(net,device,load_lq_preprocess_config(a.network_config));out=rgb_to_yuv444(rgb).copy();owner=np.zeros(rgb.shape[:2],np.float32);records=[]
    for i,(poly,score) in enumerate(zip(data['dt_polys'],data.get('dt_scores',[1.]*len(data['dt_polys'])))):
        record=restore_one(rgb,prob,np.asarray(poly,np.float32),float(score),runner,a.target_glyph_height,a.rectify,out,owner);record['line_index']=i;records.append(record)
    _write_rgb(a.output,yuv444_to_rgb(out));a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps({'rectification':a.rectify,'target_glyph_height':a.target_glyph_height,'lines':records},indent=2),encoding='utf-8');_write_gray(a.report.with_suffix('.owner.png'),owner);print(json.dumps({'lines':len(records),'updated_pixels':sum(r['updated_pixels'] for r in records),'output':str(a.output)}))
if __name__=='__main__':main()
