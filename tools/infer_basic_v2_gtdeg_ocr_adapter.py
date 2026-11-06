"""OCR-JSON-driven GT-deg inference with vertical-line rotation normalization."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from realesrgan.data.basic_text_line_full_frame import parse_ocr_json
from realesrgan.utils import YUV2RGB
from realesrgan.data.basic_text_line_y4uv_v3 import numpy_rgb_to_y4uv, y4uv_to_yuv444
from tools.basic_v2_eval_network import build_eval_network
from tools.infer_basic_text_line_full_image import rgb_to_nv21, _write_rgb

def polygon_feather_mask(shape, polygon, width=16):
    hard=np.zeros(shape,np.uint8); cv2.fillPoly(hard,[np.asarray(polygon,np.int32)],1)
    if not hard.any(): return hard.astype(np.float32)
    outside=cv2.distanceTransform(1-hard,cv2.DIST_L2,3); alpha=np.ones(shape,np.float32); band=outside<float(width); alpha[band]=.5*(1+np.cos(np.pi*np.clip(outside[band]/float(width),0,1))); alpha[outside>=float(width)]=0.; alpha[hard.astype(bool)]=1.; return alpha

def _run(model, device, tile):
    """Run either a legacy YUV444 or a native six-channel Y4UV checkpoint."""
    with torch.inference_mode():
        if hasattr(model, "expected_input_hw"):
            value = numpy_rgb_to_y4uv(tile).unsqueeze(0).to(device)
            output = y4uv_to_yuv444(model(value))
        else:
            output = model(torch.from_numpy(rgb_to_nv21(tile)).unsqueeze(0).to(device))
    return YUV2RGB(output,format="RGB",is_train=False,uv_normalized=True)[0].cpu().permute(1,2,0).numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--ocr-json',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--network-config',required=True); p.add_argument('--output',required=True); p.add_argument('--debug-dir',required=True); p.add_argument('--min-score',type=float,default=.2); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    dev=torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu'); model,digest=build_eval_network(Path(a.checkpoint),dev,Path(a.network_config))
    bgr=cv2.imread(a.input); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.; out=rgb.copy(); lines=parse_ocr_json(json.loads(Path(a.ocr_json).read_text(encoding='utf-8')),rgb.shape[:2]); debug=Path(a.debug_dir); debug.mkdir(parents=True,exist_ok=True); records=[]
    for i,line in enumerate(lines):
        if line.score < a.min_score: records.append({'line':i,'status':'skip_score','score':line.score}); continue
        q=np.asarray(line.polygon_xy); x0,y0=np.floor(q.min(0)).astype(int); x1,y1=np.ceil(q.max(0)).astype(int); x0=max(0,x0//2*2); y0=max(0,y0//2*2); x1=min(rgb.shape[1],(x1+1)//2*2); y1=min(rgb.shape[0],(y1+1)//2*2)
        crop=rgb[y0:y1,x0:x1]; vertical=crop.shape[0] > crop.shape[1]*1.2
        oriented=cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE) if vertical else crop; w=max(2,int(round(oriented.shape[1]*96/oriented.shape[0]))); resized=cv2.resize(oriented,(w,96),interpolation=cv2.INTER_AREA)
        starts=[0] if w<=480 else list(range(0,w,416)); starts[-1]=min(starts[-1],max(0,w-480)); starts=list(dict.fromkeys(starts)); acc=np.zeros_like(resized); weight=np.zeros((96,w,1),np.float32); d=debug/f'line_{i:04d}'
        for t,s in enumerate(starts):
            e=min(w,s+480); tile=resized[:,s:e]; tile=cv2.copyMakeBorder(tile,0,0,0,480-(e-s),cv2.BORDER_REPLICATE); pred=_run(model,dev,tile); acc[:,s:e]+=pred[:,:e-s]; weight[:,s:e]+=1; _write_rgb(d/f'tile_{t:03d}_input_96x480.png',tile); _write_rgb(d/f'tile_{t:03d}_output_96x480.png',pred)
        restored=cv2.resize(acc/np.maximum(weight,1),(oriented.shape[1],oriented.shape[0]),interpolation=cv2.INTER_CUBIC); restored=cv2.rotate(restored,cv2.ROTATE_90_COUNTERCLOCKWISE) if vertical else restored
        local=q-np.asarray([x0,y0]); alpha=polygon_feather_mask((y1-y0,x1-x0),local)[...,None]
        out[y0:y1,x0:x1]=alpha*restored+(1.0-alpha)*out[y0:y1,x0:x1]
        records.append({'line':i,'status':'processed','score':line.score,'vertical_rotated':vertical,'bbox':[x0,y0,x1,y1],'tiles':len(starts),'pasteback':'polygon_feather_alpha'})
    _write_rgb(Path(a.output),out); Path(str(a.output)+'.json').write_text(json.dumps({'checkpoint_sha256':digest,'ocr_json':a.ocr_json,'min_score':a.min_score,'lines':records},ensure_ascii=False,indent=2,default=int),encoding='utf-8')
if __name__=='__main__': main()
