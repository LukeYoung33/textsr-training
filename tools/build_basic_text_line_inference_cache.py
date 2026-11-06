"""One-time P17 OCR/model cache builder; this is not an offline experiment runner."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import infer_basic_text_line_full_image as source
from realesrgan.data.basic_text_line_full_frame import attach_line_index, build_p17_height96_tiled_full_frame_segments, pad_even_rgb, parse_ocr_json, tile_axis_weight

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
def array_record(path:Path)->dict[str,Any]:
    a=np.load(path,allow_pickle=False);return {'sha256':sha(path),'shape':list(a.shape),'dtype':str(a.dtype),'bytes':path.stat().st_size}
def json_record(path:Path)->dict[str,Any]:return {'sha256':sha(path),'shape':[],'dtype':'json','bytes':path.stat().st_size}
def cache_line(frame_rgb, line, segments, runner, canvas_hw):
    minx,miny,maxx,maxy=min(s.crop_xyxy[0] for s in segments),min(s.crop_xyxy[1] for s in segments),max(s.crop_xyxy[2] for s in segments),max(s.crop_xyxy[3] for s in segments)
    total=np.zeros((3,maxy-miny,maxx-minx),np.float32); weight=np.zeros(total.shape[1:],np.float32)
    for segment in segments:
        l,t,r,b=segment.crop_xyxy; crop=frame_rgb[t:b,l:r]; model_hw=segment.model_valid_hw or segment.valid_hw
        model=source._resize_hwc(crop,model_hw,interpolation=source.cv2.INTER_CUBIC); canvas=source._pad_canvas(model,segment.model_canvas_hw or canvas_hw,edge_replicate=True)
        pred=np.clip(runner.run_rgb(canvas),0,1)[:,:model_hw[0],:model_hw[1]]; pred=source._resize_chw(pred,segment.valid_hw,interpolation=source.cv2.INTER_CUBIC)
        h,w=segment.valid_hw; coef=tile_axis_weight(segment.valid_hw,segment.axis,segment.tile_count); y0,x0=t-miny,l-minx
        total[:,y0:y0+h,x0:x0+w]+=pred[:,:h,:w]*coef[None];weight[y0:y0+h,x0:x0+w]+=coef
    valid=weight>1e-6; prediction=np.zeros_like(total);prediction[:,valid]=total[:,valid]/weight[valid][None]
    return prediction,valid,(minx,miny,maxx,maxy)
def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--ocr-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--network-config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    device=source.torch.device(a.device);network,digest=source._load_network(a.checkpoint,device,a.network_config);runner=source.NativeY4UVTileRunner(network,device,source.load_lq_preprocess_config(a.network_config));canvas=source.resolve_canvas_hw(a.network_config);frames=[]
    for image_path in sorted(x for x in a.input_dir.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png')):
        ocr=a.ocr_dir/(image_path.stem+'.json'); original=source._read_rgb(image_path);frame,original_hw=pad_even_rgb(original);lines=parse_ocr_json(json.loads(ocr.read_text(encoding='utf-8')),frame.shape[:2]);frame_dir=a.output_dir/'frames'/image_path.stem;frame_dir.mkdir(parents=True,exist_ok=True);np.save(frame_dir/'frame_yuv444.npy',source.rgb_to_yuv444(frame));source._write_rgb(frame_dir/'frame_input.png',frame);line_records=[]
        for line in lines:
            segments=attach_line_index(build_p17_height96_tiled_full_frame_segments(line.polygon_xy,frame.shape[:2],canvas,overlap=64,vertical_context=.12,horizontal_context=.03)[0],line.line_index)
            if not segments:continue
            prediction,valid,crop=cache_line(frame,line,segments,runner,canvas);directory=frame_dir/'lines'/f'{line.line_index:04d}';directory.mkdir(parents=True,exist_ok=True);np.save(directory/'prediction_yuv444.npy',prediction);np.save(directory/'valid_support.npy',valid)
            geometry={'line_index':line.line_index,'crop_xyxy':list(crop),'line_to_frame_offset':[crop[0],crop[1]],'detection_score':float(line.score),'text':line.text};route={'resize_mode':'p17_height96_tile_pad','input_interpolation':'bicubic','restore_interpolation':'bicubic','segments':[{'crop_xyxy':list(s.crop_xyxy),'tile_index':s.tile_index,'tile_count':s.tile_count} for s in segments]};(directory/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding='utf-8');(directory/'route.json').write_text(json.dumps(route,ensure_ascii=False,indent=2),encoding='utf-8');artifact={name:(array_record(directory/name) if name.endswith('.npy') else json_record(directory/name)) for name in ('prediction_yuv444.npy','valid_support.npy','geometry.json','route.json')};(directory/'artifact.sha256.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8');line_records.append({'line_index':line.line_index,'directory':str(directory.relative_to(a.output_dir)).replace('\\','/')})
        frame_meta={'image_id':image_path.stem,'directory':str(frame_dir.relative_to(a.output_dir)).replace('\\','/'),'input_sha256':sha(image_path),'ocr_sha256':sha(ocr),'original_hw':list(original_hw),'lines':line_records};(frame_dir/'frame.json').write_text(json.dumps(frame_meta,ensure_ascii=False,indent=2),encoding='utf-8');frames.append(frame_meta)
    manifest={'cache_id':'p17_all96_original_r1_bglf_uv_linecache_20260923_v1','schema_version':1,'detector':'PaddleOCR v6 tiny','checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'network_config':str(a.network_config),'network_config_sha256':sha(a.network_config),'resize_mode':'p17_height96_tile_pad','input_interpolation':'bicubic','restore_interpolation':'bicubic','builder_sha256':sha(Path(__file__)),'frames':frames};(a.output_dir/'cache_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(a.output_dir/'cache_manifest.sha256').write_text(hashlib.sha256(canonical(manifest)).hexdigest(),encoding='utf-8');(a.output_dir/'cache_build_receipt.json').write_text(json.dumps({'status':'complete','cache_id':manifest['cache_id'],'runner_calls':runner.calls},indent=2),encoding='utf-8')
if __name__=='__main__':main()
