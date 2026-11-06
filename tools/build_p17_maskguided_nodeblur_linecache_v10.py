"""Build the V10 no-blur mask-guided line prediction cache once."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import build_basic_text_line_inference_cache as old
from tools import infer_basic_text_line_full_image as src
from realesrgan.data.basic_text_line_full_frame import attach_line_index,build_p17_height96_tiled_full_frame_segments,pad_even_rgb,parse_ocr_json

def _sha(p:Path)->str:
    """@luna-comment Hash one frozen input or produced artifact for the V10 cache receipt.

    Input: Existing filesystem path. Output: SHA-256 digest. Key Steps: hash bytes once.
    Key Params: None. State/Dependencies: Read-only filesystem access.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()

def _cache_line(frame,line,segments,runner,canvas):
    """@luna-comment Assemble one recovered OCR line from bicubic P17 tile predictions.

    Input: Frame, recovered line, P17 segments, model runner, and canvas shape.
    Output: Fused prediction, valid support, and frame crop bounds.
    Key Steps: Bicubic resize then clamp only the model crop before P17, restore bicubically, and blend tiles.
    Key Params: The frozen geometry uses 96x480 canvas and complementary tile weights.
    State/Dependencies: Calls the model runner once per canvas; frame pixels are never changed.
    """
    x0,y0,x1,y1=min(s.crop_xyxy[0] for s in segments),min(s.crop_xyxy[1] for s in segments),max(s.crop_xyxy[2] for s in segments),max(s.crop_xyxy[3] for s in segments); total=np.zeros((3,y1-y0,x1-x0),np.float32); weight=np.zeros((y1-y0,x1-x0),np.float32)
    for s in segments:
        l,t,r,b=s.crop_xyxy; hw=s.model_valid_hw or s.valid_hw; model=np.clip(src._resize_hwc(frame[t:b,l:r],hw,interpolation=src.cv2.INTER_CUBIC),0,1); canvas_rgb=src._pad_canvas(model,s.model_canvas_hw or canvas,edge_replicate=True); pred=src._resize_chw(np.clip(runner.run_rgb(canvas_rgb),0,1)[:,:hw[0],:hw[1]],s.valid_hw,interpolation=src.cv2.INTER_CUBIC); h,w=s.valid_hw; coef=old.tile_axis_weight(s.valid_hw,s.axis,s.tile_count); oy,ox=t-y0,l-x0; total[:,oy:oy+h,ox:ox+w]+=pred[:,:h,:w]*coef[None]; weight[oy:oy+h,ox:ox+w]+=coef
    valid=weight>1e-6; out=np.zeros_like(total); out[:,valid]=total[:,valid]/weight[valid][None]; return out,valid,(x0,y0,x1,y1)

def main()->None:
    """@luna-comment Build model predictions after the frozen raw-mask geometry recovery route.

    Input: One-image input/OCR directories, frozen no-blur YAML/checkpoint/statistics, output directory.
    Output: Cache manifest, frame arrays, recovered per-line prediction arrays, and geometry receipts.
    Key Steps: run one full-frame raw probe, recover only trusted OCR y-bounds, build 96x480 bicubic
      segments, then run each model canvas once. Key Params: context ratios are .12/.03 and no pixel
      postprocess is used. State/Dependencies: This is the only V10 stage permitted to load OCR/model.
    """
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--ocr-dir',type=Path,required=True); ap.add_argument('--network-config',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--mask-guided-geometry-stats',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--expected-images',type=int,default=1); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    images=sorted(x for x in a.input_dir.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png'))
    if len(images)!=a.expected_images: raise ValueError('unexpected image count')
    a.output_dir.mkdir(parents=True,exist_ok=True); device=torch.device(a.device); network,digest=src._load_network(a.checkpoint,device,a.network_config); config=src.load_lq_preprocess_config(a.network_config); runner=src.NativeY4UVTileRunner(network,device,config); stats=src._load_mask_guided_geometry_stats(a.mask_guided_geometry_stats); frames=[]; total_segments=0; accepted=fallback=0
    for image in images:
        ocr=a.ocr_dir/(image.stem+'.json'); original=src._read_rgb(image); frame,original_hw=pad_even_rgb(original); lines=parse_ocr_json(json.loads(ocr.read_text(encoding='utf-8')),frame.shape[:2]); probe=torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).contiguous().to(device=device,dtype=torch.float32)
        with torch.no_grad(): _,debug=src.preprocess_p17_torch_gpu_with_debug(probe,config)
        raw=debug['text_mask'][0,0].detach().cpu().numpy()>.5; directory=a.output_dir/'frames'/image.stem; directory.mkdir(parents=True,exist_ok=True); src._write_gray(directory/'p17_geometry_mask_raw.png',raw); lines,decisions=src._recover_mask_guided_ocr_lines(lines,raw,frame,stats,directory); (directory/'geometry_recovery.json').write_text(json.dumps({'stats':stats,'lines':[decisions[k] for k in sorted(decisions)]},ensure_ascii=False,indent=2),encoding='utf-8'); accepted+=sum(x['status']=='accepted' for x in decisions.values()); fallback+=sum(x['status']=='fallback' for x in decisions.values()); np.save(directory/'frame_yuv444.npy',src.rgb_to_yuv444(frame)); src._write_rgb(directory/'frame_input.png',frame); records=[]; canvas=src.resolve_canvas_hw(a.network_config)
        for line in lines:
            segments=attach_line_index(build_p17_height96_tiled_full_frame_segments(line.polygon_xy,frame.shape[:2],canvas,overlap=64,vertical_context=.12,horizontal_context=.03)[0],line.line_index)
            if not segments: continue
            prediction,valid,crop=_cache_line(frame,line,segments,runner,canvas); ld=directory/'lines'/f'{line.line_index:04d}'; ld.mkdir(parents=True,exist_ok=True); np.save(ld/'prediction_yuv444.npy',prediction); np.save(ld/'valid_support.npy',valid); geometry={'line_index':line.line_index,'crop_xyxy':list(crop),'line_to_frame_offset':[crop[0],crop[1]],'detection_score':float(line.score),'text':line.text,'geometry_recovery':decisions[line.line_index]}; route={'resize_mode':'p17_height96_tile_pad','input_interpolation':'bicubic','restore_interpolation':'bicubic','postprocess':'none','segments':[{'crop_xyxy':list(s.crop_xyxy),'tile_index':s.tile_index,'tile_count':s.tile_count} for s in segments]}; (ld/'geometry.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)); (ld/'route.json').write_text(json.dumps(route,ensure_ascii=False,indent=2)); artifact={n:(old.array_record(ld/n) if n.endswith('.npy') else old.json_record(ld/n)) for n in ('prediction_yuv444.npy','valid_support.npy','geometry.json','route.json')}; (ld/'artifact.sha256.json').write_text(json.dumps(artifact,indent=2)); records.append({'line_index':line.line_index,'directory':str(ld.relative_to(a.output_dir)).replace('\\','/')}); total_segments+=len(segments)
        meta={'image_id':image.stem,'directory':str(directory.relative_to(a.output_dir)).replace('\\','/'),'input_sha256':_sha(image),'ocr_sha256':_sha(ocr),'original_hw':list(original_hw),'lines':records}; (directory/'frame.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); frames.append(meta)
    manifest={'cache_id':'p17_maskguided_nodeblur_preprocess_input_r1_bglf_uv_linecache_v10_20260923','schema_version':1,'detector':'PaddleOCR v6 tiny','checkpoint':str(a.checkpoint),'checkpoint_sha256':_sha(a.checkpoint),'network_config':str(a.network_config),'network_config_sha256':_sha(a.network_config),'geometry_stats_sha256':_sha(a.mask_guided_geometry_stats),'resize_mode':'p17_height96_tile_pad','input_interpolation':'bicubic','restore_interpolation':'bicubic','postprocess':'none','p17_wiener_bypass_enabled':bool(config.get('p17_wiener_bypass_enabled')),'frames':frames}; (a.output_dir/'cache_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)); (a.output_dir/'cache_manifest.sha256').write_text(hashlib.sha256(old.canonical(manifest)).hexdigest()); (a.output_dir/'cache_build_receipt.json').write_text(json.dumps({'status':'complete','raw_probe_calls':len(frames),'accepted_geometry_lines':accepted,'fallback_geometry_lines':fallback,'segments':total_segments,'runner_calls':runner.calls,'checkpoint_sha256':digest},indent=2))
if __name__=='__main__': main()
