"""Run V8 cached edge-local S-curve and C1 outer-frame join."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import run_cached_text_line_bimodal_scurve as common
from tools import run_cached_text_line_bimodal_scurve_v7 as v7
MODE="cached_maskraw_edge_local_scurve_outer_c1_feather_v8"

def fill_v8(prediction: np.ndarray, frame_crop: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray,dict[str,np.ndarray],dict[str,Any]]:
    """@luna-comment Apply V8 only as a continuous luminance increment in the true transition band.

    Input: Cached prediction/frame YUV crops and valid support.
    Output: V8 crop, V7-derived raw-mask evidence, and V8 telemetry.
    Key Steps: Reuse V7's single P10/statistical hard gates; window its mapped-Y delta
      to t=.10..90; preserve prediction platforms; join only outer rings 13..28 to frame.
    Key Params: Window .10/.90 and C1 raised cosine r=13..28 are frozen by V8.
    State/Dependencies: Cache-only; no OCR/model/alpha compositor and UV stays prediction-exact.
    """
    provisional,evidence,telemetry=v7.fill_v7(prediction,frame_crop,valid)
    if telemetry["status"]!="processed": return provisional,evidence,telemetry
    b0,f0=float(telemetry["B0"]),float(telemetry["F0"]); sign=float(telemetry["polarity_sign"])
    t=np.clip(sign*(prediction[0]-b0)/abs(f0-b0),0,1)
    q=np.clip((t-.10)/.80,0,1); eta=.5-.5*np.cos(np.pi*q)
    eta[(t<=.10)|(t>=.90)]=0.0
    local=prediction.copy(); local[0]=prediction[0]+eta*(provisional[0]-prediction[0])
    rings=common._rings(evidence["raw"],valid,28); core=evidence["core"]
    output=frame_crop.copy(); output[:,core]=local[:,core]
    weights=np.zeros_like(eta,np.float32)
    for index,ring in enumerate(rings[12:],13):
        w=.5-.5*np.cos(np.pi*(index-12)/16.0)
        output[:,ring]=(1-w)*prediction[:,ring]+w*frame_crop[:,ring]; weights[ring]=w
    evidence["weight"]=weights; evidence["edge_window_weight"]=eta.astype(np.float32)
    telemetry.update({"mode":MODE,"edge_window_low":.10,"edge_window_high":.90,"edge_window_changed_pixels":int(((eta>0)&core).sum()),"flat_platform_changed_pixels":int(((eta==0)&core&(np.abs(local[0]-prediction[0])>1e-8)).sum()),"outer_join_start":13,"outer_join_end":28,"outer_join_kind":"c1_raised_cosine","uv_preserved_in_core":bool(np.array_equal(output[1:,core],prediction[1:,core]))})
    return output,evidence,telemetry

def run(manifest_path:Path,cache_root:Path,output_root:Path)->None:
    """@luna-comment Run V8 line cache processing and emit required review artifacts.

    Input: Frozen V8 manifest, immutable cache, fresh output location.
    Output: Full image, line routes, diagnostics, and receipts.
    Key Steps: Verify cache, call fill_v8 per ascending line, use first-write ownership.
    Key Params: Manifest mode and V8 edge/join geometry are fixed.
    State/Dependencies: Cache arrays only; no model or OCR is executed.
    """
    exp=json.loads(manifest_path.read_text())
    if exp.get("authorized_postprocess")!=MODE: raise ValueError("manifest mode mismatch")
    cache=common.verify_cache(cache_root); output_root.mkdir(parents=True,exist_ok=True); summary={"experiment_id":exp["experiment_id"],"mode":MODE,"images":[]}
    for frame in cache["frames"]:
        cf=np.load(cache_root/frame["directory"]/'frame_yuv444.npy'); result=cf.copy(); taken=np.zeros(cf.shape[1:],bool); routes=[]
        for line in sorted(frame["lines"],key=lambda x:x["line_index"]):
            d=cache_root/line["directory"]; p=np.load(d/'prediction_yuv444.npy'); valid=np.load(d/'valid_support.npy').astype(bool); l,t,r,b=json.loads((d/'geometry.json').read_text())["crop_xyxy"]
            q,e,tele=fill_v8(p,cf[:,t:b,l:r],valid); prior=taken[t:b,l:r]; write=valid&~prior; result[:,t:b,l:r][:,write]=q[:,write]; taken[t:b,l:r][write]=True
            dbg=output_root/'debug'/frame['image_id']/f"line_{line['line_index']:04d}"
            for name,key in (("valid_support.png","valid"),("mask_raw.png","raw"),("foreground_stat_samples.png","foreground_samples"),("background_stat_samples.png","background_samples"),("core_support.png","core"),("transition_band_before_after.png","transition_after")):
                common._png(dbg/name,np.where(e[key],255,0).astype(np.uint8))
            common._png(dbg/'outer_feather_weights.png',np.rint(e['weight']*255).astype(np.uint8)); common._png(dbg/'edge_window_weight.png',np.rint(e.get('edge_window_weight',np.zeros_like(valid))*255).astype(np.uint8))
            common._rgb_png(dbg/'prediction_before.png',p);common._rgb_png(dbg/'cache_frame_crop.png',cf[:,t:b,l:r]);common._rgb_png(dbg/'reconstructed_line.png',q);common._png(dbg/'delta_y_signed.png',common._delta_image(q[0]-p[0])); common._histogram_debug(dbg/'histogram_before_after.png',p[0,valid],q[0,valid],tele)
            tele.update({"line_index":line['line_index'],"write_pixels":int(write.sum()),"overlap_conflict_pixels":int((valid&prior).sum())});(dbg/'line_telemetry.json').write_text(json.dumps(tele,ensure_ascii=False,indent=2));routes.append(tele)
        common._rgb_png(output_root/f"{frame['image_id']}.png",result);(output_root/f"{frame['image_id']}.routes.json").write_text(json.dumps(routes,ensure_ascii=False,indent=2));summary['images'].append({'image_id':frame['image_id'],'lines':len(routes)})
    summary['manifest_sha256']=common._sha256(manifest_path);(output_root/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(output_root/'run_receipt.json').write_text(json.dumps({'manifest_sha256':summary['manifest_sha256'],'mode':MODE},indent=2));(output_root/'command.log').write_text('offline V8 cached postprocess; OCR/model not executed\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.manifest,a.cache_root,a.output_dir)
