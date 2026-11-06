"""Expand frame-level IQ560 OCR manifests into segment-level BASIC V2 manifests."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Iterable
import cv2, numpy as np
import importlib.util

_CROP_PATH = Path(__file__).resolve().parents[1] / "realesrgan" / "data" / "text_line_crop_v3.py"
_CROP_SPEC = importlib.util.spec_from_file_location("text_line_crop_v3", _CROP_PATH)
if _CROP_SPEC is None or _CROP_SPEC.loader is None:
    raise ImportError(f"Unable to load crop contract: {_CROP_PATH}")
_CROP_MODULE = importlib.util.module_from_spec(_CROP_SPEC)
import sys
sys.modules[_CROP_SPEC.name] = _CROP_MODULE
_CROP_SPEC.loader.exec_module(_CROP_MODULE)
build_line_crops = _CROP_MODULE.build_line_crops
polygon_to_array = _CROP_MODULE.polygon_to_array
assess_crop_size = _CROP_MODULE.assess_crop_size
SIZE_FILTER_VERSION = _CROP_MODULE.SIZE_FILTER_VERSION
REQUIRED_ASSETS=("lq","gt","valid_mask","text_mask","ocr")
def _sha256(path:Path)->str:
    """@shanlan-comment Hash one input manifest for reproducibility."""
    digest=hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()
def _read_image_hw(path:str)->tuple[int,int]:
    """@shanlan-comment Read image dimensions and fail on missing assets."""
    image=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if image is None: raise FileNotFoundError(path)
    return int(image.shape[0]),int(image.shape[1])


def _manifest_image_hw(record: dict[str, Any]) -> tuple[int, int]:
    """@shanlan-comment Use manifest image_hw when present, otherwise decode one LQ header."""
    image_hw = record.get("image_hw") or record.get("lq_hw")
    if image_hw and len(image_hw) >= 2:
        return int(image_hw[0]), int(image_hw[1])
    return _read_image_hw(record["lq"])
def _read_polygons(ocr_path:str)->tuple[str,list[np.ndarray],list[str]]:
    """@shanlan-comment Parse OCR polygons with source-field provenance."""
    payload=json.loads(Path(ocr_path).read_text(encoding="utf-8"))
    for field in ("rec_polys","dt_polys","rec_boxes"):
        values=payload.get(field) or []
        if not values: continue
        polygons=[]
        for value in values:
            try: polygons.append(polygon_to_array(value))
            except (TypeError,ValueError): continue
        texts=payload.get("rec_texts") or payload.get("text_word") or []
        return field,polygons,[str(item) for item in texts]
    # The 20260820 V3 OCR contract stores one normalized line object per
    # detection instead of the legacy parallel ``rec_polys`` arrays. Keep the
    # legacy precedence above unchanged so older 0812 manifests reproduce
    # exactly, then adapt only the V3 ``lines[].polygon`` representation.
    lines=payload.get("lines") or []
    if lines:
        polygons=[]; texts=[]
        for line in lines:
            if not isinstance(line,dict):
                continue
            value=line.get("polygon") or line.get("original_polygon")
            try:
                polygons.append(polygon_to_array(value))
            except (TypeError,ValueError):
                continue
            texts.append(str(line.get("text", "")))
        return "lines.polygon",polygons,texts
    return "",[],[]
def _record_assets(record:dict[str,Any],line_number:int)->None:
    """@shanlan-comment Enforce complete paired assets before expansion."""
    missing=[key for key in REQUIRED_ASSETS if not record.get(key)]
    if missing: raise ValueError(f"manifest line {line_number} missing keys: {missing}")
    for key in REQUIRED_ASSETS:
        if not Path(record[key]).is_file(): raise FileNotFoundError(f"manifest line {line_number} {key}: {record[key]}")
def expand_record(
    record:dict[str,Any],
    line_number:int,
    canvas_hw:tuple[int,int],
    overlap:int,
    min_short_side_ratio:float = 0.5,
    min_long_side_ratio:float = 0.5,
):
    """@shanlan-comment Expand one frame into one row per OCR line tile."""
    _record_assets(record,line_number); image_hw=_manifest_image_hw(record)
    if record.get("gt_hw") and tuple(map(int, record["gt_hw"][:2])) != tuple(image_hw): return [],[{"line_number":line_number,"reason":"LQ_GT_SHAPE_MISMATCH"}]
    source_field,polygons,texts=_read_polygons(record["ocr"])
    if not polygons: return [],[{"line_number":line_number,"reason":"NO_VALID_OCR_POLYGON"}]
    rows=[]; filtered=[]
    for line_index,polygon in enumerate(polygons):
        try: crops=build_line_crops(polygon,image_hw,canvas_hw,overlap,line_index=line_index)
        except (TypeError,ValueError) as exc: filtered.append({"line_number":line_number,"line_index":line_index,"reason":"CROP_BUILD_FAILED","detail":str(exc)}); continue
        text=texts[line_index] if line_index<len(texts) else ""
        for crop in crops:
            size = assess_crop_size(
                crop.valid_hw,
                canvas_hw,
                crop.orientation,
                min_short_side_ratio=min_short_side_ratio,
                min_long_side_ratio=min_long_side_ratio,
            )
            if not bool(size["accepted"]):
                filtered.append({
                    "line_number": line_number,
                    "line_index": line_index,
                    "tile_index": crop.tile_index,
                    "reason": str(size["reason"]),
                    "valid_hw": list(crop.valid_hw),
                    "short_side_ratio": float(size["short_side_ratio"]),
                    "long_side_ratio": float(size["long_side_ratio"]),
                })
                continue
            source_line_id=f"{record.get('stem','frame')}::line{line_index}"
            # Direct text-line CP works on the already cropped tile. Tiling is
            # not a rejection reason for horizontal/oblique samples; vertical
            # samples remain outside this CRNN contract for the current run.
            eligible=bool(crop.orientation in ("HORIZONTAL","OBLIQUE") and crop.text_mask.sum()>0)
            rows.append({"dataset":record.get("dataset",""),"stem":record.get("stem",""),"frame_id":record.get("stem",""),"source_line_id":source_line_id,"segment_id":f"{source_line_id}::tile{crop.tile_index}","line_index":line_index,"tile_index":crop.tile_index,"tile_count":crop.tile_count,"orientation":crop.orientation,"polygon_source":source_field,"polygon_local":crop.polygon_local.astype(float).tolist(),"crop_xyxy_even":list(crop.crop_xyxy_even),"valid_hw":list(crop.valid_hw),"canvas_hw":list(canvas_hw),"valid_area":float(crop.valid_mask.sum()),"text_area":float(crop.text_mask.sum()),"size_filter_version":SIZE_FILTER_VERSION,"min_short_side_ratio":float(min_short_side_ratio),"min_long_side_ratio":float(min_long_side_ratio),"short_side_ratio":float(size["short_side_ratio"]),"long_side_ratio":float(size["long_side_ratio"]),"cp_eligible":eligible,"cp_skip_reason":"" if eligible else "VERTICAL_OR_EMPTY","is_crop_truncated":False,"lq":record["lq"],"gt":record["gt"],"valid_mask":record["valid_mask"],"text_mask":record["text_mask"],"ocr":record["ocr"],"transcription":text,"source_manifest_line":line_number})
    return rows,filtered
def build_segment_manifest(input_jsonl:Path,output_jsonl:Path,*,canvas_hw=(96,320),overlap=64,max_records=None,min_short_side_ratio=0.5,min_long_side_ratio=0.5):
    """@shanlan-comment Expand an IQ560 frame manifest and write auditable segment rows."""
    rows=[]; filtered=[]; frame_count=0
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number,line in enumerate(handle,1):
            if not line.strip(): continue
            if max_records is not None and frame_count>=max_records: break
            expanded,rejected=expand_record(json.loads(line),line_number,canvas_hw,overlap,min_short_side_ratio,min_long_side_ratio); rows.extend(expanded); filtered.extend(rejected); frame_count+=1
    output_jsonl.parent.mkdir(parents=True,exist_ok=True)
    with output_jsonl.open("w",encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    return {"input_manifest":str(input_jsonl),"input_sha256":_sha256(input_jsonl),"output_manifest":str(output_jsonl),"canvas_hw":list(canvas_hw),"overlap":int(overlap),"size_filter_version":SIZE_FILTER_VERSION,"min_short_side_ratio":float(min_short_side_ratio),"min_long_side_ratio":float(min_long_side_ratio),"frame_count":frame_count,"segment_count":len(rows),"source_line_count":len({r["source_line_id"] for r in rows}),"multi_tile_line_count":len({r["source_line_id"] for r in rows if r["tile_count"]>1}),"cp_eligible_count":sum(bool(r["cp_eligible"]) for r in rows),"orientation_counts":{k:sum(r["orientation"]==k for r in rows) for k in ("HORIZONTAL","VERTICAL","OBLIQUE")},"filtered_count":len(filtered),"filtered_reason_counts":{reason:sum(item.get("reason")==reason for item in filtered) for reason in sorted({item.get("reason") for item in filtered})},"filtered":filtered,"output_sha256":_sha256(output_jsonl)}
def main(argv:Iterable[str]|None=None)->int:
    """@shanlan-comment Build a segment manifest and JSON summary from CLI arguments."""
    parser=argparse.ArgumentParser(); parser.add_argument("--input-jsonl",type=Path,required=True); parser.add_argument("--output-jsonl",type=Path,required=True); parser.add_argument("--summary-json",type=Path,required=True); parser.add_argument("--canvas-h",type=int,default=96); parser.add_argument("--canvas-w",type=int,default=320); parser.add_argument("--overlap",type=int,default=64); parser.add_argument("--min-short-side-ratio",type=float,default=0.5); parser.add_argument("--min-long-side-ratio",type=float,default=0.5); parser.add_argument("--max-records",type=int,default=None); args=parser.parse_args(argv)
    summary=build_segment_manifest(args.input_jsonl,args.output_jsonl,canvas_hw=(args.canvas_h,args.canvas_w),overlap=args.overlap,max_records=args.max_records,min_short_side_ratio=args.min_short_side_ratio,min_long_side_ratio=args.min_long_side_ratio); args.summary_json.parent.mkdir(parents=True,exist_ok=True); args.summary_json.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
