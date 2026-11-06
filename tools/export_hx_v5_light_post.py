"""Create a local native-resolution seven-image export without network-dependent viewer assets."""
import argparse
import copy
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_camera_2x2_viewer import render_html
from tools.light_text_postprocess import PRESETS
from tools.hx_nv12_comparison import sha


def main(args):
    """@luna-comment Input:accepted NAS evaluation/export folders. Output:exact35 full-frame PNGs,35 upright comparison PNGs, portable pages and method/provenance. Key Steps:verify baseline/final hashes, copy selected assets only, preserve IDs, render shared viewer with relative URLs, record timing and method limits. Key Params:fixed seven IDs/three global presets; no source/model modification. State/Dependencies:read-only NAS outputs and approved local export directory."""
    source=args.source;target=args.export;target.mkdir(parents=True,exist_ok=True)
    evaluation=json.loads((source/'EVALUATION.json').read_text(encoding='utf-8'));manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    ids=evaluation['ids'];assert ids==['003','014','027','035','049','055','056']
    entries={(v['id'],v['preset']):v for v in evaluation['records']}
    for folder in ['v5_baseline','lq',*PRESETS]:
        (target/folder).mkdir(exist_ok=True)
        for identity in ids:
            path=source/folder/(identity+'.png');dest=target/folder/path.name;shutil.copyfile(path,dest)
            expected=entries[(identity,'standard')]['baseline_sha256'] if folder=='v5_baseline' else (entries[(identity,folder)]['final_sha256'] if folder in PRESETS else sha(path))
            assert sha(dest)==expected
    # Full-frame view retains original camera coordinates and exact V5 baseline bytes.
    page=render_html(manifest).replace('</header>','<div style="margin-top:6px"><a href="upright.html">转正文档对比</a> · <a href="METHOD.html">Method / CPU timing</a></div></header>',1)
    (target/'index.html').write_text(page,encoding='utf-8')
    upright=copy.deepcopy(manifest);upright['title']='V5 · 转正文档对比';upright['subtitle']='相同文档坐标 · 原生像素\n基线 / 轻度 / 标准 / 增强'
    base=source.parent/'hx_upright_postblend_v2_20261022'
    asset_hashes=[]
    for row in upright['rows']:
        identity=row['id'];doc=json.loads((base/'run/documents'/(identity+'.png')/'result.json').read_text(encoding='utf-8'));item=max(doc['instances'],key=lambda value:value['confidence']);iid=f"{item['instance_id']:03d}"
        paths={'lq':base/'run/documents'/(identity+'.png')/item['image_file'],'v5':base/'results/vosr_v5_block/stages'/identity/iid/'model.png'}
        paths.update({preset:source/'stages'/preset/identity/(iid+'.png') for preset in PRESETS})
        for key,path in paths.items():
            destination=target/'upright'/key/(identity+'.png');destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
            asset_hashes.append(dict(path=str(destination.relative_to(target)),sha256=sha(destination)))
            with Image.open(destination) as pixels:
                if key=='lq':row['width'],row['height']=pixels.size
                else:assert pixels.size==(row['width'],row['height'])
            if key=='lq':row['lq']='upright/lq/'+identity+'.png'
            else:row['images'][key]='upright/'+key+'/'+identity+'.png'
    upright['models'][0]['label']='LQ · 转正文档原图'
    upright['notes']='这里显示模型实际处理的转正文档区域。\n完整相机图及原始 V5 提取文件见完整图页面和 v5_baseline 文件夹。\n所有对比使用原生 PNG，不用 JPEG 样张。\n三档参数对全部7张统一，不逐张手工调参。'
    page=render_html(upright).replace('</header>','<div style="margin-top:6px"><a href="index.html">完整相机图对比</a> · <a href="METHOD.html">Method / CPU timing</a></div></header>',1)
    (target/'upright.html').write_text(page,encoding='utf-8')
    timing=[]
    for preset in PRESETS:
        values=[v['algorithm_seconds']*1000 for v in evaluation['records'] if v['preset']==preset]
        timing.append(f"| {preset} | {np.median(values):.1f} | {min(values):.1f} | {max(values):.1f} |")
    method='''# Lightweight Classical Text Contrast and Clarity v1

## Selected sources
003, 014, 027, 035, 049, 055, 056. The v5_baseline folder contains byte-identical copies of the latest upright-replay V5 raw block256 full-frame outputs. LQ references remain direct NV12 conversions. Previous61/280 galleries are unchanged.

## Operator
1. Estimate luminance background on a 1/8 resolution image with Gaussian sigma1.5 (about12 native pixels).
2. Estimate a robust high-frequency noise floor; soft-threshold low-amplitude detail at2..6 gray levels.
3. Smoothly select dark-stroke contrast on bright backgrounds and bright-stroke contrast on dark backgrounds. Apply a bounded rational contrast gain, preserving the existing paper/screen background.
4. Apply sigma0.8 small-radius unsharp detail, limited by the adjusted3x3 local luminance envelope to bound ringing.
5. Apply one gamut-safe integer delta to all BGR channels. Pairwise channel differences are preserved exactly; no RGB-independent gain or new learned model is used.
6. With optional upstream OCR support, feather the final correction over the union of text rectangles by16 pixels. Existing detection is reused; the operator does not run OCR. Map only the additional correction onto the exact V5 baseline with the existing source-bound homography/ownership.

## Fixed presets
| Preset | Contrast gain | Sharp gain | Maximum gray-level delta |
|---|---:|---:|---:|
| mild |0.40|0.35|14|
| standard |0.75|0.65|24|
| strong |1.10|1.00|32|

Standard is an initial balanced choice; mild is useful for already clear text. Strong intentionally exposes the effect of additional gain for visual review. Parameters are fixed across all seven samples. The method increases existing stroke separation/edge steepness; it cannot reconstruct missing characters or remove model-generated color/shape artifacts.

## Runtime
OpenCV/NumPy CPU, two OpenCV threads on Remote58; no GPU or new model required. These are measured core image-processing times summed across each image's detected document regions. File decoding/encoding, inverse mapping and source-only percentile diagnostics are excluded. They are prototype CPU measurements, not a real-time throughput claim.

| Preset | Median ms/image | Min ms/image | Max ms/image |
|---|---:|---:|---:|
'''+ '\n'.join(timing)+'''

## Validation and viewing
The supplied PNGs retain native resolution. Original baseline hashes, processed hashes, per-region controls and timings are in EVALUATION.json. Synthetic tests cover flat/subthreshold noise, bright/dark text, gamut-safe common-channel correction and support boundaries. Source-only percentile contrast is a diagnostic, not PSNR/SSIM or a claim of recognition accuracy. No GT is available for this test batch.

Open upright.html for convenient upright document comparisons; index.html compares full camera frames. Both pages work without Y-drive access after export. They provide synchronized zoom/pan, ROI zoom, original-input wipe, image list and Previous/Next.

## Code
The standalone operator is tools/light_text_postprocess.py in the repository. It requires OpenCV and NumPy only. Example:

python E:/workspace/AISR/Real-ESRGAN/tools/light_text_postprocess.py --input INPUT.png --output OUTPUT.png --preset standard

Use --boxes OCR.json to reuse existing half-open text boxes. Without boxes, the operator runs on the entire provided image. The gallery evaluation applies it to upright V5 document outputs and inverse-maps the correction; direct full-camera CLI processing is a different integration route.
'''
    (target/'METHOD.md').write_text(method,encoding='utf-8');(target/'METHOD.html').write_text('<!doctype html><meta charset="utf-8"><title>Lightweight text postprocess</title><body style="max-width:1100px;margin:30px auto;font:15px/1.7 system-ui"><a href="upright.html">Upright comparison</a> · <a href="index.html">Full-frame comparison</a><pre style="white-space:pre-wrap;font:inherit">'+html.escape(method)+'</pre></body>',encoding='utf-8')
    for name in ('EVALUATION.json','presets.json'):shutil.copyfile(source/name,target/name)
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(target/'upright_manifest.json').write_text(json.dumps(upright,ensure_ascii=False,indent=2),encoding='utf-8')
    (target/'EXPORT_AUDIT.json').write_text(json.dumps(dict(status='passed',ids=ids,full_frame_pngs=35,upright_pngs=35,upright_assets=asset_hashes,html_sha256=sha(target/'index.html'),upright_html_sha256=sha(target/'upright.html')),indent=2),encoding='utf-8')
    print('PASS exported7 exact V5 baselines /21 enhanced finals / portable full-frame and upright viewers')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--export',type=Path,required=True);main(parser.parse_args())
