"""Standalone classical luminance contrast/clarity postprocess for TextSR RGB output."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

PRESETS={
    'mild':dict(contrast=.40,sharp=.35,max_delta=14.),
    'standard':dict(contrast=.75,sharp=.65,max_delta=24.),
    'strong':dict(contrast=1.10,sharp=1.00,max_delta=32.),
}


def support_alpha(shape,boxes,feather=16):
    """@luna-comment Input:image HW/optional half-open OCR boxes. Output:float union alpha. Key Steps:validate support, smoothstep padded distance. Key Params:16px inward feather; None means full image. State/Dependencies:no detector or learned asset; empty boxes intentionally mean no processing."""
    if boxes is None:return np.ones(shape,np.float32)
    mask=np.zeros(shape,np.uint8)
    for x0,y0,x1,y1 in boxes:
        if not 0<=x0<x1<=shape[1] or not 0<=y0<y1<=shape[0]:raise ValueError('Invalid support box')
        mask[y0:y1,x0:x1]=1
    distance=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    t=np.clip((distance-1)/feather,0,1)
    return t*t*(3-2*t)


def enhance_bgr(image,preset='standard',boxes=None):
    """@luna-comment
    Input:BGR uint8 image/preset/optional existing text boxes. Output:BGR8 plus source-only timing/control telemetry.
    Key Steps:1/8-scale background, robust noise floor, smooth dark/light polarity contrast, small-radius USM,
    local-envelope clamp, one feathered common-channel RGB correction limited by available gamut.
    Key Params:background sigma12px equivalent, sharp sigma.8, noise2..6, preset delta14/24/32,16px support feather.
    State/Dependencies:OpenCV/NumPy CPU only; no OCR/model/GT; source and channel differences remain unchanged.
    """
    started=time.perf_counter()
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected BGR8')
    if preset not in PRESETS:raise ValueError('Unknown preset')
    config=PRESETS[preset];h,w=image.shape[:2]
    y=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
    small=cv2.resize(y,(max(1,(w+7)//8),max(1,(h+7)//8)),interpolation=cv2.INTER_AREA)
    background=cv2.resize(cv2.GaussianBlur(small,(0,0),1.5,borderType=cv2.BORDER_REFLECT_101),(w,h),interpolation=cv2.INTER_LINEAR)
    fine=y-cv2.GaussianBlur(y,(0,0),.8,borderType=cv2.BORDER_REFLECT_101)
    samples=fine[::8,::8];sigma=float(np.median(np.abs(samples-np.median(samples)))*1.4826)
    threshold=float(np.clip(1.5*sigma,2.,6.))
    residual=y-background
    paper=np.clip((background-80.)/96.,0,1);paper=paper*paper*(3-2*paper)
    dark=-np.maximum(-residual-threshold,0)*paper
    bright=np.maximum(residual-threshold,0)*(1-paper)
    contrast=config['contrast']*(dark+bright)
    contrast=contrast/(1+np.abs(contrast)/config['max_delta'])
    adjusted=y+contrast
    detail=adjusted-cv2.GaussianBlur(adjusted,(0,0),.8,borderType=cv2.BORDER_REFLECT_101)
    detail=np.sign(detail)*np.maximum(np.abs(detail)-threshold*.5,0)
    low=cv2.erode(adjusted,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REFLECT_101)
    high=cv2.dilate(adjusted,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REFLECT_101)
    target=np.clip(adjusted+config['sharp']*detail,low,high)
    alpha=support_alpha((h,w),boxes)
    delta=np.clip(target-y,-config['max_delta'],config['max_delta'])*alpha
    # A single integer delta for all BGR channels preserves their pairwise differences exactly.
    lower=-image.min(2).astype(np.float32);upper=255-image.max(2).astype(np.float32)
    delta=np.rint(np.minimum(np.maximum(delta,lower),upper)).astype(np.int16)
    result=(image.astype(np.int16)+delta[:,:,None]).astype(np.uint8)
    assert np.array_equal(result[alpha==0],image[alpha==0])
    seconds=time.perf_counter()-started
    active=alpha>0
    telemetry=dict(preset=preset,parameters=config,seconds=seconds,megapixels=h*w/1e6,noise_sigma=sigma,noise_threshold=threshold,changed_pixels=int(np.count_nonzero(delta)),support_pixels=int(active.sum()),max_abs_delta=int(np.abs(delta).max()),outside_support_unchanged=True,common_channel_delta=True)
    if active.any():
        output_y=cv2.cvtColor(result,cv2.COLOR_BGR2YCrCb)[:,:,0]
        telemetry['support_p95_p05_before']=float(np.percentile(y[active],95)-np.percentile(y[active],5))
        telemetry['support_p95_p05_after']=float(np.percentile(output_y[active],95)-np.percentile(output_y[active],5))
    return result,telemetry


def main():
    """@luna-comment Input:image/output/preset/optional JSON boxes CLI. Output:lossless enhanced image and telemetry. Key Steps:Unicode-safe decode, CPU operator, encode and JSON report. Key Params:default standard; boxes may be list or dict.boxes. State/Dependencies:only local files/OpenCV/NumPy; no downloads."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preset',choices=PRESETS,default='standard');parser.add_argument('--boxes',type=Path);parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();cv2.setNumThreads(args.threads)
    image=cv2.imdecode(np.frombuffer(args.input.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Cannot decode input')
    boxes=json.loads(args.boxes.read_text(encoding='utf-8')) if args.boxes else None
    if isinstance(boxes,dict):boxes=boxes['boxes']
    result,report=enhance_bgr(image,args.preset,boxes)
    success,data=cv2.imencode('.png',result,[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not success:raise ValueError('Cannot encode output')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(data.tobytes());args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
