"""Exact quarter-turn coordinates and the final source-frame text-box blend."""
import cv2
import numpy as np


def rotation_matrix(width,height,k):
    """@luna-comment Input:native width/height and CCW quarter-turn count. Output:exact pixel-center homography. Key Steps:explicit four integer transforms. Key Params:np.rot90 convention; dimensions swap for odd turns. State/Dependencies:pure NumPy."""
    return np.asarray([[[1,0,0],[0,1,0],[0,0,1]],[[0,1,0],[-1,0,width-1],[0,0,1]],[[-1,0,width-1],[0,-1,height-1],[0,0,1]],[[0,-1,height-1],[1,0,0],[0,0,1]]][k%4],np.float64)


def final_blend(source,processed,boxes,width=16):
    """@luna-comment Input:matching BGR8 upright images and half-open OCR boxes. Output:one RGB residual blend/alpha. Key Steps:union support, padded distance, smoothstep, single rounding. Key Params:16px inward feather; no glyph mask. State/Dependencies:source immutable; exact source outside/first border, exact processed at full-strength interior."""
    if source.shape!=processed.shape or source.dtype!=np.uint8 or processed.dtype!=np.uint8 or width<=0:
        raise ValueError('Invalid final blend inputs')
    mask=np.zeros(source.shape[:2],np.uint8)
    for x0,y0,x1,y1 in boxes:
        if not 0<=x0<x1<=mask.shape[1] or not 0<=y0<y1<=mask.shape[0]:raise ValueError('Invalid blend box')
        mask[y0:y1,x0:x1]=1
    distance=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    t=np.clip((distance-1)/width,0,1);alpha=t*t*(3-2*t)
    result=np.clip(np.rint(source.astype(np.float32)+(processed.astype(np.float32)-source)*alpha[...,None]),0,255).astype(np.uint8)
    assert np.array_equal(result[alpha==0],source[alpha==0])
    return result,np.rint(alpha*255).astype(np.uint8)
