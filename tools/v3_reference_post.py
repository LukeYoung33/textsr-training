"""Experimental dual-polarity post curve driven only by source V3 node fields."""
import cv2
import numpy as np

from evaluation.document_box_pre640_astra_20261027 import pre640
from tools.native_y_curve_four_images import preview_bgr


NODES = np.r_[np.arange(0, 640, 8), 639].astype(np.int32)


def smoothstep01(value):
    """@luna-comment Input: scalar or array occupancy evidence. Output: clipped cubic0..1. Key Steps: continuous smoothstep. Key Params: endpoints0/1. State/Dependencies: no model-output statistics."""
    value = np.clip(value, 0.0, 1.0)
    return value*value*(3.0-2.0*value)


def anchored_dark_map(y, upper, offset=12.0):
    """@luna-comment Input: Y8-domain float values and fixed upper source rail. Output: monotone g3 anchored dark mapping. Key Steps: A=max(upper-offset,0); apply cubic polynomial only Y<A. Key Params: gamma3, offset12. State/Dependencies: source-rail-only parameter; no output histogram/calibration."""
    y = np.asarray(y, np.float32)
    anchor = np.maximum(np.asarray(upper, np.float32)-offset, 0.0)
    active = (anchor>0.0)&(y<anchor)
    t = np.zeros(np.broadcast_shapes(y.shape, anchor.shape), np.float32)
    np.divide(y, anchor, out=t, where=active)
    mapped = anchor*t*t*t*(3.0-2.0*t)
    return np.where(active,mapped,y).astype(np.float32)


def node_tables(low, high, population, strengths=(.35,1.0)):
    """@luna-comment Input: repaired L3, original H and original high-class fraction on81x81 source nodes. Output: per-strength float32 node LUTs plus fixed-field telemetry. Key Steps: source-only smooth polarity and span support; evaluate both anchored mirrors at all256 Y codes. Key Params: g3/T12/support24, lambda.35/1. State/Dependencies: output image never enters table construction; no H-confidence gate."""
    low = np.asarray(low,np.float32)
    high = np.asarray(high,np.float32)
    population = np.asarray(population,np.float32)
    if low.shape != high.shape or low.shape != population.shape or low.shape != (len(NODES),len(NODES)):
        raise ValueError('Expected matching81x81 V3 fields')
    if (not np.isfinite(low).all() or not np.isfinite(high).all() or
            not np.isfinite(population).all() or np.any(low>high) or
            np.any(low<0) or np.any(high>255) or np.any((population<0)|(population>1))):
        raise ValueError('Invalid source fields')
    y = np.arange(256,dtype=np.float32)[None,None,:]
    low3,high3,p3 = low[:,:,None],high[:,:,None],population[:,:,None]
    dark = anchored_dark_map(y,high3)
    bright = 255.0-anchored_dark_map(255.0-y,255.0-low3)
    dark_weight = smoothstep01((p3-.5)/.25)
    bright_weight = smoothstep01((.5-p3)/.25)
    span = np.maximum(high3-low3,0.0)
    support = span*span/(span*span+24.0*24.0)
    correction = support*(dark_weight*(dark-y)+bright_weight*(bright-y))
    tables = {}
    for strength in strengths:
        if not 0.0 <= strength <= 1.0:
            raise ValueError('Lambda outside [0,1]')
        table = np.broadcast_to(y,correction.shape).copy() if strength==0.0 else y+strength*correction
        table = np.asarray(table,np.float32)
        if not np.isfinite(table).all() or np.min(np.diff(table,axis=2)) < -1e-4:
            raise AssertionError('Nonmonotone or nonfinite node table')
        if np.any(np.abs(table[high==low]-np.arange(256,dtype=np.float32))>1e-5):
            raise AssertionError('Zero-span node not identity')
        tables[strength]=table
    telemetry={'nodes':int(low.size),'dark_nodes':int(np.count_nonzero(dark_weight)),
               'bright_nodes':int(np.count_nonzero(bright_weight)),
               'neutral_nodes':int(np.count_nonzero((dark_weight==0)&(bright_weight==0))),
               'low_minmax':[float(low.min()),float(low.max())],
               'high_minmax':[float(high.min()),float(high.max())],
               'population_minmax':[float(population.min()),float(population.max())],
               'max_abs_node_correction':float(np.max(np.abs(correction)))}
    return tables,telemetry


def canonical_to_stored(image, orientation):
    """@luna-comment Input: display-oriented model BGR8 and source EXIF1/6. Output: lossless stored-orientation BGR8. Key Steps: inverse exact quarter-turn for orientation6. Key Params: no interpolation. State/Dependencies: same permutation as V3 evaluation inverse."""
    if orientation not in (1,6) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Invalid canonical image/orientation')
    return image.copy() if orientation==1 else np.ascontiguousarray(np.rot90(image,k=1))


def stored_to_canonical(image, orientation):
    """@luna-comment Input: stored-orientation BGR8 and source EXIF1/6. Output: lossless display BGR8. Key Steps: same clockwise quarter-turn as frozen evaluator for orientation6. Key Params: no interpolation. State/Dependencies: inverse of canonical_to_stored."""
    if orientation not in (1,6) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Invalid stored image/orientation')
    return image.copy() if orientation==1 else np.ascontiguousarray(np.rot90(image,k=3))


def render_post(stored_model, nodes, table, strength=1.0):
    """@luna-comment Input: stored full-model BGR8 and source-bound node LUT. Output: native BGR preview and RGB clipping telemetry. Key Steps: frozen half-pixel axis map, four same-Y LUT gathers, Y-bounded signed delta and shared per-channel RGB clip. Key Params: chunk128; optional zero render strength identity. State/Dependencies: no output-statistical parameter estimation; V3 replay separately uses frozen common-gamut renderer."""
    if strength==0.0:
        return stored_model.copy(),{'rgb_clipped_channels':0,'rgb_clipped_pixels':0,
                                    'native_statistical_pixels':0,'identity':True}
    if not 0.0 < strength <= 1.0 or stored_model.dtype != np.uint8 or stored_model.ndim != 3:
        raise ValueError('Invalid image or strength')
    nodes = np.asarray(nodes,np.int32)
    if table.shape != (len(nodes),len(nodes),256):
        raise ValueError('Node LUT shape mismatch')
    height,width = stored_model.shape[:2]
    xl,xr,wx = pre640.axis_map(width,nodes)
    yl,yr,wy = pre640.axis_map(height,nodes)
    y = cv2.cvtColor(stored_model,cv2.COLOR_BGR2YCrCb)[:,:,0]
    output = np.empty_like(stored_model)
    clipped_channels=clipped_pixels=y_limited=changed=0
    for top in range(0,height,128):
        bottom=min(height,top+128)
        original_y=y[top:bottom]
        upper_left=table[yl[top:bottom,None],xl[None,:],original_y]
        upper_right=table[yl[top:bottom,None],xr[None,:],original_y]
        lower_left=table[yr[top:bottom,None],xl[None,:],original_y]
        lower_right=table[yr[top:bottom,None],xr[None,:],original_y]
        upper=upper_left+(upper_right-upper_left)*wx[None,:]
        lower=lower_left+(lower_right-lower_left)*wx[None,:]
        mapped=upper+(lower-upper)*wy[top:bottom,None]
        requested_float=(mapped-original_y)*strength
        bounded_float=np.clip(requested_float,-original_y.astype(np.float32),255.0-original_y)
        y_limited += int(np.count_nonzero(bounded_float!=requested_float))
        requested=np.rint(bounded_float).astype(np.int16)
        preview,channels=preview_bgr(stored_model[top:bottom],requested)
        output[top:bottom]=preview
        clipped_channels += channels
        base=stored_model[top:bottom].astype(np.int16)
        clipped_pixels += int(np.count_nonzero(np.any((base+requested[:,:,None]<0)|
                                                     (base+requested[:,:,None]>255),axis=2)))
        changed += int(np.count_nonzero(np.any(preview!=stored_model[top:bottom],axis=2)))
    return output,{'rgb_clipped_channels':clipped_channels,'rgb_clipped_pixels':clipped_pixels,
                   'y_headroom_limited_pixels':y_limited,'changed_pixels':changed,
                   'native_statistical_pixels':0,'output_calibration_enabled':False}
