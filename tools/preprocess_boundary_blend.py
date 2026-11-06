"""Lightweight geometric blending of existing shared-BGR preprocessing corrections."""
import numpy as np


def feather_preprocess(source, processed, boxes, fraction=.12, max_width=24):
    """@luna-comment
    Input: Matching uint8 BGR source/preprocessed arrays and native half-open boxes.
    Output: Blended BGR and uint8 alpha (0 source,255 preprocessed).
    Key Steps: Separable smoothstep ramps, maximum overlap union, one rounded
    common-channel correction in128-row chunks; preserve full-strength interiors.
    Key Params: Width12percent short side,2..24px bounded to fit small boxes.
    State/Dependencies: NumPy only; no statistics, blur, chroma repair or source mutation.
    """
    if source.shape != processed.shape or source.dtype != np.uint8 or processed.dtype != np.uint8:
        raise ValueError('Matching uint8 inputs required')
    if source.ndim != 3 or source.shape[2] != 3 or not 0 < fraction <= .5 or max_width < 1:
        raise ValueError('Invalid image geometry or feather parameters')
    height, width = source.shape[:2]
    alpha = np.zeros((height, width), np.uint8)
    for x0, y0, x1, y1 in boxes:
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError('Invalid native box')
        w, h = x1-x0, y1-y0
        radius = min(max_width, max(2, int(round(min(w, h)*fraction))), max(1., (min(w, h)-1)/2))
        x = np.minimum(np.arange(w), np.arange(w)[::-1])/radius
        y = np.minimum(np.arange(h), np.arange(h)[::-1])/radius
        x, y = np.clip(x, 0, 1), np.clip(y, 0, 1)
        ax = np.rint(255*x*x*(3-2*x)).astype(np.uint8)
        ay = np.rint(255*y*y*(3-2*y)).astype(np.uint8)
        tile = alpha[y0:y1, x0:x1]
        np.maximum(tile, np.minimum(ay[:, None], ax[None, :]), out=tile)
    output = processed.copy()
    for y0 in range(0, height, 128):
        original = source[y0:y0+128].astype(np.int16)
        delta = processed[y0:y0+128].astype(np.int16)-original
        if not (np.array_equal(delta[:, :, 0], delta[:, :, 1]) and np.array_equal(delta[:, :, 1], delta[:, :, 2])):
            raise ValueError('Input is not a common-BGR preprocessing correction')
        weight = alpha[y0:y0+128]
        active = (weight < 255) & (delta[:, :, 0] != 0)
        d = delta[:, :, 0][active].astype(np.int32)
        blended = np.sign(d)*((np.abs(d)*weight[active].astype(np.int32)+127)//255)
        output[y0:y0+128][active] = (original[active]+blended[:, None]).astype(np.uint8)
    return output, alpha
