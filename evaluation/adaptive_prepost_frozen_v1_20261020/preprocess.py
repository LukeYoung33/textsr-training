"""Fixed dual-mode preprocessing entry points and reusable per-frame context."""
import cv2
import numpy as np

from .lowres_curve import estimate_line, build_tables, prepare_mapping, apply_tables


def prepare_context(image, boxes, reference_id='', canvas_y=None):
    """@luna-comment
    Input: Native uint8 BGR, ordered integer xyxy boxes, reference ID, optional1984x1472 Y.
    Output: Frame-owned low-resolution models, target anchors/LUTs and coordinate contract.
    Key Steps: Prepare/reuse canvas, estimate dual modes, build fixed-gamma tables once per line.
    Key Params: Mode/mode, gamma1.8, pitch4, no gating; optional canvas avoids extra resize.
    State/Dependencies: Frozen snapshot; caches all line tables until caller releases context.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 BGR')
    height, width = image.shape[:2]
    if canvas_y is None:
        canvas_y = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0], (1984, 1472))
    if canvas_y.shape != (1472, 1984) or canvas_y.dtype != np.uint8:
        raise ValueError('Canvas must be1984x1472 uint8 Y with matching frame geometry')
    lines = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Invalid native box')
        cb = [int(np.floor(x0*1984/width)), int(np.floor(y0*1472/height)),
              int(np.ceil(x1*1984/width)), int(np.ceil(y1*1472/height))]
        model = estimate_line(canvas_y, cb, lower_estimator='mode')
        packed = build_tables(model, 1.8)
        lines.append(dict(index=index, bbox=[x0, y0, x1, y1], model=model, packed=packed))
    return dict(version='adaptive_dualmode_scurve_v1', reference_id=reference_id,
                source_shape=(height, width), canvas_shape=(1472, 1984), lines=lines,
                geometry='aligned full frame; independent axis scaling only',
                table_bytes=sum(line['packed']['tables'].nbytes for line in lines))


def apply_preprocess(image, context):
    """@luna-comment
    Input: Original native BGR and its prepare_context result; same frame and shape required.
    Output: New BGR, using every selected column with ordered first ownership.
    Key Steps: Compute original Y, reuse cached LUTs, map geometry, apply each unowned pixel once.
    Key Params: No histogram/powers at native resolution; no border fade or strength gate.
    State/Dependencies: Frozen lookup helpers; caller guarantees context/frame identity.
    """
    if image.shape[:2] != tuple(context['source_shape']):
        raise ValueError('Preprocessing context shape mismatch')
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    output = image.copy()
    owned = np.zeros(y.shape, bool)
    for line in context['lines']:
        x0, y0, x1, y1 = line['bbox']
        mapping = prepare_mapping(line['packed'], line['bbox'], y.shape, context['canvas_shape'])
        processed, _ = apply_tables(image[y0:y1, x0:x1], y[y0:y1, x0:x1], line['packed'], mapping)
        free = ~owned[y0:y1, x0:x1]
        output[y0:y1, x0:x1][free] = processed[free]
        owned[y0:y1, x0:x1] = True
    return output


def make_post_context(context):
    """@luna-comment
    Input: Frame context after optional preprocessing rendering.
    Output: Compact JSON-serializable reference geometry and anchors for postprocessing.
    Key Steps: Retain target anchors/centers and line IDs; omit LUTs, histograms and pixel profiles.
    Key Params: Four float values per node replace512 table bytes; no image sampling.
    State/Dependencies: Copies primitive values; caller may release the large preprocessing context.
    """
    return dict(version=context['version'], reference_id=context['reference_id'],
                source_shape=list(context['source_shape']), canvas_shape=list(context['canvas_shape']),
                geometry=context['geometry'], lines=[
                    dict(index=line['index'], bbox=list(line['bbox']),
                         model=dict(canvas_bbox=list(line['model']['canvas_bbox']),
                                    normalized_anchors=list(line['model']['normalized_anchors'])),
                         packed=dict(anchors=line['packed']['anchors'].tolist(),
                                     centers=line['packed']['centers'].tolist()))
                    for line in context['lines']])
