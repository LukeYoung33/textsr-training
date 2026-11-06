"""Valid-source-aware enhancement residual pasteback for document homographies."""
import cv2
import numpy as np


def source_support(source_hw, transform, output_hw):
    """@luna-comment
    Input: source/output heights-widths and finite original-to-warp3x3 matrix.
    Output: output-height x output-width uint8 mask,255 for fully observed
    bilinear source support and0 otherwise. Matrix maps source(x,y) to warp(x,y).
    Key Steps: warp float ones with zero exterior; require exact full support.
    Key Params: no unknown white padding is valid. State/Dependencies: OpenCV/NumPy.
    """
    matrix = np.asarray(transform, np.float64)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Invalid homography')
    height, width = map(int, output_hw)
    support = cv2.warpPerspective(np.ones(source_hw, np.float32), matrix, (width,height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(support >= 1.,255,0).astype(np.uint8)


def inverse_geometry(transform, quad):
    """@luna-comment
    Input: forward homography and original quad. Output: inverse matrix and roundtrip error.
    Key Steps: validate invertibility; transform corners forward/back; require .01px bound.
    Key Params: float64 homogeneous calculations. State/Dependencies: pure geometry.
    """
    matrix = np.asarray(transform, np.float64)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Invalid homography')
    inverse = np.linalg.inv(matrix)
    corners = np.asarray(quad, np.float64).reshape(1,4,2)
    if not np.isfinite(corners).all():
        raise ValueError('Invalid quad')
    restored = cv2.perspectiveTransform(cv2.perspectiveTransform(corners,matrix),inverse)
    error = float(np.linalg.norm(restored-corners,axis=2).max())
    if error > .01:
        raise ValueError('Homography corner roundtrip exceeds .01px')
    return inverse,error


def inverse_residual(original, warped_original, warped_result, transform, quad, valid, owned, feather=8.):
    """@luna-comment
    Input: immutable original/warp BGR8, processed warp, H, quad, valid mask, prior ownership.
    Output: source-frame float residual/alpha, new boolean support ownership and
    geometry telemetry. Ownership covers valid support even where residual is zero.
    Key Steps: erode support, smooth border weights, inverse-warp weighted differences;
    intersect original quad and unowned coverage. Never inverse-warp source RGB itself.
    Key Params:8warp-pixel border feather; confidence ordering belongs to caller.
    State/Dependencies: masks are read-only; identical processed/reference warps
    give exact zero residual. Caller adds the already-alpha-weighted residual once.
    """
    if warped_result.shape != warped_original.shape or valid.shape != warped_original.shape[:2]:
        raise ValueError('Warp result/support shape mismatch')
    if original.shape[:2] != owned.shape or feather <= 0:
        raise ValueError('Invalid ownership/feather')
    inverse,error = inverse_geometry(transform,quad)
    observed = cv2.erode((valid==255).astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    distance = cv2.distanceTransform(np.pad(observed,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    ramp = np.clip((distance-1)/feather,0,1)
    alpha = ramp*ramp*(3-2*ramp)
    delta = (warped_result.astype(np.float32)-warped_original.astype(np.float32))*alpha[...,None]
    height,width = original.shape[:2]
    mapped = cv2.warpPerspective(delta,inverse,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    mapped_alpha = cv2.warpPerspective(alpha,inverse,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    polygon = np.zeros((height,width),np.uint8)
    cv2.fillConvexPoly(polygon,np.rint(quad).astype(np.int32),1)
    take = (mapped_alpha>1e-6)&(polygon>0)&(~owned)
    mapped[~take] = 0
    mapped_alpha[~take] = 0
    return mapped,mapped_alpha,take,dict(inverse=inverse.tolist(),roundtrip_error=error,
              owned_pixels=int(take.sum()),valid_warp_fraction=float((valid==255).mean()),
              interpolation='bilinear_weighted_residual_only',document_feather_warp_pixels=feather)
