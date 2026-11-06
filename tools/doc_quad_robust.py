"""Versioned robust perspective proposals for failed document boundary fits."""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import doc_quad_infer as base

BASE_POLYGON_TO_QUAD=base.polygon_to_quad


def consensus_quad(seed, samples, diagonal):
    """@luna-comment
    Input: clockwise four-point seed, uniform contour samples and pixel diagonal.
    Output: perspective quad plus four consensus counts/fractions/spans.
    Key Steps: assign nearest edges; deterministic RANSAC; refit inlier lines;
    intersect adjacent lines twice, rejecting sparse or short-span consensus.
    Key Params:192 pairs/edge; tolerance <=0.6 percent diagonal/3 percent edge;
    angle difference <=50 degrees and span >=55 percent of seed edge.
    State/Dependencies: local fixed RNG; OpenCV/NumPy; no model or source mutation.
    """
    quad=base.order_quad(seed).astype(np.float64)
    rng=np.random.default_rng(17)
    evidence=[]
    for iteration in range(2):
        distances,projections,lengths=base.edge_measurements(quad,samples)
        assignment=distances.argmin(axis=1)
        normals=[];offsets=[];evidence=[]
        for edge in range(4):
            select=(assignment==edge)&(projections[:,edge]>=-.05)&(projections[:,edge]<=1.05)&(distances[:,edge]<=max(3.,.05*diagonal))
            points=samples[select].astype(np.float64)
            if len(points)<12:
                raise ValueError('too_few_consensus_samples')
            direction=quad[(edge+1)%4]-quad[edge]
            direction/=np.linalg.norm(direction)
            tolerance=max(1.5,min(diagonal*.006,lengths[edge]*.03))
            best=None;best_score=-1
            for pair in rng.integers(0,len(points),(192,2)):
                start,end=points[pair]
                vector=end-start;length=np.linalg.norm(vector)
                if length<lengths[edge]*.3:
                    continue
                vector/=length
                if abs(np.dot(vector,direction))<np.cos(np.deg2rad(50)):
                    continue
                normal=np.array([-vector[1],vector[0]])
                inliers=np.abs((points-start)@normal)<=tolerance
                positions=(points[inliers]-quad[edge])@direction/lengths[edge]
                span=np.ptp(positions) if len(positions) else 0
                if span<.55:
                    continue
                score=int(inliers.sum())
                if score>best_score:
                    best=inliers;best_score=score
            if best is None or best_score<12 or best_score/len(points)<.30:
                raise ValueError(f'insufficient_line_consensus_edge{edge}_inliers{best_score}_samples{len(points)}')
            line=cv2.fitLine(points[best].astype(np.float32),cv2.DIST_L2,0,.01,.01).ravel()
            normal=np.float64([-line[1],line[0]])
            normals.append(normal);offsets.append(normal@line[2:])
            evidence.append(dict(samples=len(points),inliers=best_score,fraction=best_score/len(points),
                                 span=float(np.ptp((points[best]-quad[edge])@direction/lengths[edge])),tolerance=float(tolerance)))
        normals=np.asarray(normals);offsets=np.asarray(offsets)
        quad=base.order_quad([np.linalg.solve(normals[[(i-1)%4,i]],offsets[[(i-1)%4,i]]) for i in range(4)])
    return quad,evidence


def exterior_consensus_quad(seed, samples, diagonal, image_shape):
    """@luna-comment
    Input: Four-point seed, uniform raw contour, visible diagonal and image H,W.
    Output: Four approximate exterior line intersections and per-edge support evidence.
    Key Steps: Exclude frame-clipping samples; deterministic line consensus; reject
    inward lines by signed whole-contour support; intersect and refine twice.
    Key Params: Only frame-adjacent seed edges may use20-percent observed span;
    all others retain55percent. At most3percent contour samples may lie outside
    a line beyond its noise tolerance. Coverage/area gates remain with the caller.
    State/Dependencies: NumPy/OpenCV only; no image lookup, model state or mutation.
    """
    height,width=image_shape[:2]
    frame_band=max(2.,.01*min(height,width))
    observed=(samples[:,0]>frame_band)&(samples[:,0]<width-1-frame_band)
    observed&=(samples[:,1]>frame_band)&(samples[:,1]<height-1-frame_band)
    quad=base.order_quad(seed).astype(np.float64)
    rng=np.random.default_rng(23)
    evidence=[]
    for iteration in range(2):
        distances,projections,lengths=base.edge_measurements(quad,samples)
        assignment=distances.argmin(axis=1)
        normals=[];offsets=[];evidence=[]
        for edge in range(4):
            endpoints=quad[[edge,(edge+1)%4]]
            clipped_edge=bool(np.any(endpoints<frame_band)
                or np.any(endpoints>[width-1-frame_band,height-1-frame_band]))
            minimum_span=.20 if clipped_edge else .55
            selected=observed&(assignment==edge)&(projections[:,edge]>=-.05)
            selected&=(projections[:,edge]<=1.05)&(distances[:,edge]<=max(3.,.08*diagonal))
            points=samples[selected].astype(np.float64)
            if len(points)<12:
                raise ValueError('too_few_exterior_samples')
            direction=quad[(edge+1)%4]-quad[edge]
            direction/=np.linalg.norm(direction)
            tolerance=max(1.5,min(.006*diagonal,.03*lengths[edge]))
            best=None;best_count=-1
            for pair in rng.integers(0,len(points),(384,2)):
                start,end=points[pair]
                vector=end-start;length=np.linalg.norm(vector)
                if length<minimum_span*lengths[edge]:
                    continue
                vector/=length
                if np.dot(vector,direction)<0:
                    vector=-vector
                if np.dot(vector,direction)<np.cos(np.deg2rad(50)):
                    continue
                normal=np.array([-vector[1],vector[0]])
                outside=float(np.mean((samples-start)@normal < -tolerance))
                if outside>.03:
                    continue
                inliers=np.abs((points-start)@normal)<=tolerance
                positions=(points[inliers]-quad[edge])@direction/lengths[edge]
                if len(positions)<12 or np.ptp(positions)<minimum_span:
                    continue
                count=int(inliers.sum())
                if count>best_count:
                    best=inliers;best_count=count
            minimum_fraction=.15 if clipped_edge else .30
            if best is None or best_count/len(points)<minimum_fraction:
                raise ValueError(f'insufficient_exterior_consensus_edge{edge}')
            line=cv2.fitLine(points[best].astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
            vector=line[:2].astype(np.float64)
            if np.dot(vector,direction)<0:
                vector=-vector
            normal=np.array([-vector[1],vector[0]])
            offset=float(normal@line[2:])
            exterior_limit=float(np.quantile(samples@normal,.03))+tolerance
            if offset>exterior_limit:
                if offset-exterior_limit>tolerance:
                    raise ValueError('refitted_line_requires_excessive_outward_shift')
                offset=exterior_limit
            outside=float(np.mean(samples@normal < offset-tolerance))
            if outside>.03+1/len(samples):
                raise ValueError(f'refitted_line_is_not_exterior_edge{edge}_outside{outside:.3f}')
            normals.append(normal);offsets.append(offset)
            evidence.append(dict(samples=len(points),inliers=best_count,
                fraction=best_count/len(points),span=float(np.ptp(
                    (points[best]-quad[edge])@direction/lengths[edge])),
                tolerance=float(tolerance),outside_sample_fraction=outside,
                frame_truncated_support=clipped_edge,minimum_span=minimum_span))
        normals=np.asarray(normals);offsets=np.asarray(offsets)
        quad=base.order_quad([np.linalg.solve(normals[[(i-1)%4,i]],offsets[[(i-1)%4,i]]) for i in range(4)])
    return quad,evidence


def repair_quad(polygon, image_shape, previous=None, *, margin=0.):
    """@luna-comment
    Input: original visible contour/image H,W, optional baseline geometry and pixel margin.
    Output: preserved successful baseline, or validated approximate perspective proposal.
    Key Steps: preserve boundary fits; try existing robust lines first; only failed
    frame-contact cases add exterior consensus with observed truncated-edge spans;
    require convex finite bounded corners, mask coverage >=97 percent and enclosure <=1.35.
    Key Params: consensus is approximate and never promoted to baseline boundary success;
    observed supporting lines may intersect outside the image by at most one diagonal.
    State/Dependencies: original weights/release unchanged; keep fallback and diagnostics
    when no supported proposal exists. Existing rectangle remains the final fallback.
    """
    previous=previous or BASE_POLYGON_TO_QUAD(polygon,image_shape,margin=margin)
    if previous['geometry_type']=='boundary_quad':
        return previous
    points=np.asarray(polygon,np.float32)
    samples=base.sample_contour(points,1024)
    diagonal=max(float(np.linalg.norm(np.ptp(points,axis=0))),1.)
    hull=cv2.convexHull(points).reshape(-1,2)
    seeds=[cv2.boxPoints(cv2.minAreaRect(hull))]
    for contour in (points,hull):
        perimeter=cv2.arcLength(contour,True)
        for epsilon in (.003,.005,.008,.01,.015,.02,.03,.04,.05,.06,.08):
            seed=cv2.approxPolyDP(contour,epsilon*perimeter,True).reshape(-1,2)
            if len(seed)==4 and cv2.isContourConvex(seed):
                seeds.append(seed)
    best=None;score=-1;failures=[]
    for seed in seeds:
        try:
            quad,support=consensus_quad(seed,samples,diagonal)
            quad=base.enclosing_quad(quad,samples,margin)
            height,width=image_shape[:2]
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
                raise ValueError('invalid_consensus_quad')
            outside_limit=float(np.hypot(width,height))
            if np.any(quad<-outside_limit) or np.any(quad>[width-1+outside_limit,height-1+outside_limit]):
                raise ValueError('excessive_outside_intersection')
            area=float(cv2.contourArea(quad.astype(np.float32)))
            ratio=area/max(float(cv2.contourArea(hull)),1.)
            coverage=base.mask_coverage(points,quad,image_shape)
            thresholds=previous['geometry_evidence']['thresholds']
            if coverage['mask_coverage']<thresholds['min_mask_coverage'] or ratio>thresholds['max_enclosure_area_ratio'] or ratio<.5 or area<thresholds['min_area_pixels']:
                raise ValueError('coverage_or_area_gate_failed')
            strength=sum(edge['inliers'] for edge in support)
            if strength>score:
                best=(quad,support,coverage,ratio);score=strength
        except (ValueError,np.linalg.LinAlgError) as error:
            failures.append(str(error))
    exterior_used=False
    height,width=image_shape[:2]
    frame_contact=bool(np.any(points<2.) or np.any(points>[width-3.,height-3.]))
    if best is None and frame_contact:
        for seed in seeds:
            try:
                quad,support=exterior_consensus_quad(seed,samples,diagonal,image_shape)
                quad=base.enclosing_quad(quad,samples,margin)
                if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
                    raise ValueError('invalid_exterior_quad')
                outside_limit=float(np.hypot(width,height))
                if np.any(quad<-outside_limit) or np.any(quad>[width-1+outside_limit,height-1+outside_limit]):
                    raise ValueError('excessive_outside_intersection')
                area=float(cv2.contourArea(quad.astype(np.float32)))
                ratio=area/max(float(cv2.contourArea(hull)),1.)
                coverage=base.mask_coverage(points,quad,image_shape)
                thresholds=previous['geometry_evidence']['thresholds']
                if coverage['mask_coverage']<thresholds['min_mask_coverage'] or ratio>thresholds['max_enclosure_area_ratio'] or ratio<.5 or area<thresholds['min_area_pixels']:
                    raise ValueError('exterior_coverage_or_area_gate_failed')
                strength=sum(edge['inliers'] for edge in support)
                if strength>score:
                    best=(quad,support,coverage,ratio);score=strength;exterior_used=True
            except (ValueError,np.linalg.LinAlgError) as error:
                failures.append('exterior:'+str(error))
    if best is None:
        return dict(previous,repair_evidence=dict(changed=False,seeds=len(seeds),failures=failures))
    quad,support,coverage,ratio=best
    result=dict(previous,quad=quad.astype(float).tolist(),selection_reason='robust_observed_line_consensus_perspective',
                extent_semantics='approximate_observed_line_perspective',counted_as_boundary_success=False)
    result['geometry_evidence']=dict(previous['geometry_evidence'],method='raw_and_hull_seed_ransac_perspective',
        approximation_is_physical_boundary=False,robust_edge_consensus=support,enclosure_area_ratio=ratio,**coverage)
    result['repair_evidence']=dict(changed=True,seeds=len(seeds),previous_quad=previous['quad'],failures=failures)
    if exterior_used:
        result['selection_reason']='exterior_observed_line_consensus_perspective'
        result['geometry_evidence']['method']='frame_clipped_exterior_ransac_perspective'
        result['repair_evidence']['exterior_fallback_used']=True
    return result


def polygon_to_quad(polygon,image_shape,**options):
    """@luna-comment
    Input: same visible contour/image shape/options as frozen polygon_to_quad.
    Output: robust approximate perspective proposal after failed ordinary boundary fit.
    Key Steps: run baseline first; honor strict/crop modes; otherwise invoke repair.
    Key Params: strict mode still rejects, crop-only remains diagnostic; supplied
    baseline geometry options are preserved and never reinterpret a successful fit.
    State/Dependencies: saved baseline callable avoids recursion during CLI binding.
    """
    result=BASE_POLYGON_TO_QUAD(polygon,image_shape,**options)
    if options.get('crop_only') or options.get('strict_boundary'):
        return result
    return repair_quad(polygon,image_shape,result,margin=options.get('margin',0.))


def main():
    """@luna-comment
    Input: frozen detector CLI arguments and local weights/images.
    Output: standard overlays/warps/JSON using versioned robust geometry.
    Key Steps: bind geometry adapter only inside this single-process CLI call;
    execute unchanged inference/artifact code; restore baseline binding in finally.
    Key Params: default/strict/crop mode meanings retained; no model download/training.
    State/Dependencies: scoped process-local function binding, never edits frozen files.
    """
    original=base.polygon_to_quad
    try:
        base.polygon_to_quad=polygon_to_quad
        return base.main()
    finally:
        base.polygon_to_quad=original


if __name__=='__main__':
    main()
