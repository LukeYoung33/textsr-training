"""All statistical estimation and curve construction stay on the detector canvas."""
import cv2
import numpy as np


def strip_histograms(row, pitch=4):
    """@luna-comment
    Input: Nonempty detector-resolution uint8 row and positive strip pitch.
    Output: Int32 per-strip256-bin histograms; no native-image access.
    Key Steps: Encode strip/bin indices and count the low-resolution samples once.
    Key Params: Pitch4 detector pixels; final strip retains its actual samples.
    State/Dependencies: NumPy bincount only; caller owns immutable canvas pixels.
    """
    groups = np.arange(row.shape[1],dtype=np.int32)//pitch
    encoded = row.astype(np.int32)+256*groups[None,:]
    return np.bincount(encoded.ravel(),minlength=(groups[-1]+1)*256).reshape(-1,256).astype(np.int32)


def estimate_line(canvas_y, bbox, pitch=4, lower_estimator='mode'):
    """@luna-comment
    Input: Only detector-resolution uint8 Y and its integer half-open row box.
    Output: Continuous paper profile and anchors for every column; confidence diagnostic only.
    Key Steps: Prefix strip histograms, paper evidence, normalized line histogram/Otsu.
    Key Params: Width16, plateau12/edge16 for estimation seeds only; B floor32; no coverage gate.
    State/Dependencies: Never receives native pixels; NumPy/OpenCV reference estimator.
    """
    x0,y0,x1,y1 = bbox
    row = canvas_y[y0:y1,x0:x1]
    hist = strip_histograms(row,pitch)
    prefix = np.vstack([np.zeros((1,256),np.int32),np.cumsum(hist,axis=0,dtype=np.int32)])
    indices = np.arange(len(hist))
    starts,ends = np.maximum(0,indices-2),np.minimum(len(hist),indices+2)
    window_hist = prefix[ends]-prefix[starts]
    cdf = np.cumsum(window_hist,axis=1)
    count = cdf[:,-1]
    quantiles = np.array([np.argmax(cdf>=np.ceil(count*q)[:,None],axis=1) for q in [.70,.85,.90]])
    edge_row = np.concatenate([row[:2],row[-2:]],axis=0)
    edge_hist = strip_histograms(edge_row,pitch)
    eprefix = np.vstack([np.zeros((1,256),np.int32),np.cumsum(edge_hist,axis=0,dtype=np.int32)])
    ecdf = np.cumsum(eprefix[ends]-eprefix[starts],axis=1)
    edge_q85 = np.argmax(ecdf>=np.ceil(ecdf[:,-1]*.85)[:,None],axis=1)
    candidates = quantiles[1].astype(np.float32)
    seed = (quantiles[2]-quantiles[0]<=12)&(np.abs(edge_q85-candidates)<=16)&(candidates>=32)
    centers = np.minimum(indices*pitch+(pitch-1)/2,row.shape[1]-1).astype(np.float32)+x0
    supported = np.ones(len(seed),dtype=bool)
    good = np.flatnonzero(seed)
    if len(good):
        paper = np.interp(centers,centers[good],candidates[good]).astype(np.float32)
    else:
        paper = candidates.copy()
    paper_floor_count = int((paper<32).sum())
    paper = np.maximum(paper,32)
    pixel_x = np.arange(x0,x1)
    profile = np.interp(pixel_x,centers,paper).astype(np.float32)
    eligible = np.ones(len(pixel_x),dtype=bool)
    z = np.rint(row[:,eligible].astype(np.float32)*128/profile[eligible][None,:]).astype(np.int32)
    normalized_hist = np.bincount(z.ravel(),minlength=1024).astype(np.int64) if z.size else np.zeros(1024,np.int64)
    assert len(normalized_hist)==1024
    values = np.arange(1024,dtype=np.float64)
    weight = np.cumsum(normalized_hist,dtype=np.float64)
    moment = np.cumsum(normalized_hist*values)
    total = float(weight[-1])
    denominator = weight*(total-weight)
    variance = np.divide((total*moment-weight*moment[-1])**2,denominator,
                         out=np.zeros(1024),where=denominator>0)
    t = int(np.argmax(variance))
    dark,bright = int(normalized_hist[:t+1].sum()),int(normalized_hist[t+1:].sum())
    required = max(16,.005*total)
    accepted = min(dark,bright)>=required
    low = int(np.argmax(normalized_hist[:t+1]))
    raw_mode = low
    if lower_estimator == 'p10' and dark:
        low = int(np.searchsorted(np.cumsum(normalized_hist[:t+1]),max(1,int(np.ceil(dark*.1)))))
    high = t+1+int(np.argmax(normalized_hist[t+1:])) if t<1023 else 1023
    accepted = bool(accepted and low<t+.5<high and abs(high-128)<=12)
    confidence_accepted = accepted
    numeric_fallback = not (0<=low<t+.5<high)
    if numeric_fallback:
        low,t,high = 0,63.5,128
    accepted = True
    return dict(confidence_accepted=confidence_accepted,numeric_fallback=numeric_fallback,
                paper_floor_count=paper_floor_count,canvas_bbox=list(bbox),centers=centers,paper=paper,seed=seed,supported=supported,
                eligible=eligible,profile=profile,normalized_anchors=[low,t+.5,high],accepted=accepted,
                normalized_histogram=normalized_hist,statistical_pixel_count=int(row.size),
                normalized_pixel_count=int(z.size),otsu_calls=1,dark_count=dark,bright_count=bright,
                lower_estimator=lower_estimator,raw_lower_mode=raw_mode,
                seed_count=int(seed.sum()),reason='no_gate_numeric_fallback' if numeric_fallback else 'no_gate',
                row_height=row.shape[0],paper_edge_candidates=edge_q85)


def build_tables(model, gamma=1.8):
    """@luna-comment
    Input: Low-resolution model, gamma1.2/1.8; no native pixels.
    Output: Q8.8 original-Y S-curve LUTs at every node, ordered in-range anchors.
    Key Steps: Scale anchors by paper; fit gamut with common scaling; evaluate clipped S everywhere.
    Key Params:256 entries per4-canvas-column node; gamma1.8 everywhere, no identity fallback.
    State/Dependencies: NumPy powers only in this low-resolution preparation stage.
    """
    a = model['paper'].astype(np.float32)/128
    anchors = a[:,None]*np.array(model['normalized_anchors'],np.float32)[None,:]
    anchors *= np.minimum(1.,255/np.maximum(anchors[:,2],1))[:,None]
    assert np.all(anchors[:,0]<anchors[:,1]) and np.all(anchors[:,1]<anchors[:,2])
    valid = np.ones(len(anchors),dtype=bool)
    low,mid,high = [anchors[:,i,None] for i in range(3)]
    value = np.arange(256,dtype=np.float32)[None,:]
    exponent = np.full((len(valid),1),gamma,dtype=np.float32)
    lr = ((value-low)/(mid-low)).clip(0,1)
    hr = ((high-value)/(high-mid)).clip(0,1)
    lut = np.where(value<=mid,low+(mid-low)*lr**exponent,high-(high-mid)*hr**exponent)
    assert np.all(np.diff(lut,axis=1)>=-1e-5)
    tables = np.rint(lut*256).astype(np.uint16)
    return dict(tables=tables,valid=valid,anchors=anchors,gamma=gamma,centers=model['centers'])


def prepare_mapping(packed, native_bbox, native_shape, canvas_shape):
    """@luna-comment
    Input: Prepared tables, native geometry and canvas geometry; no image samples.
    Output: Native-column lookup indices, Q12 blend weights and supported coverage.
    Key Steps: Half-pixel coordinate mapping, clamp endpoints, precompute fixed weights.
    Key Params: Separate geometry scales; Q12 interpolation, actual node centers.
    State/Dependencies: NumPy geometry only; reusable across all overlapping patches.
    """
    x0,_,x1,_ = native_bbox
    x = (np.arange(x0,x1,dtype=np.float64)+.5)*canvas_shape[1]/native_shape[1]-.5
    centers = packed['centers']
    upper = np.searchsorted(centers,x,side='right').clip(0,len(centers)-1)
    lower = (upper-1).clip(0,len(centers)-1)
    lower[x>=centers[-1]] = len(centers)-1
    span = centers[upper]-centers[lower]
    w = np.divide(x-centers[lower],span,out=np.zeros(len(x)),where=span>0).clip(0,1)
    full = (packed['valid'][lower]|(w==1))&(packed['valid'][upper]|(w==0))
    return dict(lower=lower,upper=upper,weight=np.rint(w*4096).astype(np.uint32),full=full,x=x)


def apply_tables(image, native_y, packed, mapping):
    """@luna-comment
    Input: Immutable native BGR/Y crop, Q8.8 tables and precomputed column mapping.
    Output: Native BGR correction and mapped Y; source remains immutable.
    Key Steps: Two LUT reads, Q12 integer blend, bounded common-channel brightness delta.
    Key Params: No histogram/Otsu/search/quantiles/normalization/division/power here.
    State/Dependencies: NumPy indexed integer operations and OpenCV channel arithmetic.
    """
    lower,upper,w = mapping['lower'],mapping['upper'],mapping['weight'][None,:]
    first = packed['tables'][lower[None,:],native_y].astype(np.uint32)
    second = packed['tables'][upper[None,:],native_y].astype(np.uint32)
    mapped = ((first*(4096-w)+second*w+524288)>>20).astype(np.uint8)
    delta = mapped.astype(np.int16)-native_y.astype(np.int16)
    b,g,r = cv2.split(image)
    smallest,largest = cv2.min(cv2.min(b,g),r),cv2.max(cv2.max(b,g),r)
    delta = cv2.min(cv2.max(delta,-smallest.astype(np.int16)),255-largest.astype(np.int16))
    result = cv2.merge([cv2.add(c,delta,dtype=cv2.CV_8U)for c in [b,g,r]])
    return result,mapped
