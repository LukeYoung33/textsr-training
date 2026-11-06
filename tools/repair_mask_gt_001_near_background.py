"""Nearest original-background Dirichlet repair for001 false-pixel cleanup."""
from pathlib import Path
import json,sys
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
from tools.preview_mask_guided_vosr_strokes import weighted_field
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'mask_gt_001_fill_clean_20261104'
OUT=DATA/'mask_gt_001_near_background_20261104'


def local_ink(image,mask,valid):
    """@luna-comment Input:original001RGBandM2. Output:localwhite/creaminkRGBfield/support. Key Steps:same-componentoriginalbrightdonors,normalizedlocalinterpolation. Key Params:001white-on-darkprototype;no repairedpixel/far-backgrounddonors. State/Dependencies:originalVOSRappearanceonly."""
    labels,_=ndi.label((mask>.5)&valid,np.ones((3,3)));nearest=ndi.distance_transform_edt(labels==0,return_distances=False,return_indices=True);owners=labels[tuple(nearest)]
    F=np.zeros_like(image);support=np.zeros(mask.shape,bool);y=(image*np.array([.299,.587,.114])).sum(2)
    for number,sl0 in enumerate(ndi.find_objects(labels),1):
        if sl0 is None:continue
        a=max(0,sl0[0].start-16);b=min(mask.shape[0],sl0[0].stop+16);c=max(0,sl0[1].start-16);d=min(mask.shape[1],sl0[1].stop+16);sl=np.s_[a:b,c:d]
        possible=(labels[sl]==number)&(mask[sl]>.75)&(y[sl]>.5)
        if possible.sum()<4:continue
        lo,hi=np.quantile(y[sl][possible],[.6,.98]);donor=possible&(y[sl]>=lo)&(y[sl]<=hi)
        if donor.sum()<3:continue
        field,_=weighted_field(image[sl],donor.astype(np.float32),6);domain=owners[sl]==number
        F[sl][domain]=field[domain];support[sl]|=domain
    return F,support


def harmonic_background(image,bg,unknown,ink):
    """@luna-comment Input:originalRGB/backgrounddomain/unknownfalsemask/inkfield. Output:backgroundsolution,known/unresolvedmaps andanchorcounts. Key Steps:same-background-componentanchors,sparse4neighborharmonicRGBDirichletsolve,textedgesexcluded. Key Params:immediatelyadjacentknownpixels;closedcounterneedsdeepcontrastinganchor. State/Dependencies:originalbackgroundfixed,nofarmeanfield."""
    labels,count=ndi.label(bg);known=bg&~unknown;unresolved=np.zeros(bg.shape,bool);closed_anchors=0
    for number in np.unique(labels[unknown]):
        if number==0:continue
        region=labels==number
        if known[region].any():continue
        distance=ndi.distance_transform_edt(region);deep=region&(distance>=max(1.25,.8*distance.max()))
        reliable=deep&(np.linalg.norm(image-ink,axis=2)>.12)
        if reliable.any():known|=reliable;unknown=unknown&~reliable;closed_anchors+=int(reliable.sum())
        else:unresolved|=region;unknown=unknown&~region
    ys,xs=np.where(unknown);n=len(xs);index=np.full(bg.shape,-1,np.int32);index[ys,xs]=np.arange(n);rows=[];cols=[];values=[];rhs=np.zeros((n,3),np.float64);degree=np.zeros(n,np.float64)
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        yy=ys+dy;xx=xs+dx;inside=(yy>=0)&(yy<bg.shape[0])&(xx>=0)&(xx<bg.shape[1]);ids=np.where(inside)[0];yy=yy[inside];xx=xx[inside]
        allowed=bg[yy,xx];ids=ids[allowed];yy=yy[allowed];xx=xx[allowed];degree[ids]+=1
        neighbors=index[yy,xx];free=neighbors>=0
        rows.extend(ids[free]);cols.extend(neighbors[free]);values.extend([-1.]*int(free.sum()))
        rhs[ids[~free]]+=image[yy[~free],xx[~free]]
    assert (degree>0).all()
    rows.extend(np.arange(n));cols.extend(np.arange(n));values.extend(degree)
    result=image.copy()
    if n:
        A=coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr();solution=spsolve(A,rhs);assert np.isfinite(solution).all();result[ys,xs]=solution.astype(np.float32)
    assert np.array_equal(result[known],image[known])
    return result,unknown,known,unresolved,closed_anchors


def interface_step(image,mask,bg):
    """@luna-comment Input:RGBimage/oldcleanupregion/backgrounddomain. Output:meanRGBneighborstepatoldregioninterfaceinDN. Key Steps:onlybackground-backgroundcrossings. Key Params:factualinterfacecontrast,notvisualpassscore. State/Dependencies:numpyreadonly."""
    total=0.;count=0
    for axis in [0,1]:
        a=np.take(mask,range(mask.shape[axis]-1),axis=axis);b=np.take(mask,range(1,mask.shape[axis]),axis=axis)
        ga=np.take(bg,range(mask.shape[axis]-1),axis=axis);gb=np.take(bg,range(1,mask.shape[axis]),axis=axis);edge=(a!=b)&ga&gb
        ia=np.take(image,range(mask.shape[axis]-1),axis=axis);ib=np.take(image,range(1,mask.shape[axis]),axis=axis)
        total+=float(np.abs(ia-ib)[edge].sum());count+=int(edge.sum())*3
    return total/max(count,1)*255


def main():
    """@luna-comment Input:001original/M2/filledsource/oldfalsemap. Output:twoactualnearest-backgroundcandidates andboundaryaudit. Key Steps:tightunknownmap,same-regionharmonicsolve,onesoftedge,reliableinkandfarbackgroundfreeze. Key Params:001only,noGTreplacement. State/Dependencies:ownedOUT;sourcepixelshashesbound."""
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    previous=json.loads((SOURCE/'AUDIT.json').read_text());fill_audit=json.loads((SOURCE/'fill_stage/AUDIT.json').read_text());job=fill_audit['source']
    for k,p in job['files'].items():assert digest(p)==job['sha256'][k]
    raw=cv2.cvtColor(cv2.imread(job['files']['vosr_s22']),cv2.COLOR_BGR2RGB);I=raw.astype(np.float32)/255
    filled=cv2.cvtColor(cv2.imread(str(SOURCE/'fill_stage/direct_strong.png')),cv2.COLOR_BGR2RGB);base=filled.astype(np.float32)/255
    old=cv2.cvtColor(cv2.imread(str(SOURCE/'fill_clean_strong.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    m=cv2.imread(job['files']['text_mask_plus2'],0).astype(np.float32)/255;valid=cv2.imread(job['files']['valid'],0)>=254;shape=np.clip((m-.2)/.6,0,1)
    selected=np.zeros(m.shape,bool);box=previous['crop_xyxy']
    for a,b,c,d in previous['zones'].values():selected[box[1]+max(0,b-5):box[1]+d+5,box[0]+max(0,a-5):box[0]+c+5]=True
    oldmap=cv2.imread(str(SOURCE/'fill_clean_strong_map.png'),0)>0
    bg=(shape==0)&valid;unknown=ndi.binary_dilation(oldmap,iterations=1)&bg&selected
    ink,ink_support=local_ink(I,m,valid)
    B,unknown,known,unresolved,anchor_count=harmonic_background(I,bg,unknown,ink)
    # The soft stroke edge borrows the immediately adjacent solved background side.
    nearest=ndi.distance_transform_edt(~bg,return_distances=False,return_indices=True);background=B[tuple(nearest)]
    edge=oldmap&selected&(shape>0)&(shape<.99)&ink_support&valid
    target=base.copy();target[unknown]=B[unknown];target[edge]=(background+shape[...,None]*(ink-background))[edge]
    editable=unknown|edge;deep=shape>=.99;results={}
    for key,gain in [('nearest_conservative',.65),('nearest_strong',1.)]:
        result=base.copy();result[editable]=(base+gain*(target-base))[editable]
        image=np.rint(np.clip(result,0,1)*255).astype(np.uint8);image[~editable]=filled[~editable]
        assert np.array_equal(image[deep],filled[deep]) and np.array_equal(image[known],filled[known]) and np.array_equal(image[~selected],filled[~selected])
        p=OUT/(key+'.png');cv2.imwrite(str(p),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));changed=np.any(image!=filled,2)
        results[key]=dict(path=str(p),sha256=digest(p),changed_pixels=int(changed.sum()),known_background_changed_pixels=0,deep_ink_changed_pixels=0,outside_roi_changed_pixels=0,
            previous_interface_step_dn=interface_step(old,oldmap,bg),new_interface_step_dn=interface_step(image.astype(np.float32)/255,oldmap,bg))
        cv2.imwrite(str(OUT/(key+'_map.png')),np.repeat(changed[...,None].astype(np.uint8)*255,3,2))
    donors=ndi.binary_dilation(unknown)&known
    for name,value in [('known_boundary_donors',donors),('unknown_background',unknown),('unresolved',unresolved)]:cv2.imwrite(str(OUT/(name+'.png')),np.repeat(value[...,None].astype(np.uint8)*255,3,2))
    cv2.imwrite(str(OUT/'nearest_background.png'),cv2.cvtColor(np.rint(np.clip(B,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    save(OUT/'AUDIT.json',dict(passed=True,sample='001',crop_xyxy=box,variants=results,unknown_pixels=int(unknown.sum()),immediate_known_boundary_donors=int(donors.sum()),closed_counter_anchor_pixels=anchor_count,
        unresolved_pixels=int(unresolved.sum()),background_method='same-background-component4neighborharmonicRGB;originalDirichletneighbors;no-textdonors',source_gt_replaced=False))
    save(OUT/'COMPLETE.json',dict(passed=True,samples=1,candidates=2));print(json.dumps(results))


if __name__=='__main__':main()
