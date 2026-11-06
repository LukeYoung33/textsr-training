"""Bounded eight-target mask-guided VOSR repair prototype; originals immutable."""
from pathlib import Path
import json,sys,time,hashlib,shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
OUT=Path('/AI_DNS/guang.yang/TextSR/data/mask_guided_vosr_s22_preview8_20261104')
VARIANTS=['fill_conservative','fill_strong','edge_conservative','edge_strong']
PARAMS=[dict(tau=.15,gain=.65,limit=.35,residual=.20,refs=4,mad=.06,edge_gain=.5,edge_limit=.10,edge_band=.5,gap=1.),
        dict(tau=.08,gain=1.,limit=.70,residual=.35,refs=3,mad=.10,edge_gain=.75,edge_limit=.20,edge_band=.75,gap=1.5)]


def protected_holes(mask):
    """@luna-comment Input:originalsoftmask. Output:unionoforiginalcounterregionsatthreelevels. Key Steps:binaryfilldifference. Key Params:.35/.5/.7. State/Dependencies:geometryonly,neverusedasphotometry."""
    result=np.zeros(mask.shape,bool)
    for threshold in [.35,.5,.7]:
        fg=mask>threshold;result|=ndi.binary_fill_holes(fg)&~fg
    return result


def weighted_field(image,weight,sigma):
    """@luna-comment Input:RGBfloat anddonorweights. Output:fixedrobustappearancefieldandconfidence. Key Steps:normalizedGaussian,nearest-donorfallback. Key Params:noGTorpostrepairpixels. State/Dependencies:OpenCV/scipy;read-onlyoriginal."""
    den=cv2.GaussianBlur(weight.astype(np.float32),(0,0),sigma)
    result=cv2.GaussianBlur(image*weight[...,None],(0,0),sigma)/np.maximum(den[...,None],1e-6)
    if (den<.005).any():
        nearest=ndi.distance_transform_edt(weight<=0,return_distances=False,return_indices=True)
        result[den<.005]=image[tuple(nearest)][den<.005]
    return result,den


def section_templates(alpha,component):
    """@luna-comment Input:originalcoverage/componentmask. Output:nearest-skeletonhealthyprofiletemplate/MAD/coregeometry orNone. Key Steps:skeletonPCA,normalsections,compatiblebidirectionaldonors,robustmedian. Key Params:33samples,stablecenters,excludejunctionsandtips. State/Dependencies:originaldataonly;M2notatarget."""
    skel=skeletonize(component);ys,xs=np.where(skel)
    if len(xs)<8:return None
    yy,xx=np.indices(component.shape,dtype=np.float64);w=skel.astype(np.float64)
    mass=cv2.GaussianBlur(w,(0,0),2.5)+1e-9
    mx=cv2.GaussianBlur(w*xx,(0,0),2.5)/mass;my=cv2.GaussianBlur(w*yy,(0,0),2.5)/mass
    cxx=cv2.GaussianBlur(w*xx*xx,(0,0),2.5)/mass-mx*mx;cyy=cv2.GaussianBlur(w*yy*yy,(0,0),2.5)/mass-my*my
    cxy=cv2.GaussianBlur(w*xx*yy,(0,0),2.5)/mass-mx*my
    theta=.5*np.arctan2(2*cxy,cxx-cyy);tangent=np.stack([np.cos(theta[ys,xs]),np.sin(theta[ys,xs])],1)
    normal=np.stack([-tangent[:,1],tangent[:,0]],1)
    anis=np.sqrt((cxx-cyy)**2+4*cxy*cxy)/(cxx+cyy+1e-6)
    degree=ndi.convolve(skel.astype(np.uint8),np.ones((3,3),np.uint8))-skel
    excluded=ndi.binary_dilation(skel&(degree!=2),iterations=1)
    radius=ndi.distance_transform_edt(component)[ys,xs].clip(.75,20)
    u=np.linspace(-1.8,1.8,33).astype(np.float32)
    sx=xs[:,None]+normal[:,0,None]*radius[:,None]*u;sy=ys[:,None]+normal[:,1,None]*radius[:,None]*u
    profiles=ndi.map_coordinates(alpha,[sy,sx],order=1,mode='nearest').clip(0,1)
    center=profiles[:,13:20].mean(1)
    healthy=(center>.70)&(profiles[:,[0,-1]].max(1)<.35)&(anis[ys,xs]>.35)&~excluded[ys,xs]
    # Subpixel center agreement is measured on healthy central sections only.
    q=profiles[:,8:25];offset=(q*(u[8:25][None]*radius[:,None])).sum(1)/q.sum(1).clip(1e-6)
    if healthy.sum()<4 or np.median(np.abs(offset[healthy]))>.5 or np.percentile(np.abs(offset[healthy]),95)>1.:return None
    points=np.stack([xs,ys],1);tree=cKDTree(points);neighbors=tree.query_ball_point(points,24)
    beta=np.zeros_like(profiles);mad=np.ones_like(profiles);counts=np.zeros(len(xs),np.int16);gap=np.ones(len(xs),np.float32)*1e6
    for i,around in enumerate(neighbors):
        if excluded[ys[i],xs[i]] or anis[ys[i],xs[i]]<.35:continue
        idx=np.asarray(around);delta=points[idx]-points[i];along=delta@tangent[i];across=np.abs(delta@normal[i]);distance=np.linalg.norm(delta,axis=1)
        good=healthy[idx]&(distance>1)&(distance<=max(8,min(24,4*radius[i])))&(np.abs(tangent[idx]@tangent[i])>np.cos(np.pi/6))&(across<max(.8,.5*radius[i]))&(radius[idx]/radius[i]>.8)&(radius[idx]/radius[i]<1.25)
        idx=idx[good];along=along[good]
        if len(idx)<3 or not (along>0).any() or not (along<0).any():continue
        order=np.argsort(np.abs(along))[:16];idx=idx[order];along=along[order]
        if not (along>0).any() or not (along<0).any():continue
        beta[i]=np.median(profiles[idx],axis=0);mad[i]=np.median(np.abs(profiles[idx]-beta[i]),axis=0);counts[i]=len(idx)
        gap[i]=along[along>0].min()-along[along<0].max()
    if not counts.any():return None
    nearest=ndi.distance_transform_edt(~skel,return_distances=False,return_indices=True);index=np.zeros(component.shape,np.int32);index[ys,xs]=np.arange(len(xs));owner=index[tuple(nearest)]
    ox=xx-xs[owner];oy=yy-ys[owner];position=(ox*normal[owner,0]+oy*normal[owner,1])/radius[owner]
    coordinate=np.clip((position+1.8)/3.6*32,0,32);left=np.floor(coordinate).astype(int).clip(0,31);frac=coordinate-left
    target=beta[owner,left]*(1-frac)+beta[owner,left+1]*frac;uncertainty=mad[owner,left]*(1-frac)+mad[owner,left+1]*frac
    slope=np.abs(beta[owner,np.minimum(left+1,32)]-beta[owner,np.maximum(left-1,0)])/(.225*radius[owner])
    width=(profiles>.5).sum(1)*.1125*radius
    return dict(target=target,mad=uncertainty,refs=counts[owner],gap=gap[owner],radius=radius[owner],slope=slope,
        width=width[owner],core_distance=ndi.distance_transform_edt(~skel),center=center[owner],supported_sections=int((counts>0).sum()))


def topology_guard(alpha,fill,edge,allowed):
    """@luna-comment Input:originalalpha/positivefill/signededge/localdomain. Output:edge withnegativechangesrejectediffragmentation/newholes. Key Steps:threecoveragelevels,comparefillversusedge. Key Params:originalcountersfrozenelsewhere. State/Dependencies:component-localdiagnosticguard."""
    rejected=False
    for level in [.3,.5,.7]:
        a=(alpha+fill>level)&allowed;b=(alpha+fill+edge>level)&allowed
        _,na=ndi.label(a,np.ones((3,3)));_,nb=ndi.label(b,np.ones((3,3)))
        ha=ndi.label(ndi.binary_fill_holes(a)&~a)[1];hb=ndi.label(ndi.binary_fill_holes(b)&~b)[1]
        if nb>na or hb>ha:rejected=True;break
    if rejected:edge=np.maximum(edge,0)
    return edge,rejected


def prevent_new_merges(alpha,delta,labels,support):
    """@luna-comment Input:originalcoverage/candidatestep/M0labels/support. Output:stepwithnovelcross-M0mergesrevertedandcount. Key Steps:threelevels,baselinecomponentmembershipintersection,newcomponentcheck. Key Params:existingjoinsallowed;localcandidateonlyreverted. State/Dependencies:readonlyoriginalgeometry."""
    rejected=0
    for threshold in [.3,.5,.7]:
        before,_=ndi.label((alpha>threshold)&support,np.ones((3,3)));after,_=ndi.label((alpha+delta>threshold)&support,np.ones((3,3)))
        selected=(labels>0)&(before>0)
        pairs=np.unique(np.stack([labels[selected],before[selected]],1),axis=0);membership={}
        for owner,part in pairs:membership.setdefault(int(owner),set()).add(int(part))
        selected=(labels>0)&(after>0);pairs=np.unique(np.stack([after[selected],labels[selected]],1),axis=0);groups={}
        for part,owner in pairs:groups.setdefault(int(part),set()).add(int(owner))
        bad=[]
        for part,owners in groups.items():
            if len(owners)<2:continue
            common=set.intersection(*(membership.get(owner,set()) for owner in owners))
            if not common:bad.append(part)
        if bad:
            region=np.isin(after,bad)&(delta!=0);delta=delta.copy();delta[region]=0;rejected+=len(bad)
    return delta,rejected


def process(job):
    """@luna-comment Input:oneauditedS2.2/M0/M2/validtuple. Output:fourfinitecandidatePNGs,masks/diffs/telemetry. Key Steps:appearancefromIonly,geometry/templates,hardlocality/anchor/countergates,scalarRGBupdates. Key Params:independenttwofill/twoedgebranches;nooriginalGTpixels. State/Dependencies:isolatedOUT;sourcehashchecks."""
    cv2.setNumThreads(1);started=time.time();identity=job['id'];folder=OUT/'images'/identity;folder.mkdir(parents=True,exist_ok=True)
    for k,p in job['files'].items():assert digest(p)==job['sha256'][k]
    raw=cv2.cvtColor(cv2.imread(job['files']['vosr_s22']),cv2.COLOR_BGR2RGB);image=raw.astype(np.float32)/255
    m0=cv2.imread(job['files']['text_mask_original'],0).astype(np.float32)/255;m2=cv2.imread(job['files']['text_mask_plus2'],0).astype(np.float32)/255;valid=cv2.imread(job['files']['valid'],0)>=254
    assert image.shape[:2]==m0.shape==m2.shape==valid.shape and np.all(m2>=m0)
    fg=(m0>.5)&valid;labels,count=ndi.label(fg,np.ones((3,3)));protected=protected_holes(m0);background=(m2<.025)&valid
    B,_=weighted_field(image,background.astype(np.float32),8)
    nearest=ndi.distance_transform_edt(~fg,return_distances=False,return_indices=True);owners=labels[tuple(nearest)]
    conflict=(ndi.maximum_filter(owners,3)!=ndi.minimum_filter(owners,3))&~fg
    frozen=protected|~valid|(m2<=.025);anchors=np.zeros(fg.shape,bool);direction=np.zeros_like(image);proposals=np.zeros((4,*fg.shape),np.float32);coverage=np.zeros(fg.shape,np.float32)
    eligibility=np.zeros(fg.shape,np.uint8);statistics=dict(components=count,appearance_skipped=0,template_skipped=0,supported_components=0,supported_sections=0,edge_topology_rejections=0)
    for number,slices in enumerate(ndi.find_objects(labels),1):
        if slices is None:continue
        y0=max(0,slices[0].start-24);y1=min(fg.shape[0],slices[0].stop+24);x0=max(0,slices[1].start-24);x1=min(fg.shape[1],slices[1].stop+24);sl=np.s_[y0:y1,x0:x1]
        component=labels[sl]==number
        if component.sum()<12:statistics['appearance_skipped']+=1;continue
        I=image[sl];bg=B[sl];dist=ndi.distance_transform_edt(component);amp=np.linalg.norm(I-bg,axis=2)
        possible=component&(m0[sl]>.7)&(dist>=max(.75,.4*dist.max()))&(amp>.03)
        if possible.sum()<8:statistics['appearance_skipped']+=1;continue
        lo,hi=np.quantile(amp[possible],[.45,.95]);donor=possible&(amp>=lo)&(amp<=hi)
        if donor.sum()<6:statistics['appearance_skipped']+=1;continue
        F,_=weighted_field(I,donor.astype(np.float32),max(3,min(10,2*np.median(dist[component]))))
        D=F-bg;den=(D*D).sum(2);alpha=((I-bg)*D).sum(2)/(den+1e-7);color=np.linalg.norm(I-bg-alpha[...,None]*D,axis=2)/np.sqrt(den+1e-7)
        coverage[sl][owners[sl]==number]=alpha[owners[sl]==number]
        template=section_templates(alpha,component)
        if template is None:statistics['template_skipped']+=1;continue
        owned=(owners[sl]==number)&~frozen[sl]&~conflict[sl]&~donor&(den>.03**2)
        anchors[sl]|=donor;direction[sl][owners[sl]==number]=D[owners[sl]==number]
        statistics['supported_components']+=1;statistics['supported_sections']+=template['supported_sections']
        target=template['target'];error=target-alpha
        for strength,param in enumerate(PARAMS):
            good=owned&(template['refs']>=param['refs'])&(template['mad']<=param['mad'])&(color<=param['residual'])&(target>.08)
            gap_ok=(template['center']>.2)|(template['gap']<=param['gap']*2*template['radius'])
            good&=gap_ok
            confidence=np.clip(1-color/param['residual'],0,1)
            fill=np.where(good&(error>param['tau']),np.minimum(param['gain']*confidence*np.maximum(error,0),param['limit']),0).astype(np.float32)
            fill=np.minimum(fill,np.maximum(1-alpha,0));fill[donor]=0
            proposals[strength,sl[0],sl[1]]=np.maximum(proposals[strength,sl[0],sl[1]],fill)
            edge_band=np.abs(target-.5)<=param['edge_band']*template['slope']
            edge_good=good&edge_band&(template['core_distance']>.75)
            difference=target-(alpha+fill)
            edge=np.where(edge_good&(np.abs(difference)>(.15 if strength==0 else .10)),np.clip(param['edge_gain']*confidence*difference,-param['edge_limit'],param['edge_limit']),0).astype(np.float32)
            thin=(template['width']<2)|(dist.max()<1.2)
            edge[thin]=np.maximum(edge[thin],0)
            edge,rejected=topology_guard(alpha,fill,edge,(owners[sl]==number)&(m2[sl]>.025));statistics['edge_topology_rejections']+=int(rejected)
            total=np.clip(fill+edge,np.minimum(0,-alpha),np.maximum(0,1-alpha));total[~owned]=0
            # Signed candidates do not use maximum-composition: ownership is unique.
            target_field=proposals[2+strength,sl[0],sl[1]];target_field[owned]=total[owned]
            eligibility[sl][good]=255
    frozen|=anchors
    records={}
    for index,name in enumerate(VARIANTS):
        delta=proposals[index];delta[frozen]=0
        delta,merges=prevent_new_merges(coverage,delta,labels,(m2>.025)&valid)
        # Bound the whole RGB scalar step, retaining direction and channel relations.
        low=np.full(delta.shape,-1e6,np.float32);high=np.full(delta.shape,1e6,np.float32)
        for c in range(3):
            d=direction[...,c];a=np.where(np.abs(d)>1e-8,-image[...,c]/np.where(np.abs(d)>1e-8,d,1),-1e6);b=np.where(np.abs(d)>1e-8,(1-image[...,c])/np.where(np.abs(d)>1e-8,d,1),1e6)
            low=np.maximum(low,np.minimum(a,b));high=np.minimum(high,np.maximum(a,b))
        delta=np.clip(delta,low,high);result=np.rint(np.clip(image+delta[...,None]*direction,0,1)*255).astype(np.uint8);result[frozen]=raw[frozen]
        assert np.array_equal(result[frozen],raw[frozen]) and np.array_equal(result[m2<=.025],raw[m2<=.025])
        path=folder/(name+'.png');assert cv2.imwrite(str(path),cv2.cvtColor(result,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
        difference=result.astype(np.int16)-raw.astype(np.int16);changed=np.any(difference!=0,2)
        signed=(difference*np.array([.299,.587,.114])).sum(2);vis=np.zeros_like(raw);vis[...,0]=np.clip(np.maximum(-signed,0)*8,0,255);vis[...,2]=np.clip(np.maximum(signed,0)*8,0,255)
        cv2.imwrite(str(folder/(name+'_diff.png')),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
        records[name]=dict(path=str(path),sha256=digest(path),changed_pixels=int(changed.sum()),mean_y_change_dn=float(signed.mean()),max_channel_change_dn=int(np.abs(difference).max()),protected_changed_pixels=0,outside_m2_changed_pixels=0,actual_negative_pixels=int((delta<0).sum()),cross_component_merge_rejections=merges)
    shutil.copy2(job['files']['vosr_s22'],folder/'original.png')
    for name,m in [('mask_original',m0),('mask_plus2',m2),('eligible',eligibility.astype(np.float32)/255)]:
        cv2.imwrite(str(folder/(name+'.png')),np.repeat(np.rint(m[...,None]*255).astype(np.uint8),3,axis=2))
    save(folder/'receipt.json',dict(id=identity,sample_id=job['sample_id'],dataset=job['dataset'],source_sha256=job['sha256'],statistics=statistics,variants=records,seconds=time.time()-started,shape=list(raw.shape),foreground_anchors=int(anchors.sum()),protected_counters=int(protected.sum())))
    return dict(id=identity,statistics=statistics,variants=records,seconds=time.time()-started)


def main():
    """@luna-comment Input:audited8sourcejobs. Output:independent32candidateimages/progress/completion. Key Steps:twoCPUworkers,sourcehash/localitychecks,nonidentitycoveragegate. Key Params:prototypeonly,notbulkGTreplacement. State/Dependencies:ownedOUT,sourceimmutable."""
    OUT.mkdir(parents=True,exist_ok=True);audit=json.loads((OUT/'INPUT_AUDIT.json').read_text());assert audit['passed'] and len(audit['jobs'])==8
    started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(process,audit['jobs']):
            results.append(result);save(OUT/'PROGRESS.json',dict(done=len(results),total=8,seconds=time.time()-started,results=results));print('DONE',result['id'],{k:v['changed_pixels'] for k,v in result['variants'].items()},flush=True)
    assert sum(r['variants']['fill_strong']['changed_pixels'] for r in results)>0,'All candidates inactive;do not present as repairs'
    save(OUT/'RESULTS.json',dict(passed=True,images=8,candidates=32,parameters=PARAMS,results=results,source_gt_replaced=False,limitations=['component-local appearance and nearest-normal-section prototype','unreliable/no-reference regions skipped','no global topology optimizer']))
    save(OUT/'COMPLETE.json',dict(passed=True,images=8,candidates=32,seconds=time.time()-started))


if __name__=='__main__':main()
