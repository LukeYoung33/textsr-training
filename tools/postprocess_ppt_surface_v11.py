"""P13: clean-exterior anchored harmonic background, preserving all stroke guards."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes,distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare,smoothstep,write_png
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_crossseam_v9 import sample


def harmonic_surface(anchors,unknown,roi,initial,tolerance=.001,max_iterations=2000):
    """@luna-comment Input: colors/masks/guess. Output: anchored harmonic field/support.
    Key Steps: discard unanchored components, Jacobi valid-neighbor solve, residual.
    Key Params: four neighbors, no-flux ROI boundary, fixed known data, max2000.
    State/Dependencies: OpenCV float arrays; initial estimate never an anchor.
    """
    kernel=np.array([[0,1,0],[1,0,1],[0,1,0]],np.float32)
    known=roi&~unknown
    count,labels=cv2.connectedComponents(unknown.astype(np.uint8),connectivity=4)
    adjacent=cv2.filter2D(known.astype(np.float32),-1,kernel,borderType=cv2.BORDER_CONSTANT)>0
    supported_ids=np.unique(labels[unknown&adjacent])
    supported=np.isin(labels,supported_ids)&unknown
    rf=roi.astype(np.float32);degree=cv2.filter2D(rf,-1,kernel,borderType=cv2.BORDER_CONSTANT)
    supported &= degree>0
    field=initial.copy();field[known]=anchors[known]
    residual=0.;iteration=0
    for iteration in range(1,max_iterations+1):
        avg=cv2.filter2D(field*rf[...,None],-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.maximum(degree[...,None],1)
        if iteration%10==0:
            residual=float(np.abs(field[supported]-avg[supported]).max()) if supported.any() else 0.
            if residual<=tolerance:break
        field[supported]=avg[supported]
    avg=cv2.filter2D(field*rf[...,None],-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.maximum(degree[...,None],1)
    residual=float(np.abs(field[supported]-avg[supported]).max()) if supported.any() else 0.
    return field,known|supported,dict(iterations=iteration,residual=residual,converged=residual<=.02,unsupported_components=int(count-1-len(supported_ids)),known_pixels=int(known.sum()))


def reconstruct(base,prepared,alpha):
    """@luna-comment Input: V4 and original buffers. Output: surface/final diagnostics.
    Key Steps: exclude complete patch from anchors, harmonic solve, guarded composite.
    Key Params: outer24, stroke6..12, bounded luma detail2; quarter-resolution solve.
    State/Dependencies: no old-background limiter; no text reconstruction.
    """
    src,y,roi,core,bg,_,_=prepared;by=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    clear=distance_transform_edt(~(roi&(by-y>12))).astype(np.float32)
    seed=(roi&(core|(alpha>=.5))).astype(np.uint8)
    patch=binary_fill_holes(cv2.morphologyEx(seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))))&roi
    dp=distance_transform_edt(~patch);region=roi&(dp<24);unknown=roi&((dp<24)|(clear<16))
    low=cv2.GaussianBlur(src,(0,0),2);h,w=y.shape;size=(w//4,h//4)
    q=cv2.resize(low,size,interpolation=cv2.INTER_AREA)
    rq=cv2.resize(roi.astype(np.float32),size,interpolation=cv2.INTER_AREA)>.999999
    uq=(cv2.resize(unknown.astype(np.float32),size,interpolation=cv2.INTER_AREA)>0)&rq
    initial=cv2.resize(bg,size,interpolation=cv2.INTER_AREA)
    surface,support,report=harmonic_surface(q,uq,rq,initial)
    if not report['converged']:raise RuntimeError(f'Harmonic solver did not converge: {report}')
    target=cv2.resize(surface,(w,h),interpolation=cv2.INTER_LINEAR)
    supported=cv2.resize(support.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    weight=region*supported*smoothstep((clear-6)/6)*smoothstep(distance_transform_edt(region)/12)*smoothstep((distance_transform_edt(roi)-12)/12)
    detail=np.clip(y-cv2.GaussianBlur(y,(0,0),2),-2,2)*smoothstep((clear-10)/6)
    plain=base+weight[...,None]*(target-base)
    final=base+weight[...,None]*(target+detail[...,None]-base)
    outputs=[np.clip(np.rint(im),0,255).astype(np.uint8) for im in (plain,final)]
    reference=np.clip(np.rint(base),0,255).astype(np.uint8)
    protected=(clear<=6)|~region|~supported
    for out in outputs:assert np.array_equal(out[protected],reference[protected])
    report.update(protected_changed=0,clipped=int(np.count_nonzero((final<0)|(final>255))),changed=int(np.any(outputs[1]!=reference,axis=2).sum()),max_change=float(np.abs(final-base).max()))
    return outputs,dict(target=target,old_background=bg,weight=weight,anchors=roi&~unknown,unknown=unknown,region=region,protected=protected,detail=detail),report


def main():
    """@luna-comment Input: two-sample source/output. Output: P13 variants/debug/profiles.
    Key Steps: V4 verification, surface reconstruction, exact historical profile replay.
    Key Params: same crops and coordinates, surface/final comparison, no batch.
    State/Dependencies: prior images immutable, new output directory only.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);reports=[]
    root=Path('results/ppt_dehalo_20260924');oldreports=json.loads((root/'sample_v10_wide_final/receipt.json').read_text())
    for ident in ('044920599','052927799'):
        path=next(args.input.glob(f'*{ident}_preprocessInput.png'));src=read_image(path);prepared=prepare(src,robust_seed=True)
        v4,maps,_=normal_correct(prepared);base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0]);name=path.stem+'_dehalo.png'
        assert np.array_equal(v4,read_image(root/'sample_v4_normal'/name))
        v10=read_image(root/'sample_v10_wide_final'/name)
        (plain,result),debug,report=reconstruct(base,prepared,maps['correction_weight'])
        write_png(args.output/name,result);write_png(args.output/'surface_only'/name,plain);assert np.array_equal(result,read_image(args.output/name))
        folder=args.output/'debug'/path.stem
        for key in ('weight','anchors','unknown','region','protected'):
            write_png(folder/(key+'.png'),np.clip(np.rint(debug[key]*255),0,255).astype(np.uint8))
        write_png(folder/'surface.png',np.clip(np.rint(debug['target']),0,255).astype(np.uint8))
        write_png(folder/'oldB_minus_surface.png',np.clip(np.rint(128+4*(debug['old_background']-debug['target'])),0,255).astype(np.uint8))
        report['profiles']=[]
        previous=next(r for r in oldreports if r['image']==name)
        for prof in previous['profiles']:
            x=np.array(prof['x'],np.float32);yy=np.array(prof['y'],np.float32);entry=dict(pixel=prof['pixel'],x=prof['x'],y=prof['y'])
            for key,im in [('input',src),('v4',v4),('v10',v10),('surface',debug['target']),('final',result)]:
                colors=sample(im.astype(np.float32),x,yy);luma=colors@np.array([.114,.587,.299])
                entry[key]=dict(rgb=colors.tolist(),y=luma.tolist(),peak_step=float(np.abs(np.diff(luma)).max()))
            report['profiles'].append(entry)
        report['quality']={}
        for label,(x0,y0,x1,y1) in dict(title=(950,450,2750,760),practice=(950,740,2150,990),degrees=(1750,1260,2950,1480),color=(600,1690,2700,1890)).items():
            sl=np.s_[y0:y1,x0:x1];panels=[]
            for title,im in [('V10',v10),('V11 surface only',plain),('V11 reconstructed background + detail',result)]:
                panel=cv2.copyMakeBorder(im[sl],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30));cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
            mask=debug['weight'][sl]>.95;target=cv2.cvtColor(debug['target'][sl],cv2.COLOR_BGR2GRAY)
            metrics={}
            for key,im in [('v10',v10),('final',result)]:
                error=cv2.cvtColor(im[sl].astype(np.float32),cv2.COLOR_BGR2GRAY)-target
                metrics[key]=float(np.median(np.abs(error[mask]))) if mask.any() else None
            report['quality'][label]=metrics
        report['image']=name;reports.append(report);(args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='profiles'}),flush=True)


if __name__=='__main__':main()
