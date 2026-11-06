"""Independent outside-anchor plane checks and exact same-position detail previews."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_crossseam_v9 import sample
from tools.postprocess_ppt_dehalo_20260924 import write_png


def main():
    """@luna-comment Input: original source and completed P13 folder. Output: QA evidence.
    Key Steps: independent local exterior-anchor plane, replay profiles, enlarged crops.
    Key Params: anchor window radius100; lowpass2; no reconstruction-target fitting.
    State/Dependencies: read-only originals/results; writes evaluation artifacts only.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=Path('results/ppt_dehalo_20260924');receipts=json.loads((args.output/'receipt.json').read_text());evidence=[]
    prior=json.loads((root/'sample_v4_normal/receipt.json').read_text());cv2.setNumThreads(4)
    for rec in receipts:
        name=rec['image'];stem=name.removesuffix('_dehalo.png');path=args.input/(stem+'.png')
        original=read_image(path);low=cv2.GaussianBlur(original.astype(np.float32),(0,0),2)
        debug=args.output/'debug'/stem
        anchors=cv2.imread(str(debug/'anchors.png'),0)>0;weight=cv2.imread(str(debug/'weight.png'),0)/255.
        v10=read_image(root/'sample_v10_wide_final'/name);final=read_image(args.output/name);plain=read_image(args.output/'surface_only'/name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();baseline=next(r for r in prior if r['source'].endswith(stem+'.png'))
        assert digest==baseline['sha256']
        imdict=dict(v10=v10,plain=plain,final=final)
        report=dict(image=name,source_hash_match=True,profiles=[])
        for prof in rec['profiles']:
            x,y=prof['pixel'];x0=max(0,x-100);x1=min(low.shape[1],x+101);y0=max(0,y-100);y1=min(low.shape[0],y+101)
            ay,ax=np.nonzero(anchors[y0:y1,x0:x1]);ax=ax+x0;ay=ay+y0
            if len(ax)<100:continue
            A=np.column_stack([np.ones(len(ax)),(ax-x)/100.,(ay-y)/100.])
            coef=np.linalg.lstsq(A,low[ay,ax],rcond=None)[0]
            px=np.array(prof['x'],np.float32);py=np.array(prof['y'],np.float32)
            trend=np.column_stack([np.ones(len(px)),(px-x)/100.,(py-y)/100.])@coef
            yy=trend@np.array([.114,.587,.299]);eligible=sample(weight.astype(np.float32),px,py)>.5
            entry=dict(pixel=prof['pixel'],anchor_count=len(ax),excluded_fraction=float(1-eligible.mean()),metrics={})
            for key,im in imdict.items():
                colors=sample(cv2.GaussianBlur(im.astype(np.float32),(0,0),2),px,py);luma=colors@np.array([.114,.587,.299])
                entry['metrics'][key]=dict(median_absolute_trend_error=float(np.median(np.abs(luma[eligible]-yy[eligible]))) if eligible.any() else None,peak_step=float(np.abs(np.diff(luma)).max()),endpoint_difference=float(luma[-1]-luma[0]))
            report['profiles'].append(entry)
        for label,(x0,y0,x1,y1) in dict(marked_seam=(1780,495,2110,665),practice=(1430,760,1960,970)).items():
            panels=[];texture={}
            for caption,im in imdict.items():
                crop=im[y0:y1,x0:x1];big=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
                panel=cv2.copyMakeBorder(big,26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30));cv2.putText(panel,caption,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
                gy=cv2.cvtColor(crop.astype(np.float32),cv2.COLOR_BGR2GRAY);hp=gy-cv2.GaussianBlur(gy,(0,0),2);mask=weight[y0:y1,x0:x1]>.95
                texture[caption]=float(hp[mask].std()) if mask.any() else None
            write_png(args.output/'comparisons'/(stem+'_'+label+'_200pct.png'),np.concatenate(panels));report[label+'_highpass_std']=texture
        evidence.append(report)
    (args.output/'independent_evaluation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    for r in evidence:
        print(r['image'],'source unchanged:',r['source_hash_match'])
        for p in r['profiles'][:5]:print(p)


if __name__=='__main__':main()
