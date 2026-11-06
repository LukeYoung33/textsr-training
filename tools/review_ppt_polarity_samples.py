"""Export local original/V8/adaptive comparisons without modifying processing."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_dehalo_20260924 import write_png


def main():
    """@luna-comment Input:shared sample/source/old roots,local output. Output:review PNGs.
    Key Steps:decode6 outputs,consistent crops and sizes,aggregate numeric receipts.
    Key Params:100percent detail,1000px full preview; originals untouched.
    State/Dependencies:reads finished remote files via share,local diagnostics only.
    """
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('samples','source','old','output'): p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False); reports=[]
    for path in sorted((args.samples/'diagnostics').glob('*/*.json')):
        r=json.loads(path.read_text()); assert r['status']=='complete',r
        group=r['group']; name=Path(r['source']).name
        src=read_image(args.source/group/name); old=read_image(args.old/group/(Path(name).stem+'_dehalo_v8.png')); new=read_image(args.samples/r['output'])
        h,w=src.shape[:2]; images=[src,old,new]
        if group=='blackwhite': box=(int(.20*w),int(.28*h),int(.60*w),int(.43*h))
        else: box=(950,450,min(w,2150),min(h,800))
        x0,y0,x1,y1=box
        for label in ('full','detail'):
            panels=[]
            for title,im in zip(('Input','V8 previous','V8 adaptive'),images):
                crop=cv2.resize(im,(1000,round(h*1000/w)),interpolation=cv2.INTER_AREA) if label=='full' else im[y0:y1,x0:x1]
                panel=cv2.copyMakeBorder(crop,28,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
                cv2.putText(panel,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1); panels.append(panel)
            write_png(args.output/f'{group}_{Path(name).stem}_{label}.png',np.concatenate(panels))
        for core in ('dark_core','light_core'):
            mask=read_image(path.parent/Path(name).stem/(core+'.png'))[:,:,0]>0
            r[core+'_changed']=int(np.count_nonzero(np.any(new!=src,axis=2)&mask))
            assert r[core+'_changed']==0
        reports.append(r)
    (args.output/'review.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(dict(reviewed=len(reports),output=str(args.output))))


if __name__=='__main__': main()
