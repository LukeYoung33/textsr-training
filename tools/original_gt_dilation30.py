"""Read-only original aligned GT lineage and subpixel foreground dilation diagnostic."""
from pathlib import Path
import sys,json,math,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'original_gt_dilation30_20261102'


def thicken(gt,mask,valid,radius):
    """@luna-comment Input:uint8 RGB/BGR image,soft foreground/valid masks,radius px. Output:expanded-stroke image and support. Key Steps:background-weighted local polarity;8 shifted grayscale extrema;copy winning RGB triplet only near mask. Key Params:0.5/1/1.5px radius,valid>=254. State/Dependencies:OpenCV;source immutable,no PSF fitting."""
    assert gt.dtype==np.uint8 and mask.shape==valid.shape==gt.shape[:2] and 0<=radius<=1.5
    if radius==0:return gt.copy(),np.zeros(mask.shape,bool)
    gray=cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY).astype(np.float32);m=mask.astype(np.float32)/255
    bgw=(1-m)*(valid>=254);den=cv2.GaussianBlur(bgw,(0,0),8)
    background=cv2.GaussianBlur(gray*bgw,(0,0),8)/np.maximum(den,1e-6)
    foreground=cv2.GaussianBlur((gray-background)*m,(0,0),4)
    bright=foreground>=0
    n=int(math.ceil(radius));support=(cv2.dilate((m>.05).astype(np.uint8),np.ones((2*n+1,2*n+1),np.uint8))>0)&(valid>=254)
    best=gray.copy();result=gt.astype(np.float32).copy()
    for angle in np.arange(8)*math.pi/4:
        shift=np.array([[1,0,radius*math.cos(angle)],[0,1,radius*math.sin(angle)]],np.float32)
        candidate=cv2.warpAffine(gt.astype(np.float32),shift,(gt.shape[1],gt.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        cm=cv2.warpAffine(m,shift,(gt.shape[1],gt.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        cv=cv2.warpAffine(valid,shift,(gt.shape[1],gt.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        level=cv2.cvtColor(candidate,cv2.COLOR_BGR2GRAY)
        choose=support&(cm>.025)&(cv>=254)&np.where(bright,level>best,level<best)
        best[choose]=level[choose];result[choose]=candidate[choose]
    result=np.clip(np.rint(result),0,255).astype(np.uint8)
    assert np.array_equal(result[~support],gt[~support])
    return result,support


def main():
    """@luna-comment Input:authoritative pre-teacher manifest. Output:30 stratified original/LQ/mask/three dilation sets and hash receipts. Key Steps:even samples within four sources;strict hash/shape verification;native morphology. Key Params:8HXA/8HXB/7Lenovo0926/7Lenovo1011,no source edits. State/Dependencies:CPU-only separate output."""
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    manifest=DATA/'vosr_2u_block256_v1_20261023/source_view/manifests/train.jsonl'
    rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()];selected=[]
    for name,count in [('HXA',8),('HXB',8),('lenovo0926',7),('lenovo1011',7)]:
        group=sorted([r for r in rows if r['dataset']==name],key=lambda r:r['sample_id'])
        selected.extend(group[i] for i in np.linspace(0,len(group)-1,count,dtype=int))
    assert len(selected)==30
    save(OUT/'SELECTION.json',dict(manifest=str(manifest),manifest_sha256=digest(manifest),rule='stratified equally spaced sorted sample IDs;8/8/7/7',rows=selected))
    results=[];start=time.time()
    for index,row in enumerate(selected,1):
        key=f'{index:03d}';folder=OUT/'images'/key;folder.mkdir(parents=True,exist_ok=True)
        paths={'original':row['original_gt_lineage']['path'],'lq':row['files']['lq'],'mask':row['files']['text_mask'],'valid':row['files']['valid_mask']}
        expected={'original':row['original_gt_lineage']['sha256'],'lq':row['output_sha256']['lq'],'mask':row['output_sha256']['text_mask'],'valid':row['output_sha256']['valid_mask']}
        for k,path in paths.items():assert digest(path)==expected[k],(key,k)
        gt=cv2.imread(paths['original']);lq=cv2.imread(paths['lq']);mask=cv2.imread(paths['mask'],0);valid=cv2.imread(paths['valid'],0)
        assert gt is not None and gt.shape==lq.shape and gt.shape[:2]==mask.shape==valid.shape,(key,'shape')
        outputs={'original':gt,'lq':lq,'mask':cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)};checks={}
        for label,radius in [('width1',.5),('width2',1.),('width3',1.5)]:
            value,support=thicken(gt,mask,valid,radius);outputs[label]=value
            checks[label]=dict(radius=radius,nominal_width_increase=2*radius,changed_pixels=int(np.any(value!=gt,axis=2).sum()),outside_support_unchanged=bool(np.array_equal(value[~support],gt[~support])))
        hashes={}
        for k,value in outputs.items():
            path=folder/(k+'.png');assert cv2.imwrite(str(path),value,[cv2.IMWRITE_PNG_COMPRESSION,3]);hashes[k]=digest(path)
        results.append(dict(id=key,name=row['dataset']+' / '+row['sample_id'],width=gt.shape[1],height=gt.shape[0],lq=f'images/{key}/lq.png',images={k:f'images/{key}/{k}.png' for k in outputs if k!='lq'},source=paths,source_sha256=expected,output_sha256=hashes,checks=checks))
        save(OUT/'STATUS.json',dict(completed=index,total=30,seconds=time.time()-start));print(key,flush=True)
    save(OUT/'RESULTS.json',dict(passed=True,rows=results,scope='Original aligned displayfit GT before VOSR/V3;not unaligned camera originals. Native pixel radii;approximate width change,not PSF equivalence.'))


if __name__=='__main__':main()
