"""Input-referenced exterior changes on the eight original heldout Mojito frames."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.diagnose_halo003_images import OUT,path_of,luma,mask_of
from tools.run_uniform20_steep_scurve import save


def main():
    """@luna-comment Input:eightheldoutIDs/currentauditedfour500Koutputs. Output:fixedinputmaskwidth/exteriorresidualmetricsandnativecrops. Key Steps:EXIFdecode,relativewhiteboardROI,fourcleanpairedvariants. Key Params:inputreferenceonly,notGTaccuracyorautomaticqualityscore. State/Dependencies:readonlyexistingPNGs,noinference/training."""
    data=json.loads((OUT/'source_manifest.json').read_text());selection=json.loads((OUT.parent/'mojito280_subset32_20261104/selection.json').read_text())['samples']
    ids=[s['id'] for s in selection if s['split']=='val'];assert ids==['001','002','003','004','269','270','271','272']
    keys=['v3_input','v3m_l1_raw','v3m_mf_only_raw','v3m_uv_only_raw','v3m_mf_uv_raw'];records=[]
    for identity in ids:
        row=next(r for r in data['rows'] if r['id']==identity);w,h=row['width'],row['height'];box=(round(w*.10),round(h*.20),round(w*.80),round(h*.78));images={}
        for key in keys:
            with Image.open(path_of(row['images'][key])) as image:images[key]=np.asarray(ImageOps.exif_transpose(image).convert('RGB').crop(box))
        y=luma(images['v3_input']);mask,threshold,eligible,skel,dist=mask_of(y);outside=(dist>2)&(dist<=8);wa=2*__import__('scipy').ndimage.distance_transform_edt(mask)[skel]
        result=dict(id=identity,roi=box,models={})
        folder=OUT/'mojito_validation'/identity;folder.mkdir(parents=True,exist_ok=True)
        for key,rgb in images.items():
            yy=luma(rgb);m=(yy<threshold)&eligible;wb=2*__import__('scipy').ndimage.distance_transform_edt(m)[skel];res=yy-y
            centered=res-cv2.GaussianBlur(res,(0,0),32)
            result['models'][key]=dict(area_ratio=float(m.sum()/max(mask.sum(),1)),width_change_mean_px=float((wb-wa).mean()),
                exterior_residual_mean_dn=float(res[outside].mean()),exterior_centered_bright_excess_dn=float(np.maximum(centered[outside]-2,0).mean()),
                exterior_centered_dark_excess_dn=float(np.maximum(-centered[outside]-2,0).mean()))
            # Small display crop retains original pixels;allmetricsusefullfixedROI.
            crop=rgb[max(0,rgb.shape[0]//2-192):rgb.shape[0]//2+192,:1024]
            cv2.imwrite(str(folder/(key+'.png')),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
        records.append(result);print('MOJITO_HELDOUT',identity,flush=True)
    save(OUT/'MOJITO_VAL8.json',dict(passed=True,images=8,ids=ids,reference='V3 input,notGT;lowfrequencyresidualremovedforouterringcontrast',records=records))


if __name__=='__main__':main()
