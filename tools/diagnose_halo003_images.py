"""Read-only original-pixel halo/width audit on fixed Mojito003 model outputs."""
from pathlib import Path
from urllib.parse import unquote
import json
import sys
import hashlib
import html
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image, ImageOps

OUT = Path('/AI_DNS/guang.yang/TextSR/data/halo003_diagnostic_20261104')
ROIS = {'poem': (650, 1900, 2700, 2370), 'small': (80, 1450, 2450, 1690),
        'medium': (100, 1180, 3000, 1400), 'large': (150, 570, 2800, 790),
        'blank': (3150, 750, 3950, 1250)}
KEYS = ['lq', 'v3_input', 'v3m_l1_raw', 'v3m_mf_only_raw', 'v3m_full_noguard_raw',
        'v3m_mf_uv_raw', 'v3m_mf_uv_mix50_raw', 'v3m_original_gt_dilate2_l1_raw',
        'v3m_dilate2_r1_width_370k_raw', 'v3_original_aligned_gt_full_250k_raw']


def path_of(url):
    """@luna-comment Input:currentmappedYfileURL. Output:exactNASpath. Key Steps:decodeandreplaceonlymappedprefix. Key Params:rejectotherroots. State/Dependencies:read-onlymanifest."""
    assert url.startswith('file:///Y:/')
    return Path('/AI_DNS')/unquote(url[len('file:///Y:/'):])


def luma(rgb):
    """@luna-comment Input:HWC RGB8. Output:float32Rec601YinDN. Key Steps:fixedcoefficients. Key Params:0..255domain. State/Dependencies:numpy,noimagechanges."""
    return np.einsum('hwc,c->hw', rgb.astype(np.float32), np.array([.299,.587,.114],np.float32))


def mask_of(y):
    """@luna-comment Input:inputYROI. Output:fixedinputhalfcontrastmask/threshold/candidate/skeleton/distance. Key Steps:31pxmorphologicalbackground,15pxlocalcontrast,sharedthreshold. Key Params:contrast>20DN;masknotGTandnotaccuracy. State/Dependencies:OpenCV/scipy/skimage."""
    bg = cv2.morphologyEx(y, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    contrast = cv2.dilate(np.maximum(bg-y,0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    eligible = contrast>20
    threshold = bg-.5*contrast
    mask = (y<threshold)&eligible
    return mask, threshold, eligible, skeletonize(mask), ndi.distance_transform_edt(~mask)


def main():
    """@luna-comment Input:localgallerymanifestcopiedtoowneddiagnosticroot. Output:003fixedROIresidual/annulus/widthmetricsandChineseHTMLcrops. Key Steps:losslessEXIFdecode,sharedinputmask,10models/sameROIs,signedDNandskeletonwidth. Key Params:noGTclaim;norefittingpermodelthreshold. State/Dependencies:immutableauditedsourcePNGs;noinferenceoroptimizer."""
    OUT.mkdir(exist_ok=True)
    data = json.loads((OUT/'source_manifest.json').read_text())
    row = next(r for r in data['rows'] if r['id']=='003')
    values = {};bindings={}
    for key in KEYS:
        p = path_of(row['lq'] if key=='lq' else row['images'][key])
        with Image.open(p) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        assert rgb.shape[:2] == (row['height'],row['width'])
        bindings[key]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        values[key]={name:rgb[y0:y1,x0:x1].copy() for name,(x0,y0,x1,y1) in ROIS.items()}
    results={};profile_rows=[]
    for name in ROIS:
        source=luma(values['v3_input'][name]);mask,threshold,eligible,skel,distance=mask_of(source)
        reference_width=2*ndi.distance_transform_edt(mask)[skel]
        results[name]={}
        for key in KEYS:
            rgb=values[key][name];y=luma(rgb);residual=y-source
            m=(y<threshold)&eligible
            widths=2*ndi.distance_transform_edt(m)[skel]
            item=dict(mean_residual_dn=float(residual.mean()),residual_abs_dn=float(np.abs(residual).mean()),
                residual_sigma8_std_dn=float(cv2.GaussianBlur(residual,(0,0),8).std()))
            if name!='blank':
                item.update(input_mask_pixels=int(mask.sum()),area_ratio=float(m.sum()/max(mask.sum(),1)),
                    input_width_median_px=float(np.median(reference_width)),width_median_px=float(np.median(widths)),
                    width_change_mean_px=float(np.mean(widths-reference_width)),rings=[])
                for lo,hi in [(0,2),(2,4),(4,8),(8,16),(16,32)]:
                    ring=(distance>lo)&(distance<=hi)
                    v=residual[ring]
                    info=dict(lo=lo,hi=hi,pixels=int(ring.sum()),mean_dn=float(v.mean()),
                        dark_fraction_below_minus2=float((v< -2).mean()),bright_fraction_above2=float((v>2).mean()),
                        negative_mean_dn=float(np.minimum(v,0).mean()),positive_mean_dn=float(np.maximum(v,0).mean()))
                    item['rings'].append(info)
            results[name][key]=item
            folder=OUT/'crops'/name;folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(folder/(key+'.png'))
            if key not in ['lq','v3_input']:
                signed=np.zeros((*residual.shape,3),np.uint8);amp=np.clip(np.abs(residual)*12,0,255).astype(np.uint8)
                signed[...,0]=np.where(residual<0,amp,0);signed[...,2]=np.where(residual>0,amp,0)
                Image.fromarray(signed).save(folder/(key+'_residual.png'))
    report=dict(sample='003',rois=ROIS,bindings=bindings,metrics=results,
        contract='All DN/width comparisons relative to V3 input under one fixed input-derived mask. Not GT accuracy. Negative residual darkens;positive brightens. Width is twice EDT at fixed input skeleton,not literal font ground truth.')
    (OUT/'IMAGE_METRICS.json').write_text(json.dumps(report,indent=2))
    labels={m['key']:m['label'] for m in data['models']}
    body='<!doctype html><meta charset="utf-8"><title>003 黑影、halo及笔画变粗诊断</title><style>body{font:16px/1.6 sans-serif;margin:24px}section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}article{border:1px solid #aaa;padding:8px}img{width:100%;image-rendering:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style><h1>003 黑影、halo及笔画变粗：像素证据</h1><p>全部为模型原始输出；对照V3输入。残差图红色=变暗，蓝色=变亮，放大12倍。当前图没有对应可信GT，以下量化描述变化，不代表修复准确率。</p>'
    for name in ROIS:
        body+='<h2>'+html.escape(name)+' · 原图坐标 '+str(ROIS[name])+'</h2><table><tr><th>模型</th><th>前景面积比</th><th>平均宽度变化(px)</th><th>4–8px环平均残差(DN)</th></tr>'
        for key in KEYS:
            r=results[name][key]
            body+='<tr><td>'+html.escape(labels.get(key,key))+'</td><td>'+str(round(r.get('area_ratio',0),3))+'</td><td>'+str(round(r.get('width_change_mean_px',0),3))+'</td><td>'+str(round(r.get('rings',[{},{},{'mean_dn':0}])[2]['mean_dn'],3))+'</td></tr>'
        body+='</table><section>'
        for key in KEYS:
            body+='<article><b>'+html.escape(labels.get(key,key))+'</b><img src="crops/'+name+'/'+key+'.png">'
            if key not in ['lq','v3_input']:body+='<img src="crops/'+name+'/'+key+'_residual.png">'
            body+='</article>'
        body+='</section>'
    (OUT/'images.html').write_text(body,encoding='utf-8')
    print(json.dumps(dict(passed=True,models=len(KEYS),rois=len(ROIS))))


if __name__=='__main__':main()
