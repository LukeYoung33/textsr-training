"""Show original donor-boundary background reconstruction versus old cleanup."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_camera_2x2_viewer import render_html,digest
SOURCE=Path('Y:/guang.yang/TextSR/data/mask_gt_001_near_background_20261104')
OUT=ROOT/'evaluation/mask_gt_001_m2_direct_20261104'


def main():
    """@luna-comment Input:twoauditednear-backgroundcandidates/current001page. Output:samepageold/newcomparisonplusdonorlocations. Key Steps:newhash/frozenboundarychecks,samecrop,actualdonoroverlay,render. Key Params:001only;oldresultsretained. State/Dependencies:NASnewassets/localviewer."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    audit=json.loads((SOURCE/'AUDIT.json').read_text());box=tuple(audit['crop_xyxy']);data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));row=data['rows'][0]
    for v in audit['variants'].values():
        assert digest(SOURCE/Path(v['path']).name)==v['sha256']
        assert v['known_background_changed_pixels']==v['deep_ink_changed_pixels']==v['outside_roi_changed_pixels']==0
    entries=[('nearest_conservative','nearest_conservative.png','近邻背景修补 · 保守'),('nearest_strong','nearest_strong.png','近邻背景修补 · 强诊断'),
        ('near_unknown','unknown_background.png','本次背景待修位置'),('near_unresolved','unresolved.png','无法取得可信背景的位置'),('near_background_only','nearest_background.png','仅近邻背景修补（诊断）')]
    keys={k for k,_,_ in entries}|{'near_donors'};data['models']=[m for m in data['models'] if m['key'] not in keys]
    for key,filename,label in entries:
        with Image.open(SOURCE/filename) as im:crop=im.crop(box)
        target=OUT/(key+'.png');crop.save(target);row['images'][key]=target.as_uri();data['models'].append(dict(key=key,label=label,note='同一背景区域的紧邻原像素；已知背景值固定，不跨笔画取色'))
    original=np.asarray(Image.open(OUT/'lq.png').convert('RGB')).copy()
    with Image.open(SOURCE/'known_boundary_donors.png') as im:donors=np.asarray(im.crop(box))[:,:,0]>0
    with Image.open(SOURCE/'unknown_background.png') as im:unknown=np.asarray(im.crop(box))[:,:,0]>0
    contour,_=cv2.findContours(unknown.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(original,contour,-1,(255,145,25),1);original[donors]=(0,230,255)
    target=OUT/'near_donors.png';Image.fromarray(original).save(target);row['images']['near_donors']=target.as_uri();data['models'].append(dict(key='near_donors',label='近邻原背景取色点（青色）',note='橙色待修边缘；青色为相邻已知背景像素'))
    for m in data['models']:
        if m['key']=='background_estimate':m['label']='旧版：远背景平滑估计（问题对照）'
        if m['key']=='fill_clean_strong':m['label']='旧版：远背景清理 · 强档'
    s=audit['variants']['nearest_strong']
    data.update(title='001 · 补齐笔画＋近邻原背景修补',subtitle='原背景边界固定 · 字腔独立取色 · 新旧对照',defaults='lq,mask_plus2,near_donors,expanded_fill_strong,fill_clean_strong,nearest_strong')
    data['notes']='仅训练GT样例001。旧版8px外取色再高斯平滑会引入不同背景色，本版改用修补区紧邻原背景像素。\n背景区域按M2隔开，文字像素不能作背景取色；字腔优先固定自身深处原背景。修补区周边已知背景RGB严格不变，内部连续插值。\n上排原GT/M2/实际近邻取色位置，下排只补齐/旧远背景清理/新近邻背景修补。\n已知背景、深笔画和操作区外像素与只补齐版本逐像素一致；不重新拉亮度，不修改源GT。\n已知边界取色点'+str(audit['immediate_known_boundary_donors'])+'个，字腔内部原背景锚点'+str(audit['closed_counter_anchor_pixels'])+'个，无法可靠修补的背景像素'+str(audit['unresolved_pixels'])+'个。\n旧修补区边界处相邻RGB色差均值：'+f'{s["previous_interface_step_dn"]:.2f} → {s["new_interface_step_dn"]:.2f}'+' DN。此数值是连续性检查，不替代看图判断。'
    text=render_html(data);assert text.count('for(let i=0;i<4;i++)')==1
    text=text.replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;').replace('与 LQ','与原GT').replace('与LQ','与原GT')
    (OUT/'index.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'NEAREST_BACKGROUND_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(passed=True,sample='001',candidates=2,options=len(data['models']))))


if __name__=='__main__':main()
