"""Append real fill-and-background-cleanup candidates to the existing001 page."""
from pathlib import Path
import json,sys
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_camera_2x2_viewer import render_html,digest
SOURCE=Path('Y:/guang.yang/TextSR/data/mask_gt_001_fill_clean_20261104')
OUT=ROOT/'evaluation/mask_gt_001_m2_direct_20261104'


def main():
    """@luna-comment Input:new001cleanup2variants/completion andexistingpage. Output:updatedsamepagewithpreviousversions/cleanupmaps. Key Steps:newonlyhash/invariantchecks,nativecrop,sixpanelrender. Key Params:001only,oldassetsretained,actual8pxbanddisclosed. State/Dependencies:NASnewPNGs/localmanifest."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    audit=json.loads((SOURCE/'AUDIT.json').read_text());box=tuple(audit['crop_xyxy'])
    for v in audit['variants'].values():
        assert digest(SOURCE/Path(v['path']).name)==v['sha256']
        assert v['deep_ink_changed_pixels']==v['far_background_changed_pixels']==v['outside_roi_changed_pixels']==0
    data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));row=data['rows'][0]
    raw=np.asarray(Image.open(OUT/'lq.png').convert('RGB'));mask=np.asarray(Image.open(OUT/'mask_plus2.png').convert('RGB'))[...,0]
    overlay=raw.copy();contours,_=cv2.findContours((mask>127).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(80,255,50),1)
    for a,b,c,d in audit['zones'].values():cv2.rectangle(overlay,(max(0,a-5),max(0,b-5)),(min(383,c+5),min(383,d+5)),(255,140,30),1)
    Image.fromarray(overlay).save(OUT/'cleanup_contour.png')
    entries=[('expanded_fill_strong',SOURCE/'fill_stage/direct_strong.png','同范围：只补齐（含月）'),
        ('fill_clean_conservative',SOURCE/'fill_clean_conservative.png','补齐＋清理mask外 · 保守'),
        ('fill_clean_strong',SOURCE/'fill_clean_strong.png','补齐＋清理mask外 · 强诊断'),
        ('cleanup_diff',SOURCE/'fill_clean_strong_diff.png','清外增量差分 ×4（红暗／蓝亮）'),
        ('cleanup_map',SOURCE/'fill_clean_strong_map.png','清外实际修改位置'),
        ('background_estimate',SOURCE/'background_estimate.png','局部背景B估计（诊断）')]
    new_keys={e[0] for e in entries}|{'cleanup_contour'};data['models']=[m for m in data['models'] if m['key'] not in new_keys]
    for key,path,label in entries:
        with Image.open(path) as im:crop=im.crop(box)
        target=OUT/(key+'.png');crop.save(target);row['images'][key]=target.as_uri()
        data['models'].append(dict(key=key,label=label,note='001英文、岁、月、在；近轮廓8px内局部背景恢复'))
    row['images']['cleanup_contour']=(OUT/'cleanup_contour.png').as_uri();data['models'].append(dict(key='cleanup_contour',label='M2轮廓＋本次清外范围',note='绿色M2轮廓，橙色允许操作区域'))
    data.update(title='001 · 补齐mask内＋清理mask外',subtitle='局部VOSR墨色和背景取色 · 旧版保留对照',defaults='lq,mask_plus2,cleanup_contour,expanded_fill_strong,fill_clean_conservative,fill_clean_strong')
    strong=audit['variants']['fill_clean_strong']
    data['notes']='仅训练GT样例001，非Mojito001。\n本版在英文、岁、月、在区域采用M2补齐笔画，同时清除轮廓及字腔附近的多余墨迹。\n背景从距M2至少8px的VOSR原背景估计；近5px清理，5–8px渐变到零，不向纯黑或纯白推。强档另处理近4px内颜色方向一致的反极性阴影。\nM2深笔画核心、远处背景和操作区域外与同范围只补齐版本逐像素一致。\n保守档仅减弱外部多余像素，强档恢复到局部背景/软边缘目标。旧版只补齐和上一轮结果仍在下拉菜单。\n上排原GT/M2/范围，下排同范围只补齐/补齐清外保守/补齐清外强档。\n强档近5px检测到外部墨色残留'+str(strong['exterior_ink_candidates'])+'像素，处理后仍超过覆盖率0.02的为'+str(strong['remaining_exterior_alpha_above_point02'])+'像素；这是程序检查，不替代视觉评审。'
    text=render_html(data);assert text.count('for(let i=0;i<4;i++)')==1
    text=text.replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;').replace('与 LQ','与原GT').replace('与LQ','与原GT')
    (OUT/'index.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'CLEANUP_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(passed=True,samples=1,new_candidates=2,options=len(data['models']),page=str(OUT/'index.html'))))


if __name__=='__main__':main()
