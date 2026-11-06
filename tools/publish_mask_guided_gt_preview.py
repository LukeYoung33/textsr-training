"""Publish actual native GT-repair candidates with local detail crops andfull links."""
from pathlib import Path
from urllib.parse import unquote
import json,sys,shutil
import numpy as np
from PIL import Image
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_camera_2x2_viewer import render_html,file_url,digest
SOURCE=Path('Y:/guang.yang/TextSR/data/mask_guided_vosr_s22_preview8_20261104')
OUT=ROOT/'evaluation/mask_guided_vosr_s22_design_20261104'
MODELS=[dict(key='lq',label='原 VOSR＋S2.2 GT',note='原始GT，仅作为修复基准'),
    dict(key='mask_original',label='原始 TextMask M0',note='笔画结构与字腔参考'),dict(key='mask_plus2',label='+2px TextMask M2',note='操作范围上限，不作为目标笔画宽度'),
    dict(key='fill_conservative',label='仅补齐 · 保守',note='局部VOSR取色；填缺损'),dict(key='fill_strong',label='仅补齐 · 强诊断',note='更大覆盖率补齐；结构保护不放宽'),
    dict(key='edge_conservative',label='补齐＋边缘修整 · 保守',note='独立窄边缘带小幅双向修整'),dict(key='edge_strong',label='补齐＋边缘修整 · 强诊断',note='强补齐＋边缘；细笔画禁止削薄'),
    dict(key='fill_strong_diff',label='仅补齐强档 · 差分×8',note='红色=变暗，蓝色=变亮'),dict(key='edge_strong_diff',label='补齐＋边缘强档 · 差分×8',note='红色=变暗，蓝色=变亮'),
    dict(key='eligible',label='允许修复区域',note='通过局部参考检查，不等于实际修改位置'),dict(key='changed',label='实际修改位置（强边缘档）',note='白色为原图与强边缘候选有差异的像素')]


def mapped(path):
    """@luna-comment Input:ownedremoteNASpath. Output:WindowsYpath. Key Steps:exactprefixtranslation. Key Params:nootherroots. State/Dependencies:mappedshare."""
    assert str(path).startswith('/AI_DNS/')
    return Path('Y:/'+str(path)[len('/AI_DNS/'):])


def render(data,path):
    """@luna-comment Input:8rowmanifest/targetHTML. Output:accepted6panelnativeviewer. Key Steps:existingrenderer,panelcount-onlyadaptation,GTtoolbarwording. Key Params:noresizeinthefullviewer. State/Dependencies:sharedrendererunchanged."""
    text=render_html(data);assert text.count('for(let i=0;i<4;i++)')==1
    text=text.replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;').replace('与 LQ','与原GT').replace('与LQ','与原GT')
    path.write_text(text,encoding='utf-8')


def main():
    """@luna-comment Input:completed32realcandidatePNGs/8auditedsources. Output:actual8samplecropviewer/fullviewer/pathaudit/contactsheet. Key Steps:checkcandidatehashes/nativegeometry/nonidentity/localityreceipts,cropmax-changearea,renderallbranches. Key Params:384pxnativecrops,originalsourcesimmutable. State/Dependencies:newlocalcrops/NASresults only."""
    complete=json.loads((SOURCE/'COMPLETE.json').read_text());assert complete['passed'] and complete['candidates']==32
    audit=json.loads((SOURCE/'INPUT_AUDIT.json').read_text());records=[];rows=[];full_rows=[]
    crop_root=OUT/'crops';crop_root.mkdir(exist_ok=True)
    for job in audit['jobs']:
        identity=job['id'];folder=SOURCE/'images'/identity;receipt=json.loads((folder/'receipt.json').read_text())
        original=np.asarray(Image.open(folder/'original.png').convert('RGB'));strong=np.asarray(Image.open(folder/'edge_strong.png').convert('RGB'));h,w=original.shape[:2]
        score=np.abs(strong.astype(np.float32)-original).sum(2);smoothed=cv2.boxFilter(score,-1,(192,192),normalize=False)
        y,x=np.unravel_index(np.argmax(smoothed),score.shape);size=min(384,w,h);x0=int(np.clip(x-size//2,0,w-size));y0=int(np.clip(y-size//2,0,h-size));box=(x0,y0,x0+size,y0+size)
        changed=np.any(strong!=original,2);Image.fromarray(np.repeat(changed[...,None].astype(np.uint8)*255,3,2)).save(crop_root/(identity+'_changed_full.png'))
        images={};full_images={}
        for m in MODELS:
            key=m['key'];name='original' if key=='lq' else key
            path=crop_root/(identity+'_changed_full.png') if key=='changed' else folder/(name+'.png')
            with Image.open(path) as im:
                assert im.size==(w,h)
                target=crop_root/(identity+'_'+key+'.png');im.crop(box).save(target)
            images[key]=target.as_uri();full_images[key]=path.absolute().as_uri()
        for variant,r in receipt['variants'].items():
            assert digest(mapped(r['path']))==r['sha256'] and r['protected_changed_pixels']==r['outside_m2_changed_pixels']==0
        assert sum(r['changed_pixels'] for r in receipt['variants'].values())>0
        name=job['dataset']+' / '+job['sample_id']
        row=dict(id=identity,name=name+'【训练GT样例，非Mojito编号】',width=size,height=size,lq=images['lq'],images={k:v for k,v in images.items() if k!='lq'})
        rows.append(row);full_rows.append(dict(row,width=w,height=h,lq=full_images['lq'],images={k:v for k,v in full_images.items() if k!='lq'}));records.append(dict(id=identity,crop_xyxy=box,receipt=receipt))
    notes='这是8张训练GT小样，不是Mojito32。首屏是每张修改最集中的384px原生裁切，所有候选同位置。\n原VOSR+S2.2、M0、M2、4个修复候选、差分和修复位置均可从下拉选择。\n修复颜色仅取自原VOSR图；正常笔画锚点、受保护字腔、有效区外及M2外像素保持不变。\n首版是组件局部健康截面原型，不足以代表全量效果。缺乏可靠截面或取色的组件明确跳过；不自动补整条不可见笔画。\n红色差分=变暗，蓝色=变亮，放大8倍。效果请看图判断。\n完整原生大图入口：同目录 full.html；统计与跳过记录：AUDIT.json。'
    manifest=dict(title='VOSR＋S2.2 · Mask引导笔画修复 · 8张小样',subtitle='真实候选图 · 原图不覆盖 · 4种修复分支',notes=notes,
        defaults='lq,mask_original,mask_plus2,fill_conservative,fill_strong,edge_strong',models=MODELS,rows=rows)
    if (OUT/'index.html').exists() and not (OUT/'design.html').exists():shutil.copy2(OUT/'index.html',OUT/'design.html')
    render(manifest,OUT/'index.html');(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    full=dict(manifest,title='VOSR＋S2.2 · 笔画修复8张 · 完整原生大图',rows=full_rows);render(full,OUT/'full.html');(OUT/'full_manifest.json').write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'AUDIT.json').write_text(json.dumps(dict(passed=True,images=8,candidates=32,records=records,source_gt_replaced=False),indent=2))
    thumbs=[]
    for key in ['lq','fill_strong','edge_strong']:
        im=Image.open(crop_root/('028_'+key+'.png'));thumbs.append(im)
    contact=Image.new('RGB',(sum(im.width for im in thumbs),thumbs[0].height))
    x=0
    for im in thumbs:contact.paste(im,(x,0));x+=im.width
    contact.save(OUT/'sample028_contact.png')
    print(json.dumps(dict(passed=True,images=8,candidates=32,page=str(OUT/'index.html'))))


if __name__=='__main__':main()
