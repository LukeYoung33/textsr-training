"""Original/old/new native crop comparison for user-marked training001 repairs."""
from pathlib import Path
import json,sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_camera_2x2_viewer import render_html,digest
SOURCE=Path('Y:/guang.yang/TextSR/data/mask_gt_001_m2_direct_20261104')
OLD=Path('Y:/guang.yang/TextSR/data/mask_guided_vosr_s22_preview8_20261104/images/001')
OUT=ROOT/'evaluation/mask_gt_001_m2_direct_20261104'


def main():
    """@luna-comment Input:twoaudited001directM2variantsandoldsource. Output:one-sample6panelnativecropviewer/audit. Key Steps:verifynewhashes/frozenchecks,exactsamecropforallbranches,explicitlocalmetrics. Key Params:384px,userapprovedzonesonly;old8resultsretained. State/Dependencies:acceptedrenderer/NASassets."""
    assert json.loads((SOURCE/'COMPLETE.json').read_text())['passed']
    audit=json.loads((SOURCE/'AUDIT.json').read_text());box=tuple(audit['crop_xyxy']);OUT.mkdir(exist_ok=True)
    for v in audit['variants'].values():
        p=SOURCE/Path(v['path']).name;assert digest(p)==v['sha256'] and v['outside_roi_changed_pixels']==v['background_changed_pixels']==v['healthy_changed_pixels']==0
    paths={'lq':OLD/'original.png','old_strong':OLD/'edge_strong.png','mask_original':OLD/'mask_original.png','mask_plus2':OLD/'mask_plus2.png',
        'direct_conservative':SOURCE/'direct_conservative.png','direct_strong':SOURCE/'direct_strong.png','contour':SOURCE/'mask_contour.png',
        'diff':SOURCE/'direct_strong_diff.png','eligible':SOURCE/'fill_eligible.png','unresolved':SOURCE/'unresolved.png'}
    labels={'lq':'原 VOSR＋S2.2 GT','old_strong':'上一版 · 补齐＋边缘强档','mask_original':'原始 TextMask M0（参考）','mask_plus2':'+2px TextMask M2（本次几何目标）',
        'direct_conservative':'新：按 M2 补齐 · 保守','direct_strong':'新：按 M2 补齐 · 强诊断','contour':'M2 轮廓＋本次操作范围','diff':'新强档差分 ×4（红暗／蓝亮）','eligible':'实际允许填充区域','unresolved':'缺少取色参考的 M2 内部位置'}
    urls={}
    for key,p in paths.items():
        with Image.open(p) as im:crop=im.crop(box)
        target=OUT/(key+'.png');crop.save(target);urls[key]=target.as_uri()
    models=[dict(key=k,label=labels[k],note='同一001原生裁切；仅用户指出的英文、岁、在区域') for k in paths]
    notes='001是HXA/hx_1013_A_000006训练GT样例，不是Mojito001。\n本版针对用户圈定的英文、岁、在，直接采用M2明确笔画几何；M0和健康截面不再阻止这些位置的补齐。\n强档在M2内部向同一笔画的VOSR局部墨色F收敛，边缘只应用一次软覆盖；不直接设成纯白，不使用原始GT像素。\n其余字、操作范围外、M2背景和取色锚点保持原像素。允许M2明确表示的笔画连接，不做全图增粗。\n上排原图/M2/轮廓，下排旧强档/新保守/新强档。差分和未填位置可下拉选择。\n覆盖率达标是程序指标，不代表视觉质量；请重点看笔画是否均匀、是否过粗和墨色是否匹配。'
    for name,m in audit['variants']['direct_strong']['per_zone'].items():
        notes+='\n强档 '+name+'：核心欠覆盖'+str(m['core_deficit_pixels'])+'像素，剩余未达目标'+str(m['remaining_core_deficit_pixels'])+'像素，无取色参考'+str(m['unresolved_no_donor_pixels'])+'像素。'
    data=dict(title='001 · 用户圈定缺笔 · M2直接补齐',subtitle='原VOSR亮度取色 · 原图／旧版／新两档',notes=notes,
        models=models,defaults='lq,mask_plus2,contour,old_strong,direct_conservative,direct_strong',
        rows=[dict(id='001',name='HXA / hx_1013_A_000006【用户圈定区域】',width=box[2]-box[0],height=box[3]-box[1],lq=urls['lq'],images={k:v for k,v in urls.items() if k!='lq'})])
    text=render_html(data);assert text.count('for(let i=0;i<4;i++)')==1
    text=text.replace('for(let i=0;i<4;i++)','for(let i=0;i<6;i++)').replace('grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;','grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr;').replace('与 LQ','与原GT').replace('与LQ','与原GT')
    (OUT/'index.html').write_text(text,encoding='utf-8');(OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(passed=True,samples=1,new_candidates=2,page=str(OUT/'index.html'))))


if __name__=='__main__':main()
