"""Publish two native five-step interpolation ladders for the fixed32 subset."""
from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('Y:/guang.yang/TextSR/data/mojito32_s22_thin_blend_20261104')
OUT = ROOT/'evaluation/mojito32_s22_thin_blend_20261104'


def main():
    """@luna-comment Input:completed32blend receipts. Output:10panel native viewer/manifest. Key Steps:bindexactnativepaths;reuse sharedrenderer with2x5layout;annotatefractions/orientation. Key Params:twoGTarms,0/25/50/75/100percentGT. State/Dependencies:Yimages,ownedlocalHTML;no imageconversion."""
    result = json.loads((SOURCE/'RESULTS.json').read_text(encoding='utf-8'))
    assert result['passed'] and len(result['rows']) == 32
    title = 'Mojito32 · V3输入 → 缩窄GT · 中间形态'
    subtitle = '上排：−1px GT；下排：−2px GT\n每排：V3输入 → 25% → 50% → 75% → GT'
    notes = 'GT占比α：中间图=(1−α)×V3输入+α×GT。\n直接在现有RGB像素值域插值，浮点计算后一次取整；无额外S-curve、锐化或金字塔处理。\n−1/−2px为笔画总宽度的名义缩窄量。两条路线各保留25%、50%、75%中间形态。\n8张方向6的输入顺时针旋转90°，旋转后与GT生成时输入逐像素相同；其他24张直接一致。\n这是可控的中间目标序列，不等同于真实相机PSF。\n所有窗口同步缩放，按空格查看V3输入，100%查看原生像素。图片从Y盘加载。'
    models = [dict(key='lq', label='V3预处理输入 · GT 0%', note='融合起点；原生坐标')]
    defaults = []
    for arm, width in [('minus1',1), ('minus2',2)]:
        defaults.append('lq')
        for percent in (25,50,75):
            key = arm+f'_a{percent}'
            models.append(dict(key=key,label=f'−{width}px路线 · GT {percent}%', note=f'V3输入 {100-percent}% + 缩窄GT {percent}%'))
            defaults.append(key)
        models.append(dict(key=arm,label=f'−{width}px GT · GT 100%',note=f'VOSR + S2.2后笔画约缩窄{width}px；终点'))
        defaults.append(arm)
    models.append(dict(key='original_gt',label='VOSR + S2.2 原版GT',note='未缩窄参考；不属于上述融合序列'))
    rows=[]
    for row in result['rows']:
        paths={k:(SOURCE/v['path']).as_uri() for k,v in row['outputs'].items()}
        rows.append(dict(id=row['id'],name=row['name'],width=row['width'],height=row['height'],lq=paths.pop('v3_input'),images=paths))
    data=dict(title=title,subtitle=subtitle,notes=notes,models=models,defaults=','.join(defaults),rows=rows)
    text=(ROOT/'evaluation/near_val056_train558_gt_comparison_20261028/index.html').read_text(encoding='utf-8')
    text=re.sub(r'const DATA=.*?;</script>',lambda _:'const DATA='+json.dumps(data,ensure_ascii=False)+';</script>',text,count=1,flags=re.S)
    text=re.sub(r'<title>.*?</title>','<title>'+title+'</title>',text,count=1)
    text=re.sub(r'<aside><h1>.*?</small>','<aside><h1>'+title+'</h1><small>'+subtitle.replace('\n','<br>')+'</small>',text,count=1,flags=re.S)
    text=re.sub(r'<div id="notes">.*?</div>','<div id="notes">'+html.escape(notes).replace('\n','<br>')+'</div>',text,count=1,flags=re.S)
    text=text.replace('grid-template-columns:1fr 1fr 1fr','grid-template-columns:repeat(5,minmax(0,1fr))').replace('i<6;i++','i<10;i++')
    text=text.replace('与原始 GT 滑动对比','与V3输入滑动对比').replace('LQ  |  模型结果','V3输入 | 中间形态')
    text=text.replace('href="PROVENANCE.json"','href="RESULTS.json"')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'index.html').write_text(text,encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'index.html')


if __name__=='__main__':
    main()
