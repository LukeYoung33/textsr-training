"""Create a local zoomable comparison page from exported 33-image results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
HEIGHTS = (96, 64)
ROUTES = ("normal", "large_tile", "overlap_tile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=ROUTES)
    args = parser.parse_args()
    root = args.root
    choices = []
    samples = None
    for arm in ARMS:
        for height in HEIGHTS:
            name = f"{arm}_h{height}"
            summary = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
            if summary.get("completed") != 33 * len(args.routes):
                raise ValueError(f"incomplete summary: {name}")
            found = {row["sample_id"] for row in summary["rows"]}
            if len(found) != 33 or (samples is not None and samples != found):
                raise ValueError(f"sample mismatch: {name}")
            samples = found
            for route in args.routes:
                choices.append({"key": f"{name}/{route}", "label": f"{name} / {route}"})
    dataset = {"samples": sorted(samples), "choices": choices}
    for choice in choices:
        for sample in samples:
            directory = root / choice["key"] / sample
            if not (directory / "output_full.jpg").is_file() or not (directory / "preview.jpg").is_file():
                raise ValueError(f"missing visual: {directory}")
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>白板 33 张图：8 checkpoint × 3 推理路径</title><style>
*{box-sizing:border-box}body{margin:0;background:#111923;color:#eef3f7;font:14px system-ui,sans-serif}header{position:sticky;top:0;z-index:3;background:#1a2633;padding:8px 12px;border-bottom:1px solid #405165}nav{display:flex;align-items:center;gap:8px;flex-wrap:wrap}button,select{color:#eef3f7;background:#2a3b4e;border:1px solid #566b80;border-radius:5px;padding:5px;font:inherit}button{cursor:pointer}select{max-width:38vw}.muted{color:#aebecb}main{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}.pane{min-width:0;border:1px solid #405165}.title{padding:6px;background:#223040;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.viewport{position:relative;height:calc(100vh - 150px);min-height:300px;overflow:hidden;background:#0d131a;touch-action:none;cursor:grab}.viewport.drag{cursor:grabbing}.viewport img{position:absolute;max-width:none;max-height:none;pointer-events:none;user-select:none}.viewport .overlay{display:none}.holding #viewA .overlay{display:block}a{color:#9bc8ff}@media(max-width:750px){main{grid-template-columns:1fr}.viewport{height:42vh}}
</style></head><body><header><nav><b>白板 33 张：50K checkpoints / 三条推理路径</b><span class="muted" id="counter"></span></nav><nav>
<button id="prev">上一张</button><select id="sample"></select><button id="next">下一张</button>
<label>左 <select id="left"></select></label><label>右 <select id="right"></select></label>
<button id="fit">适配</button><button id="actual">1:1</button><button id="hold">按住右图覆盖左图</button>
<span class="muted">滚轮缩放 · 拖动画面 · 空格按住覆盖</span></nav></header>
<main><section class="pane"><div class="title" id="titleA"></div><div class="viewport" id="viewA"><img id="imgA"><img class="overlay" id="overlay"></div></section>
<section class="pane"><div class="title" id="titleB"></div><div class="viewport" id="viewB"><img id="imgB"></div></section></main>
<script>const data=__DATA__;const $=id=>document.getElementById(id);const sample=$('sample'),left=$('left'),right=$('right');
data.samples.forEach((x,i)=>sample.add(new Option(`${i+1}. ${x}`,x)));for(const sel of [left,right])data.choices.forEach(x=>sel.add(new Option(x.label,x.key)));
left.value=data.choices[0].key;right.value=data.choices[1].key;const panes=[{box:$('viewA'),img:$('imgA')},{box:$('viewB'),img:$('imgB')}];const overlay=$('overlay');let zoom=1,cx=.5,cy=.5;
function fit(p){return p.img.naturalWidth?Math.min(p.box.clientWidth/p.img.naturalWidth,p.box.clientHeight/p.img.naturalHeight):1}
function layout(){for(const p of [...panes,{box:$('viewA'),img:overlay}]){if(!p.img.naturalWidth)continue;const w=p.img.naturalWidth*fit(p)*zoom,h=p.img.naturalHeight*fit(p)*zoom;p.img.style.width=w+'px';p.img.style.height=h+'px';p.img.style.left=(w<=p.box.clientWidth?(p.box.clientWidth-w)/2:Math.max(p.box.clientWidth-w,Math.min(0,p.box.clientWidth/2-cx*w)))+'px';p.img.style.top=(h<=p.box.clientHeight?(p.box.clientHeight-h)/2:Math.max(p.box.clientHeight-h,Math.min(0,p.box.clientHeight/2-cy*h)))+'px'}}
function show(){const s=sample.value;[left,right].forEach((sel,i)=>{panes[i].img.src=`${sel.value}/${s}/output_full.jpg`;panes[i].img.onload=layout;$('title'+(i?'B':'A')).innerHTML=`${sel.value} · <a href="${sel.value}/${s}/preview.jpg" target="_blank">LQ / 输出 / GT 预览</a>`});overlay.src=panes[1].img.src;overlay.onload=layout;$('counter').textContent=`${sample.selectedIndex+1}/${data.samples.length}`}
for(const sel of [sample,left,right])sel.onchange=show;$('prev').onclick=()=>{sample.selectedIndex=Math.max(0,sample.selectedIndex-1);show()};$('next').onclick=()=>{sample.selectedIndex=Math.min(sample.options.length-1,sample.selectedIndex+1);show()};$('fit').onclick=()=>{zoom=1;cx=.5;cy=.5;layout()};$('actual').onclick=()=>{zoom=Math.max(1,1/fit(panes[0]));layout()};const hold=$('hold');hold.onpointerdown=e=>{hold.setPointerCapture(e.pointerId);document.body.classList.add('holding')};for(const ev of ['pointerup','pointercancel','lostpointercapture'])hold.addEventListener(ev,()=>document.body.classList.remove('holding'));
for(const p of panes){let drag=null;p.box.onwheel=e=>{e.preventDefault();zoom=Math.max(1,Math.min(50,zoom*(e.deltaY<0?1.2:1/1.2)));layout()};p.box.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};p.box.setPointerCapture(e.pointerId);p.box.classList.add('drag')};p.box.onpointermove=e=>{if(!drag)return;cx=Math.max(0,Math.min(1,cx-(e.clientX-drag.x)/(p.img.naturalWidth*fit(p)*zoom)));cy=Math.max(0,Math.min(1,cy-(e.clientY-drag.y)/(p.img.naturalHeight*fit(p)*zoom)));drag={x:e.clientX,y:e.clientY};layout()};p.box.onpointerup=p.box.onpointercancel=()=>{drag=null;p.box.classList.remove('drag')}}
window.onkeydown=e=>{if(e.code==='Space'){e.preventDefault();document.body.classList.add('holding')}if(e.code==='ArrowLeft')$('prev').click();if(e.code==='ArrowRight')$('next').click()};window.onkeyup=e=>{if(e.code==='Space')document.body.classList.remove('holding')};window.onresize=layout;show();</script></body></html>'''
    (root / "index.html").write_text(page.replace("__DATA__", json.dumps(dataset, ensure_ascii=False)), encoding="utf-8")
    print(root / "index.html")


if __name__ == "__main__":
    main()
