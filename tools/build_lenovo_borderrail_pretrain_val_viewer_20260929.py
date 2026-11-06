"""Build a pairwise full-frame viewer from eight pretrain validation runs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_pretrain250k_val_fullbox_complete_20260930")
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")


def main() -> None:
    paths = {}
    for arm in ARMS:
        for height in (96, 64):
            name = f"{arm}_h{height}"
            summary = json.loads((ROOT / name / "summary.json").read_text(encoding="utf-8"))
            if summary["completed"] != summary["expected"]:
                raise ValueError(f"incomplete: {name}")
            for row in summary["rows"]:
                key = f"{row['dataset']}/{row['sample_id']}"
                paths.setdefault(key, {})[name] = str(Path(row["output"]).relative_to(ROOT))
                paths[key].setdefault(f"LQ_{arm}", str((Path(row["output"]).parent / "lq_used.png").relative_to(ROOT)))
                paths[key].setdefault("GT_source", str((Path(row["output"]).parent / "gt.png").relative_to(ROOT)))
    for key, choices in paths.items():
        if len(choices) != 13 or any(not (ROOT / path).is_file() for path in choices.values()):
            raise ValueError(f"missing visual: {key}")
    data = json.dumps(paths, ensure_ascii=False)
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Borderrail 250K 验证集</title>
<style>*{box-sizing:border-box}body{margin:0;background:#111923;color:#eef3f7;font:14px system-ui}header{padding:8px;position:sticky;top:0;background:#1c2b3a;z-index:2}nav{display:flex;flex-wrap:wrap;gap:8px;margin:5px}button,select{background:#2b3d50;color:white;border:1px solid #52687b;border-radius:4px;padding:5px}main{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}.pane{min-width:0}.view{height:calc(100vh - 155px);position:relative;overflow:hidden;background:#0b1119;touch-action:none}.view img{position:absolute;max-width:none;max-height:none;pointer-events:none}.overlay{display:none}.holding .overlay{display:block}@media(max-width:700px){main{grid-template-columns:1fr}.view{height:40vh}}</style>
<header><nav><b>八组 250K 验证集大图对照</b><button id="prev">上一张</button><select id="sample"></select><button id="next">下一张</button><span id="counter"></span></nav><nav>左<select id="left"></select>右<select id="right"></select><button id="fit">适配</button><button id="actual">1:1</button><button id="hold">按住右图覆盖左图</button><span>滚轮缩放 · 拖动画面 · 空格覆盖</span></nav></header>
<main><div class="pane"><div id="titleA"></div><div class="view" id="viewA"><img id="imgA"><img class="overlay" id="over"></div></div><div class="pane"><div id="titleB"></div><div class="view" id="viewB"><img id="imgB"></div></div></main>
<script>const data=__DATA__,$=id=>document.getElementById(id),sample=$('sample'),left=$('left'),right=$('right');let zoom=1,cx=.5,cy=.5;
Object.keys(data).forEach((s,i)=>sample.add(new Option((i+1)+'. '+s,s)));Object.keys(data[sample.value]).forEach(c=>{left.add(new Option(c,c));right.add(new Option(c,c))});left.value='LQ_v5_borderrail';right.value='v5_borderrail_h96';
const panes=[{box:$('viewA'),img:$('imgA')},{box:$('viewB'),img:$('imgB')}],over=$('over');function fit(p){return p.img.naturalWidth?Math.min(p.box.clientWidth/p.img.naturalWidth,p.box.clientHeight/p.img.naturalHeight):1}
function layout(){for(const p of [...panes,{box:$('viewA'),img:over}]){if(!p.img.naturalWidth)continue;let w=p.img.naturalWidth*fit(p)*zoom,h=p.img.naturalHeight*fit(p)*zoom;p.img.style.width=w+'px';p.img.style.height=h+'px';p.img.style.left=(w<=p.box.clientWidth?(p.box.clientWidth-w)/2:Math.max(p.box.clientWidth-w,Math.min(0,p.box.clientWidth/2-cx*w)))+'px';p.img.style.top=(h<=p.box.clientHeight?(p.box.clientHeight-h)/2:Math.max(p.box.clientHeight-h,Math.min(0,p.box.clientHeight/2-cy*h)))+'px'}}
function show(){for(const [i,c] of [left.value,right.value].entries()){let p=panes[i];p.img.src=data[sample.value][c];p.img.onload=layout;$('title'+(i?'B':'A')).textContent=c}over.src=panes[1].img.src;over.onload=layout;$('counter').textContent=(sample.selectedIndex+1)+'/'+sample.options.length}
for(const x of [sample,left,right])x.onchange=show;$('prev').onclick=()=>{sample.selectedIndex=Math.max(0,sample.selectedIndex-1);show()};$('next').onclick=()=>{sample.selectedIndex=Math.min(sample.options.length-1,sample.selectedIndex+1);show()};$('fit').onclick=()=>{zoom=1;cx=.5;cy=.5;layout()};$('actual').onclick=()=>{zoom=Math.max(1,1/fit(panes[0]));layout()};
const hold=$('hold');hold.onpointerdown=e=>{hold.setPointerCapture(e.pointerId);document.body.classList.add('holding')};for(const ev of ['pointerup','pointercancel','lostpointercapture'])hold.addEventListener(ev,()=>document.body.classList.remove('holding'));
for(const p of panes){let drag=null;p.box.onwheel=e=>{e.preventDefault();zoom=Math.max(1,Math.min(40,zoom*(e.deltaY<0?1.2:1/1.2)));layout()};p.box.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};p.box.setPointerCapture(e.pointerId)};p.box.onpointermove=e=>{if(!drag)return;cx=Math.max(0,Math.min(1,cx-(e.clientX-drag.x)/(p.img.naturalWidth*fit(p)*zoom)));cy=Math.max(0,Math.min(1,cy-(e.clientY-drag.y)/(p.img.naturalHeight*fit(p)*zoom)));drag={x:e.clientX,y:e.clientY};layout()};p.box.onpointerup=p.box.onpointercancel=()=>drag=null}
window.onkeydown=e=>{if(e.code==='Space'){e.preventDefault();document.body.classList.add('holding')}if(e.code==='ArrowLeft')$('prev').click();if(e.code==='ArrowRight')$('next').click()};window.onkeyup=e=>{if(e.code==='Space')document.body.classList.remove('holding')};window.onresize=layout;show();</script></html>'''
    target = ROOT / "index.html"
    target.write_text(html.replace("__DATA__", data), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
