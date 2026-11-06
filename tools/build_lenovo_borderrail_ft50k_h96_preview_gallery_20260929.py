"""Build an immediate local gallery for the completed h96 preview images."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929"
ARMS = ("v5_borderrail_h96", "v6_borderrail_h96", "gtfit_v5_borderrail_h96", "gtfit_v6_borderrail_h96")
ROUTES = ("normal", "large_tile", "overlap_tile")


def main() -> None:
    names = sorted(directory.name for directory in (ROOT / ARMS[0] / ROUTES[0]).iterdir() if directory.is_dir())
    if len(names) != 33:
        raise ValueError(f"expected 33 images, got {len(names)}")
    choices = [f"{arm}/{route}" for arm in ARMS for route in ROUTES]
    for choice in choices:
        if any(not (ROOT / choice / name / "preview.jpg").is_file() for name in names):
            raise ValueError(f"missing preview in {choice}")
    data = json.dumps({"samples": names, "choices": choices}, ensure_ascii=False)
    html = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>白板33张：96高四组预览</title><style>
body{margin:0;background:#101820;color:#eef;font:14px system-ui}header{position:sticky;top:0;background:#1d2a37;padding:9px;z-index:2}nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap}select,button{background:#304156;color:#eef;border:1px solid #668;border-radius:4px;padding:5px;font:inherit}main{display:grid;grid-template-columns:1fr 1fr;gap:7px;padding:7px}figure{margin:0;min-width:0}figcaption{padding:7px}img{display:block;width:100%;height:calc(100vh - 125px);object-fit:contain;background:#17212b}a{color:#bde}@media(max-width:700px){main{grid-template-columns:1fr}img{height:45vh}}
</style></head><body><header><nav><b>96 高：四组模型 × 三条路径</b><button id="prev">上一张</button><select id="sample"></select><button id="next">下一张</button><label>左 <select id="left"></select></label><label>右 <select id="right"></select></label><span id="count"></span></nav></header><main><figure><figcaption id="captionA"></figcaption><a id="linkA" target="_blank"><img id="imageA"></a></figure><figure><figcaption id="captionB"></figcaption><a id="linkB" target="_blank"><img id="imageB"></a></figure></main><script>
const data=__DATA__;const $=id=>document.getElementById(id);for(const [i,s] of data.samples.entries())$('sample').add(new Option(`${i+1}. ${s}`,s));for(const id of ['left','right'])for(const v of data.choices)$(id).add(new Option(v,v));$('left').value=data.choices[0];$('right').value=data.choices[2];function show(){for(const [sel,img,cap,link] of [['left','imageA','captionA','linkA'],['right','imageB','captionB','linkB']]){const path=`${$(sel).value}/${$('sample').value}/preview.jpg`;$(img).src=path;$(link).href=path;$(cap).textContent=$(sel).value}$('count').textContent=`${$('sample').selectedIndex+1}/33`};for(const id of ['sample','left','right'])$(id).onchange=show;$('prev').onclick=()=>{$('sample').selectedIndex=Math.max(0,$('sample').selectedIndex-1);show()};$('next').onclick=()=>{$('sample').selectedIndex=Math.min(32,$('sample').selectedIndex+1);show()};window.onkeydown=e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()};show();</script></body></html>'''
    (ROOT / "index_previews_h96.html").write_text(html.replace("__DATA__", data), encoding="utf-8")
    print(ROOT / "index_previews_h96.html")


if __name__ == "__main__":
    main()
