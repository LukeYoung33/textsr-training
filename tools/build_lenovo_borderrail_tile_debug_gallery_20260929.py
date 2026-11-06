"""Create a local tile-by-tile input/output browser for whiteboard image 00007."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929/debug_00007_gtfit_v6_20260929"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    records = []
    for height in (96, 64):
        for route in ("large_tile", "overlap_tile"):
            for sample_dir in sorted((root / f"h{height}" / route).iterdir()):
                if not sample_dir.is_dir():
                    continue
                lines = []
                for line_dir in sorted((sample_dir / "tile_debug").glob("line_*")):
                    tiles = []
                    for input_file in sorted(line_dir.glob("tile_*_input.png")):
                        output_file = input_file.with_name(input_file.name.replace("_input.png", "_output.png"))
                        if not output_file.is_file():
                            raise ValueError(f"missing output: {output_file}")
                        tiles.append({"name": input_file.stem.replace("_input", ""),
                                      "input": input_file.relative_to(root).as_posix(),
                                      "output": output_file.relative_to(root).as_posix()})
                    if tiles:
                        lines.append({"name": line_dir.name, "tiles": tiles,
                                      "source": (line_dir / "source_patch.png").relative_to(root).as_posix(),
                                      "merged": (line_dir / "merged_output.png").relative_to(root).as_posix(),
                                      "metadata": (line_dir / "metadata.json").relative_to(root).as_posix()})
                records.append({"label": f"h{height} / {route} / {sample_dir.name.split('__')[-1]}",
                                "sample": sample_dir.name, "lines": lines,
                                "preview": (sample_dir / "preview.jpg").relative_to(root).as_posix(),
                                "full": (sample_dir / "output_full.jpg").relative_to(root).as_posix(),
                                "merge_report": ((sample_dir / "merge_report.json").relative_to(root).as_posix()
                                                 if (sample_dir / "merge_report.json").is_file() else None)})
    if len(records) != 8 or not all(record["lines"] for record in records):
        raise ValueError("unexpected debug case or tile count")
    payload = json.dumps(records, ensure_ascii=False)
    html = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>00007 tile 输入输出调试</title><style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#e8f1f8;font:14px system-ui}header{position:sticky;top:0;background:#1c2b3a;padding:9px;z-index:2}nav{display:flex;gap:9px;flex-wrap:wrap;align-items:center}select,button{background:#304356;color:inherit;border:1px solid #627990;border-radius:4px;padding:5px;font:inherit}main{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}figure{margin:0;background:#192431;min-width:0}figcaption{padding:6px}img{display:block;max-width:100%;margin:auto;image-rendering:pixelated}#input,#output{width:min(70vh,100%)}#source,#merged{max-height:28vh;object-fit:contain}a{color:#a9d1ff}.muted{color:#b6c5d2}@media(max-width:700px){main{grid-template-columns:1fr}}
</style></head><body><header><nav><b>00007 · gtfit_v6 逐 tile 调试</b><label>结果 <select id="case"></select></label><label>文本行 <select id="line"></select></label><button id="prev">上一 tile</button><label>Tile <select id="tile"></select></label><button id="next">下一 tile</button><span id="info" class="muted"></span><a id="preview" target="_blank">整图三联预览</a><a id="full" target="_blank">原尺寸输出</a><a id="merge" target="_blank">合并报告</a></nav></header><main>
<figure><figcaption>模型输入 tile（RGB 可视化）</figcaption><img id="input"></figure><figure><figcaption>模型输出 tile</figcaption><img id="output"></figure><figure><figcaption>该行原图区域</figcaption><img id="source"></figure><figure><figcaption>该行融合输出</figcaption><img id="merged"></figure></main><script>
const data=__DATA__,$=id=>document.getElementById(id);data.forEach((x,i)=>$('case').add(new Option(x.label,i)));function fill(select,items){select.replaceChildren(...items.map((x,i)=>new Option(x.name,i)))}function caseChange(){const r=data[Number($('case').value)];fill($('line'),r.lines);$('line').value='0';$('preview').href=r.preview;$('full').href=r.full;$('merge').href=r.merge_report||'#';$('merge').style.display=r.merge_report?'':'none';lineChange()}function lineChange(){const r=data[Number($('case').value)],l=r.lines[Number($('line').value)];fill($('tile'),l.tiles);$('tile').value='0';$('source').src=l.source;$('merged').src=l.merged;tileChange()}function tileChange(){const r=data[Number($('case').value)],l=r.lines[Number($('line').value)],t=l.tiles[Number($('tile').value)];$('input').src=t.input;$('output').src=t.output;$('info').textContent=`${l.tiles.length} tiles · ${Number($('tile').value)+1}/${l.tiles.length} · ${r.sample}`}$('case').onchange=caseChange;$('line').onchange=lineChange;$('tile').onchange=tileChange;$('prev').onclick=()=>{$('tile').selectedIndex=Math.max(0,$('tile').selectedIndex-1);tileChange()};$('next').onclick=()=>{$('tile').selectedIndex=Math.min($('tile').options.length-1,$('tile').selectedIndex+1);tileChange()};window.onkeydown=e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()};caseChange();
</script></body></html>'''
    output = root / "index_tiles.html"
    output.write_text(html.replace("__DATA__", payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
