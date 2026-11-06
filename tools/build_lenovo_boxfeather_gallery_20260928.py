"""Build a local gallery comparing OCR-mask and expanded-box validation pasteback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath


ARMS = ("scurve_h96", "scurve_h64", "raw_h96", "raw_h64")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()
    new_root, old_root = args.new_root.resolve(), args.old_root.resolve()
    gallery_root = new_root.parent
    records: dict[tuple[str, str], dict] = {}

    def relative(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        return os.path.relpath(path, gallery_root).replace("\\", "/")

    for arm in ARMS:
        old_summary = json.loads((old_root / arm / "summary.json").read_text(encoding="utf-8"))
        old_rows = {(row["dataset"], row["sample_id"]): row for row in old_summary["rows"]}
        new_summary = json.loads((new_root / arm / "summary.json").read_text(encoding="utf-8"))
        if new_summary["passed"] != 231 or new_summary["failed"]:
            raise ValueError(f"incomplete new arm: {arm}")
        for row in new_summary["rows"]:
            key = row["dataset"], row["sample_id"]
            record = records.setdefault(key, {"dataset": key[0], "sample": key[1], "arms": {}})
            new_dir = new_root / arm / Path(*PurePosixPath(row["output"]).parts[-4:-1])
            old_row = old_rows[key]
            old_dir = old_root / arm / Path(*PurePosixPath(old_row["output"]).parts[-4:-1])
            record["arms"][arm] = {
                "new": relative(new_dir / "output.png"),
                "old": relative(old_dir / "output.png"),
                "lines_new": row["lines_processed"], "lines_old": old_row["lines_processed"],
            }
            if arm == "scurve_h96":
                record["lq"] = relative(new_dir / "lq_original.png")
                record["gt"] = relative(new_dir / "gt.png")
                record["lq_used"] = relative(new_dir / "lq_used.png")
    if len(records) != 231 or any(len(record["arms"]) != 4 for record in records.values()):
        raise ValueError("sample or arm mismatch")
    data = json.dumps([records[key] for key in sorted(records)], ensure_ascii=False).replace("</", "<\\/")
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Lenovo OCR box 回贴对照</title>
<style>body{margin:0;background:#121b24;color:#e7eef6;font:14px Arial,sans-serif}header{position:sticky;top:0;background:#1d2b38;padding:10px;z-index:2;border-bottom:1px solid #536575}select{background:#263a4c;color:white;padding:5px}main{padding:12px}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.panel{background:#1c2a36;padding:7px;overflow:auto}.panel img{display:block;width:100%;height:auto}.panel a{color:#a2dfff}#zoom{width:160px}@media(max-width:800px){.row{grid-template-columns:1fr}}</style>
<header>样本 <select id="sample"></select>　模型 <select id="arm"></select>　放大 <input id="zoom" type="range" min="100" max="400" value="100"><span id="zoomlabel">100%</span>　<span id="meta"></span><br>左：旧版笔画 mask 回贴；右：新版外扩 box 内完整回贴、向单行裁图边界羽化。点击图像标题可打开原尺寸。</header>
<main><div class="row" id="images"></div><details><summary>原始 LQ / S-curve 输入 / GT</summary><div class="row" id="references"></div></details></main>
<script>const DATA=__DATA__;const samples=document.getElementById('sample'),arms=document.getElementById('arm'),zoom=document.getElementById('zoom');DATA.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+'/231 '+r.dataset+' / '+r.sample;samples.append(o)});['scurve_h96','scurve_h64','raw_h96','raw_h64'].forEach(a=>{let o=document.createElement('option');o.value=a;o.textContent=a;arms.append(o)});function panel(name,path){return '<div class="panel"><a target="_blank" href="'+path+'">'+name+'（原尺寸）</a><img src="'+path+'"></div>'}function render(){let r=DATA[+samples.value],a=r.arms[arms.value];document.getElementById('meta').textContent='旧版 '+a.lines_old+' 行 / 新版 '+a.lines_new+' 行';document.getElementById('images').innerHTML=panel('旧版',a.old)+panel('新版',a.new);document.getElementById('references').innerHTML=panel('原始 LQ',r.lq)+panel('S-curve 输入',r.lq_used)+panel('GT',r.gt);setZoom()}function setZoom(){let v=zoom.value;document.getElementById('zoomlabel').textContent=v+'%';document.querySelectorAll('.panel img').forEach(x=>x.style.width=v+'%')}samples.onchange=arms.onchange=render;zoom.oninput=setZoom;render();</script></html>'''
    (gallery_root / "index.html").write_text(html.replace("__DATA__", data), encoding="utf-8")
    print(gallery_root / "index.html")


if __name__ == "__main__":
    main()
