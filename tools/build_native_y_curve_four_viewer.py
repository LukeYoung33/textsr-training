"""Publish the bounded native-Y-formula offline experiment without changing source assets."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_camera_2x2_viewer import render_html

BASE = ROOT / "output/camera280_six_block_comparison_20261026"
OUT = BASE / "postprocess_4_20261026"
IDS = ("051", "052", "099", "100")


def sha256(path):
    """@luna-comment Input:file path. Output:SHA256. Key Steps:stream file. Key Params:1MiB chunks. State/Dependencies:read-only local asset."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def build():
    """@luna-comment
    Input:frozen six-model manifest and 48 completed pilot PNGs. Output:four-row viewer and original-gallery link.
    Key Steps:verify inventory, retain source URLs, render model triples, back up and minimally link original HTML.
    Key Params:051/052/099/100 only; default VOSR V5 triple plus LQ. State/Dependencies:shared offline renderer;
    source manifest must retain its original SHA256; neither quality nor native-device performance is accepted here.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = BASE / "manifest.json"
    before = sha256(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    families = [m for m in source["models"] if m["key"] != "lq"]
    assert len(families) == 6
    rows = [copy.deepcopy(r) for r in source["rows"] if r["id"] in IDS]
    assert tuple(r["id"] for r in rows) == IDS
    models = [copy.deepcopy(next(m for m in source["models"] if m["key"] == "lq"))]
    suffixes = {"current": "当前后处理（部署公式离线复现）", "proposed": "新曲线（探索参数）"}
    for family in families:
        models.append({**family, "label": family["label"] + " · 无后处理"})
        for mode, label in suffixes.items():
            key = family["key"] + "_" + mode
            models.append({"key": key, "label": family["label"] + " · " + label,
                           "note": "PNG 派生 BT.601 全范围 Y；与手机原生 YUV 非逐字节等价。"})
            for row in rows:
                path = OUT / mode / (family["key"] + "_" + row["id"] + ".png")
                if not path.is_file():
                    raise FileNotFoundError(path)
                row["images"][key] = path.as_uri()
    data = {"title": "4张图 · 轻量亮度对比度后处理", "subtitle": "051 / 052 / 099 / 100 · 六组模型\n无后处理 / 当前后处理 / 新曲线",
            "notes": "无后处理直接引用原始融合结果。当前后处理与新曲线均为 PNG 派生 Y 的离线复现，非手机端原生 YUV 输出。\n共用16×16背景网格与两轮小网格平滑。新曲线仅对 |Y−B| 从16到48平滑增加增益0.75→0.825、软限幅24→26，不增加锐化或空间滤波。\n本轮为探索参数；是否改善由人工评审决定。PC耗时不能验证手机≤140ms。\n上方“模型三路联动”可同时切换前三窗；第四窗可自由选择。空格目标可选任意版本。",
            "defaults": "vosr_v5,vosr_v5_current,vosr_v5_proposed,lq", "models": models, "rows": rows}
    html = render_html(data)
    options = "".join('<option value="' + m["key"] + '">' + m["label"] + '</option>' for m in families)
    control = '<label>模型三路联动 <select id="familySelect" aria-label="模型三路联动">' + options + '</select></label>'
    html = html.replace('<div class="toolbar">', '<div class="toolbar">' + control, 1)
    extra = '''<script>
// @luna-comment Input:model-family selection. Output:first three synchronized model choices. Key Steps:load none/current/proposed from same family without resetting viewport; retain fourth panel and Space target. Key Params:six frozen families. State/Dependencies:existing viewer panel state and hash serializer.
function selectNativeYFamily(){const base=$('familySelect').value;[base,base+'_current',base+'_proposed'].forEach((key,i)=>{panels[i].select.value=key;loadPanel(panels[i]);});updateHash();}
$('familySelect').value=panels[0].select.value.replace(/_(current|proposed)$/,'');$('familySelect').onchange=selectNativeYFamily;
</script>'''
    assert html.count('</html>') == 1
    html = html.replace('</html>', extra + '</html>')
    html = html.replace('<div id="notes">', '<div id="notes"><a href="../index.html">返回280张六模型总览</a><br><a href="selftest_current_vs_proposed.html">当前／新曲线客观差异与评审报告</a><br>')
    (OUT / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "index.html").write_text(html, encoding="utf-8")
    index_path = BASE / "index.html"
    original = index_path.read_bytes()
    backup = OUT / "original_index_before_link.html"
    if not backup.exists():
        backup.write_bytes(original)
    text = backup.read_bytes().decode("utf-8")
    marker = 'id="native-y-four-link"'
    if marker not in text:
        anchor = '<a id="native-y-four-link" href="postprocess_4_20261026/index.html">051 / 052 / 099 / 100：无后处理／当前后处理／新曲线对照</a><br>'
        assert '<div id="notes">' in text
        index_path.write_bytes(text.replace('<div id="notes">', '<div id="notes">' + anchor, 1).encode("utf-8"))
    assert sha256(manifest_path) == before
    receipt = {"ids": IDS, "families": [m["key"] for m in families], "new_pngs": 48,
               "source_manifest_sha256": before, "source_manifest_unchanged": True,
               "original_index_backup_sha256": sha256(backup), "viewer_sha256": sha256(OUT / "index.html")}
    (OUT / "viewer_publication.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    build()
