"""Build the 33-image normal versus full-box V5 postprocess viewer."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evaluation"
SOURCE = ROOT / "lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929"
OUTPUT = ROOT / "lenovo_borderrail_ft50k_whiteboard33_normal_v5post_rulefix_20260929"
ARM = "gtfit_v6_borderrail_h96"


def main() -> None:
    page = (SOURCE / "index.html").read_text(encoding="utf-8")
    match = re.search(r"const data=(\{.*?\});const \$=", page)
    if match is None:
        raise ValueError("source viewer data not found")
    samples = json.loads(match.group(1))["samples"]
    if len(samples) != 33 or len(set(samples)) != 33:
        raise ValueError("expected 33 distinct samples")
    for sample in samples:
        before = SOURCE / ARM / "normal" / sample / "output_full.jpg"
        after = OUTPUT / ARM / sample / "output_full.jpg"
        if not before.is_file() or not after.is_file():
            raise FileNotFoundError(f"missing normal or V5 post image for {sample}")
    choices = [
        {"key": f"../{SOURCE.name}/{ARM}/normal", "label": "normal"},
        {"key": ARM, "label": "V5 post / board frame and horizontal rule corrected"},
    ]
    data = json.dumps({"samples": samples, "choices": choices}, ensure_ascii=False)
    page = page[:match.start()] + "const data=" + data + ";const $=" + page[match.end():]
    page = re.sub(r"<title>.*?</title>", "<title>33 张白板图：normal 与 V5 后处理</title>", page, count=1)
    page = re.sub(r"<b>.*?</b>", "<b>33 张白板图：gtfit_v6_borderrail_h96 / normal 与 V5 后处理</b>", page, count=1)
    page = page.replace("left.value='v5_borderrail_h96/normal';right.value='v5_borderrail_h96/overlap_tile';",
                        f"left.value='{choices[0]['key']}';right.value='{choices[1]['key']}';")
    path = OUTPUT / "index_gtfit_v6_h96_33.html"
    path.write_text(page, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
