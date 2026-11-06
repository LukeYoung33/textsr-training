"""Build a 33-image viewer comparing normal, V5 postprocess, and small-text-safe results."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evaluation"
SOURCE = ROOT / "lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929"
BASE = ROOT / "lenovo_borderrail_ft50k_whiteboard33_normal_v5post_boardframe_20260929"
OUTPUT = ROOT / "lenovo_borderrail_ft50k_whiteboard33_smalltext_outline_20260929"
ARM = "gtfit_v6_borderrail_h96"


def main() -> None:
    page = (BASE / "index_gtfit_v6_h96_33.html").read_text(encoding="utf-8")
    match = re.search(r"const data=(\{.*?\});const \$=", page)
    if match is None:
        raise ValueError("source viewer data not found")
    samples = json.loads(match.group(1))["samples"]
    if len(samples) != 33:
        raise ValueError("expected 33 images")
    choices = [
        {"key": f"../{SOURCE.name}/{ARM}/normal", "label": "normal"},
        {"key": f"../{BASE.name}/{ARM}", "label": "previous V5 post"},
        {"key": ARM, "label": "small-text outline safe"},
    ]
    for sample in samples:
        for root, choice in ((SOURCE, f"{ARM}/normal"), (BASE, ARM), (OUTPUT, ARM)):
            if not (root / choice / sample / "output_full.jpg").is_file():
                raise FileNotFoundError(f"{root}/{choice}/{sample}")
    replacement = json.dumps({"samples": samples, "choices": choices}, ensure_ascii=False)
    page = page[:match.start()] + "const data=" + replacement + ";const $=" + page[match.end():]
    page = re.sub(r"<title>.*?</title>", "<title>33 张白板图：细字轮廓对比</title>", page, count=1)
    page = re.sub(r"<b>.*?</b>", "<b>33 张白板图：细字轮廓对比</b>", page, count=1)
    page = re.sub(r"left.value=.*?;right.value=.*?;",
                  f"left.value='{choices[1]['key']}';right.value='{choices[2]['key']}';", page, count=1)
    path = OUTPUT / "index_smalltext_outline_33.html"
    path.write_text(page, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
