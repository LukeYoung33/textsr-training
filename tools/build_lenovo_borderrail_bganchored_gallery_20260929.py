"""Build a full-resolution comparison viewer for background-anchored V5 results."""

from __future__ import annotations

import json
import re
from pathlib import Path


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
RAW = EVALUATION / "lenovo_borderrail_ft50k_whiteboard33_v5post_rawdelta_20260929"
ANCHORED = EVALUATION / "lenovo_borderrail_ft50k_whiteboard33_v5post_bganchored_20260929"
ARM = "gtfit_v6_borderrail_h96"


def main() -> None:
    page = (RAW / "index_raw_v5_33.html").read_text(encoding="utf-8")
    match = re.search(r"const data=(\{.*?\});const \$=", page)
    if match is None:
        raise ValueError("source viewer data not found")
    samples = json.loads(match.group(1))["samples"]
    if len(samples) != 33:
        raise ValueError("expected 33 samples")
    for sample in samples:
        if not (ANCHORED / ARM / sample / "output.png").is_file():
            raise FileNotFoundError(sample)
    choices = [
        {"key": f"../lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929/{ARM}/normal",
         "label": "normal"},
        {"key": f"../{RAW.name}/{ARM}", "label": "V5 raw / no filter"},
        {"key": ARM, "label": "V5 / background anchored"},
    ]
    payload = json.dumps({"samples": samples, "choices": choices}, ensure_ascii=False)
    page = page[:match.start()] + "const data=" + payload + ";const $=" + page[match.end():]
    page = re.sub(r"<title>.*?</title>", "<title>33 whiteboard images: V5 box blend</title>", page, count=1)
    page = re.sub(r"<b>.*?</b>", "<b>33 whiteboard images: V5 raw vs background anchored</b>", page, count=1)
    page = re.sub(r"left.value='[^']*';right.value='[^']*';",
                  f"left.value='{choices[1]['key']}';right.value='{choices[2]['key']}';", page, count=1)
    output = ANCHORED / "index_v5_box_blend_33.html"
    output.write_text(page, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
