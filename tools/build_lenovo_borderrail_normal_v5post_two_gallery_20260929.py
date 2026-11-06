"""Build a zoomable before/after browser for the two V5 postprocessed images."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evaluation"
SOURCE = ROOT / "lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929"
OUTPUT = ROOT / "lenovo_borderrail_ft50k_whiteboard33_normal_v5post_fullbox_20260929"
ARMS = tuple(f"{arm}_h{height}" for arm in
             ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
             for height in (96, 64))
SAMPLES = ("lenovo_ppt_IMG_20260610_052710373",
           "mojito_4160_00008_ss_IMG_20260611_061919404__textsr_mojito_tile512_u1")


def main() -> None:
    choices = []
    for arm in ARMS:
        before = f"../{SOURCE.name}/{arm}/normal"
        after = arm
        choices.extend(({"key": before, "label": f"{arm} / normal"},
                        {"key": after, "label": f"{arm} / V5 post full box"}))
        for sample in SAMPLES:
            if not (OUTPUT / before / sample / "output_full.jpg").is_file():
                raise ValueError(f"missing normal: {arm}/{sample}")
            if not (OUTPUT / after / sample / "output_full.jpg").is_file():
                raise ValueError(f"missing V5 post: {arm}/{sample}")
    page = (SOURCE / "index.html").read_text(encoding="utf-8")
    data = json.dumps({"samples": SAMPLES, "choices": choices}, ensure_ascii=False)
    page, count = re.subn(r"const data=\{.*?\};const \$=", "const data=" + data + ";const $=", page, count=1)
    if count != 1:
        raise ValueError("source gallery script structure changed")
    page = page.replace("白板 33 张图：8 checkpoint × 3 推理路径", "两张白板图：normal 与 V5 后处理")
    page = page.replace("白板 33 张：50K checkpoints / 三条推理路径", "两张白板图：8 组 checkpoint / normal 与 V5 后处理")
    page = page.replace("left.value='v5_borderrail_h96/normal';right.value='v5_borderrail_h96/overlap_tile';",
                        f"left.value='{choices[0]['key']}';right.value='{choices[1]['key']}';")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
