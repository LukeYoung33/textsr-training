"""Save readable original-resolution crops of the two requested V5-post samples."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "evaluation"
BEFORE = ROOT / "lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929"
AFTER = ROOT / "lenovo_borderrail_ft50k_whiteboard33_normal_v5post_fullbox_20260929"
ARM = "gtfit_v6_borderrail_h96"
SAMPLES = ("lenovo_ppt_IMG_20260610_052710373",
           "mojito_4160_00008_ss_IMG_20260611_061919404__textsr_mojito_tile512_u1")


def main() -> None:
    for sample in SAMPLES:
        before = cv2.imread(str(BEFORE / ARM / "normal" / sample / "output_full.jpg"))
        after = cv2.imread(str(AFTER / ARM / sample / "output.png"))
        if before is None or after is None or before.shape != after.shape:
            raise ValueError(f"missing or mismatched image: {sample}")
        height, width = after.shape[:2]
        x0, x1 = int(width * .08), int(width * .48)
        y0, y1 = int(height * .07), int(height * .23)
        panels = [before[y0:y1, x0:x1], after[y0:y1, x0:x1]]
        label = np.full((35, 2 * (x1 - x0), 3), 245, np.uint8)
        cv2.putText(label, "NORMAL", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .75, (25, 25, 25), 2)
        cv2.putText(label, "V5 POST", (x1 - x0 + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, .75, (25, 25, 25), 2)
        output = AFTER / f"focus_{sample}.jpg"
        cv2.imwrite(str(output), np.vstack((label, np.hstack(panels))), [cv2.IMWRITE_JPEG_QUALITY, 96])
        print(output)


if __name__ == "__main__":
    main()
