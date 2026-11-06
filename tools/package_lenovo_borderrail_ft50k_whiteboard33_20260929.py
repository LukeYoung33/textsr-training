"""Wait for all render jobs, then package viewable JPEGs and debug metadata."""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path


ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929")
ARMS = ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
ROUTES = ("normal", "large_tile", "overlap_tile")
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=ROUTES)
    parser.add_argument("--archive-name", default="whiteboard33_ft50k_visuals.tar")
    args = parser.parse_args()
    root = args.root
    archive = root / args.archive_name
    while True:
        status_file = root / "status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status.get("stage") == "complete":
                break
            if status.get("stage") == "failed":
                raise RuntimeError("render jobs failed; refusing to package partial results")
        time.sleep(30)
    names = [f"{arm}_h{height}" for arm in ARMS for height in (96, 64)]
    for name in names:
        summary = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
        if summary.get("completed") != 33 * len(args.routes):
            raise ValueError(f"incomplete: {name}")
    with tarfile.open(archive.with_suffix(".tar.tmp"), "w") as tar:
        tar.add(root / "status.json", arcname="status.json")
        for name in names:
            tar.add(root / name / "summary.json", arcname=f"{name}/summary.json")
            for route in args.routes:
                for sample_dir in sorted((root / name / route).iterdir()):
                    if not sample_dir.is_dir():
                        continue
                    for filename in ("output_full.jpg", "preview.jpg", "result.json", "merge_report.json"):
                        path = sample_dir / filename
                        if path.is_file():
                            tar.add(path, arcname=f"{name}/{route}/{sample_dir.name}/{filename}")
    archive.with_suffix(".tar.tmp").replace(archive)
    print(f"archive={archive} bytes={archive.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
