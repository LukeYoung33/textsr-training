"""Bundle the compact four-arm Lenovo full-frame validation review assets."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path


ARMS = ("scurve_h96", "scurve_h64", "raw_h96", "raw_h64")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--mirror-root", type=Path)
    args = parser.parse_args()
    if (args.archive is None) == (args.mirror_root is None):
        raise ValueError("select exactly one of --archive or --mirror-root")
    destination = args.archive if args.archive is not None else args.mirror_root
    if destination.exists():
        raise FileExistsError(destination)
    files: list[Path] = []
    for arm in ARMS:
        arm_root = args.root / arm
        summary_path = arm_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["parents"] != 231 or summary["passed"] != 231 or summary["failed"]:
            raise ValueError(f"incomplete validation: {arm}")
        files.append(summary_path)
        for row in summary["rows"]:
            sample = Path(row["output"]).parent
            files.extend((sample / "output.png", sample / "comparison_lq_used_output_gt.jpg"))
            if arm.startswith("scurve"):
                files.append(sample / "lq_used.png")
            if arm == "scurve_h96":
                files.extend((sample / "lq_original.png", sample / "gt.png", sample / "changed_pixels.png"))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.mirror_root is not None:
        for index, path in enumerate(files, 1):
            target = destination / path.relative_to(args.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
            if index % 200 == 0:
                print(f"linked {index}/{len(files)}", flush=True)
    else:
        with tarfile.open(destination, "w") as archive:
            for index, path in enumerate(files, 1):
                archive.add(path, arcname=path.relative_to(args.root.parent), recursive=False)
                if index % 200 == 0:
                    print(f"packed {index}/{len(files)}", flush=True)
    print(f"prepared {len(files)} files at {destination}", flush=True)


if __name__ == "__main__":
    main()
