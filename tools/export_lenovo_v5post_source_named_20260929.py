"""Export the 33 final V5-postprocessed PNGs under their camera source names."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evaluation"
DEFAULT_RUN = ROOT / "lenovo_borderrail_ft50k_whiteboard33_normal_v5post_boardframe_20260929"
ARM = "gtfit_v6_borderrail_h96"
MANIFEST = DEFAULT_RUN / "source_manifest" / "train.jsonl"
CAMERA_STEM = re.compile(r"IMG_\d{8}_\d{9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_root
    destination = run / ARM / "outputs_by_source_name"
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 33 or len({row["sample_id"] for row in rows}) != 33:
        raise ValueError("expected 33 distinct source records")
    stems = []
    for row in rows:
        source_name = Path(row["source_lq"]).name
        match = CAMERA_STEM.search(source_name)
        if match is None:
            raise ValueError(f"camera filename not found: {source_name}")
        stems.append(match.group(0))
    totals = Counter(stems)
    indices: defaultdict[str, int] = defaultdict(int)
    destination.mkdir(parents=True, exist_ok=True)
    mapping = []
    for row, stem in zip(rows, stems):
        indices[stem] += 1
        suffix = f"_idx{indices[stem]:02d}" if totals[stem] > 1 else ""
        target = destination / f"{stem}{suffix}.png"
        source = run / ARM / row["sample_id"] / "output.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.is_file():
            raise FileExistsError(target)
        shutil.copy2(source, target)
        mapping.append({"output_filename": target.name, "camera_stem": stem,
                        "duplicate_index": indices[stem] if totals[stem] > 1 else None,
                        "sample_id": row["sample_id"], "source_lq": row["source_lq"]})
    if len(list(destination.glob("*.png"))) != 33:
        raise ValueError("exported PNG count does not match 33 samples")
    (destination / "filename_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(mapping)} files written to {destination}")


if __name__ == "__main__":
    main()
