"""Copy aligned raw GT crops into all direct arms and publish local pair paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.materialize_lenovo_mild_scurve_lines import _load_jsonl


DIRECT_ARMS = ("v5", "v6", "v5_borderrail", "v6_borderrail")


def _atomic_text(path: Path, content: str) -> None:
    """@luna-comment
    Input: Canonical manifest or pair-list path and complete UTF-8 content.
    Output: Atomically replaced text file.
    Key Steps: Write a sibling temporary file, then replace the destination.
    Key Params: The temporary name is private to this process.
    State/Dependencies: Mutates only the selected dataset manifest directory.
    """
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _copy_and_check(item: tuple[Path, Path, tuple[int, int]]) -> int:
    """@luna-comment
    Input: Shared raw GT PNG, arm-local destination, and expected height/width.
    Output: Verified byte count for one unchanged local GT crop.
    Key Steps: Copy to a temporary PNG if absent; compare source/destination bytes and decode destination dimensions.
    Key Params: Exact source bytes are retained; no resampling or re-encoding occurs.
    State/Dependencies: Writes only one destination under its direct border-rail arm.
    """
    source, target, expected_hw = item
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    source_bytes, target_bytes = source.read_bytes(), target.read_bytes()
    if source_bytes != target_bytes:
        raise ValueError(f"local GT differs from raw crop: {target}")
    decoded = cv2.imread(str(target), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != expected_hw:
        raise ValueError(f"local GT cannot decode at expected size: {target}")
    return len(target_bytes)


def localize(root: Path) -> dict:
    """@luna-comment
    Input: Completed OCR matrix root with verified base and border-rail arms.
    Output: Local GT trees, revised canonical/shard pair lists and a verification report.
    Key Steps: Validate exact original pairing, copy and check every GT, then atomically publish arm-local paths in all manifests.
    Key Params: Direct V5/V6 and their border-rail revisions receive exact raw-crop GT bytes.
    State/Dependencies: Mutates only direct-arm GT trees and their canonical manifest/pair-list files.
    """
    if not (root / "verification.json").is_file() or not (root / "borderrail_verification.json").is_file():
        raise FileNotFoundError("verified base and border-rail matrix required")
    prepared: dict[tuple[str, str], list[dict]] = {}
    work = []
    for arm in DIRECT_ARMS:
        for split in ("train", "val"):
            manifest = root / arm / "manifests" / f"{split}.jsonl"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            rows = _load_jsonl(manifest)
            for row in rows:
                line_id = row["sample_id"]
                expected_rel = f"../base/{split}/GT_raw/{line_id}.png"
                local_rel = f"{split}/GT/{line_id}.png"
                current_rel = row["files"]["gt"]
                if current_rel not in (expected_rel, local_rel):
                    raise ValueError(f"unexpected GT source path: {arm}/{line_id}: {current_rel}")
                lq_path = root / arm / row["files"]["lq"]
                if not lq_path.is_file():
                    raise FileNotFoundError(lq_path)
                source = root / "base" / split / "GT_raw" / f"{line_id}.png"
                target = root / arm / local_rel
                width, height = row["cropped_size_wh"]
                work.append((source, target, (height, width)))
                row["files"]["gt"] = local_rel
                row["preprocess"]["gt_storage"] = "arm_local_raw_copy"
            prepared[(arm, split)] = rows
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, size in enumerate(pool.map(_copy_and_check, work), start=1):
            total_bytes += size
            if index % 2000 == 0:
                print(json.dumps({"verified_local_gt": index, "total": len(work)}), flush=True)
    counts = {}
    manifest_hashes = {}
    shard_manifests = 0
    for (arm, split), rows in prepared.items():
        directory = root / arm / "manifests"
        manifest = directory / f"{split}.jsonl"
        pairs = directory / f"{split}_pairs.txt"
        _atomic_text(manifest, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        _atomic_text(pairs, "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in rows))
        canonical_ids = {row["sample_id"] for row in rows}
        for shard_manifest in sorted(directory.glob(f"{split}.shard*.jsonl")):
            shard_rows = _load_jsonl(shard_manifest)
            for row in shard_rows:
                if row["sample_id"] not in canonical_ids:
                    raise ValueError(f"unknown shard GT pairing: {shard_manifest}/{row['sample_id']}")
                row["files"] = {name: path.replace("\\", "/") for name, path in row["files"].items()}
                row["files"]["gt"] = f"{split}/GT/{row['sample_id']}.png"
                row["preprocess"]["gt_storage"] = "arm_local_raw_copy"
            _atomic_text(shard_manifest, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard_rows))
            shard_pairs = shard_manifest.with_name(f"{shard_manifest.stem}_pairs.txt")
            _atomic_text(shard_pairs, "".join(
                f"{row['files']['lq']} {row['files']['gt']}\n" for row in shard_rows))
            shard_manifests += 1
        counts[f"{arm}_{split}"] = len(rows)
        manifest_hashes[f"{arm}_{split}"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = {"schema": "lenovo_direct_local_gt_v1", "passed": True,
              "verified_local_gt": len(work), "copied_bytes": total_bytes,
              "rewritten_shard_manifests": shard_manifests,
              "counts": counts, "manifest_sha256": manifest_hashes}
    _atomic_text(root / "direct_local_gt_verification.json", json.dumps(report, indent=2))
    return report


def main() -> int:
    """@luna-comment
    Input: CLI matrix root.
    Output: Printed local-GT verification report and process exit status.
    Key Steps: Parse the root and run the safe copy-then-publish operation.
    Key Params: Four direct arms and both train/val splits.
    State/Dependencies: Writes only to the matrix root's direct arms.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(localize(args.root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
