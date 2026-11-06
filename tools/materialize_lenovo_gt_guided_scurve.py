"""Build paired Lenovo A/B manifests with a GT-derived S-curve on raw LQ."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from tools.text_lq_gt_guided_scurve import DEFAULT_CONFIG, preprocess_text_lq_from_gt


SHARED_NAMES = ("LQ_raw", "GT", "OCR", "text_mask", "valid_mask", "lq_foreground_mask", "fit_support_mask")


def _worker_init() -> None:
    """@luna-comment
    Input: New worker process.
    Output: Single-threaded OpenCV execution in that process.
    Key Steps: Avoid nested native threads across independent line workers.
    Key Params: Process count is selected by the CLI.
    State/Dependencies: Changes only process-local OpenCV settings.
    """
    cv2.setNumThreads(1)


def _load_rows(path: Path) -> list[dict]:
    """@luna-comment
    Input: One completed baseline JSONL manifest.
    Output: Ordered decoded records.
    Key Steps: Decode nonempty lines without altering the source.
    Key Params: Source rows have absolute paired asset paths.
    State/Dependencies: Filesystem read only.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prepare_links(source: Path, output: Path) -> None:
    """@luna-comment
    Input: Completed baseline root and new variant root.
    Output: Directory links for immutable paired assets.
    Key Steps: Link only directories present in the source; reject existing mismatched targets.
    Key Params: Raw LQ, GT, OCR, and masks are never copied or processed.
    State/Dependencies: Writes links under the new variant root only.
    """
    for split in ("train", "val"):
        (output / split).mkdir(parents=True, exist_ok=True)
        for name in SHARED_NAMES:
            old = source / split / name
            if not old.is_dir():
                continue
            new = output / split / name
            if new.is_symlink():
                if new.resolve() != old.resolve():
                    raise ValueError(f"shared directory link points elsewhere: {new}")
            elif new.exists():
                raise FileExistsError(f"shared directory path already exists: {new}")
            else:
                new.symlink_to(old, target_is_directory=True)


def _process_chunk(task: tuple[str, list[dict], str]) -> tuple[str, list[dict], dict[str, int], int]:
    """@luna-comment
    Input: Split, ordered source-line rows, and variant output root.
    Output: New manifest rows, fallback counts, and reused image count.
    Key Steps: Fit each GT curve, apply to raw LQ, atomically save processed LQ, rewrite row paths.
    Key Params: Every chunk owns disjoint line IDs; any completed PNG can be reused on restart.
    State/Dependencies: Reads linked source assets and writes only its own LQ PNGs.
    """
    split, source_rows, output_text = task
    output = Path(output_text)
    rows = []
    fallbacks: Counter[str] = Counter()
    reused = 0
    for source_row in source_rows:
        sample_id = source_row["sample_id"]
        files = source_row["files"]
        raw = cv2.imread(str(files["lq_raw"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(files["gt"]), cv2.IMREAD_COLOR)
        if raw is None or gt is None:
            raise ValueError(f"unreadable raw LQ or GT: {sample_id}")
        processed, info = preprocess_text_lq_from_gt(raw, gt)
        target = output / split / "LQ_scurve" / f"{sample_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 0:
            reused += 1
        else:
            temporary = target.with_suffix(".tmp.png")
            if not cv2.imwrite(str(temporary), processed, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise OSError(f"cannot write processed LQ: {temporary}")
            temporary.replace(target)
        row = copy.deepcopy(source_row)
        row["release_root"] = str(output)
        row["source_variant_root"] = str(Path(source_row["release_root"]))
        row["baseline_preprocess"] = row.pop("preprocess", None)
        row["files"] = {
            name: str(target) if name == "lq" else str(output / split / Path(path).parent.name / Path(path).name)
            for name, path in files.items()
        }
        row["preprocess"] = info
        rows.append(row)
        fallbacks[str(info["fallback_reason"])] += 1
    return split, rows, dict(fallbacks), reused


def materialize(source: Path, output: Path, workers: int = 6, chunk_size: int = 32) -> dict:
    """@luna-comment
    Input: Baseline root, new variant root, worker count, and source-line chunk size.
    Output: Complete A/B variant summary and rewritten manifests.
    Key Steps: Check baseline, link immutable assets, dispatch disjoint lines, publish ordered manifests.
    Key Params: Existing completed output summary is never replaced; interrupted image work resumes.
    State/Dependencies: Writes only within the new output root, apart from reading the baseline.
    """
    if workers < 1 or chunk_size < 1:
        raise ValueError("workers and chunk_size must be positive")
    source = source.resolve()
    if not (source / "manifests" / "verification.json").is_file():
        raise ValueError("baseline dataset must have a verification report")
    baseline_report = json.loads((source / "manifests" / "verification.json").read_text(encoding="utf-8"))
    if baseline_report.get("status") != "passed":
        raise ValueError("baseline verification did not pass")
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() == source:
        raise ValueError("output must differ from the baseline root")
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    if (manifest_dir / "manifest.json").exists():
        raise FileExistsError("completed variant manifest already exists")
    _prepare_links(source, output)
    source_rows = {split: _load_rows(source / "manifests" / f"{split}.jsonl") for split in ("train", "val")}
    tasks = [(split, rows[start:start + chunk_size], str(output))
             for split, rows in source_rows.items() for start in range(0, len(rows), chunk_size)]
    completed = 0
    ordered: dict[str, list[dict]] = {"train": [], "val": []}
    fallbacks: Counter[str] = Counter()
    reused = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = {executor.submit(_process_chunk, task): (task[0], index)
                   for index, task in enumerate(tasks)}
        chunks: dict[int, list[dict]] = {}
        for future in as_completed(futures):
            split, index = futures[future]
            actual_split, rows, counts, reused_count = future.result()
            if split != actual_split:
                raise ValueError("worker split mismatch")
            chunks[index] = rows
            fallbacks.update(counts)
            reused += reused_count
            completed += 1
            if completed % 100 == 0:
                print(json.dumps({"completed_chunks": completed, "total_chunks": len(tasks),
                                  "reused_lines": reused}), flush=True)
    for index, task in enumerate(tasks):
        ordered[task[0]].extend(chunks[index])
    for split, rows in ordered.items():
        if len(rows) != len(source_rows[split]):
            raise ValueError(f"{split} row count changed")
        files = {
            f"{split}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            f"{split}_pairs.txt": "".join(f"{row['files']['lq']} {row['files']['gt']}\n" for row in rows),
            f"{split}_raw_pairs.txt": "".join(f"{row['files']['lq_raw']} {row['files']['gt']}\n" for row in rows),
        }
        for name, content in files.items():
            temporary = manifest_dir / (name + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(manifest_dir / name)
    excluded = source / "manifests" / "review_excluded.jsonl"
    excluded_bytes = excluded.read_bytes() if excluded.is_file() else b""
    (manifest_dir / "review_excluded.jsonl").write_bytes(excluded_bytes)
    summary = {
        "schema": "lenovo_gt_guided_scurve_ab_v1", "source_variant_root": str(source),
        "output_root": str(output), "preprocess_version": DEFAULT_CONFIG.version,
        "curve_source": "gt", "curve_config": vars(DEFAULT_CONFIG),
        "counts": {f"{split}_lines": len(rows) for split, rows in ordered.items()},
        "fallback_counts": dict(fallbacks), "workers": workers, "reused_existing_lines": reused,
        "excluded_source_images": sum(1 for line in excluded_bytes.splitlines() if line.strip()),
        "shared_asset_mode": "directory_symlink", "complete": True,
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """@luna-comment
    Input: Baseline and destination paths with process options.
    Output: Printed summary and process exit status.
    Key Steps: Parse arguments and execute the GT-guided A/B materializer.
    Key Params: Six workers and 32-line chunks are the default remote batch settings.
    State/Dependencies: Only the specified output root receives new data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_root, args.output_root, args.workers, args.chunk_size),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
