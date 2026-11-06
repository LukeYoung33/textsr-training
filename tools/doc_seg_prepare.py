"""Prepare a provenance-preserving single-class YOLO segmentation dataset."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np


def parse_label(path, allowed_classes):
    """@luna-comment
    Input: UTF-8 YOLO polygon file and audited allowed integer class IDs.
    Output: remapped rows and legacy IDs; raises ValueError for malformed labels.
    Key Steps: parse finite normalized points, validate area and uniqueness.
    Key Params: allowed_classes prevents importing unrelated detection semantics.
    State/Dependencies: reads only the specified file; numpy/OpenCV geometry.
    """
    rows, legacy = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise ValueError(f"line {line_number}: not a polygon coordinate-pair row")
        class_id = int(tokens[0])
        if class_id not in allowed_classes:
            raise ValueError(f"line {line_number}: unaudited class {class_id}")
        points = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(points).all() or points.min() < 0 or points.max() > 1:
            raise ValueError(f"line {line_number}: nonfinite/out-of-range coordinates")
        if len(np.unique(points, axis=0)) < 3 or abs(cv2.contourArea(points.astype(np.float32))) < 1e-8:
            raise ValueError(f"line {line_number}: degenerate polygon")
        rows.append("0 " + " ".join(f"{value:.8f}" for value in points.ravel()))
        legacy.append(class_id)
    if not rows:
        raise ValueError("empty label is not an audited negative")
    return rows, legacy


def find_image(label, label_root, image_root):
    """@luna-comment
    Input: label path and corresponding labels/images roots.
    Output: unique paired image path, or raises for missing/ambiguous matches.
    Key Steps: Probe suffixes and deduplicate filesystem-equivalent path spellings.
    Key Params: root mapping prevents accidental cross-source pairing.
    State/Dependencies: read-only filesystem metadata.
    """
    relative = label.relative_to(label_root).with_suffix("")
    candidates = [
        image_root / (str(relative) + suffix)
        for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".webp")
    ]
    found_by_path = {}
    for path in candidates:
        if path.is_file():
            found_by_path.setdefault(os.path.normcase(str(path.resolve())), path)
    found = list(found_by_path.values())
    if len(found) != 1:
        raise ValueError(f"image pairing count={len(found)}")
    return found[0]


def group_key(source, relative):
    """@luna-comment
    Input: audited source name and image-relative path.
    Output: deterministic conservative identifiable-document group string.
    Key Steps: group 5k render suffixes and WarpDoc transformations; others by stem.
    Key Params: legacy split components are removed from identity.
    State/Dependencies: no external state; unknown capture identity remains documented.
    """
    stem = relative.stem
    if source == "5k":
        stem = re.sub(r"_\d+$", "", stem)
    parts = [p for p in relative.parts[:-1] if p not in ("train", "val", "test")]
    if "WarpDoc" in parts:
        parts = ["WarpDoc"]
    return source + "/" + "/".join(parts + [stem])


def cache_image_bytes(cache_directory, image, raw, digest):
    """@luna-comment
    Input: Cache directory, original image path, original bytes and their SHA256.
    Output: Complete content-addressed cache path; raises on cache storage failure.
    Key Steps: Write a unique sibling temporary file, close, then atomically replace.
    Key Params: Original suffix is retained; concurrent identical writers are safe.
    State/Dependencies: Writes cache only; originals untouched; no partial final file.
    """
    destination = cache_directory / (digest + image.suffix.lower())
    temporary = None
    try:
        if destination.is_file() and destination.stat().st_size == len(raw):
            return destination
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=digest + ".", suffix=".tmp", dir=cache_directory,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        os.replace(temporary, destination)
        return destination
    except OSError as error:
        raise RuntimeError(f"Image cache write failed: {destination}: {error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def inspect_pair(label, label_root, image_root, source, classes, cache_directory):
    """@luna-comment
    Input: One label, paired roots, source/classes and optional output cache directory.
    Output: Provenance record containing metadata only, never image buffers.
    Key Steps: Parse; read/decode/hash; release decoded pixels; optionally cache bytes.
    Key Params: Original image/label paths and hashes remain authoritative provenance.
    State/Dependencies: Worker-local image buffers; optional atomic cache file writes.
    """
    rows, legacy = parse_label(label, classes)
    image = find_image(label, label_root, image_root)
    raw = image.read_bytes()
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or min(decoded.shape[:2]) < 16:
        raise ValueError("unreadable/tiny image")
    height, width = decoded.shape[:2]
    pixel_digest = hashlib.sha256()
    pixel_digest.update(memoryview(decoded))
    pixel_digest.update(str(decoded.shape).encode())
    del decoded
    raw_digest = hashlib.sha256(raw).hexdigest()
    cached_image = None
    if cache_directory is not None:
        cached_image = str(cache_image_bytes(cache_directory, image, raw, raw_digest))
    del raw
    relative = image.relative_to(image_root)
    original_split = next((p for p in relative.parts if p in ("train", "val", "test")), None)
    return {
        "source": source, "image": str(image), "label": str(label),
        "cached_image": cached_image,
        "sha256": raw_digest, "pixel_sha256": pixel_digest.hexdigest(),
        "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
        "rows": rows, "legacy_classes": legacy,
        "group": group_key(source, relative), "original_split": original_split,
        "width": width, "height": height,
    }


def iter_labels(label_root, walk_errors):
    """@luna-comment
    Input: Source labels root and mutable directory-walk error collection.
    Output: Lazy deterministic sequence of label paths.
    Key Steps: Sort walk directories and filenames, then yield text labels.
    Key Params: Lazy iteration supports a bounded worker submission queue.
    State/Dependencies: Filesystem metadata; no label/image contents retained.
    """
    for directory, directories, files in os.walk(label_root, onerror=walk_errors.append):
        directories.sort()
        for filename in sorted(files):
            if filename.lower().endswith(".txt"):
                yield Path(directory) / filename


def inspect_source(root, source, classes, exclusions, workers=8, cache_directory=None):
    """@luna-comment
    Input: Dataset roots/classes, exclusions, positive worker count and optional cache.
    Output: valid paired image/polygon records with content hashes and grouping evidence.
    Key Steps: Submit bounded inspection jobs, collect results and sort stable records.
    Key Params: At most workers*2 futures; log source start and every 100 completions.
    State/Dependencies: Thread pool, OpenCV and optional cache; storage failures abort.
    """
    if workers < 1:
        raise ValueError("workers must be at least one")
    started_at = time.monotonic()
    print(json.dumps({
        "source": source, "stage": "inspection_started", "workers": workers,
        "cache_images": cache_directory is not None,
    }), flush=True)
    label_root, image_root = root / source / "labels", root / source / "images"
    records, source_exclusions = [], []
    if not label_root.is_dir():
        exclusions.append({"source": source, "reason": "labels_directory_missing"})
        return records
    if cache_directory is not None:
        cache_directory = Path(cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
    walk_errors = []
    labels = iter_labels(label_root, walk_errors)
    completed = 0
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="doc-inspect") as executor:
        pending = {}
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                label = next(labels, None)
                if label is None:
                    exhausted = True
                    break
                future = executor.submit(
                    inspect_pair, label, label_root, image_root, source, classes, cache_directory,
                )
                pending[future] = label
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                label = pending.pop(future)
                try:
                    records.append(future.result())
                except (ValueError, OSError, cv2.error) as exc:
                    source_exclusions.append({"source": source, "label": str(label), "reason": str(exc)})
                completed += 1
                if completed % 100 == 0:
                    print(json.dumps({
                        "source": source, "completed": completed, "valid": len(records),
                        "excluded": len(source_exclusions),
                        "elapsed_seconds": round(time.monotonic() - started_at, 2),
                    }), flush=True)
    source_exclusions.extend({"source": source, "reason": str(error)} for error in walk_errors)
    exclusions.extend(sorted(source_exclusions, key=lambda item: (item.get("label", ""), item["reason"])))
    print(json.dumps({
        "source": source, "finished_valid": len(records), "completed": completed,
        "excluded": len(source_exclusions), "walk_errors": len(walk_errors),
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
    }), flush=True)
    return sorted(records, key=lambda record: record["image"])


def assign_splits(records, seed):
    """@luna-comment
    Input: validated records and stable split seed.
    Output: unique-image records, duplicate/conflict audit and group-consistent splits.
    Key Steps: union exact pixel duplicates with document groups; preserve val/test priority.
    Key Params: new groups use 80/10/10; legacy train groups reserve 10 percent test.
    State/Dependencies: only in-memory records are mutated; no source label writes.
    """
    parents = {}

    def find(key):
        """@luna-comment
        Input: group identifier. Output: union-find representative.
        Key Steps: compress parent paths. Key Params: none.
        State/Dependencies: enclosing parents mapping only.
        """
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    by_pixels = defaultdict(list)
    for record in records:
        find(record["group"])
        by_pixels[record["pixel_sha256"]].append(record)
    for duplicate_records in by_pixels.values():
        roots = sorted({find(record["group"]) for record in duplicate_records})
        for representative in roots[1:]:
            parents[representative] = roots[0]
    priorities = {"test": 3, "val": 2, "train": 1, None: 0}
    group_priority = defaultdict(int)
    for record in records:
        group_priority[find(record["group"])] = max(group_priority[find(record["group"])], priorities[record["original_split"]])
    unique, duplicates = [], []
    for duplicate_records in by_pixels.values():
        chosen = max(duplicate_records, key=lambda r: (priorities[r["original_split"]], len(r["rows"]), r["source"], r["image"]))
        conflict = len({tuple(r["rows"]) for r in duplicate_records}) > 1
        if len(duplicate_records) > 1:
            duplicates.append({
                "kept": chosen["image"], "others": [r["image"] for r in duplicate_records if r is not chosen],
                "label_conflict": conflict,
                "disposition": "exclude_all_pending_review" if conflict else "keep_one",
            })
        if conflict:
            continue
        group = find(chosen["group"])
        score = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 2**32
        priority = group_priority[group]
        split = "test" if priority == 3 else "val" if priority == 2 else (
            ("test" if score < .1 else "train") if priority == 1 else
            ("test" if score < .1 else "val" if score < .2 else "train")
        )
        chosen.update(split=split, split_group=group)
        unique.append(chosen)
    return sorted(unique, key=lambda r: r["image"]), duplicates


def prepare(args):
    """@luna-comment
    Input: CLI namespace with read-only source root and new output directory.
    Output: data.yaml, derived labels/image links, manifest, audit and sample lists.
    Key Steps: Concurrent inspection/cache, split, materialize, verify leakage, publish audit.
    Key Params: Workers default to eight; cache-images redirects final links to raw cache.
    State/Dependencies: Output descendants only; cache alone is not a prepared dataset.
    """
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {output}")
    output.mkdir(parents=True)
    root = Path(args.root).resolve()
    workers = getattr(args, "workers", 8)
    if workers < 1:
        raise ValueError("workers must be at least one")
    cache_directory = output / "cache_images" if getattr(args, "cache_images", False) else None
    records, excluded = [], []
    for source in args.sources:
        if Path(source).name != source:
            raise ValueError("sources must be immediate dataset directory names")
        records.extend(inspect_source(
            root, source, set(args.classes), excluded, workers=workers, cache_directory=cache_directory,
        ))
    records, duplicates = assign_splits(records, args.seed)
    counts = Counter(record["split"] for record in records)
    if not counts["train"] or not counts["val"]:
        raise ValueError(f"Insufficient train/val data: {counts}")
    by_source = defaultdict(Counter)
    manifest = []
    for record in records:
        split = record["split"]
        identifier = record["pixel_sha256"][:24]
        image = Path(record["image"])
        target_image = output / "images" / split / (identifier + image.suffix.lower())
        target_label = output / "labels" / split / (identifier + ".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_image.symlink_to(Path(record["cached_image"]) if record.get("cached_image") else image)
        target_label.write_text("\n".join(record["rows"]) + "\n", encoding="utf-8")
        by_source[record["source"]][split] += 1
        manifest.append({**{k: v for k, v in record.items() if k != "rows"},
                         "prepared_image": str(target_image), "prepared_label": str(target_label),
                         "instances": len(record["rows"])})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for split in ("train", "val", "test"):
        paths = [r["prepared_image"] for r in manifest if r["split"] == split]
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (output / "data.yaml").write_text(
        f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\n"
        + ("test: images/test\n" if counts["test"] else "")
        + "names:\n  0: text_region\n", encoding="utf-8")
    for field in ("sha256", "pixel_sha256", "split_group"):
        assignment = defaultdict(set)
        for record in manifest:
            assignment[record[field]].add(record["split"])
        assert all(len(splits) == 1 for splits in assignment.values()), field
    summary = {
        "source_root": str(root), "seed": args.seed, "splits": counts,
        "by_source": dict(by_source), "instances": sum(r["instances"] for r in manifest),
        "excluded_count": len(excluded), "duplicate_count": sum(len(r["others"]) for r in duplicates),
        "conflicting_pixel_groups_excluded": sum(r["label_conflict"] for r in duplicates),
        "exact_image_and_group_leakage": 0, "original_classes": args.classes,
        "class_mapping": {str(c): 0 for c in args.classes},
        "grouping_limit": "5k render stems and WarpDoc source stems grouped; other capture/document identity unknown. Near-duplicate leakage not certified.",
        "annotation_limit": "Existing polygons retained; visible/amodal and page-vs-spread inconsistency not manually relabeled.",
        "negative_policy": "Empty labels excluded until explicitly audited.",
        "sources_requested": args.sources,
        "inspection_workers": workers,
        "cache_images_enabled": cache_directory is not None,
        "cache_directory": str(cache_directory) if cache_directory is not None else None,
        "cache_semantics": "Original encoded bytes; not a completed dataset until split/materialization/audit finish.",
    }
    for name, data in (("audit.json", summary), ("excluded.json", excluded), ("duplicates.json", duplicates)):
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    """@luna-comment
    Input: command-line dataset root/output, sources/classes and seed.
    Output: preparation artifacts or nonzero process exit on invalid inputs.
    Key Steps: parse explicit scope and execute preparation.
    Key Params: Four audited source defaults, workers=8, optional original-byte caching.
    State/Dependencies: delegates to prepare; OpenCV CPU thread count capped.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", nargs="+", default=["5k", "RWMD_dataset_yolo", "document_task_yolo", "documents_last_version"])
    parser.add_argument("--classes", nargs="+", type=int, default=[0, 11])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent image inspection workers (default: 8).")
    parser.add_argument("--cache-images", action="store_true", help="Atomically cache original bytes inside output; link training images to cache.")
    cv2.setNumThreads(1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    prepare(args)


if __name__ == "__main__":
    main()
