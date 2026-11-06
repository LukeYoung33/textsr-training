"""Build immutable V5 frame manifests for the BASIC native line-crop cache.

The historical ``training_lists_full_20260820`` files define the approved
train/validation split.  This adapter preserves those rows and replaces their
V3 five-tuple paths with the matching topology-bg V5 candidate paths.  It
never copies pixels and refuses incomplete or ambiguous V5 candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


ASSET_KEYS = ("lq", "gt", "valid_mask", "text_mask", "ocr")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: row is not an object")
        rows.append(row)
    return rows


def _parse_candidate(value: str) -> tuple[str, Path]:
    batch_id, separator, raw_path = value.partition("=")
    if not separator or not batch_id or not raw_path:
        raise ValueError("--candidate must be BATCH_ID=/absolute/path/dataset_V5_manifest.json")
    return batch_id, Path(raw_path)


def _candidate_index(batch_id: str, manifest_path: Path) -> dict[str, Path]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "training_publish_v5_candidate":
        raise ValueError(f"{manifest_path}: not a V5 candidate manifest")
    if manifest.get("algorithm") != "topology_bg_20260901":
        raise ValueError(f"{manifest_path}: unexpected algorithm {manifest.get('algorithm')!r}")
    if manifest.get("batch_id") != batch_id:
        raise ValueError(f"{manifest_path}: batch id does not match {batch_id!r}")
    if int(manifest.get("quarantine_count", -1)) != 0:
        raise ValueError(f"{manifest_path}: candidate contains quarantine samples")
    directories = manifest.get("directories")
    samples = manifest.get("samples")
    if not isinstance(directories, dict) or not isinstance(samples, list):
        raise ValueError(f"{manifest_path}: missing directories/samples")
    result: dict[str, Path] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"{manifest_path}: invalid sample entry")
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"{manifest_path}: invalid or duplicate sample id {sample_id!r}")
        result[sample_id] = Path(str(directories["gt"])) / f"{sample_id}.png"
        for key, suffix in (("lq", ".png"), ("gt", ".png"), ("valid_mask", ".png"), ("text_mask", ".png"), ("ocr", ".json")):
            asset = Path(str(directories[key])) / f"{sample_id}{suffix}"
            if not asset.is_file():
                raise FileNotFoundError(f"{batch_id}/{sample_id}: missing V5 {key}: {asset}")
    return result


def _v5_paths(manifest: dict[str, Any], sample_id: str) -> dict[str, str]:
    directories = manifest["directories"]
    return {
        "lq": str(Path(str(directories["lq"])) / f"{sample_id}.png"),
        "gt": str(Path(str(directories["gt"])) / f"{sample_id}.png"),
        "valid_mask": str(Path(str(directories["valid_mask"])) / f"{sample_id}.png"),
        "text_mask": str(Path(str(directories["text_mask"])) / f"{sample_id}.png"),
        "ocr": str(Path(str(directories["ocr"])) / f"{sample_id}.json"),
    }


def build(source_root: Path, output_root: Path, candidate_args: Iterable[str]) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    candidates: dict[str, dict[str, Any]] = {}
    indexes: dict[str, dict[str, Path]] = {}
    for argument in candidate_args:
        batch_id, manifest_path = _parse_candidate(argument)
        if batch_id in candidates:
            raise ValueError(f"duplicate candidate batch: {batch_id}")
        manifest = _read_json(manifest_path)
        indexes[batch_id] = _candidate_index(batch_id, manifest_path)
        candidates[batch_id] = manifest

    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"staging root exists: {staging}")
    staging.mkdir(parents=True)
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    try:
        for split in ("train", "val"):
            source_path = source_root / f"{split}.jsonl"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            output_rows: list[dict[str, Any]] = []
            for number, row in enumerate(_read_jsonl(source_path), 1):
                batch_id = str(row.get("dataset", ""))
                sample_id = str(row.get("stem", row.get("frame_id", "")))
                if batch_id not in candidates:
                    raise ValueError(f"{source_path}:{number}: no V5 candidate for dataset {batch_id!r}")
                if sample_id not in indexes[batch_id]:
                    raise ValueError(f"{source_path}:{number}: V5 sample missing: {batch_id}/{sample_id}")
                identity = (batch_id, sample_id)
                if identity in seen:
                    raise ValueError(f"source split overlap or duplicate: {identity}")
                seen.add(identity)
                output = dict(row)
                output.update(_v5_paths(candidates[batch_id], sample_id))
                output["source_v3_record"] = {key: row.get(key) for key in ASSET_KEYS}
                output["v5_algorithm"] = "topology_bg_20260901"
                output_rows.append(output)
            target = staging / f"{split}.jsonl"
            target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
            counts[split] = len(output_rows)
        payload = {
            "schema_version": "basic_v5_0901_source_manifest_v1",
            "algorithm": "topology_bg_20260901",
            "source_training_lists": str(source_root),
            "candidate_manifests": {key: str(value["candidate_root"]) for key, value in sorted(candidates.items())},
            "split_counts": counts,
            "split_sha256": {split: _sha256(staging / f"{split}.jsonl") for split in counts},
        }
        (staging / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_root)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_root, args.candidate), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
