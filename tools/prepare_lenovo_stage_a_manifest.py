"""Materialise a local Lenovo validation manifest for the Stage A screen.

The source manifest contains historical ``/AI_DNS`` paths and nested variants.
This adapter keeps the source-group identity, resolves assets below the local
manifest root, and emits the JSONL contract consumed by the Stage A tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_MANIFEST = Path(
    "datasets/generated_images/lenovo_validation_aspect_2to5_stronger_c5_priority_128_20260911/manifest.json"
)


def _load_manifest(path: Path) -> dict[str, Any]:
    """@luna-comment Load the nested local-validation manifest object.

    Input: JSON manifest path.
    Output: Parsed object containing the root ``records`` list.
    Key Steps: Read UTF-8 JSON and reject non-object or missing-record payloads.
    Key Params: No path rewriting occurs at this stage.
    State/Dependencies: Reads one manifest file without modifying it.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("manifest must be an object with a records list")
    return payload


def _local_asset_path(raw_path: str, manifest_root: Path, marker: str) -> tuple[str, Path]:
    """@luna-comment Convert one historical asset path to a safe local relative path.

    Input: Historical path string, local manifest root, and ``gt`` or ``lq`` marker.
    Output: POSIX relative path plus its resolved local filesystem path.
    Key Steps: Find the final asset marker, anchor its suffix below the root, and
      require the resulting file to exist within the manifest directory.
    Key Params: Marker prevents accidental retention of ``/AI_DNS`` prefixes.
    State/Dependencies: Filesystem existence check only; no image decoding.
    """
    normalised = str(raw_path).replace("\\", "/")
    parts = PurePosixPath(normalised).parts
    positions = [index for index, value in enumerate(parts) if value == marker]
    if not positions:
        raise ValueError(f"asset path has no {marker} component: {raw_path}")
    relative = PurePosixPath(*parts[positions[-1] :])
    root = manifest_root.resolve()
    candidate = (root / Path(*relative.parts)).resolve()
    if root not in candidate.parents:
        raise ValueError(f"asset escaped manifest root: {raw_path}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return relative.as_posix(), candidate


def _category(variant: dict[str, Any]) -> tuple[Any, str]:
    """@luna-comment Preserve variant categories with a profile fallback.

    Input: One nested variant object.
    Output: JSON-safe category value and its source label.
    Key Steps: Prefer non-empty ``problem_categories``; otherwise retain ``profile``.
    Key Params: Lists remain lists so multi-category samples are not flattened.
    State/Dependencies: No filesystem or mutable state.
    """
    categories = variant.get("problem_categories")
    if isinstance(categories, list) and categories:
        return list(categories), "problem_categories"
    profile = variant.get("profile")
    if profile is not None and str(profile):
        return str(profile), "profile"
    return None, "missing"


def adapt_manifest(
    manifest_path: Path, expected_sources: int = 128, expected_variants: int = 3
) -> list[dict[str, Any]]:
    """@luna-comment Flatten 128 root records into Stage-A JSONL pair records.

    Input: Nested Lenovo manifest and expected source/variant cardinalities.
    Output: One row per source/variant with local relative RGB paths.
    Key Steps: Validate root count, map source_id to source_group, resolve GT/LQ,
      preserve category/profile metadata, and enforce three variants per group.
    Key Params: Defaults require 128 sources and variants 1/2/3; test fixtures may
      override counts explicitly without changing the production default.
    State/Dependencies: Reads local file metadata only; does not run Stage A.
    """
    manifest_path = manifest_path.expanduser().resolve()
    payload = _load_manifest(manifest_path)
    root = manifest_path.parent
    records = payload["records"]
    if len(records) != int(expected_sources):
        raise ValueError(f"expected {expected_sources} source records, got {len(records)}")
    if int(expected_variants) < 1:
        raise ValueError("expected_variants must be positive")
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"root record {index} is not an object")
        if "source_id" not in record or not str(record["source_id"]):
            raise ValueError(f"root record {index} has no source_id")
        source_id = str(record["source_id"])
        source_group = source_id
        if source_group in seen_groups:
            raise ValueError(f"duplicate source_id/source_group: {source_group}")
        seen_groups.add(source_group)
        gt_relative, gt_path = _local_asset_path(str(record.get("gt", "")), root, "gt")
        variants = record.get("variants")
        if not isinstance(variants, list) or len(variants) != int(expected_variants):
            raise ValueError(
                f"source {source_id} must retain {expected_variants} variants, got {len(variants) if isinstance(variants, list) else 'non-list'}"
            )
        variant_ids: list[int] = []
        for variant in variants:
            if not isinstance(variant, dict) or "variant" not in variant:
                raise ValueError(f"source {source_id} contains an invalid variant")
            variant_id = int(variant["variant"])
            variant_ids.append(variant_id)
        expected_ids = set(range(1, int(expected_variants) + 1))
        if set(variant_ids) != expected_ids:
            raise ValueError(f"source {source_id} variants must be {sorted(expected_ids)}, got {sorted(set(variant_ids))}")
        for variant in variants:
            variant_id = int(variant["variant"])
            lq_relative, lq_path = _local_asset_path(str(variant.get("lq", "")), root, "lq")
            category, category_source = _category(variant)
            rows.append(
                {
                    "sample_id": f"{source_id}_v{variant_id}",
                    "source_group": source_group,
                    "source_id": source_id,
                    "variant": variant_id,
                    "lq_rgb_path": lq_relative,
                    "gt_rgb_path": gt_relative,
                    "category": category,
                    "category_source": category_source,
                    "profile": variant.get("profile"),
                    "problem_categories": variant.get("problem_categories"),
                    "local_gt_exists": gt_path.is_file(),
                    "local_lq_exists": lq_path.is_file(),
                }
            )
    if len(rows) != int(expected_sources) * int(expected_variants):
        raise AssertionError("adapter output cardinality mismatch")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """@luna-comment Write adapted Stage-A records as deterministic JSONL.

    Input: Output path and JSON-safe flattened rows.
    Output: None; one UTF-8 record per line.
    Key Steps: Create the parent directory and serialise each row without NaN.
    Key Params: Sorted keys and newline termination make manifests reproducible.
    State/Dependencies: Writes only the requested output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """@luna-comment Build the local-manifest adapter CLI parser.

    Input: None.
    Output: Parser with manifest, output, and cardinality options.
    Key Steps: Bind the repository-local default and explicit test overrides.
    Key Params: Production defaults enforce the supplied 128x3 contract.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True, help="Stage-A JSONL output path")
    parser.add_argument("--expected-sources", type=int, default=128)
    parser.add_argument("--expected-variants", type=int, default=3)
    return parser


def main() -> None:
    """@luna-comment Adapt one local nested manifest and print cardinality evidence.

    Input: CLI paths and expected 128x3 cardinalities.
    Output: JSONL file plus a JSON completion summary on stdout.
    Key Steps: Parse, flatten, write, and report source/variant counts.
    Key Params: No Stage-A algorithm or screen execution occurs here.
    State/Dependencies: Calls adapt_manifest and _write_jsonl only.
    """
    args = build_parser().parse_args()
    rows = adapt_manifest(args.manifest, args.expected_sources, args.expected_variants)
    _write_jsonl(args.output.expanduser().resolve(), rows)
    groups = sorted({row["source_group"] for row in rows})
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "records": len(rows),
                "source_groups": len(groups),
                "variants_per_source": int(args.expected_variants),
                "output": str(args.output.expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
