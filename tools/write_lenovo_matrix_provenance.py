"""Refresh manifest hashes after a direct-arm GT localization pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("v5", "v6", "gtfit_v5", "gtfit_v6")


def _hash(path: Path) -> str:
    """@luna-comment
    Input: One manifest path.
    Output: Its SHA-256 digest.
    Key Steps: Stream file content in bounded chunks.
    Key Params: One MiB read chunks.
    State/Dependencies: Read-only filesystem access.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(root: Path, source: Path) -> dict:
    """@luna-comment
    Input: Verified matrix root and its exact parent-manifest directory.
    Output: Refreshed base and border-rail manifest provenance documents.
    Key Steps: Require successful local-GT verification, hash source and canonical arm manifests.
    Key Params: Both splits and all four arms per generation are recorded.
    State/Dependencies: Writes only two provenance JSON files beneath the matrix root.
    """
    local_report = json.loads((root / "direct_local_gt_verification.json").read_text(encoding="utf-8"))
    if local_report.get("passed") is not True:
        raise ValueError("direct local GT verification has not passed")
    source_hashes = {split: _hash(source / f"{split}.jsonl") for split in ("train", "val")}
    matrix_hashes = {arm: {split: _hash(root / arm / "manifests" / f"{split}.jsonl")
                           for split in ("train", "val")} for arm in ARMS}
    border_hashes = {arm: {split: _hash(root / f"{arm}_borderrail" / "manifests" / f"{split}.jsonl")
                           for split in ("train", "val")} for arm in ARMS}
    base = {"source_manifest_sha256": source_hashes,
            "output_manifest_sha256": matrix_hashes,
            "verification": "verification.json",
            "direct_local_gt_verification": "direct_local_gt_verification.json"}
    border = {"schema": "lenovo_border_rail_provenance_v1",
              "reference_verification": "verification.json",
              "revision_verification": "borderrail_verification.json",
              "direct_local_gt_verification": "direct_local_gt_verification.json",
              "manifest_sha256": border_hashes}
    (root / "provenance.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    (root / "borderrail_provenance.json").write_text(json.dumps(border, indent=2), encoding="utf-8")
    return {"source": source_hashes, "base_arms": len(matrix_hashes), "borderrail_arms": len(border_hashes)}


def main() -> int:
    """@luna-comment
    Input: Matrix root and exact source-manifest directory CLI arguments.
    Output: Refreshed provenance summary and exit status.
    Key Steps: Parse paths, write hashes, and print compact result.
    Key Params: No implicit source-path guessing.
    State/Dependencies: Writes only the requested matrix root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_provenance(args.root, args.source)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
