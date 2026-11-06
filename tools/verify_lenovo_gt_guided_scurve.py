"""Verify GT-guided A/B pairing, raster integrity, and the paired reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.verify_lenovo_mild_scurve_lines import smoke_training_reader, verify as verify_rasters


def verify_ab(root: Path) -> dict:
    """@luna-comment
    Input: Completed GT-guided dataset root.
    Output: Verification report with baseline pairing evidence.
    Key Steps: Verify all rasters, align every row with baseline, and check shared asset identity.
    Key Params: Only processed LQ paths may differ; split order and sample IDs must match exactly.
    State/Dependencies: Reads both datasets without mutating them.
    """
    summary = json.loads((root / "manifests" / "manifest.json").read_text(encoding="utf-8"))
    baseline = Path(summary["source_variant_root"])
    result = verify_rasters(root)
    result["curve_source"] = summary["curve_source"]
    for split in ("train", "val"):
        original = (baseline / "manifests" / f"{split}.jsonl").open(encoding="utf-8")
        current = (root / "manifests" / f"{split}.jsonl").open(encoding="utf-8")
        with original, current:
            for index, (old_line, new_line) in enumerate(zip(original, current, strict=True)):
                old = json.loads(old_line)
                new = json.loads(new_line)
                if old["sample_id"] != new["sample_id"] or old["crop_xywh"] != new["crop_xywh"]:
                    raise ValueError(f"A/B sample or crop mismatch: {split} row {index}")
                if new["preprocess"].get("curve_source") != "gt":
                    raise ValueError(f"curve source is not GT: {split} row {index}")
                for name, old_path in old["files"].items():
                    new_path = new["files"][name]
                    if name == "lq":
                        if "/LQ_scurve/" not in new_path or Path(new_path).resolve() == Path(old_path).resolve():
                            raise ValueError(f"processed LQ did not move to the variant: {split} row {index}")
                    elif Path(new_path).resolve() != Path(old_path).resolve():
                        raise ValueError(f"shared asset changed: {split} row {index} {name}")
    result["ab_pairing"] = "passed"
    result["paired_loader_smoke"] = smoke_training_reader(root, result["counts"]["train"])
    return result


def main() -> int:
    """@luna-comment
    Input: GT-guided output root on the command line.
    Output: JSON verification report and process exit status.
    Key Steps: Run full paired checks and publish the report after success.
    Key Params: All line rasters and both split manifests are inspected.
    State/Dependencies: Writes only the final verification report under the output root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_ab(args.root)
    (args.root / "manifests" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
