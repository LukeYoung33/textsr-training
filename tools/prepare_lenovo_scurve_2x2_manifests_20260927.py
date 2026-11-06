"""Create immutable paired-LQ 2x2 manifests from the two materialized line sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    "lenovo260926": Path("/home/guang.yang/data/lenovo_260926_validcrop_v4_all3_mild_scurve_v1_20260927"),
    "lenovo20260920": Path("/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop_v4_mild_scurve_v1_lines"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"manifest output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    report = {"schema": "lenovo_mild_scurve_2x2_manifests_v1", "datasets": {}}
    for source_name, root in SOURCES.items():
        dataset_report = {"root": str(root), "splits": {}}
        verification = json.loads((root / "manifests" / "verification.json").read_text(encoding="utf-8"))
        if verification.get("status") != "passed":
            raise RuntimeError(f"source verification failed: {root}")
        for split in ("train", "val"):
            source_manifest = root / "manifests" / f"{split}.jsonl"
            source_sha256 = digest(source_manifest)
            selected = []
            total = 0
            for line in source_manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                bbox = row.get("ocr_crop_xyxy")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"missing crop-local OCR bbox: {source_manifest}:{total}")
                height = float(bbox[3]) - float(bbox[1])
                if source_name == "lenovo260926" and height < 128:
                    continue
                files = row.get("files", {})
                if "LQ_scurve" not in str(files.get("lq", "")):
                    raise ValueError(f"processed LQ missing: {source_manifest}:{total}")
                if "LQ_raw" not in str(files.get("lq_raw", "")):
                    raise ValueError(f"raw LQ missing: {source_manifest}:{total}")
                selected.append(row)
            if not selected:
                raise ValueError(f"empty selected split: {source_name}/{split}")
            for mode in ("scurve", "raw"):
                output = args.output_dir / f"{source_name}_{split}_{mode}.jsonl"
                with output.open("w", encoding="utf-8", newline="\n") as handle:
                    for source_row in selected:
                        row = dict(source_row)
                        files = dict(row["files"])
                        if mode == "raw":
                            files["lq"] = files["lq_raw"]
                        row["files"] = files
                        row.pop("lq", None)
                        row["input_selection"] = {
                            "mode": mode,
                            "source_manifest_sha256": source_sha256,
                            "min_ocr_height": 128 if source_name == "lenovo260926" else None,
                        }
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                dataset_report["splits"].setdefault(split, {})[mode] = {
                    "manifest": str(output), "sha256": digest(output), "selected": len(selected),
                    "source_total": total, "filtered_out": total - len(selected),
                }
            dataset_report["splits"][split]["source_sha256"] = source_sha256
        report["datasets"][source_name] = dataset_report
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
