"""Prepare matched GT-guided S-curve manifests and V5 training options."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


SOURCES = {
    "lenovo260926": Path("/home/guang.yang/data/lenovo_260926_validcrop_v4_all3_gt_guided_scurve_v1_20260928"),
    "lenovo20260920": Path("/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop_v4_gt_guided_scurve_v1_lines"),
}
OLD_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_mild_scurve_2x2_20260927")
NEW_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/lenovo_gt_guided_scurve_ab_20260928")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-base-dir", type=Path, required=True)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_dir = NEW_ROOT / "input_manifests"
    if manifest_dir.exists():
        raise FileExistsError(manifest_dir)
    manifest_dir.mkdir(parents=True)
    report = {"schema": "lenovo_gtguided_ab_training_v1", "datasets": {}, "experiment_root": str(NEW_ROOT)}
    counts: dict[str, dict[str, int]] = {}
    for source, root in SOURCES.items():
        verification = json.loads((root / "manifests" / "verification.json").read_text(encoding="utf-8"))
        if verification.get("status") != "passed":
            raise RuntimeError(f"source verification did not pass: {root}")
        report["datasets"][source] = {"root": str(root), "splits": {}}
        counts[source] = {}
        for split in ("train", "val"):
            manifest = root / "manifests" / f"{split}.jsonl"
            new_rows = rows(manifest)
            selected = []
            for row in new_rows:
                bbox = row.get("ocr_crop_xyxy")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"missing OCR bbox: {row.get('sample_id')}")
                if source == "lenovo260926" and float(bbox[3]) - float(bbox[1]) < 128:
                    continue
                if "LQ_scurve" not in str(row.get("files", {}).get("lq", "")):
                    raise ValueError(f"processed LQ missing: {row.get('sample_id')}")
                selected.append(row)
            old = rows(OLD_ROOT / "input_manifests" / f"{source}_{split}_scurve.jsonl")
            if len(selected) != len(old):
                raise ValueError(f"A/B selection count mismatch: {source}/{split}: {len(selected)} vs {len(old)}")
            for a, b in zip(selected, old):
                if a.get("sample_id") != b.get("sample_id") or a.get("ocr_crop_xyxy") != b.get("ocr_crop_xyxy"):
                    raise ValueError(f"A/B sample or OCR geometry mismatch: {source}/{split}: {a.get('sample_id')}")
            output = manifest_dir / f"{source}_{split}_gtguided.jsonl"
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                for row in selected:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[source][split] = len(selected)
            report["datasets"][source]["splits"][split] = {
                "source_manifest": str(manifest), "source_sha256": sha(manifest),
                "selected_manifest": str(output), "selected_sha256": sha(output),
                "source_count": len(new_rows), "selected_count": len(selected),
                "filtered_count": len(new_rows) - len(selected),
            }
    args.output_config_dir.mkdir(parents=True, exist_ok=True)
    for height in (96, 64):
        previous = args.config_base_dir / f"train_scurve_h{height}_250k_20260927.yml"
        config = yaml.safe_load(previous.read_text(encoding="utf-8"))
        name = f"lenovo_v5_r1_bglf_uv_gtguided_h{height}_250k_20260928"
        config["name"] = name
        config["path"]["experiments_root"] = str(NEW_ROOT)
        config["dist_params"]["port"] = 31200 + height
        for split in ("train", "val"):
            dataset = config["datasets"][split]
            dataset["name"] = f"{name}_{split}"
            dataset["epoch_size"] = sum(counts[source][split] for source in SOURCES)
            for source_opt in dataset["sources"]:
                source = source_opt["name"]
                source_opt["root"] = str(SOURCES[source])
                source_opt["manifest"] = str(manifest_dir / f"{source}_{split}_gtguided.jsonl")
                source_opt["weight"] = counts[source][split]
        if config["train"]["total_iter"] != 250000 or config["lq_preprocess"]["enabled"]:
            raise ValueError("inherited training contract changed")
        output = args.output_config_dir / f"train_gtguided_h{height}_250k_20260928.yml"
        output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        report.setdefault("configs", []).append({"height": height, "local_path": str(output), "name": name})
    (manifest_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
