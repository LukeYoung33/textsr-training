"""Build an offline gallery of four model arms and their validation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


ARMS = ("scurve_h96", "scurve_h64", "raw_h96", "raw_h64")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records: dict[tuple[str, str, str], dict] = {}
    for arm in ARMS:
        summary = json.loads((root / arm / "summary.json").read_text(encoding="utf-8"))
        if summary["passed"] != 231 or summary["failed"]:
            raise ValueError(f"incomplete arm: {arm}")
        for row in summary["rows"]:
            remote = PurePosixPath(row["output"])
            parts = remote.parts
            marker = parts.index(root.name)
            relative = PurePosixPath(*parts[marker + 1:])
            dataset, source, sample = relative.parts[1:4]
            key = (dataset, source, sample)
            record = records.setdefault(key, {"dataset": dataset, "source": source, "sample": sample, "arms": {}})
            arm_dir = relative.parent
            output = root / Path(*relative.parts)
            comparison = root / Path(*arm_dir.parts) / "comparison_lq_used_output_gt.jpg"
            if not output.is_file() or not comparison.is_file():
                raise FileNotFoundError(output if not output.is_file() else comparison)
            record["arms"][arm] = {"output": (PurePosixPath(root.name) / relative).as_posix(),
                                    "comparison": (PurePosixPath(root.name) / arm_dir / "comparison_lq_used_output_gt.jpg").as_posix(),
                                    "lines": row["lines_processed"]}
    if len(records) != 231 or any(len(record["arms"]) != 4 for record in records.values()):
        raise ValueError("four arms do not contain the same 231 parent images")
    input_root = root.parent / "inputs_gallery"
    for record in records.values():
        relative = Path(record["dataset"]) / record["source"] / record["sample"]
        sources = {
            "lq": input_root / relative / "lq_original.png",
            "lq_scurve_h96": root / "scurve_h96" / relative / "lq_used.png",
            "lq_scurve_h64": root / "scurve_h64" / relative / "lq_used.png",
            "gt": input_root / relative / "gt.png",
        }
        for name, path in sources.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            record["arms"][name] = {"output": path.relative_to(root.parent).as_posix()}
    data = json.dumps([records[key] for key in sorted(records)], ensure_ascii=False).replace("</", "<\\/")
    html = Path(__file__).with_name("lenovo_scurve_2x2_gallery.html").read_text(encoding="utf-8")
    (root.parent / "index.html").write_text(html.replace("__DATA__", data), encoding="utf-8")
    print(root.parent / "index.html")


if __name__ == "__main__":
    main()
