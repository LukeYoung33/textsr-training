"""Decode one real pair from each source of all eight border-rail train configs."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_lenovo_borderrail_8arm_configs_20260928 import OUTPUT
from realesrgan.data.basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset


def main() -> None:
    report = []
    with tempfile.TemporaryDirectory(prefix="lenovo_borderrail_reader_") as temporary:
        directory = Path(temporary)
        for config_path in sorted(OUTPUT.glob("*.yml")):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            height = config["network_g"]["canvas_hw"][0]
            for split in ("train", "val"):
                opt = copy.deepcopy(config["datasets"][split])
                opt["epoch_size"] = len(opt["sources"])
                for index, source in enumerate(opt["sources"]):
                    manifest = Path(source["manifest"])
                    with manifest.open(encoding="utf-8") as handle:
                        first = handle.readline()
                    if not first:
                        raise ValueError(f"empty manifest: {manifest}")
                    sampled = directory / f"{config_path.stem}_{split}_{index}.jsonl"
                    sampled.write_text(first, encoding="utf-8")
                    source["manifest"] = str(sampled)
                    source["weight"] = 1
                dataset = BasicTextLineRemote58Dataset(opt)
                if len(dataset) != 2:
                    raise ValueError(f"unexpected sampled size: {config_path}/{split}")
                for index in range(2):
                    item = dataset[index]
                    if (tuple(item["lq"].shape) != (1, height * 3 // 2, 480)
                            or tuple(item["gt"].shape) != (3, height, 480)
                            or tuple(item["text_mask"].shape) != (1, height, 480)
                            or not torch.isfinite(item["lq"]).all()
                            or not torch.isfinite(item["gt"]).all()):
                        raise ValueError(f"invalid reader output: {config_path}/{split}/{index}")
                    report.append({"config": config_path.name, "split": split,
                                   "source": item["source_name"], "line_id": item["line_id"],
                                   "lq_shape": list(item["lq"].shape), "gt_shape": list(item["gt"].shape)})
    print(json.dumps({"passed": True, "pairs_decoded": len(report), "samples": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
