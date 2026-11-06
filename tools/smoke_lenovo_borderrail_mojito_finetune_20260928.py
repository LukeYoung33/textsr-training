"""Check all eight finetune configs and decode real whiteboard lines."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_lenovo_borderrail_mojito_finetune_configs_20260928 import FINETUNE_OUTPUT
from realesrgan.data.basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset


def main() -> None:
    report = []
    with tempfile.TemporaryDirectory(prefix="mojito_finetune_smoke_") as temp:
        for config_path in sorted(FINETUNE_OUTPUT.glob("*.yml")):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            assert config["train"]["total_iter"] == 100000
            assert "val" not in config and "val" not in config["datasets"]
            dataset_opt = copy.deepcopy(config["datasets"]["train"])
            source = dataset_opt["sources"][0]
            lines = Path(source["manifest"]).read_text(encoding="utf-8").splitlines()
            assert len(lines) == 405
            sampled = Path(temp) / f"{config_path.stem}.jsonl"
            sampled.write_text(lines[0] + "\n" + lines[-1] + "\n", encoding="utf-8")
            source["manifest"] = str(sampled)
            source["weight"] = 2
            dataset_opt["epoch_size"] = 2
            dataset = BasicTextLineRemote58Dataset(dataset_opt)
            for index in range(2):
                item = dataset[index]
                height = config["network_g"]["canvas_hw"][0]
                assert tuple(item["lq"].shape) == (1, height * 3 // 2, 480)
                assert tuple(item["gt"].shape) == (3, height, 480)
                assert torch.isfinite(item["lq"]).all() and torch.isfinite(item["gt"]).all()
                report.append({"config": config_path.name, "sample": item["line_id"]})
    print(json.dumps({"passed": True, "decoded": len(report), "samples": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
