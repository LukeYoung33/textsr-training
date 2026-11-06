"""Run one real post-ramp P17/OCR trainer step without saving a checkpoint."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.losses  # noqa: F401
import realesrgan.models  # noqa: F401


def _number(value):
    return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-iter", type=int, default=7000)
    args = parser.parse_args()
    option = yaml.safe_load(Path(args.opt).read_text(encoding="utf-8"))
    dataset_opt = copy.deepcopy(option["datasets"]["train"])
    dataset_opt.update({"phase": "train", "batch_size_per_gpu": 1, "num_worker_per_gpu": 0})
    dataset = build_dataset(dataset_opt)
    sample = dataset[0]
    if not sample["structure_available"] or not sample["ocr_instance_id"]:
        raise RuntimeError("smoke did not receive a verified OCR line with a structure mask")
    batch = dataset.collate_fn([sample])
    runtime = copy.deepcopy(option)
    runtime.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1, "num_gpu": 1,
                    "root_path": str(Path(__file__).resolve().parents[1])})
    output = Path(args.output_dir)
    runtime["path"] = copy.deepcopy(runtime.get("path", {}))
    runtime["path"].update({"resume_state": None, "models": str(output / "models"),
                            "training_states": str(output / "states"), "log": str(output / "logs")})
    model = build_model(runtime)
    model.feed_data(batch)
    model.optimize_parameters(args.current_iter)
    logs = {key: _number(value) for key, value in model.get_current_log().items()}
    required = ("l_uv_strong", "l_missing", "l_false", "l_bg_lf", "l_cldice", "l_stroke_core")
    checks = {
        "finite": bool(torch.isfinite(model.output).all()) and all(math.isfinite(value) for value in logs.values()),
        "verified_ocr_line": bool(batch["structure_available"][0]) and bool(batch["ocr_instance_id"][0]),
        "auxiliary_losses_active": all(logs.get(key, 0.0) > 0.0 for key in required),
    }
    result = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
              "dataset_length": len(dataset), "line_id": batch["line_id"][0],
              "ocr_instance_id": batch["ocr_instance_id"][0], "logs": logs}
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
