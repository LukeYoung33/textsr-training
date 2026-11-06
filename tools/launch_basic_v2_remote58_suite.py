"""Launch one unattended Remote58 BASIC V2 training queue.

Each invocation owns one visible GPU and runs the selected jobs sequentially.
Every successful train is followed by exactly one two-source validation and
one frozen HX test stage.  A failed job is recorded and does not prevent the
next queued job from starting.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_basic_v2_post_eval_contract import build_contract
from tools.run_unattended_training import run_unattended


PYTHON = Path("/home/guang.yang/ViT_env/bin/python")
EXPERIMENTS_ROOT = Path("/home/guang.yang/workspace/Real-ESRGAN/experiments")
DEFAULT_SUITE_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/unattended/basic_v2_remote58_96x480_20260831")
HX_ROOT = Path("/AI_DNS/guang.yang/TextSR/data/Artguru_textHD/HX_testset_training")
HX_MANIFEST = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/basic_text_line_v3_hx250k_common_20260817/segments.jsonl")
HX_SUMMARY = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/basic_text_line_v3_hx250k_common_20260817/summary.json")
DATA_ROOTS = {
    "v2": Path("/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"),
    "real": Path("/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune"),
}
RUN_CONFIGS = {
    "gtdeg": Path("options/train_bucket_3/train_basic_v2_gtdeg_real_finetune_96x480_20260831.yml"),
    "paired": Path("options/train_bucket_3/train_basic_v2_real_finetune_96x480_20260831.yml"),
    "large10": Path("options/train_bucket_3/train_basic_v2_large_10x_real_finetune_96x480_20260831.yml"),
    "large20": Path("options/train_bucket_3/train_basic_v2_large_20x_real_finetune_96x480_20260831.yml"),
    "large50": Path("options/train_bucket_3/train_basic_v2_large_50x_real_finetune_96x480_20260831.yml"),
}


def _q(value: Path | str) -> str:
    return shlex.quote(str(value))


def _suite_run_name(name: str) -> str:
    """@guang-yang-comment Keep the new config name unchanged inside its suite root."""
    return name


def _validation_document(train_config: dict[str, Any], checkpoint: Path, suite_root: Path, run_name: str | None = None) -> dict[str, Any]:
    """@guang-yang-comment Build a frozen two-source post-training validation YAML."""
    name = str(run_name or _suite_run_name(str(train_config["name"])))
    network = deepcopy(train_config["network_g"])
    network["canvas_hw"] = [96, 480]
    common = {
        "type": "BasicTextLineRemote58Dataset",
        "split": "val",
        "canvas_hw": [96, 480],
        "resize_mode": "stretch",
        "batch_size_per_gpu": 1,
        "num_worker_per_gpu": 0,
    }
    return {
        "name": f"{name}_post_validation",
        "model_type": "BasicTextLineTrainingModelV3",
        "scale": 1,
        "num_gpu": 1,
        "manual_seed": int(train_config.get("manual_seed", 20260829)),
        "color": "nv21",
        "network_g": network,
        "path": {
            "pretrain_network_g": str(checkpoint),
            "strict_load_g": True,
            "results_root": str(suite_root / name / "validation_results"),
        },
        "datasets": {
            "val_v2": {"name": "real_finetune_v2_val", "root": str(DATA_ROOTS["v2"]), **common},
            "val_real": {"name": "real_finetune_val", "root": str(DATA_ROOTS["real"]), **common},
        },
        "val": {"save_img": True, "suffix": None, "metrics": {}},
        "dist_params": {"backend": "nccl", "port": 29540},
    }


def _prepare_run(key: str, suite_root: Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """@guang-yang-comment Resolve one train config and materialize its validation contract."""
    train_path = (REPO_ROOT / RUN_CONFIGS[key]).resolve()
    train_config = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    name = str(train_config["name"])
    run_name = _suite_run_name(name)
    configured_canvas = train_config.get("network_g", {}).get("canvas_hw")
    if configured_canvas is None or [int(value) for value in configured_canvas] != [96, 480]:
        raise ValueError(f"{key} is not a 96x480 training config: canvas_hw={configured_canvas}")
    checkpoint = EXPERIMENTS_ROOT / name / "models" / "net_g_250000.pth"
    run_root = suite_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    validation_path = run_root / "generated_validation.yml"
    validation_path.write_text(
        yaml.safe_dump(_validation_document(train_config, checkpoint, suite_root, run_name), sort_keys=False),
        encoding="utf-8",
    )
    contract = build_contract(
        validation_path,
        checkpoint,
        train_path,
        HX_MANIFEST,
        HX_SUMMARY,
        250000,
        HX_ROOT / "lq",
        HX_ROOT / "ocr",
        expected_segments=669,
        expected_images=52,
        device="cuda",
        python_command=str(PYTHON),
    )
    (run_root / "post_eval_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return train_path, train_config, validation_path, contract


def run_queue(run_keys: Sequence[str], suite_root: Path, retry_failed: bool = False) -> int:
    """@guang-yang-comment Execute every requested run and retain a queue summary."""
    suite_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    queue_status_path = suite_root / f"queue_status_{'-'.join(run_keys)}.json"
    for key in run_keys:
        train_path, train_config, validation_path, contract = _prepare_run(key, suite_root)
        name = _suite_run_name(str(train_config["name"]))
        train_command = (
            f"{_q(PYTHON)} realesrgan/train.py -opt {_q(train_path)} --launcher none"
        )
        code = run_unattended(
            suite_root / name,
            train_command,
            contract["validation"]["command"],
            contract["testset"]["command"],
            config_paths=[train_path, validation_path],
            testset_id="HX_testset_training_669_segments_52_frames",
            testset_manifest=HX_MANIFEST,
            cwd=REPO_ROOT,
            env={"PYTHONUNBUFFERED": "1"},
            retry_failed=retry_failed,
        )
        outcomes.append({"key": key, "name": name, "exit_code": int(code)})
        queue_status_path.write_text(
            json.dumps({"pid": os.getpid(), "outcomes": outcomes}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if all(item["exit_code"] == 0 for item in outcomes) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", choices=sorted(RUN_CONFIGS), required=True)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    return run_queue(args.runs, args.suite_root.expanduser().resolve(), retry_failed=args.retry_failed)


if __name__ == "__main__":
    raise SystemExit(main())
