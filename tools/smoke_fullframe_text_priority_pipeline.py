"""Exercise configuration, the extended train loop, resume and detection-free CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.make_fullframe_text_priority_configs import make_configs


def create_fixtures(directory):
    """@luna-comment
    Input: fresh repository-local evidence directory.
    Output: paired native RGB images, stroke masks and distinct train/val manifests.
    Key Steps: create mixed-polarity variable-width strokes and no-text/hard examples.
    Key Params: validation size 53x75 tests odd image inference; no source split overlap.
    State/Dependencies: synthetic OpenCV assets only; no production data/model changes.
    """
    directory.mkdir(parents=True)
    train, val = [], []
    for name, stratum, shape in (("train_text", "text", (64, 80)), ("train_hard", "hard", (64, 80)),
                                 ("train_other", "other", (64, 80)), ("val_text", "text", (53, 75))):
        h, w = shape
        gt = np.full((h, w, 3), 210, np.uint8)
        mask = np.zeros((h, w), np.uint8)
        instances = []
        if stratum != "other":
            gt[8:28, 8:28] = 240
            gt[12:24, 13:15] = 20
            gt[12:14, 13:24] = 20
            mask[12:24, 13:15] = 255
            mask[12:14, 13:24] = 255
            gt[28:48, 32:56] = 30
            gt[31:44, 36:40] = 230
            gt[31:35, 36:51] = 230
            mask[31:44, 36:40] = 255
            mask[31:35, 36:51] = 255
            instances = [{"id": "small", "bbox": [12, 11, 25, 25], "confidence": 1,
                          "orientation": "horizontal", "single_line": True, "char_height": 14},
                         {"id": "large", "bbox": [35, 30, 52, 45], "confidence": .8,
                          "orientation": "vertical", "single_line": True, "char_height": 24}]
        lq = cv2.GaussianBlur(gt, (3, 3), .6)
        for suffix, image in (("gt", gt), ("lq", lq), ("mask", mask)):
            if not cv2.imwrite(str(directory / f"{name}_{suffix}.png"), image):
                raise RuntimeError("Fixture image encoding failed")
        row = {"lq": f"{name}_lq.png", "gt": f"{name}_gt.png", "text_mask": f"{name}_mask.png",
               "source_id": name, "capture_group": name, "stratum": stratum,
               "annotation_status": "no_text" if stratum == "other" else "annotated", "instances": instances}
        (val if name.startswith("val") else train).append(row)
    for split, rows in (("train", train), ("val", val)):
        (directory / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return directory / "train.jsonl", directory / "val.jsonl", directory / "val_text_lq.png"


def run_command(command, log_path):
    """@luna-comment
    Input: argv array and output log path. Output: completed command evidence or error.
    Key Steps: run with bounded CPU threading, capture stdout/stderr, assert zero exit.
    Key Params: 300-second timeout bounds each short synthetic pipeline command.
    State/Dependencies: subprocess cwd stays at repository root; no shell interpolation.
    """
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=300)
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}), see {log_path}: {result.stderr[-1500:]}")
    if "Initializing recognizer" in result.stdout or "loading pretrained TPG" in result.stdout:
        raise AssertionError("Unexpected legacy OCR initialization in the new pipeline")
    return {"command": command, "exit_code": result.returncode, "log": str(log_path)}


def run_smoke(output):
    """@luna-comment
    Input: fresh evidence path inside this repository. Output: acceptance.json report.
    Key Steps: generate arms, run B for two steps, resume to step three, execute C2,
        run full-frame inference, and compare frozen old source/config hashes.
    Key Params: tiny V5 geometry/channels bound CPU cost; pilot uses explicit random init.
    State/Dependencies: only new local synthetic artifacts; old loop/builders are reused.
    """
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Choose a fresh output directory inside the repository")
    output.mkdir(parents=True)
    train, val, image = create_fixtures(output / "data")
    report = make_configs(train_manifest=train, val_manifest=val, root=train.parent,
                          output=output / "configs", experiments_root=output / "runs",
                          canvas_hw=(32, 48), halo=8, num_gpu=0, total_iter=2,
                          arms=("A", "B", "C1", "C2"))
    for path in report["configs"]:
        option_path = Path(path)
        options = yaml.safe_load(option_path.read_text(encoding="utf-8"))
        options["network_g"].update(stage_channels=[2, 8, 32], local_channels=4)
        options["datasets"]["train"].update(batch_size_per_gpu=2, num_worker_per_gpu=0)
        options["datasets"]["train"]["sampling"]["num_samples"] = 8
        options["logger"].update(print_freq=1, save_checkpoint_freq=1)
        options["val"].update(val_freq=1, metrics={"psnr": {"type": "calculate_psnr", "crop_border": 0}})
        option_path.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
    b_path = output / "configs/train_B.yml"
    commands = [run_command([sys.executable, "tools/train_fullframe_text_priority.py", "-opt", str(b_path)],
                            output / "train_B.log")]
    options = yaml.safe_load(b_path.read_text(encoding="utf-8"))
    run_root = output / "runs" / options["name"]
    if not (run_root / "training_states/2.state").is_file():
        raise AssertionError("The real loop did not save step-two training state")
    options["train"]["total_iter"] = 5
    b_path.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
    commands.append(run_command([sys.executable, "tools/train_fullframe_text_priority.py", "-opt", str(b_path),
                                 "--auto_resume"], output / "resume_B.log"))
    if not (run_root / "training_states/5.state").is_file():
        raise AssertionError("Resume did not advance from step two across an epoch to step five")
    continuous = dict(options)
    continuous["name"] = "fullframe_continuous_reference"
    continuous_path = output / "configs/train_continuous.yml"
    continuous_path.write_text(yaml.safe_dump(continuous, sort_keys=False), encoding="utf-8")
    commands.append(run_command([sys.executable, "tools/train_fullframe_text_priority.py", "-opt", str(continuous_path)],
                                 output / "continuous_B.log"))
    import torch
    resumed_parameters = torch.load(run_root / "models/net_g_5.pth", map_location="cpu", weights_only=True)["params"]
    continuous_parameters = torch.load(output / "runs" / continuous["name"] / "models/net_g_5.pth",
                                       map_location="cpu", weights_only=True)["params"]
    if not all(torch.equal(value, continuous_parameters[key]) for key, value in resumed_parameters.items()):
        raise AssertionError("Interrupted-and-resumed training differs from continuous training")
    c_path = output / "configs/train_C2.yml"
    commands.append(run_command([sys.executable, "tools/train_fullframe_text_priority.py", "-opt", str(c_path)],
                                 output / "train_C2.log"))
    c1_path = output / "configs/train_C1.yml"
    commands.append(run_command([sys.executable, "tools/train_fullframe_text_priority.py", "-opt", str(c1_path)],
                                 output / "train_C1.log"))
    path_hits = {}
    for log_name, field in (("train_B.log", "roi_roi_valid"), ("train_C2.log", "roi_effective_sdf"),
                            ("train_C1.log", "roi_effective_missing")):
        text = (output / log_name).read_text(encoding="utf-8")
        values = [float(value) for value in re.findall(re.escape(field) + r": ([0-9.eE+-]+)", text)]
        path_hits[field] = bool(values and max(values) > 0)
        if not path_hits[field]:
            raise AssertionError(f"Actual training never hit the required ROI path: {field}")
    checkpoint = run_root / "models/net_g_latest.pth"
    commands.append(run_command([sys.executable, "tools/infer_fullframe_text_priority.py", "--config", str(b_path),
                                 "--checkpoint", str(checkpoint), "--input", str(image),
                                 "--output", str(output / "inference"), "--device", "cpu"], output / "inference.log"))
    prediction = cv2.imread(str(output / "inference" / (image.name + ".restored.png")))
    if prediction is None or prediction.shape[:2] != (53, 75):
        raise AssertionError("Odd-sized full-frame output does not match input geometry")
    legacy_snapshot = ROOT / ".artifacts/fullframe_20261014/legacy_hashes.json"
    changed = []
    protected = []
    if legacy_snapshot.is_file():
        protected = json.loads(legacy_snapshot.read_text(encoding="utf-8-sig"))
        for item in protected:
            actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            if actual.upper() != item["sha256"].upper():
                changed.append(item["path"])
        if changed:
            raise AssertionError(f"Frozen legacy files changed: {changed}")
    result = {"passed": True, "commands": commands, "protected_files": len(protected),
              "changed_legacy_files": changed, "output_shape": list(prediction.shape),
              "continuous_resume_parameter_equality": True,
              "path_hits": path_hits,
              "checkpoint": str(checkpoint), "note": "Pipeline acceptance only, not production image quality."}
    (output / "acceptance.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv=None):
    """@luna-comment
    Input: --output fresh repository-local evidence path. Output: JSON acceptance report.
    Key Steps: parse and execute synthetic smoke with actual subprocess entry points.
    Key Params: output isolation protects existing evidence and production assets.
    State/Dependencies: invokes local implemented training/inference only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
