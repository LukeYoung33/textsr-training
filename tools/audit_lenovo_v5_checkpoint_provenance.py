"""Read-only provenance audit for the C0/R1 checkpoint texture scan.

The archived experiment YAML files contain one known invalid bare ``5000``
line.  This tool records the raw hash and line number, removes only that exact
line in memory, and never modifies the archived file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archived_yaml(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    removed = [i + 1 for i, line in enumerate(raw.splitlines()) if line.strip() == "5000"]
    if len(removed) != 1:
        raise ValueError(f"expected exactly one bare 5000 in {path}, got {removed}")
    cleaned = "\n".join(line for line in raw.splitlines() if line.strip() != "5000") + "\n"
    return yaml.safe_load(cleaned), {
        "path": str(path), "raw_sha256": sha256(path), "removed_bare_5000_lines": removed,
        "normalized_in_memory_sha256": hashlib.sha256(cleaned.encode()).hexdigest(),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        answer: dict[str, Any] = {}
        for key in sorted(value, key=str):
            answer.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return answer
    if isinstance(value, list):
        answer = {}
        for index, item in enumerate(value):
            answer.update(flatten(item, f"{prefix}[{index}]"))
        return answer
    return {prefix: value}


def semantic_diff(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    a, b = flatten(left), flatten(right)
    return [{"path": key, "left": a.get(key, "<ABSENT>"), "right": b.get(key, "<ABSENT>")}
            for key in sorted(set(a) | set(b)) if a.get(key, "<ABSENT>") != b.get(key, "<ABSENT>")]


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype),
                "numel": value.numel(), "scalar": value.item() if value.numel() == 1 else None}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(map(str, value.keys()))}
    return {"type": type(value).__name__}


def state_summary(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    result: dict[str, Any] = {"path": str(path), "sha256": sha256(path),
                              "top_level_keys": sorted(map(str, state.keys()))}
    for name in ("iter", "epoch"):
        if name in state:
            result[name] = scalar(state[name])
    optimizers = state.get("optimizers", [])
    result["optimizer_count"] = len(optimizers)
    result["optimizer_param_groups"] = [
        [{key: scalar(value) for key, value in group.items() if key != "params"} |
         {"parameter_count": len(group.get("params", []))} for group in optimizer.get("param_groups", [])]
        for optimizer in optimizers]
    schedulers = state.get("schedulers", [])
    result["scheduler_count"] = len(schedulers)
    result["scheduler_states"] = [{key: scalar(value) for key, value in scheduler.items()}
                                  for scheduler in schedulers]
    return result


def checkpoint_inventory(models: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(models.glob("net_g_[0-9]*.pth"), key=lambda p: int(p.stem.split("_")[-1])):
        state = torch.load(path, map_location="cpu")
        parameters = state.get("params", state)
        candidates = [(key, value) for key, value in parameters.items() if key.endswith("context_scale")]
        scales = {key: float(value.item()) for key, value in candidates}
        alpha = {key: 0.5 * float(torch.tanh(value).item()) for key, value in candidates
                 if key == "context_scale"}
        rows.append({"iteration": int(path.stem.split("_")[-1]), "path": str(path), "sha256": sha256(path),
                     "context_scale": scales, "eval_alpha": alpha})
    return rows


def log_evidence(path: Path) -> dict[str, Any]:
    patterns = re.compile(r"(seed|manual_seed|Random seed|lr:|optim|scheduler|model_type|recall|missing|false)", re.I)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = [line for line in lines if patterns.search(line)]
    return {"path": str(path), "sha256": sha256(path), "matching_lines": selected[:250],
            "matching_line_count": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-c0", type=Path, required=True)
    parser.add_argument("--actual-r1", type=Path, required=True)
    parser.add_argument("--repo-c0", type=Path, required=True)
    parser.add_argument("--repo-r1", type=Path, required=True)
    parser.add_argument("--c0-models", type=Path, required=True)
    parser.add_argument("--r1-models", type=Path, required=True)
    parser.add_argument("--c0-state", type=Path, required=True)
    parser.add_argument("--r1-state", type=Path, required=True)
    parser.add_argument("--c0-log", type=Path, required=True)
    parser.add_argument("--r1-log", type=Path, required=True)
    parser.add_argument("--lq", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    actual_c0, c0_meta = load_archived_yaml(args.actual_c0)
    actual_r1, r1_meta = load_archived_yaml(args.actual_r1)
    repo_c0 = yaml.safe_load(args.repo_c0.read_text(encoding="utf-8"))
    repo_r1 = yaml.safe_load(args.repo_r1.read_text(encoding="utf-8"))
    out = {
        "status": "COMPLETE", "scope": "read-only archived provenance audit",
        "archived_yaml": {"c0": c0_meta, "r1": r1_meta},
        "semantic_diffs": {
            "actual_c0_vs_actual_r1": semantic_diff(actual_c0, actual_r1),
            "actual_c0_vs_repo_c0": semantic_diff(actual_c0, repo_c0),
            "actual_r1_vs_repo_r1": semantic_diff(actual_r1, repo_r1),
        },
        "input_lq_sha256": sha256(args.lq),
        "checkpoint_inventory": {"c0": checkpoint_inventory(args.c0_models), "r1": checkpoint_inventory(args.r1_models)},
        "state_150000": {"c0": state_summary(args.c0_state), "r1": state_summary(args.r1_state)},
        "training_logs": {"c0": log_evidence(args.c0_log), "r1": log_evidence(args.r1_log)},
        "source_sha256": {str(path): sha256(path) for path in args.source},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "c0_r1_semantic_diffs": len(out["semantic_diffs"]["actual_c0_vs_actual_r1"])}))


if __name__ == "__main__":
    main()
