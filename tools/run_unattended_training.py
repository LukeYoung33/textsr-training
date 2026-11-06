"""Run training followed by one validation and one frozen-testset pass.

The wrapper is intentionally evaluator-agnostic.  Commands receive
``AUTO_EVAL_RUN_ROOT``, ``AUTO_EVAL_STAGE``, and ``AUTO_EVAL_STAGE_ROOT`` so
the training/evaluation entry points can place all outputs inside the run
bundle.  A stage is terminal after its first attempt (success or failure),
which makes a restart safe and prevents completed visualizations from being
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


STAGES = ("train", "validation", "testset")
TERMINAL_STAGES = ("succeeded", "failed")
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXIT_RESTART_BLOCKED = 75


def utc_now() -> str:
    """@guang-yang-comment Return an ISO-8601 UTC timestamp for run records.

    Input: none. Output: second-resolution UTC timestamp with a trailing ``Z``.
    Key Steps: read the system clock and normalize it to UTC. Key Params: none.
    State/Dependencies: system clock only.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """@guang-yang-comment Hash one file in fixed-size chunks for provenance.

    Input: an existing regular file. Output: lowercase SHA-256 digest. Key
    Steps: stream 1 MiB chunks so large checkpoints do not fill memory. Key
    Params: chunk size is fixed at 1 MiB. State/Dependencies: filesystem read.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    """@guang-yang-comment Atomically persist a JSON status or marker file.

    Input: destination path and JSON-compatible mapping. Output: replaced file
    containing deterministic indented JSON. Key Steps: write a sibling temporary
    file, flush it, then use ``os.replace``. Key Params: UTF-8 and sorted keys.
    State/Dependencies: destination directory must be writable.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _read_json(path: Path) -> Dict[str, object]:
    """@guang-yang-comment Read a previously persisted wrapper state mapping.

    Input: existing UTF-8 JSON status file. Output: decoded mutable mapping.
    Key Steps: parse JSON and reject non-object state. Key Params: none.
    State/Dependencies: filesystem read and valid JSON schema.
    """

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _command_display(command: str) -> str:
    """@guang-yang-comment Normalize a command for human-readable manifests.

    Input: shell command string. Output: shell-quoted display string where
    possible. Key Steps: tokenize with platform-appropriate shell rules and
    re-quote tokens. Key Params: POSIX quoting is used on Remote58/Linux.
    State/Dependencies: malformed quoting falls back to the original command.
    """

    try:
        return " ".join(shlex.quote(token) for token in shlex.split(command, posix=os.name != "nt"))
    except ValueError:
        return command


def _artifact_manifest(root: Path) -> Dict[str, object]:
    """@guang-yang-comment Capture hashes and visualization counts for a stage.

    Input: completed stage directory. Output: manifest with every regular file,
    byte size, SHA-256, and image count. Key Steps: deterministic relative-path
    traversal and chunked hashing. Key Params: image suffix set is explicit.
    State/Dependencies: stage files are expected to be immutable after return.
    """

    files: List[Dict[str, object]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in {"artifact_manifest.json", "stage_complete.json", "stage_failed.json"}:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    visuals = [item for item in files if Path(str(item["path"])).suffix.lower() in IMAGE_SUFFIXES]
    return {"file_count": len(files), "visualization_count": len(visuals), "files": files}


def _snapshot_tree(source: Path, destination: Path) -> None:
    """@guang-yang-comment Copy evaluator outputs into an immutable attempt tree.

    Input: completed source directory and empty destination directory. Output:
    recursively copied files preserving metadata. Key Steps: reject an existing
    destination file, then copy each file with ``copy2``. Key Params: no source
    deletion or replacement is allowed. State/Dependencies: source is local.
    """

    if not source.exists():
        return
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite artifact {destination}")
        shutil.copy2(source, destination)
        return
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite artifact {target}")
            shutil.copy2(item, target)


def _copy_configs(run_root: Path, config_paths: Iterable[Path]) -> List[Dict[str, object]]:
    """@guang-yang-comment Preserve config copies and their source hashes.

    Input: run root and config paths supplied by the launcher. Output: metadata
    records containing source/copy paths, byte counts, and SHA-256. Key Steps:
    copy each config exactly once, disambiguating duplicate basenames. Key Params:
    config copies are never overwritten. State/Dependencies: source files exist.
    """

    config_root = run_root / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for index, source in enumerate(config_paths, start=1):
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Config file does not exist: {source}")
        target = config_root / source.name
        if target.exists():
            target = config_root / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        records.append({"source": str(source), "copy": str(target.relative_to(run_root)), "bytes": source.stat().st_size, "sha256": sha256_file(source)})
    return records


def _new_state(run_root: Path, run_id: str, configs: List[Dict[str, object]], testset_id: str, testset_manifest: Optional[Path]) -> Dict[str, object]:
    """@guang-yang-comment Build the initial run state and provenance manifest.

    Input: run identity, copied config records, and frozen testset metadata.
    Output: schema-versioned state with all stages pending. Key Steps: record
    host/time, testset hash, and immutable command-independent metadata. Key
    Params: three fixed stage names. State/Dependencies: optional manifest file.
    """

    testset_record: Dict[str, object] = {"id": testset_id}
    if testset_manifest is not None:
        manifest = testset_manifest.expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Testset manifest does not exist: {manifest}")
        testset_record.update({"path": str(manifest), "bytes": manifest.stat().st_size, "sha256": sha256_file(manifest)})
    now = utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "RUNNING",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": now,
        "updated_at": now,
        "configs": configs,
        "frozen_testset": testset_record,
        "stages": {stage: {"status": "pending", "attempts": []} for stage in STAGES},
    }


def _save_state(run_root: Path, state: Dict[str, object]) -> None:
    """@guang-yang-comment Update the authoritative run status manifest.

    Input: mutable wrapper state mapping. Output: atomic ``status.json`` update.
    Key Steps: refresh timestamp and serialize through the atomic writer. Key
    Params: status path is fixed at run root. State/Dependencies: local disk.
    """

    state["updated_at"] = utc_now()
    _write_json_atomic(run_root / "status.json", state)


def _stage_status(state: Mapping[str, object], stage: str) -> str:
    """@guang-yang-comment Return one stage's persisted status value.

    Input: wrapper state and one name from ``STAGES``. Output: pending/running or
    terminal status. Key Steps: validate the stage mapping and default to pending.
    Key Params: fixed stage names prevent accidental later-stage execution.
    State/Dependencies: state schema version 1.
    """

    stages = state.get("stages", {})
    record = stages.get(stage, {}) if isinstance(stages, dict) else {}
    return str(record.get("status", "pending")) if isinstance(record, dict) else "pending"


def _run_stage(
    run_root: Path,
    state: Dict[str, object],
    stage: str,
    command: str,
    artifact_source: Optional[Path],
    base_env: Mapping[str, str],
    cwd: Optional[Path],
) -> int:
    """@guang-yang-comment Execute exactly one terminal stage attempt.

    Input: pending stage, shell command, optional output source, environment, and
    working directory. Output: process exit code plus immutable log/manifest and
    marker files. Key Steps: create unique attempt dir, persist running state,
    stream output, snapshot artifacts, then persist success/failure. Key Params:
    no stage is retried after any terminal result. State/Dependencies: subprocess
    and filesystem; caller enforces train -> validation -> testset ordering.
    """

    current = _stage_status(state, stage)
    if current in TERMINAL_STAGES:
        record = state["stages"][stage]
        attempt = record["attempts"][-1]
        return int(attempt.get("exit_code", EXIT_RESTART_BLOCKED))
    if current == "running":
        return EXIT_RESTART_BLOCKED

    stage_dir = run_root / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(stage_dir.glob("attempt-*"))
    attempt_number = len(existing) + 1
    attempt_root = stage_dir / f"attempt-{attempt_number:03d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    log_path = attempt_root / "command.log"
    stage_env = dict(base_env)
    stage_env.update({
        "AUTO_EVAL_RUN_ROOT": str(run_root),
        "AUTO_EVAL_STAGE": stage,
        "AUTO_EVAL_STAGE_ROOT": str(attempt_root),
        "REAL_ESRGAN_DEFER_FINAL_VALIDATION": "1" if stage == "train" else stage_env.get("REAL_ESRGAN_DEFER_FINAL_VALIDATION", "0"),
    })
    record: Dict[str, object] = {
        "attempt": attempt_number,
        "status": "running",
        "command": command,
        "command_display": _command_display(command),
        "started_at": utc_now(),
        "pid": None,
        "attempt_root": str(attempt_root.relative_to(run_root)),
        "log": str(log_path.relative_to(run_root)),
    }
    state["stages"][stage]["status"] = "running"
    state["stages"][stage]["attempts"].append(record)
    _save_state(run_root, state)

    return_code = 127
    try:
        with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as log_handle:
            log_handle.write(f"COMMAND: {command}\n")
            log_handle.write(f"STAGE_ROOT: {attempt_root}\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                env=stage_env,
                shell=True,
                executable=None if os.name == "nt" else "/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            record["pid"] = process.pid
            _save_state(run_root, state)
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
            return_code = process.wait()
    except Exception as exc:  # pragma: no cover - exercised by environment failures
        with log_path.open("a", encoding="utf-8", errors="replace") as log_handle:
            log_handle.write(f"\nWRAPPER_EXCEPTION: {type(exc).__name__}: {exc}\n")
        record["wrapper_exception"] = f"{type(exc).__name__}: {exc}"
        return_code = 1

    record["ended_at"] = utc_now()
    record["exit_code"] = return_code
    record["status"] = "succeeded" if return_code == 0 else "failed"
    if artifact_source is not None:
        source = artifact_source.expanduser().resolve()
        if source != attempt_root.resolve():
            _snapshot_tree(source, attempt_root / "artifacts")
    manifest = _artifact_manifest(attempt_root)
    record["artifacts"] = manifest
    _write_json_atomic(attempt_root / "artifact_manifest.json", manifest)
    marker_name = "stage_complete.json" if return_code == 0 else "stage_failed.json"
    marker = {"stage": stage, "status": record["status"], "exit_code": return_code, "attempt": attempt_number, "completed_at": record["ended_at"], "artifacts": manifest}
    _write_json_atomic(attempt_root / marker_name, marker)
    state["stages"][stage]["status"] = record["status"]
    if return_code != 0:
        state["status"] = "FAILED"
        state["failed_stage"] = stage
        state["failure_exit_code"] = return_code
    _save_state(run_root, state)
    return return_code


def run_unattended(
    run_root: Path,
    train_command: str,
    validation_command: str,
    test_command: str,
    config_paths: Sequence[Path] = (),
    testset_id: str = "frozen-testset",
    testset_manifest: Optional[Path] = None,
    artifact_sources: Optional[Mapping[str, Optional[Path]]] = None,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    retry_failed: bool = False,
) -> int:
    """@guang-yang-comment Drive the failure-safe train/evaluate state machine.

    Input: unique run root and three commands; optional configs, frozen testset,
    artifact roots, cwd, and environment. Output: 0 only after all three stages
    succeed and ``COMPLETE`` is persisted; nonzero on any failure. Key Steps:
    initialize/reload state, skip successful stages, and gate each later stage
    on the prior success. Key Params: an explicit retry may reset only a failed
    validation/test stage while preserving prior attempts. State/Dependencies:
    status.json is the restart authority.
    """

    run_root = run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    if status_path.exists():
        state = _read_json(status_path)
        if state.get("status") == "COMPLETE":
            return 0
        if state.get("status") == "FAILED":
            if not retry_failed:
                return int(state.get("failure_exit_code", 1)) or 1
            failed_stage = str(state.get("failed_stage", ""))
            if failed_stage not in {"validation", "testset"}:
                return int(state.get("failure_exit_code", 1)) or 1
            failed_index = STAGES.index(failed_stage)
            if any(_stage_status(state, stage) != "succeeded" for stage in STAGES[:failed_index]):
                return EXIT_RESTART_BLOCKED
            failed_record = state["stages"][failed_stage]
            if not isinstance(failed_record, dict) or failed_record.get("status") != "failed":
                return EXIT_RESTART_BLOCKED
            recovery_events = state.setdefault("recovery_events", [])
            if not isinstance(recovery_events, list):
                return EXIT_RESTART_BLOCKED
            recovery_events.append({
                "at": utc_now(),
                "stage": failed_stage,
                "prior_exit_code": int(state.get("failure_exit_code", 1)),
            })
            failed_record["status"] = "pending"
            state["status"] = "RUNNING"
            state.pop("failed_stage", None)
            state.pop("failure_exit_code", None)
            _save_state(run_root, state)
    else:
        configs = _copy_configs(run_root, config_paths)
        run_id = run_root.name
        state = _new_state(run_root, run_id, configs, testset_id, testset_manifest)
        _save_state(run_root, state)
        _write_json_atomic(run_root / "run_meta.json", {"schema_version": 1, "run_id": run_id, "created_at": state["created_at"], "configs": configs, "frozen_testset": state["frozen_testset"], "wrapper": str(Path(__file__).resolve())})

    base_env = dict(os.environ)
    if env:
        base_env.update(env)
    sources = dict(artifact_sources or {})
    stages_and_commands = (("train", train_command), ("validation", validation_command), ("testset", test_command))
    for stage, command in stages_and_commands:
        if state.get("status") == "FAILED":
            return int(state.get("failure_exit_code", 1)) or 1
        code = _run_stage(run_root, state, stage, command, sources.get(stage), base_env, cwd)
        if code != 0:
            return code

    state["status"] = "COMPLETE"
    state["completed_at"] = utc_now()
    _save_state(run_root, state)
    _write_json_atomic(run_root / "completion_manifest.json", {"status": "COMPLETE", "run_id": state["run_id"], "completed_at": state["completed_at"], "stages": {stage: state["stages"][stage] for stage in STAGES}})
    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """@guang-yang-comment Parse the unattended-run command-line contract.

    Input: optional argv sequence. Output: validated argparse namespace. Key
    Steps: require three stage commands and parse optional provenance paths. Key
    Params: one run root, optional repeated config paths. State/Dependencies:
    argparse only.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--train-command", required=True)
    parser.add_argument("--validation-command", required=True)
    parser.add_argument("--test-command", required=True)
    parser.add_argument("--config", action="append", default=[], type=Path)
    parser.add_argument("--testset-id", default="frozen-testset")
    parser.add_argument("--testset-manifest", type=Path)
    parser.add_argument("--train-artifacts", type=Path)
    parser.add_argument("--validation-artifacts", type=Path)
    parser.add_argument("--test-artifacts", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """@guang-yang-comment Launch the CLI state machine and return its exit code.

    Input: command-line arguments accepted by ``_parse_args``. Output: process
    exit code suitable for nohup/systemd supervision. Key Steps: parse args and
    bind stage artifact sources. Key Params: failures propagate unchanged.
    State/Dependencies: local run-root and subprocess commands.
    """

    args = _parse_args(argv)
    sources = {"train": args.train_artifacts, "validation": args.validation_artifacts, "testset": args.test_artifacts}
    return run_unattended(
        args.run_root,
        args.train_command,
        args.validation_command,
        args.test_command,
        args.config,
        args.testset_id,
        args.testset_manifest,
        sources,
        args.cwd,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
