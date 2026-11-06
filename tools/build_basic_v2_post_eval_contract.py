"""Generate ST-04 commands for one validation and one frozen HX testset pass.

The generated commands are intended for ``run_unattended_training.py``.  They
write under ``$AUTO_EVAL_STAGE_ROOT`` and use ``&&`` for testset short-circuiting.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml


STAGE_ROOT = '"$AUTO_EVAL_STAGE_ROOT"'


def _q(value: Path | str) -> str:
    """@guang-yang-comment Quote a fixed path for a generated POSIX command.

    Input: local or Remote58 path. Output: shell-safe single-quoted token.
    Key Steps: delegate quoting to ``shlex``. Key Params: commands target bash.
    State/Dependencies: none.
    """

    return shlex.quote(str(value))


def _validation_names(config: Path) -> list[str]:
    """@guang-yang-comment Extract validation dataset names from a YAML config.

    Input: training/test YAML containing a mapping-valued ``datasets`` section.
    Output: names whose key or phase denotes validation. Key Steps: parse YAML,
    inspect mapping entries, and preserve declaration order. Key Params: at
    least two source-specific validation entries are required. State/Dependencies:
    PyYAML and a readable config.
    """

    with config.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    datasets = document.get("datasets") if isinstance(document, Mapping) else None
    if not isinstance(datasets, Mapping):
        raise ValueError(f"config must contain mapping datasets: {config}")
    names = []
    for name, value in datasets.items():
        phase = value.get("phase") if isinstance(value, Mapping) else None
        if str(name).split("_")[0] == "val" or str(phase).split("_")[0] == "val":
            names.append(str(name))
    if len(names) < 2:
        raise ValueError(f"validation config must expose both source datasets; found {names}")
    return names


def build_contract(
    validation_config: Path,
    checkpoint: Path,
    network_config: Path,
    hx_manifest: Path,
    hx_manifest_summary: Optional[Path],
    hx_iteration: int,
    full_input_dir: Path,
    full_ocr_dir: Path,
    expected_segments: int = 669,
    expected_images: int = 52,
    device: str = "cuda",
    python_command: str = "python",
    resize_mode: str = "stretch",
) -> dict[str, Any]:
    """@guang-yang-comment Build the machine-readable post-evaluation contract.

    Input: two-source validation config, Large-compatible checkpoint/config, HX
    manifest, and 52-image full-frame inputs. Output: commands and expected
    counts for wrapper integration. Key Steps: validate source count and compose
    one save-image validation command plus a short-circuit test command. Key
    Params: default frozen counts are 669 segments and 52 full images. State/
    Dependencies: generated commands require the repository root as cwd.
    """

    validation_config = validation_config.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    network_config = network_config.expanduser().resolve()
    hx_manifest = hx_manifest.expanduser().resolve()
    if hx_manifest_summary is not None:
        hx_manifest_summary = hx_manifest_summary.expanduser().resolve()
    names = _validation_names(validation_config)
    with network_config.open("r", encoding="utf-8") as handle:
        network_document = yaml.safe_load(handle)
    configured_canvas = (network_document.get("network_g", {}).get("canvas_hw")
                         if isinstance(network_document, Mapping) else None)
    if configured_canvas is not None and [int(value) for value in configured_canvas] != [96, 480]:
        raise ValueError(f"post-evaluation network canvas must be [96, 480], got {configured_canvas}")
    resize_mode = str(resize_mode).strip().lower()
    if resize_mode != "stretch":
        raise ValueError("the 96x480 post-evaluation contract requires resize_mode=stretch")
    python = python_command
    validation_command = (
        f"{python} realesrgan/test.py -opt {_q(validation_config)} --launcher none "
        f"--force_yml val:save_img=true path:results_root=\"$AUTO_EVAL_STAGE_ROOT/validation\""
    )
    summary_arg = f" --manifest-summary {_q(hx_manifest_summary)}" if hx_manifest_summary else ""
    test_command = (
        f"{python} tools/dump_hx_basic_text_lines_v3.py --manifest {_q(hx_manifest)}"
        f"{summary_arg} --checkpoint {_q(checkpoint)} --network-config {_q(network_config)}"
        f" --output-root \"$AUTO_EVAL_STAGE_ROOT/hx_segments\" --iteration {int(hx_iteration)}"
        f" --device {_q(device)} --expected-segments {int(expected_segments)} --resize-mode {resize_mode}"
        f" && {python} tools/run_basic_text_line_full_image_batch.py"
        f" --input-dir {_q(full_input_dir)} --ocr-dir {_q(full_ocr_dir)}"
        f" --output-dir \"$AUTO_EVAL_STAGE_ROOT/full_image\" --debug-root \"$AUTO_EVAL_STAGE_ROOT/full_image/debug\""
        f" --checkpoint {_q(checkpoint)} --network-config {_q(network_config)}"
        f" --device {_q(device)} --expected-images {int(expected_images)}"
        f" --resize-mode {resize_mode}"
    )
    return {
        "schema_version": 1,
        "validation": {
            "command": validation_command,
            "dataset_names": names,
            "save_img": True,
            "output_root": "$AUTO_EVAL_STAGE_ROOT/validation",
            "canvas_hw": [96, 480],
            "resize_mode": resize_mode,
        },
        "testset": {
            "command": test_command,
            "expected_hx_segments": int(expected_segments),
            "expected_full_images": int(expected_images),
            "outputs": [
                "$AUTO_EVAL_STAGE_ROOT/hx_segments",
                "$AUTO_EVAL_STAGE_ROOT/full_image",
            ],
            "short_circuit": "validation command must exit 0; HX dump && full-image batch",
            "canvas_hw": [96, 480],
            "resize_mode": resize_mode,
        },
        "network_config": str(network_config),
        "checkpoint": str(checkpoint),
        "stage_root_contract": "$AUTO_EVAL_STAGE_ROOT",
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """@guang-yang-comment Parse contract-generator arguments and output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--hx-manifest", type=Path, required=True)
    parser.add_argument("--hx-manifest-summary", type=Path)
    parser.add_argument("--hx-iteration", type=int, required=True)
    parser.add_argument("--full-input-dir", type=Path, required=True)
    parser.add_argument("--full-ocr-dir", type=Path, required=True)
    parser.add_argument("--expected-segments", type=int, default=669)
    parser.add_argument("--expected-images", type=int, default=52)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-command", default="python",
                        help="Interpreter token in generated command (Remote58 defaults to python)")
    parser.add_argument("--resize-mode", choices=("stretch",), default="stretch")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """@guang-yang-comment Generate and print or persist the ST-04 contract."""

    args = _parse_args(argv)
    contract = build_contract(
        args.validation_config, args.checkpoint, args.network_config, args.hx_manifest,
        args.hx_manifest_summary, args.hx_iteration, args.full_input_dir, args.full_ocr_dir,
        args.expected_segments, args.expected_images, args.device, args.python_command, args.resize_mode,
    )
    payload = json.dumps(contract, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
