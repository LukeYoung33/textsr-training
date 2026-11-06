"""Standalone launcher for the BASIC 2.0 fixed-shape training pipeline.

The launcher imports only the V2 registry modules before delegating to BasicSR's
training driver.  It does not alter the legacy ``realesrgan.train`` entrypoint.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


V2_REGISTRY_MODULES = (
    "realesrgan.archs.basic_text_line_restore_v2_arch",
    "realesrgan.data.basic_text_line_dataset_v2",
    "realesrgan.losses.basic_text_line_losses_v2",
    "realesrgan.models.realesrnet_model_v2",
)


def _load_v2_registrations() -> None:
    """Import V2 registry modules before BasicSR constructs datasets/models.

    Input: no arguments; the repository root must be importable from ``sys.path``.
    Output: all available V2 modules are imported, while a missing module raises a
        clear error instead of silently falling back to a legacy registration.
    Key Steps: import each module in deterministic order and report the first
        missing implementation with its expected path.
    Key Params: ``V2_REGISTRY_MODULES`` is the only module inventory.
    State/Dependencies: mutates Python's import/registry state.
    """
    for module_name in V2_REGISTRY_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise RuntimeError(
                    f"BASIC V2 module is missing: {module_name}. "
                    "Install the complete V2 implementation before training.") from exc
            raise


def _build_parser() -> argparse.ArgumentParser:
    """Build the launcher parser without duplicating BasicSR option parsing.

    Input: none.
    Output: parser accepting the V2 YAML path and optional dry-run mode.
    Key Steps: expose a repository-relative default and preserve extra arguments
        for BasicSR's own parser.
    Key Params: ``--opt`` selects the training configuration.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--opt",
        default="options/train_bucket_3/train_basic_text_line_v2.yml",
        help="BASIC V2 YAML passed to BasicSR (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import V2 modules and validate the config without starting training.",
    )
    return parser


def _validate_config(config_path: Path) -> None:
    """Validate the minimal V2 config contract before launching training.

    Input: existing YAML path.
    Output: none; raises ``ValueError``/``FileNotFoundError`` on contract errors.
    Key Steps: parse YAML and verify fixed canvas, V2 model, and queue settings.
    Key Params: ``model_type``, ``network_g.canvas_hw``, and ``queue_size``.
    State/Dependencies: PyYAML is required by the repository's config system.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model_type = str(config.get("model_type", ""))
    canvas = config.get("network_g", {}).get("canvas_hw")
    if model_type != "RealESRNetModelV2":
        raise ValueError("BASIC V2 config must use model_type=RealESRNetModelV2")
    if not isinstance(canvas, list) or len(canvas) != 2:
        raise ValueError("network_g.canvas_hw must be [Hc, Wc]")
    if any(int(value) <= 0 or int(value) % 32 for value in canvas):
        raise ValueError("network_g.canvas_hw values must be positive multiples of 32")
    if int(config.get("queue_size", config.get("train", {}).get("queue_size", 0))) != 0:
        raise ValueError("BASIC V2 requires queue_size=0")


def main(argv=None) -> int:
    """Load V2 modules, validate the config, and invoke BasicSR training.

    Input: optional argv sequence excluding the executable name.
    Output: process-style integer exit code; training exceptions propagate.
    Key Steps: resolve repository root, import registrations, validate config,
        then delegate to ``basicsr.train.train_pipeline``.
    Key Params: ``--dry-run`` prevents any optimizer or dataloader execution.
    State/Dependencies: imports V2 registries and may start the BasicSR runtime.
    """
    parser = _build_parser()
    args, passthrough = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    config_path = Path(args.opt)
    if not config_path.is_absolute():
        config_path = root / config_path
    _validate_config(config_path)
    if args.dry_run:
        return 0
    _load_v2_registrations()

    from basicsr.train import train_pipeline

    sys.argv = [sys.argv[0], "-opt", str(config_path), *passthrough]
    train_pipeline(str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
