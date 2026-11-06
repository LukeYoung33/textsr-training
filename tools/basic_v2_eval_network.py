"""Shared registry-backed network construction for BASIC V2 evaluators."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

from basicsr.archs import build_network

# Importing these modules registers both the baseline and Large architectures.
import realesrgan.archs  # noqa: F401

from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint


BASELINE_NETWORK_CONFIG = {
    "type": "BasicTextLineR4StudentV2",
    "canvas_hw": [96, 320],
    "in_channel": 12,
    "out_channel": 12,
    "layer_num": 3,
    "expand_channel": [0.5, 1.0, 2.0, 4.0],
    "depth_factor": 1,
    "scale_factor": 2,
    "with_idt": False,
    "act_type": "prelu",
    "scale": 2,
    "with_proj": True,
}


def load_network_config(config_path: Optional[Path]) -> dict[str, Any]:
    """@guang-yang-comment Load ``network_g`` from a training YAML or baseline defaults.

    Input: optional training option path containing a mapping-valued ``network_g``.
    Output: detached network constructor mapping. Key Steps: parse YAML, require
    a type, and preserve all architecture knobs for Large variants. Key Params:
    absent config selects the legacy BASIC V2 defaults. State/Dependencies:
    PyYAML and the registry modules imported above.
    """

    if config_path is None:
        return deepcopy(BASELINE_NETWORK_CONFIG)
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping) or not isinstance(document.get("network_g"), Mapping):
        raise ValueError(f"network config must contain mapping network_g: {path}")
    network = dict(document["network_g"])
    if not network.get("type"):
        raise ValueError(f"network_g.type is required: {path}")
    return network


def build_eval_network(
    checkpoint: Path,
    device: torch.device,
    network_config: Optional[Path] = None,
) -> tuple[torch.nn.Module, str]:
    """@guang-yang-comment Build a registry network and strictly load its checkpoint.

    Input: checkpoint, target device, and optional training YAML. Output: eval-mode
    network plus checkpoint SHA-256. Key Steps: resolve ``network_g``, call the
    shared registry builder, and invoke strict state loading. Key Params: default
    config preserves the old BasicTextLineR4StudentV2 graph. State/Dependencies:
    registered ``realesrgan.archs`` classes and torch checkpoint format.
    """

    network = build_network(load_network_config(network_config)).to(device)
    checkpoint_sha256 = load_strict_checkpoint(network, checkpoint, map_location="cpu")
    network.eval()
    return network, checkpoint_sha256


def resolve_canvas_hw(network_config: Optional[Path] = None) -> tuple[int, int]:
    """@guang-yang-comment Resolve and validate the fixed canvas for an evaluator.

    Input: optional training YAML. Output: positive even ``(height, width)``.
    Key Steps: load ``network_g.canvas_hw`` and validate its two dimensions. Key
    Params: absent config keeps the legacy 96x320 canvas. State/Dependencies:
    network configuration parser above.
    """

    config = load_network_config(network_config)
    canvas = config.get("canvas_hw", [96, 320])
    if not isinstance(canvas, (list, tuple)) or len(canvas) != 2:
        raise ValueError("network_g.canvas_hw must contain two dimensions")
    result = (int(canvas[0]), int(canvas[1]))
    if any(value <= 0 or value % 2 for value in result):
        raise ValueError(f"network_g.canvas_hw must be positive even dimensions: {result}")
    return result
