"""Configurable RGB BCHW adapter for the frozen Lenovo P17-B path."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .lenovo_p17_stage_a_exact import _p14_kernel_ab
from .lenovo_p17_torch_gpu import preprocess_p17_torch_gpu

DEFAULT_LQ_PREPROCESS: dict[str, Any] = {
    "enabled": False, "method": "lenovo_p17_regional_otsu", "implementation": "cpu_exact", "lambda": 0.03, "alpha": 1.0,
    "residual_input_source": "preprocessed",
    "p17_patch_size": 96, "p17_patch_stride": 32, "p17_stats_gaussian_sigma": 0.6,
    "p17_min_class_fraction": 0.12, "p16_write_full_radius": 16.0, "p16_write_zero_radius": 24.0,
    "mask_sigma": 0.6, "kernel_sigma_min": 0.3, "kernel_sigma_max": 4.0,
    "kernel_sigma_step": 0.05, "kernel_center_limit": 4.0, "kernel_center_step": 0.25,
    "kernel_huber_delta": 0.05, "kernel_max_fit_rmse": 0.08, "kernel_min_contrast": 0.12,
    "kernel_max_reverse_steps": 8, "profile_half_width": 16.0, "sample_step": 0.5,
    "boundary_stride": 1, "min_platform_run": 3.0, "platform_scan_limit": 24.0,
    "platform_tolerance_abs": 0.0078431373, "platform_tolerance_rel": 0.03,
    "min_profile_contrast": 0.12, "max_profiles": 128, "p14_min_field_support": 0.001,
    "p14_min_profile_contrast": 0.12, "p14_sample_step": 0.5, "p14_boundary_stride": 1,
    "p14_max_radius": 24.0, "p14_max_profiles": 4096, "p14_interpolation_gap_px": 4.0,
    "p14_interpolation_angle_deg": 30.0, "p14_endpoint_feather_px": 1.0,
    "p17_wiener_antiring_direction_enabled": False, "p17_wiener_antiring_envelope_enabled": False,
    "p17_wiener_global_clip_enabled": False, "p17_wiener_bypass_enabled": False, "p17_wiener_antiring_delta_floor": 2.0 / 255.0,
    "p17_wiener_antiring_envelope_epsilon": 2.0 / 255.0, "p17_wiener_antiring_radius_sigma": 3.0,
    "p17_wiener_antiring_radius_max": 12,
}


def normalize_lq_preprocess_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """@luna-comment
    Input: Optional root YAML lq_preprocess mapping.
    Output: A validated flat P17 configuration with explicit selector and candidate fields.
    Key Steps: Merges defaults, validates the P17 method/implementation/residual route,
    normalizes scalar controls, and binds the selected lambda/alpha candidate.
    Key Params: Experimental Wiener anti-ringing and bypass controls default to disabled
    and are valid only for the CUDA route so CPU exact behavior cannot silently diverge.
    State/Dependencies: Pure mapping/NumPy validation; no model or image state.
    """
    result = dict(DEFAULT_LQ_PREPROCESS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("lq_preprocess must be a mapping")
        result.update(dict(config))
    if result.get("method") != "lenovo_p17_regional_otsu":
        raise ValueError("only lenovo_p17_regional_otsu is supported")
    implementation = str(result.get("implementation", "cpu_exact")).strip().lower()
    if implementation not in {"cpu_exact", "torch_gpu"}:
        raise ValueError("lq_preprocess.implementation must be cpu_exact or torch_gpu")
    result["implementation"] = implementation
    residual_source = str(result.get("residual_input_source", "preprocessed")).strip().lower()
    if residual_source not in {"preprocessed", "original"}:
        raise ValueError("lq_preprocess.residual_input_source must be preprocessed or original")
    result["residual_input_source"] = residual_source
    result["enabled"], result["lambda"], result["alpha"] = bool(result["enabled"]), float(result["lambda"]), float(result["alpha"])
    for key in ("p17_wiener_antiring_direction_enabled", "p17_wiener_antiring_envelope_enabled", "p17_wiener_global_clip_enabled", "p17_wiener_bypass_enabled"):
        result[key] = bool(result[key])
    for key in ("p17_wiener_antiring_delta_floor", "p17_wiener_antiring_envelope_epsilon", "p17_wiener_antiring_radius_sigma"):
        result[key] = float(result[key])
    result["p17_wiener_antiring_radius_max"] = int(result["p17_wiener_antiring_radius_max"])
    result["p17_patch_size"], result["p17_patch_stride"] = max(3, int(result["p17_patch_size"])), max(1, int(result["p17_patch_stride"]))
    if not np.isfinite(result["lambda"]) or result["lambda"] <= 0.0: raise ValueError("lq_preprocess.lambda must be finite and positive")
    if not np.isfinite(result["alpha"]) or not 0.0 < result["alpha"] <= 1.0: raise ValueError("lq_preprocess.alpha must be within (0, 1]")
    if any((not np.isfinite(result[key]) or result[key] < 0.0) for key in ("p17_wiener_antiring_delta_floor", "p17_wiener_antiring_envelope_epsilon")) or not np.isfinite(result["p17_wiener_antiring_radius_sigma"]) or result["p17_wiener_antiring_radius_sigma"] <= 0.0 or result["p17_wiener_antiring_radius_max"] < 1:
        raise ValueError("invalid P17 Wiener anti-ringing parameters")
    if implementation != "torch_gpu" and any(result[key] for key in ("p17_wiener_antiring_direction_enabled", "p17_wiener_antiring_envelope_enabled", "p17_wiener_global_clip_enabled", "p17_wiener_bypass_enabled")):
        raise ValueError("P17 Wiener anti-ringing guards require implementation=torch_gpu")
    result["lambda_candidates"], result["alpha_candidates"], result["p17_mode"] = [result["lambda"]], [result["alpha"]], True
    return result


def preprocess_lq_rgb(lq_rgb: torch.Tensor, config: Mapping[str, Any] | None = None) -> torch.Tensor:
    """@luna-comment
    Input: Finite RGB BCHW LQ tensor in [0, 1] and an optional P17 configuration.
    Output: Same-shape RGB BCHW tensor after exactly one selected P17 preprocessing pass.
    Key Steps: Normalizes config, returns identity when disabled, dispatches CUDA P17
    directly, or executes the frozen CPU reference then restores device/dtype.
    Key Params: implementation selects explicit CUDA or CPU behavior; disabled
    anti-ringing controls leave the frozen CUDA result unchanged.
    State/Dependencies: Calls the frozen CPU reference or CUDA P17 module; no model state.
    """
    normalized = normalize_lq_preprocess_config(config)
    if not normalized["enabled"]: return lq_rgb
    if not isinstance(lq_rgb, torch.Tensor) or lq_rgb.ndim != 4 or lq_rgb.shape[1] != 3: raise ValueError("enabled lq_preprocess requires RGB BCHW tensor")
    if not torch.isfinite(lq_rgb).all(): raise ValueError("lq_rgb contains non-finite values")
    if float(lq_rgb.detach().amin()) < -1e-6 or float(lq_rgb.detach().amax()) > 1.0 + 1e-6: raise ValueError("lq_rgb must be in [0,1]")
    if normalized["implementation"] == "torch_gpu":
        return preprocess_p17_torch_gpu(lq_rgb, normalized)
    device, dtype = lq_rgb.device, lq_rgb.dtype; outputs: list[torch.Tensor] = []
    for sample in lq_rgb.detach().to(device="cpu", dtype=torch.float32):
        rgb = sample.permute(1, 2, 0).contiguous().numpy(); _, meta = _p14_kernel_ab(rgb, normalized); stages = meta.get("_p14_stage_outputs", []); b_stages = stages[0].get("b_stages", []) if stages else []
        output = rgb if not b_stages else np.asarray(b_stages[0]["output"], dtype=np.float32)
        if output.shape != rgb.shape or not np.isfinite(output).all(): raise ValueError("P17 preprocessing produced invalid RGB output")
        outputs.append(torch.from_numpy(output).permute(2, 0, 1).contiguous())
    return torch.stack(outputs, dim=0).to(device=device, dtype=dtype)


__all__ = ["DEFAULT_LQ_PREPROCESS", "normalize_lq_preprocess_config", "preprocess_lq_rgb"]
