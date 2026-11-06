"""Reproducible Torch implementation of the confirmed Lenovo degradation set.

Each call samples a physically ordered capture chain plus compatible ISP
post-processing.  It never treats the eight preview categories as an arbitrary
permutation: capture/sampling precedes chroma ISP, sharpening follows chroma
processing, and secondary capture wraps a completed first capture.  C6 is
deliberately pinned until a physical source for edge irregularity is confirmed.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping

import numpy as np
from realesrgan import lenovo_atomic as atoms
from realesrgan import lenovo_atomic_torch as torch_atoms
import torch
from torch.nn import functional as F


DEFAULT_LENOVO_DEGRADATION_CONFIG: dict[str, Any] = {
    "always": {"shot_noise": [.00035, .00090], "read_noise": [.00020, .00060],
               "jitter_px": [.03, .22], "zoom_scale": [.82, .94]},
    "composition": {
        "capture_weights": {"c1": .30, "c2": .15, "c1_c2": .40, "zoom_c2": .15},
        "post_probabilities": {"c3": .22, "c4": .18, "c5": .20, "c7": .12},
        "max_post_branches": 2,
        "secondary_capture_probability": .08,
        "enforce_problem_count": True,
    },
    "c1": {"gaussian_sigma": [.35, 1.45], "motion_length": [3, 9], "defocus_radius": [1, 3], "zoom_source_scale": [.45, .75]},
    "c2": {"psf_sigma": [.82, 1.15], "sample_scale": [.50, .72], "phase_px": .42},
    "c3": {"strength": [6., 10.], "sigmas": [1.5, 4., 9.], "scale": .5, "rho": -.75,
           "version": "c3_multiscale_edge_protected_v1"},
    "c4": {"uv_blur_sigma": [1.5, 4.0], "midfreq_strength": [1., 4.]},
    "c5": {"usm_sigma": [1.4, 2.6], "usm_amount": [.25, .75], "jpeg_quality": [82, 96]},
    "c7": {"shift_x": [1., 3.], "shift_y": [.5, 2.], "weight": [.12, .35]},
    "c8": {"display_scale": [.50, .75], "display_jpeg_quality": [88, 96], "camera_blur_sigma": [.35, .9], "final_jpeg_quality": [82, 94], "perspective_fraction": [.0, .008]},
}


def _uniform(generator: torch.Generator, device: torch.device, low: float, high: float) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high), generator=generator).item())


def _choice(generator: torch.Generator, device: torch.device, count: int) -> int:
    return int(torch.randint(count, (), device=device, generator=generator).item())


def _kernel_gaussian(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-axis.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[:, None] * kernel[None, :]


def _filter(image: torch.Tensor, kernel: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Apply one spatial kernel, optionally extending image values at its boundary."""
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    kernel = kernel[None, None].expand(image.shape[1], 1, -1, -1)
    padding = kernel.shape[-1] // 2
    if padding_mode == "zeros":
        return F.conv2d(image, kernel, padding=padding, groups=image.shape[1])
    if padding_mode not in {"replicate", "reflect"}:
        raise ValueError(f"unsupported filter padding mode: {padding_mode}")
    padded = F.pad(image, (padding, padding, padding, padding), mode=padding_mode)
    return F.conv2d(padded, kernel, groups=image.shape[1])


def _gaussian(image: torch.Tensor, sigma: float) -> torch.Tensor:
    return _filter(image, _kernel_gaussian(max(float(sigma), .05), image.device, image.dtype))


def _motion_kernel(length: int, angle: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size = max(3, int(length) | 1)
    axis = torch.arange(size, device=device, dtype=dtype) - size // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    theta = math.radians(angle)
    along = xx * math.cos(theta) + yy * math.sin(theta)
    across = -xx * math.sin(theta) + yy * math.cos(theta)
    kernel = ((along.abs() <= length / 2) & (across.abs() <= .6)).to(dtype)
    return kernel / kernel.sum().clamp_min(1.)


def _disk_kernel(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = (xx.square() + yy.square() <= radius * radius).to(dtype)
    return kernel / kernel.sum().clamp_min(1.)


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= .04045, value / 12.92, ((value + .055) / 1.055).pow(2.4))


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0., 1.)
    return torch.where(value <= .0031308, value * 12.92, 1.055 * value.pow(1. / 2.4) - .055)


def _rgb_to_ycrcb(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = .299 * r + .587 * g + .114 * b
    cr = .5 * r - .418688 * g - .081312 * b + .5
    cb = -.168736 * r - .331264 * g + .5 * b + .5
    return torch.cat((y, cr, cb), dim=1)


def _ycrcb_to_rgb(value: torch.Tensor) -> torch.Tensor:
    y, cr, cb = value[:, 0:1], value[:, 1:2] - .5, value[:, 2:3] - .5
    return torch.cat((y + 1.402 * cr, y - .714136 * cr - .344136 * cb, y + 1.772 * cb), dim=1)


def _bandpass(shape: tuple[int, int], sigma_low: float, sigma_high: float, reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn((1, 1, *shape), device=reference.device, dtype=reference.dtype, generator=generator)
    band = _gaussian(noise, sigma_low) - _gaussian(noise, sigma_high)
    return band / band.std().clamp_min(1e-6)


class LenovoDegradationPipeline:
    """Sample C1-C5/C7/C8 on a batch using only an explicit Torch generator."""

    def __init__(self, config: Mapping[str, Any] | None = None, jpeg: Callable[[torch.Tensor, float], torch.Tensor] | None = None):
        self.config = copy.deepcopy(DEFAULT_LENOVO_DEGRADATION_CONFIG)
        if config:
            for key, value in config.items():
                if isinstance(value, Mapping) and isinstance(self.config.get(key), dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        self.jpeg = jpeg
        self._torch_jpeg = None
        self._validate()

    def _validate(self) -> None:
        if "band_sigma" in self.config["c3"]:
            raise ValueError("C3 frequency schema changed: replace legacy band_sigma with sigmas")
        if self.config["c3"].get("version") != "c3_multiscale_edge_protected_v1":
            raise ValueError("Unsupported C3 version")
        if any(float(v) != 0 for v in self.config["c5"].get("edge_noise", [0])):
            raise ValueError("C5 extra edge noise is not part of the certified reference")
        composition = self.config.get("composition")
        if not isinstance(composition, Mapping):
            raise ValueError("composition must be a mapping")
        capture = composition.get("capture_weights")
        required_capture = {"c1", "c2", "c1_c2", "zoom_c2"}
        if not isinstance(capture, Mapping) or set(capture) != required_capture:
            raise ValueError("capture_weights must contain c1, c2, c1_c2 and zoom_c2")
        if any(float(value) < 0. for value in capture.values()) or abs(sum(map(float, capture.values())) - 1.) > 1e-6:
            raise ValueError("capture_weights must be non-negative and sum to 1")
        post = composition.get("post_probabilities")
        if not isinstance(post, Mapping) or set(post) != {"c3", "c4", "c5", "c7"}:
            raise ValueError("post_probabilities must contain exactly c3, c4, c5 and c7")
        if any(not 0. <= float(value) <= 1. for value in post.values()):
            raise ValueError("post probabilities must be in [0, 1]")
        if int(composition.get("max_post_branches", 0)) not in {0, 1, 2}:
            raise ValueError("max_post_branches must be 0, 1 or 2")
        if not 0. <= float(composition.get("secondary_capture_probability", -1.)) <= 1.:
            raise ValueError("secondary_capture_probability must be in [0, 1]")
        forced_c1 = composition.get("forced_c1_subbranch")
        if forced_c1 is not None and forced_c1 not in {"gaussian", "motion", "zoom", "defocus"}:
            raise ValueError("forced_c1_subbranch must be gaussian, motion, zoom or defocus")
        # Retained as an explicit pin even though C6 has no handler in this graph.
        c6 = self.config.get("c6", {})
        if isinstance(c6, Mapping) and float(c6.get("weight", 0.)) != 0.0:
            raise ValueError("C6 is pinned and must have weight 0")

    def _always_capture(self, image: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg, device = self.config["always"], image.device
        shot = _uniform(generator, device, *cfg["shot_noise"]); read = _uniform(generator, device, *cfg["read_noise"])
        noise = torch.randn(image.shape, device=device, dtype=image.dtype, generator=generator)
        output = (image + noise * (read + shot * image.clamp_min(0.).sqrt())).clamp(0., 1.)
        jitter = _uniform(generator, device, *cfg["jitter_px"]); dx = _uniform(generator, device, -jitter, jitter); dy = _uniform(generator, device, -jitter, jitter)
        h, w = image.shape[-2:]; theta = image.new_tensor([[[1., 0., 2 * dx / w], [0., 1., 2 * dy / h]]])
        output = F.grid_sample(output, F.affine_grid(theta, output.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)
        scale = _uniform(generator, device, *cfg["zoom_scale"])
        small = F.interpolate(output, size=(max(2, round(h * scale)), max(2, round(w * scale))), mode="area")
        return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0., 1.), {"branch": "always", "shot_noise": shot, "read_noise": read, "jitter_xy": [dx, dy], "zoom_scale": scale}

    def atomic(self, image, name, seed, params=None):
        """Exact CPU adapter; input/output RGB BCHW, atom BGR uint8."""
        if name == "c6":
            raise ValueError("C6 is pinned")
        functions = {"gaussian": atoms.gaussian_blur, "motion": atoms.motion_blur,
                     "zoom": atoms.zoom_resample, "defocus": atoms.defocus_blur,
                     "c2": atoms.fine_structure_loss, "c3": atoms.flat_luma_noise,
                     "c4": atoms.color_anomaly, "c5": atoms.halo_ringing,
                     "c7": atoms.directional_ghosting, "c8": atoms.secondary_capture}
        if image.shape[0] != 1:
            raise ValueError("atomic adapter expects one sample")
        rgb = image[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
        result, record = functions[name](np.ascontiguousarray(rgb[:, :, ::-1]),
                                        np.random.default_rng(seed), params)
        output = torch.from_numpy(result[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(image.device, image.dtype) / 255
        record.update({"branch": "c1" if name in ("gaussian", "motion", "zoom", "defocus") else name,
                       "atomic_version": atoms.ATOMIC_VERSION, "parameter_schema": atoms.PARAMETER_SCHEMA,
                       "atomic_seed": seed, "parameters": params or {}, "backend": "opencv_reference"})
        if record["branch"] == "c1":
            record["subbranch"] = name
        return output, record

    def _shared(self, image, generator, name, mapping):
        cfg = self.config["c1" if name in ("gaussian", "motion", "zoom", "defocus") else name]
        params = {}
        for target, key in mapping.items():
            value = cfg[key]
            params[target] = _uniform(generator, image.device, *value) if isinstance(value, (list, tuple)) else value
        if name == "motion":
            params["angle"] = _uniform(generator, image.device, 20., 160.)
        return self.torch_atomic(image, name, generator, params)

    def torch_atomic(self, image, name, generator, params=None, random_inputs=None):
        """Device-resident port; CPU reference remains available only for audits."""
        def jpeg(value, quality):
            if self.jpeg is not None:
                return self.jpeg(value, quality)
            if self._torch_jpeg is None:
                self._torch_jpeg = torch_atoms.ReferenceJpeg().to(image.device)
            return self._torch_jpeg(value, float(quality))
        output, record = torch_atoms.apply(image, name, generator, params, random_inputs, jpeg)
        record.update({"branch": "c1" if name in ("gaussian", "motion", "zoom", "defocus") else name,
                       "atomic_version": torch_atoms.VERSION, "parameter_schema": atoms.PARAMETER_SCHEMA,
                       "backend": "torch", "parameters": params or {}})
        if name == 'c3':
            record['atomic_version'] = record['c3_version']
            record['parameter_schema'] = record['c3_schema']
        if record["branch"] == "c1":
            record["subbranch"] = name
        return output, record

    def _c1(self, image, generator, subbranch=None):
        name = subbranch or ("gaussian", "motion", "zoom", "defocus")[_choice(generator, image.device, 4)]
        mapping = {"gaussian": {"sigma": "gaussian_sigma"}, "motion": {"length": "motion_length"},
                   "zoom": {"source_scale": "zoom_source_scale"}, "defocus": {"radius": "defocus_radius"}}
        return self._shared(image, generator, name, mapping[name])

    def _c2(self, image, generator):
        return self._shared(image, generator, "c2", {"optical_psf_sigma": "psf_sigma", "sample_scale": "sample_scale", "phase_px": "phase_px"})

    def _c3(self, image, generator):
        cfg = self.config['c3']
        amplitude = _uniform(generator, image.device, *cfg['strength'])
        seed = int(torch.randint(0, 2**31, (), device=image.device, generator=generator).item())
        local_generator = torch.Generator(device=image.device).manual_seed(seed)
        output, record = self.torch_atomic(image, 'c3', local_generator,
                                 {'strength': amplitude, 'sigmas': cfg['sigmas'],
                                  'scale': cfg['scale'], 'rho': cfg['rho']})
        record['seed'] = seed
        return output, record

    def _c4(self, image, generator):
        return self._shared(image, generator, "c4", {"uv_blur_sigma": "uv_blur_sigma", "midfreq_strength": "midfreq_strength"})

    def _c5(self, image, generator):
        return self._shared(image, generator, "c5", {"usm_sigma": "usm_sigma", "usm_amount": "usm_amount", "jpeg_quality": "jpeg_quality"})

    def _c7(self, image, generator):
        return self._shared(image, generator, "c7", {"shift_x": "shift_x", "shift_y": "shift_y", "weight": "weight"})

    def _c8(self, image, generator):
        return self._shared(image, generator, "c8", {key: key for key in ("display_scale", "display_jpeg_quality", "final_jpeg_quality", "camera_blur_sigma", "perspective_fraction")})

    @torch.no_grad()
    def __call__(self, gt: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if gt.ndim != 4 or gt.shape[1] != 3:
            raise ValueError("Lenovo degradation requires RGB BCHW input")
        if gt.shape[-2] % 2 or gt.shape[-1] % 2:
            raise ValueError("Lenovo degradation requires even spatial dimensions")
        outputs, metadata = [], []
        handlers = {"c1": self._c1, "c2": self._c2, "c3": self._c3, "c4": self._c4, "c5": self._c5, "c7": self._c7, "c8": self._c8}
        composition = self.config["composition"]
        forced_c1 = composition.get("forced_c1_subbranch")
        capture_labels = tuple(composition["capture_weights"])
        capture_weights = torch.tensor([composition["capture_weights"][name] for name in capture_labels], device=gt.device, dtype=torch.float32)
        # C5 is executed after the other post stages, but it receives priority
        # when the per-sample problem budget has to discard candidates.  Wide
        # black/bright edge bands are common in the target capture set.
        post_order = ("c7", "c4", "c3", "c5")
        post_selection_priority = ("c5", "c7", "c4", "c3")
        for index in range(gt.shape[0]):
            capture = capture_labels[int(torch.multinomial(capture_weights, 1, generator=generator).item())]
            output = gt[index:index + 1]
            output, record = self._always_capture(output, generator)
            stages: list[dict[str, Any]] = [record]
            if capture == "c1":
                output, record = self._c1(output, generator, forced_c1); stages.append(record)
            elif capture == "c2":
                output, record = self._c2(output, generator); stages.append(record)
            elif capture == "c1_c2":
                output, record = self._c1(output, generator, forced_c1); stages.append(record)
                output, record = self._c2(output, generator); stages.append(record)
            else:  # zoom is a source-resolution loss and therefore precedes C2 sampling.
                output, record = self._c1(output, generator, subbranch="zoom"); stages.append(record)
                output, record = self._c2(output, generator); stages.append(record)
            selected_post = [branch for branch in post_selection_priority
                             if _uniform(generator, gt.device, 0., 1.) < float(composition["post_probabilities"][branch])]
            if composition.get("enforce_problem_count", True):
                # Count the capture atoms, excluding the fixed acquisition base.
                # Add a sampling/ghost atom when needed, rather than turning the
                # low-probability colour-diffusion stage into a mandatory one.
                if len(stages) == 2 and not selected_post:
                    selected_post = ["c7"]
                selected_post = selected_post[:max(0, 3 - (len(stages) - 1))]
            if len(selected_post) > int(composition["max_post_branches"]):
                order = torch.randperm(len(selected_post), device=gt.device, generator=generator).tolist()
                selected_post = [selected_post[position] for position in order[:int(composition["max_post_branches"])]]
                selected_post.sort(key=post_order.index)
            for branch in selected_post:
                output, record = handlers[branch](output, generator); stages.append(record)
            if _uniform(generator, gt.device, 0., 1.) < float(composition["secondary_capture_probability"]):
                output, record = self._c8(output, generator); stages.append(record)
            outputs.append(output)
            metadata.append({"atomic_version": torch_atoms.VERSION, "parameter_schema": atoms.PARAMETER_SCHEMA, "capture_scenario": capture, "stages": stages, "c6_pinned": True})
        return torch.cat(outputs, dim=0).clamp(0., 1.), metadata


__all__ = ["DEFAULT_LENOVO_DEGRADATION_CONFIG", "LenovoDegradationPipeline"]
