"""GT-only BASIC V2 Dataset that supplies standard Real-ESRGAN kernel tensors."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils.registry import DATASET_REGISTRY

from .basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset


STANDARD_KERNEL_DEFAULTS = {
    "blur_kernel_size": 21,
    "kernel_list": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob": 0.1,
    "blur_sigma": [0.2, 3],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
    "blur_kernel_size2": 21,
    "kernel_list2": ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob2": 0.1,
    "blur_sigma2": [0.2, 1.5],
    "betag_range2": [0.5, 4],
    "betap_range2": [1, 2],
    "final_sinc_prob": 0.8,
}


@contextmanager
def _sample_seed(seed: int | None) -> Iterator[None]:
    """@shanlan-comment
    Input: optional integer seed for one sample's kernel generation.
    Output: restores caller RNG states after the enclosed sampling operation.
    Key Steps: snapshot Python/numpy/torch RNG states, seed all when requested,
    yield control, then restore the snapshots.
    Key Params: seed=None preserves ordinary worker-random sampling behavior.
    State/Dependencies: temporarily mutates global Python, numpy, and torch RNG state.
    """
    if seed is None:
        yield
        return
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def _sample_kernel(opt: dict[str, Any], *, suffix: str = "") -> torch.Tensor:
    """@shanlan-comment
    Input: standard Real-ESRGAN kernel options and optional second-stage suffix.
    Output: one 21x21 float32 blur kernel sampled with the standard distribution.
    Key Steps: choose an odd 7..21 size, select sinc or mixed kernel, then zero-pad.
    Key Params: sinc_prob, kernel_list/probability, sigma, beta ranges use x4plus values.
    State/Dependencies: Python/numpy RNG and BasicSR standard degradation helpers.
    """
    key = lambda name: f"{name}{suffix}"
    kernel_size = random.choice([2 * value + 1 for value in range(3, 11)])
    if np.random.uniform() < float(opt[key("sinc_prob")]):
        omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            opt[key("kernel_list")], opt[key("kernel_prob")], kernel_size,
            opt[key("blur_sigma")], opt[key("blur_sigma")], [-np.pi, np.pi],
            opt[key("betag_range")], opt[key("betap_range")], noise_range=None)
    pad = (21 - kernel_size) // 2
    return torch.from_numpy(np.pad(kernel, ((pad, pad), (pad, pad))).astype(np.float32))


def sample_standard_realesrgan_kernels(opt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """@shanlan-comment
    Input: merged GT-degradation Dataset options.
    Output: standard first/second blur and final-sinc 21x21 tensors.
    Key Steps: invoke the same mixed/sinc kernel sampling rules as RealESRGANDataset
    for both stages, then choose an active final sinc or the pulse kernel.
    Key Params: all defaults mirror options/train_realesrgan_x4plus.yml.
    State/Dependencies: _sample_kernel, circular_lowpass_kernel, and RNG state.
    """
    kernel1 = _sample_kernel(opt)
    kernel2 = _sample_kernel(opt, suffix="2")
    if np.random.uniform() < float(opt["final_sinc_prob"]):
        kernel_size = random.choice([2 * value + 1 for value in range(3, 11)])
        sinc = torch.from_numpy(circular_lowpass_kernel(np.random.uniform(np.pi / 3, np.pi), kernel_size, pad_to=21).astype(np.float32))
    else:
        sinc = torch.zeros(21, 21, dtype=torch.float32)
        sinc[10, 10] = 1.0
    return kernel1, kernel2, sinc


@DATASET_REGISTRY.register()
class BasicTextLineGTDegradeDataset(BasicTextLineRemote58Dataset):
    """Emit clean fixed-canvas RGB GT, standard kernels, and direct-tile metadata."""

    def __init__(self, opt: dict[str, Any]):
        """@shanlan-comment
        Input: Remote58 GT root/manifest options plus optional standard-kernel overrides.
        Output: initialized fixed-canvas GT-only BASIC V2 Dataset.
        Key Steps: initialize the paired reader for GT geometry/metadata, merge x4plus
        kernel defaults, and retain an optional deterministic per-index seed.
        Key Params: standard kernel values follow x4plus; kernel_seed enables reproducibility.
        State/Dependencies: parent Dataset reads paired source files but synthetic LQ ignores source LQ.
        """
        super().__init__(opt)
        self.degradation_opt = dict(STANDARD_KERNEL_DEFAULTS)
        self.degradation_opt.update({key: value for key, value in opt.items() if key in STANDARD_KERNEL_DEFAULTS})
        self.kernel_seed = opt.get("kernel_seed")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """@shanlan-comment
        Input: integer fixed-canvas sample index.
        Output: clean RGB GT, standard kernels, masks, and BASIC V2 CP tile metadata.
        Key Steps: reuse parent geometry/metadata, replace paired LQ/GT tensors with
        clean RGB GT for GPU synthesis, then sample standard kernels.
        Key Params: kernel_seed plus index makes kernel draws reproducible for tests.
        State/Dependencies: parent GT reader and standard Real-ESRGAN kernel sampler.
        """
        sample = super().__getitem__(index)
        seed = None if self.kernel_seed is None else int(self.kernel_seed) + int(index)
        with _sample_seed(seed):
            kernel1, kernel2, sinc_kernel = sample_standard_realesrgan_kernels(self.degradation_opt)
        clean_rgb = sample["gt_rgb"]
        sample["gt"] = clean_rgb
        sample["kernel1"] = kernel1
        sample["kernel2"] = kernel2
        sample["sinc_kernel"] = sinc_kernel
        sample["degradation_mode"] = "standard_realesrgan_gt_only"
        return sample

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """@shanlan-comment Stack standard kernels in addition to the BASIC V2 batch.

        Input: GT-only samples returned by ``__getitem__``. Output: the parent
        BASIC V2 collate mapping with batched 21x21 kernel tensors. Key Steps:
        delegate fixed-canvas tensors/metadata to the parent, then stack all
        three degradation kernels. Key Params: kernel keys must be present for
        every sample. State/Dependencies: consumed by the canonical GPU
        degradation path in ``BasicTextLineGTDegradationTrainingModel``.
        """
        result = super().collate_fn(batch)
        for key in ("kernel1", "kernel2", "sinc_kernel"):
            result[key] = torch.stack([sample[key] for sample in batch], dim=0)
        return result


__all__ = ["BasicTextLineGTDegradeDataset", "STANDARD_KERNEL_DEFAULTS", "sample_standard_realesrgan_kernels"]
