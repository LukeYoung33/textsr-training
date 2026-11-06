"""Compare frozen preview, shared atoms and training adapter; never auto-certify visuals."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realesrgan import lenovo_atomic as atoms
from realesrgan.lenovo_degradation import LenovoDegradationPipeline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(source, result):
    ycc = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb).astype(float)
    delta = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb).astype(float) - ycc
    uv = delta[:, :, 1:]
    y = ycc[:, :, 0]
    local = cv2.GaussianBlur(y, (0, 0), 3.2)
    dark = (local - y > 10) & (y < 150)
    flat = (np.abs(local - y) < 1) & (y > 180)
    energy = np.sum(uv**2, axis=2)
    spectrum = np.abs(np.fft.fft2(uv, axes=(0, 1)))**2
    fy, fx = np.meshgrid(np.fft.fftfreq(y.shape[0]), np.fft.fftfreq(y.shape[1]), indexing='ij')
    radius = np.sqrt(fx**2 + fy**2)
    total = spectrum.sum() + 1e-12
    vectors = uv / np.maximum(np.linalg.norm(uv, axis=2, keepdims=True), 1e-9)
    active = (energy[:, 1:] > 4) & (energy[:, :-1] > 4) & dark[:, 1:] & dark[:, :-1]
    hue_cos = np.sum(vectors[:, 1:] * vectors[:, :-1], axis=2)
    return {
        'uv_energy_dark': float(energy[dark].mean()) if dark.any() else None,
        'uv_energy_flat': float(energy[flat].mean()) if flat.any() else None,
        'dark_coverage': float((energy[dark] > 1).mean()) if dark.any() else None,
        'y_residual_rms': float(np.sqrt(np.mean(delta[:, :, 0]**2))),
        'uv_spectrum_bands': [float(spectrum[(radius >= lo) & (radius < hi)].sum()/total)
                              for lo, hi in [(0, .05), (.05, .25), (.25, 1)]],
        'uv_neighbor_difference': float(np.mean(np.diff(uv, axis=1)**2)),
        'dark_hue_neighbor_cosine': float(hue_cos[active].mean()) if active.any() else None,
    }


def main():
    frozen = ROOT/'tests/fixtures/lenovo_atomic/preview_r7_frozen.py'
    spec = importlib.util.spec_from_file_location('frozen_lenovo', frozen)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    out = ROOT/'datasets/lenovo_atomic_equivalence_v1'
    out.mkdir(parents=True, exist_ok=True)
    # Hand-selected sensitive contents: thin Chinese serif text, dark and blue
    # writing on a bright surface, and bright writing on a dark board.
    inputs = [ROOT/'datasets/generated_images'/name for name in [
        'page_0031.png', 'ChatGPT Image 2026年9月7日 19_11_57.png',
        'ChatGPT Image 2026年9月7日 16_07_28.png']]
    records = []
    pipe = LenovoDegradationPipeline()
    for index, path in enumerate(inputs):
        full = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        if full is None:
            raise ValueError(path)
        # Preserve native sampling; never resize the diagnostic GT.
        y, x = (95, 180) if index == 0 else (full.shape[0]//3, full.shape[1]//3)
        source = full[y:y+192, x:x+384].copy()
        gt_path = out/f'gt_{index}.png'
        cv2.imwrite(str(gt_path), source)
        tensor = torch.from_numpy(source[:, :, ::-1].copy()).permute(2, 0, 1)[None].float()/255
        panels = [source]
        for key, (_, func) in atoms.CATEGORIES.items():
            if key.startswith('c6'):
                continue
            name = {'c1a': 'gaussian', 'c1b': 'motion', 'c1c': 'zoom', 'c1d': 'defocus'}.get(key.split('_')[0], key.split('_')[0])
            for seed in (17, 20260910):
                expected, old_meta = old.CATEGORIES[key][1](source, np.random.default_rng(seed))
                actual, new_meta = func(source, np.random.default_rng(seed))
                training, train_meta = pipe.atomic(tensor, name, seed)
                decoded = training[0].permute(1, 2, 0).mul(255).round().byte().numpy()[:, :, ::-1]
                assert np.array_equal(expected, actual), (key, 'shared')
                assert np.array_equal(expected, decoded), (key, 'adapter')
                assert old_meta == new_meta, (key, 'metadata')
                expected_stats, actual_stats = statistics(source, expected), statistics(source, decoded)
                assert expected_stats == actual_stats
                dest = out/f'{index}_{key}_{seed}.png'
                cv2.imwrite(str(dest), expected)
                records.append({'atom': key, 'seed': seed, 'gt': gt_path.name, 'gt_sha256': sha(gt_path),
                                'lq': dest.name, 'lq_sha256': sha(dest), 'metadata': train_meta,
                                'statistics': actual_stats, 'max_pixel_error': 0})
                if seed == 17:
                    # Explicit before/reference/adapter triplet.
                    panels.extend([expected, decoded])
        cv2.imwrite(str(out/f'montage_{index}.png'), np.concatenate(panels, axis=0))
        combined, combined_meta = pipe(tensor, torch.Generator().manual_seed(20260910))
        combined_bgr = combined[0].permute(1, 2, 0).mul(255).round().byte().numpy()[:, :, ::-1]
        cv2.imwrite(str(out/f'composition_{index}.png'), np.concatenate([source, combined_bgr], axis=1))
        (out/f'composition_{index}.json').write_text(json.dumps(combined_meta, indent=2))
    manifest = {'version': atoms.ATOMIC_VERSION, 'schema': atoms.PARAMETER_SCHEMA,
                'frozen_sha256': sha(frozen), 'atomic_sha256': sha(ROOT/'realesrgan/lenovo_atomic.py'),
                'adapter_sha256': sha(ROOT/'realesrgan/utils/lenovo_degradation.py'),
                'automatic_equivalence': 'passed', 'records': records,
                'visual_review': 'pending',
                'c3_semantic_review': 'pending: frozen independent UV fields need hue/denoise-confidence review',
                'formal_training_authorized': False}
    manifest_path = out/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'manifest.sha256').write_text(sha(manifest_path))
    print(f'PASS: {len(records)} exact reference/shared/training comparisons; {manifest_path}')


if __name__ == '__main__':
    main()
