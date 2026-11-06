"""Isolated BasicSR entry for the four native-region V3 experiments."""
from pathlib import Path
import sys
import importlib
import types
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def calculate_metric(data, opt):
    """@luna-comment Input: metric data/options. Output: registered metric result.
    Key Steps: copy options and dispatch. Key Params: type. State/Dependencies: registry.
    """
    from basicsr.utils.registry import METRIC_REGISTRY
    options = deepcopy(opt)
    return METRIC_REGISTRY.get(options.pop('type'))(**data, **options)


def bootstrap():
    """@luna-comment Input: clean process. Output: explicit V3/dataset/loss/model registrations.
    Key Steps: use safe process-local bootstrap and register only required extensions.
    Key Params: CP enabled. State/Dependencies: process registry, no legacy source changes.
    """
    for name in ('basicsr', 'realesrgan', 'basicsr.metrics', 'realesrgan.data',
                 'realesrgan.archs', 'realesrgan.losses', 'realesrgan.models', 'realesrgan.metrics'):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT.joinpath(*name.split('.')))]
            package.__package__ = name
            sys.modules[name] = package
    import basicsr.utils as utilities
    metrics = sys.modules['basicsr.metrics']
    metrics.calculate_metric = calculate_metric
    image_metrics = importlib.import_module('basicsr.metrics.psnr_ssim')
    metrics.calculate_psnr = image_metrics.calculate_psnr
    metrics.calculate_ssim = image_metrics.calculate_ssim
    original_scan = utilities.scandir
    allowed = {'archs': set(), 'data': set(), 'models': {'sr_model.py'}, 'losses': {'basic_loss.py', 'gan_loss.py'}}

    def selected_scan(directory, *args, **kwargs):
        """@luna-comment Input: registry directory and scanner args. Output: filtered iterator.
        Key Steps: restrict BasicSR discovery to required modules. Key Params: local allowlist.
        State/Dependencies: original scanner; temporary process-local binding only.
        """
        path = Path(directory).resolve()
        choices = allowed.get(path.name) if path.parent == ROOT/'basicsr' else None
        entries = original_scan(directory, *args, **kwargs)
        return entries if choices is None else (p for p in entries if Path(p).name in choices)

    try:
        utilities.scandir = selected_scan
        for name in ('basicsr.archs', 'basicsr.losses', 'basicsr.data', 'basicsr.models'):
            importlib.import_module(name)
    finally:
        utilities.scandir = original_scan
    for name in ('realesrgan.archs.basic_text_line_r4_student_y4uv_v3_arch',
                 'realesrgan.losses.crnn_content_perceptual_loss',
                 'realesrgan.losses.background_low_frequency_loss',
                 'realesrgan.data.native_text_region_dataset',
                 'realesrgan.models.native_region_0921_model'):
        importlib.import_module(name)


def main():
    """@luna-comment Input: BasicSR -opt CLI. Output: independent training artifacts.
    Key Steps: register extensions and delegate original train_pipeline.
    Key Params: all training settings from YAML. State/Dependencies: BasicSR and CUDA.
    """
    bootstrap()
    from basicsr.train import train_pipeline
    train_pipeline(str(ROOT))


if __name__ == '__main__':
    main()
