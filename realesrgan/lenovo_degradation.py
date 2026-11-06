"""Import bridge for the Lenovo degradation module.

This repository already has ``realesrgan/utils.py`` as a module, so Python
cannot import the plan's filesystem location ``realesrgan/utils/...`` as a
subpackage.  Keep the implementation at that reviewed location and expose a
normal package import for model code.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_path = Path(__file__).with_name("utils") / "lenovo_degradation.py"
_spec = spec_from_file_location("realesrgan._lenovo_degradation_impl", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Lenovo degradation implementation from {_path}")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

DEFAULT_LENOVO_DEGRADATION_CONFIG = _module.DEFAULT_LENOVO_DEGRADATION_CONFIG
LenovoDegradationPipeline = _module.LenovoDegradationPipeline

__all__ = ["DEFAULT_LENOVO_DEGRADATION_CONFIG", "LenovoDegradationPipeline"]
