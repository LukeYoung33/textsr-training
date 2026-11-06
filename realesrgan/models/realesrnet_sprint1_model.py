"""Auto-discovery wrapper for RealESRNetModelSprint1.

`realesrgan.models.__init__` scans filenames ending in `_model.py` only.
The versioned implementation is intentionally kept in
`realesrnet_model_sprint1.py`; importing it here registers the Sprint-1 model
without changing the baseline model module.
"""

from .realesrnet_model_sprint1 import RealESRNetModelSprint1

__all__ = ['RealESRNetModelSprint1']
