"""Auto-discovery wrapper for the Sprint-1 RealCE dataset.

`realesrgan.data.__init__` imports only modules whose filenames end in
`_dataset.py`.  The implementation intentionally lives in the versioned
module `realesrgan_paired_dataset_sprint1.py`; importing it here registers
`RealESRGANPairedTextCenterDegradeDatasetSprint1` without modifying existing
source modules.
"""

from .realesrgan_paired_dataset_sprint1 import RealESRGANPairedTextCenterDegradeDatasetSprint1

__all__ = ['RealESRGANPairedTextCenterDegradeDatasetSprint1']
