"""Registry wrapper for the Sprint-1 ECBSRUnet_StdV6 architecture.

The implementation resides in `ecbsr_Unet_arch_server.py`, which is not part
of the project's `_arch.py` auto-discovery convention.  This module registers
the exact same implementation under the Sprint-1-specific name
`ECBSRUnet_StdV6Sprint1`, avoiding a collision with any legacy registry entry.
"""

from basicsr.utils.registry import ARCH_REGISTRY
from .ecbsr_Unet_arch_server import ECBSRUnet_StdV6


@ARCH_REGISTRY.register()
class ECBSRUnet_StdV6Sprint1(ECBSRUnet_StdV6):
    """Registry-visible alias of the validated ECBSRUnet_StdV6 implementation."""

    pass
