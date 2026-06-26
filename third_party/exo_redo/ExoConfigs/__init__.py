"""ExoConfigs module - Robot exoskeleton configurations with ArUco markers."""

from .exoskeleton import ExoskeletonConfig, LinkConfig
from .so100_holemounts import SO100_CONFIG
from .so100_adhesive import SO100_ADHESIVE_CONFIG
from .arx import ARX_CONFIG
from .puck import PUCK_CONFIG
from .alignment_board import ALIGNMENT_BOARD_CONFIG
from .umi_so100 import UMI_SO100_CONFIG
from .agilex_piper import AGILEX_PIPER_BASE_ONLY_CONFIG
from .panda_exo import PANDA_BASE_ONLY_CONFIG
from .yam_exo import YAM_BASE_ONLY_CONFIG
# Registry of available exoskeleton configurations
EXOSKELETON_CONFIGS = {
    "so100_holemounts": SO100_CONFIG,
    "so100_adhesive": SO100_ADHESIVE_CONFIG,
    "arx": ARX_CONFIG,
    "umi_so100": UMI_SO100_CONFIG,
    "agilex_piper_base_only": AGILEX_PIPER_BASE_ONLY_CONFIG,
    "panda_base_only": PANDA_BASE_ONLY_CONFIG,
    "yam_base_only": YAM_BASE_ONLY_CONFIG,
}

__all__ = [
    "ExoskeletonConfig",
    "LinkConfig",
    "SO100_CONFIG",
    "SO100_ADHESIVE_CONFIG",
    "ARX_CONFIG",
    "PUCK_CONFIG",
    "ALIGNMENT_BOARD_CONFIG",
    "UMI_SO100_CONFIG",
    "AGILEX_PIPER_BASE_ONLY_CONFIG",
    "PANDA_BASE_ONLY_CONFIG",
    "YAM_BASE_ONLY_CONFIG",
    "EXOSKELETON_CONFIGS",
]

