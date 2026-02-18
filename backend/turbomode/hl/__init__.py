"""
HL Layer Package

Higher-Level (HL) analytics layer - READ-ONLY, ADVISORY-ONLY.

IMPORTANT CONSTRAINTS:
- HL does NOT generate trading signals
- HL does NOT open or close positions
- HL does NOT modify core engine behavior
- HL is purely informational and analytical
"""

from .hl_model import (
    HLOutput,
    HLTrendBlock,
    HLVolumeBlock,
    HLVolatilityBlock,
    HLProbabilityDriftBlock,
    HLStructureBlock
)
from .hl_service import build_hl_output

__all__ = [
    'HLOutput',
    'HLTrendBlock',
    'HLVolumeBlock',
    'HLVolatilityBlock',
    'HLProbabilityDriftBlock',
    'HLStructureBlock',
    'build_hl_output'
]
