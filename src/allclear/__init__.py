"""ALLClear Stage1 cloud/shadow removal implementation.

The mainline model uses explicit ALLClear clear/shadow/cloud masks for routing,
SoftShadow-style soft shadow removal, and DADIGAN/UFFC SAR-guided cloud generation.
"""

from src.allclear.model import AllClearTGDADSoftShadow, DADIGANBaseline

__all__ = ["AllClearTGDADSoftShadow", "DADIGANBaseline"]
