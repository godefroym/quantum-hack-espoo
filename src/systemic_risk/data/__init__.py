"""Data loaders and synthetic system generation."""

from systemic_risk.data.bank_asset_adapter import bank_asset_to_system_spec
from systemic_risk.data.huang_2008 import (
    HUANG_ASSET_NAMES,
    make_huang_2008_style_system,
)
from systemic_risk.data.synthetic import make_scalable_system, make_synthetic_system

__all__ = [
    "HUANG_ASSET_NAMES",
    "bank_asset_to_system_spec",
    "make_huang_2008_style_system",
    "make_scalable_system",
    "make_synthetic_system",
]
