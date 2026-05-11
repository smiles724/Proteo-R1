"""
Parse module for converting input files to Structure objects.

This module provides functions to parse YAML input files and convert them
to Structure objects compatible with the upstream Boltz1 data processing pipeline.
"""

from .schema import (
    parse_boltz_schema,
    Target,
    ParsedAtom,
    ParsedBond,
    ParsedResidue,
    ParsedChain,
)
from .yaml import parse_yaml

__all__ = [
    "parse_boltz_schema",
    "parse_yaml",
    "Target",
    "ParsedAtom",
    "ParsedBond",
    "ParsedResidue",
    "ParsedChain",
]
