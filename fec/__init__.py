"""
FEC Package - Forward Error Correction Schemes

This package provides various XOR-based FEC implementations for UDP packet transmission.
"""

from .base_fec import BaseFEC
from .xor_simple import XORSimpleFEC
from .xor_interleaved import XORInterleavedFEC
from .xor_dual_parity import XORDualParityFEC

__all__ = [
    'BaseFEC',
    'XORSimpleFEC',
    'XORInterleavedFEC',
    'XORDualParityFEC',
]
