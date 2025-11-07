"""
Base FEC (Forward Error Correction) Abstract Class

This module defines the interface that all FEC schemes must implement.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseFEC(ABC):
    """
    Abstract base class for FEC schemes.
    
    All FEC implementations must inherit from this class and implement
    the encode() and decode() methods.
    """
    
    def __init__(self, block_size: int):
        """
        Initialize the FEC scheme.
        
        Args:
            block_size: Number of data packets per block
        """
        self.block_size = block_size
    
    @abstractmethod
    def encode(self, packets: List[bytes]) -> List[bytes]:
        """
        Encode data packets by adding FEC/parity packets.
        
        Args:
            packets: List of data packets (each packet is bytes)
            
        Returns:
            List of packets including both data and parity packets
        """
        pass
    
    @abstractmethod
    def decode(self, packets: List[bytes], loss_map: List[int]) -> List[bytes]:
        """
        Decode and recover missing packets using FEC.
        
        Args:
            packets: List of received packets (None for missing packets)
            loss_map: List of indices indicating which packets were lost
            
        Returns:
            List of recovered data packets (original data only, no parity)
        """
        pass
    
    def get_overhead(self) -> float:
        """
        Calculate the overhead ratio introduced by this FEC scheme.
        
        Returns:
            Overhead ratio (parity_packets / data_packets)
        """
        return self.get_parity_count() / self.block_size
    
    @abstractmethod
    def get_parity_count(self) -> int:
        """
        Get the number of parity packets added per block.
        
        Returns:
            Number of parity packets
        """
        pass
