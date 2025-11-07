"""
Dual Parity XOR FEC Implementation

This scheme creates two independent parity packets per block:
1. Even-indexed packets XOR
2. Odd-indexed packets XOR

This provides better recovery capability for up to 2 packet losses per block.
"""

from typing import List
from .base_fec import BaseFEC


class XORDualParityFEC(BaseFEC):
    """
    Dual Parity XOR FEC: Two independent parity packets per block.
    
    Creates two parity packets:
    - Parity 1: XOR of even-indexed packets (0, 2, 4, ...)
    - Parity 2: XOR of odd-indexed packets (1, 3, 5, ...)
    
    Can recover from up to 2 packet losses per block in some cases.
    """
    
    def __init__(self, block_size: int):
        """
        Initialize Dual Parity XOR FEC.
        
        Args:
            block_size: Number of data packets per block
        """
        super().__init__(block_size)
    
    def encode(self, packets: List[bytes]) -> List[bytes]:
        """
        Encode by adding two parity packets (even and odd).
        
        Args:
            packets: List of data packets
            
        Returns:
            List containing data packets + 2 parity packets
        """
        if not packets:
            return []
        
        # Pad all packets to the same length
        max_len = max(len(p) for p in packets)
        padded_packets = [p + b'\x00' * (max_len - len(p)) for p in packets]
        
        # Calculate even-indexed parity (parity 1)
        parity_even = bytearray(max_len)
        for i in range(0, len(padded_packets), 2):
            for j, byte in enumerate(padded_packets[i]):
                parity_even[j] ^= byte
        
        # Calculate odd-indexed parity (parity 2)
        parity_odd = bytearray(max_len)
        for i in range(1, len(padded_packets), 2):
            for j, byte in enumerate(padded_packets[i]):
                parity_odd[j] ^= byte
        
        # Return data packets + both parity packets
        return packets + [bytes(parity_even), bytes(parity_odd)]
    
    def decode(self, packets: List[bytes], loss_map: List[int]) -> List[bytes]:
        """
        Decode and recover missing packets using dual parity.
        
        Args:
            packets: List of packets (None for missing packets)
            loss_map: Indices of lost packets
            
        Returns:
            List of recovered data packets (excluding parity)
        """
        num_data = len(packets) - 2
        
        # If no losses, return data packets only
        if not loss_map:
            return packets[:num_data]
        
        # Separate data and parity packets
        data_packets = list(packets[:num_data])
        parity_even = packets[num_data]
        parity_odd = packets[num_data + 1]
        
        # Filter for data packet losses only
        data_losses = [idx for idx in loss_map if idx < num_data]
        
        # Cannot recover if more than 2 data packets are lost
        if len(data_losses) > 2:
            return [p if p is not None else b'' for p in data_packets]
        
        # Recover lost packets
        for lost_idx in data_losses:
            max_len = max(len(p) for p in packets if p is not None)
            recovered = bytearray(max_len)
            
            # Determine which parity to use
            if lost_idx % 2 == 0:  # Even index - use even parity
                if parity_even is None:
                    continue
                
                # XOR with even parity
                padded_parity = parity_even + b'\x00' * (max_len - len(parity_even))
                for j, byte in enumerate(padded_parity):
                    recovered[j] ^= byte
                
                # XOR with all other even-indexed packets
                for i in range(0, num_data, 2):
                    if i != lost_idx and data_packets[i] is not None:
                        padded = data_packets[i] + b'\x00' * (max_len - len(data_packets[i]))
                        for j, byte in enumerate(padded):
                            recovered[j] ^= byte
            
            else:  # Odd index - use odd parity
                if parity_odd is None:
                    continue
                
                # XOR with odd parity
                padded_parity = parity_odd + b'\x00' * (max_len - len(parity_odd))
                for j, byte in enumerate(padded_parity):
                    recovered[j] ^= byte
                
                # XOR with all other odd-indexed packets
                for i in range(1, num_data, 2):
                    if i != lost_idx and data_packets[i] is not None:
                        padded = data_packets[i] + b'\x00' * (max_len - len(data_packets[i]))
                        for j, byte in enumerate(padded):
                            recovered[j] ^= byte
            
            data_packets[lost_idx] = bytes(recovered)
        
        return data_packets
    
    def get_parity_count(self) -> int:
        """Returns 2 parity packets per block."""
        return 2
