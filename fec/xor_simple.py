"""
Simple XOR FEC Implementation

This scheme creates one parity packet per block by XORing all data packets.
Recovery is possible if only one packet is lost per block.

Formula: p = pkt1 ⊕ pkt2 ⊕ ... ⊕ pktN
"""

from typing import List
from .base_fec import BaseFEC


class XORSimpleFEC(BaseFEC):
    """
    Simple XOR FEC: 1 parity packet per block.
    
    The parity packet is the XOR of all data packets in the block.
    Can recover from a single packet loss per block.
    """
    
    def __init__(self, block_size: int):
        """
        Initialize Simple XOR FEC.
        
        Args:
            block_size: Number of data packets per block
        """
        super().__init__(block_size)
    
    def encode(self, packets: List[bytes]) -> List[bytes]:
        """
        Encode by adding one XOR parity packet at the end.
        
        Args:
            packets: List of data packets
            
        Returns:
            List containing data packets + 1 parity packet
        """
        if not packets:
            return []
        
        # Pad all packets to the same length
        max_len = max(len(p) for p in packets)
        padded_packets = [p + b'\x00' * (max_len - len(p)) for p in packets]
        
        # Calculate XOR parity
        parity = bytearray(max_len)
        for packet in padded_packets:
            for i, byte in enumerate(packet):
                parity[i] ^= byte
        
        # Return data packets + parity packet
        return packets + [bytes(parity)]
    
    def decode(self, packets: List[bytes], loss_map: List[int]) -> List[bytes]:
        """
        Decode and recover missing packet using XOR parity.
        
        Args:
            packets: List of packets (None for missing packets)
            loss_map: Indices of lost packets
            
        Returns:
            List of recovered data packets (excluding parity)
        """
        # If no losses, just return data packets (exclude parity)
        if not loss_map:
            return packets[:-1]
        
        # Can only recover if exactly one packet is lost
        if len(loss_map) > 1:
            # Cannot recover multiple losses with simple XOR
            # Return what we have (excluding parity)
            return [p if p is not None else b'' for p in packets[:-1]]
        
        lost_idx = loss_map[0]
        
        # If parity packet is lost, no recovery needed
        if lost_idx == len(packets) - 1:
            return packets[:-1]
        
        # Recover the lost data packet
        max_len = max(len(p) for p in packets if p is not None)
        recovered = bytearray(max_len)
        
        for i, packet in enumerate(packets):
            if packet is not None:
                padded = packet + b'\x00' * (max_len - len(packet))
                for j, byte in enumerate(padded):
                    recovered[j] ^= byte
        
        # Insert recovered packet at the correct position
        result = []
        for i in range(len(packets) - 1):
            if i == lost_idx:
                result.append(bytes(recovered))
            else:
                result.append(packets[i])
        
        return result
    
    def get_parity_count(self) -> int:
        """Returns 1 parity packet per block."""
        return 1
