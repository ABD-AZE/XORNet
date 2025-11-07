"""
Interleaved XOR FEC Implementation

This scheme creates parity packets across multiple blocks for better
resilience against burst losses. Parity packets protect packets from
different blocks in an interleaved pattern.

Example with 2 blocks:
- Parity 1: Block1_Pkt0 ⊕ Block2_Pkt1
- Parity 2: Block1_Pkt1 ⊕ Block2_Pkt0
"""

from typing import List
from .base_fec import BaseFEC


class XORInterleavedFEC(BaseFEC):
    """
    Interleaved XOR FEC: Cross-block parity for burst loss protection.
    
    Creates parity packets that span across multiple blocks, providing
    better protection against consecutive packet losses.
    """
    
    def __init__(self, block_size: int, interleave_depth: int = 2):
        """
        Initialize Interleaved XOR FEC.
        
        Args:
            block_size: Number of data packets per block
            interleave_depth: Number of blocks to interleave (default: 2)
        """
        super().__init__(block_size)
        self.interleave_depth = interleave_depth
    
    def encode(self, packets: List[bytes]) -> List[bytes]:
        """
        Encode with interleaved parity packets.
        
        Args:
            packets: List of data packets
            
        Returns:
            List containing data packets + interleaved parity packets
        """
        if not packets:
            return []
        
        # Pad all packets to the same length
        max_len = max(len(p) for p in packets)
        padded_packets = [p + b'\x00' * (max_len - len(p)) for p in packets]
        
        # Create interleaved parity packets
        parity_packets = []
        
        # Create one parity packet for each position in the block
        for offset in range(self.block_size):
            parity = bytearray(max_len)
            
            # XOR packets at different offsets (interleaved pattern)
            for i in range(len(padded_packets)):
                # Interleave pattern: use (i + offset) % block_size
                if (i % self.block_size) == offset:
                    for j, byte in enumerate(padded_packets[i]):
                        parity[j] ^= byte
            
            parity_packets.append(bytes(parity))
        
        # Return data packets + all interleaved parity packets
        return packets + parity_packets
    
    def decode(self, packets: List[bytes], loss_map: List[int]) -> List[bytes]:
        """
        Decode and recover missing packets using interleaved parity.
        
        Args:
            packets: List of packets (None for missing packets)
            loss_map: Indices of lost packets
            
        Returns:
            List of recovered data packets (excluding parity)
        """
        num_data = len(packets) - self.block_size
        
        # If no losses, return data packets only
        if not loss_map:
            return packets[:num_data]
        
        # Separate data and parity packets
        data_packets = packets[:num_data]
        parity_packets = packets[num_data:]
        
        # Try to recover lost data packets
        recovered_data = list(data_packets)
        
        for lost_idx in loss_map:
            if lost_idx >= num_data:
                # Lost packet is a parity packet, skip recovery
                continue
            
            # Find which parity packet can recover this lost packet
            parity_idx = lost_idx % self.block_size
            
            if parity_idx < len(parity_packets) and parity_packets[parity_idx] is not None:
                # Recover using the corresponding parity packet
                max_len = max(len(p) for p in packets if p is not None)
                recovered = bytearray(max_len)
                
                # XOR with parity packet
                parity = parity_packets[parity_idx]
                padded_parity = parity + b'\x00' * (max_len - len(parity))
                for j, byte in enumerate(padded_parity):
                    recovered[j] ^= byte
                
                # XOR with all other packets in the same interleave group
                for i in range(num_data):
                    if i != lost_idx and (i % self.block_size) == parity_idx:
                        if data_packets[i] is not None:
                            padded = data_packets[i] + b'\x00' * (max_len - len(data_packets[i]))
                            for j, byte in enumerate(padded):
                                recovered[j] ^= byte
                
                recovered_data[lost_idx] = bytes(recovered)
        
        return recovered_data
    
    def get_parity_count(self) -> int:
        """Returns block_size parity packets per block."""
        return self.block_size
