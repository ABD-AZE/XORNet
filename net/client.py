"""
UDP Client

This module implements a UDP receiver that simulates packet loss
and applies FEC decoding to recover lost packets.
"""

import socket
import time
import random
from typing import Optional, Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fec.base_fec import BaseFEC
from metrics import TransmissionMetrics
from utils import parse_packet, get_logger, Timer


logger = get_logger(__name__)


class UDPClient:
    """
    UDP client that receives data with simulated packet loss and FEC recovery.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 10000,
                 fec: Optional[BaseFEC] = None, loss_rate: float = 0.0):
        """
        Initialize UDP client.
        
        Args:
            host: Client host address
            port: Client port
            fec: FEC scheme to use (None for no FEC)
            loss_rate: Simulated packet loss rate (0.0 to 1.0)
        """
        self.host = host
        self.port = port
        self.fec = fec
        self.loss_rate = loss_rate
        self.socket = None
        self.metrics = TransmissionMetrics()
    
    def start(self):
        """Start the UDP client."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(5.0)  # 5 second timeout
        logger.info(f"UDP Client started on {self.host}:{self.port}")
        logger.info(f"FEC: {type(self.fec).__name__ if self.fec else 'None'}")
        logger.info(f"Loss rate: {self.loss_rate:.2%}")
    
    def receive_data(self, expected_blocks: Optional[int] = None) -> bytes:
        """
        Receive data from server with simulated loss and FEC recovery.
        
        Args:
            expected_blocks: Expected number of blocks (None for auto-detect)
            
        Returns:
            Received data
        """
        if not self.socket:
            raise RuntimeError("Client not started. Call start() first.")
        
        timer = Timer()
        timer.start()
        self.metrics.transmission_start = time.time()
        
        received_packets = []
        packet_map = {}  # seq_num -> (data, is_parity, received_time)
        
        logger.info("Waiting for packets...")
        
        try:
            while True:
                try:
                    # Receive packet
                    packet, addr = self.socket.recvfrom(65535)
                    receive_time = time.time()
                    
                    # Parse packet
                    seq_num, is_parity, data = parse_packet(packet)
                    
                    # Check for end marker
                    if seq_num == 0xFFFFFFFF:
                        logger.info("Received end-of-transmission marker")
                        break
                    
                    # Simulate packet loss
                    if random.random() < self.loss_rate:
                        logger.debug(f"Simulated loss: packet {seq_num}")
                        self.metrics.record_lost()
                        continue
                    
                    # Record received packet
                    self.metrics.record_received()
                    self.metrics.record_sent(is_parity, len(data))
                    packet_map[seq_num] = (data, is_parity, receive_time)
                    
                except socket.timeout:
                    logger.info("Receive timeout")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        self.metrics.transmission_end = time.time()
        timer.stop()
        
        logger.info(f"Received {len(packet_map)} packets in {timer.elapsed():.2f}s")
        
        # Reconstruct data
        data = self._reconstruct_data(packet_map)
        
        return data
    
    def _reconstruct_data(self, packet_map: Dict) -> bytes:
        """
        Reconstruct data from received packets with FEC recovery.
        
        Args:
            packet_map: Dictionary of seq_num -> (data, is_parity, time)
            
        Returns:
            Reconstructed data
        """
        if not packet_map:
            return b''
        
        # Sort packets by sequence number
        sorted_seqs = sorted(packet_map.keys())
        
        if not self.fec:
            # No FEC - just concatenate received data packets
            data = b''
            for seq in sorted_seqs:
                packet_data, is_parity, _ = packet_map[seq]
                if not is_parity:
                    data += packet_data
            return data
        
        # With FEC - process in blocks
        block_size = self.fec.block_size
        parity_count = self.fec.get_parity_count()
        full_block_size = block_size + parity_count
        
        recovered_data = b''
        
        # Group packets into blocks
        max_seq = max(sorted_seqs)
        num_blocks = (max_seq + full_block_size) // full_block_size
        
        for block_idx in range(num_blocks):
            block_start = block_idx * full_block_size
            block_end = block_start + full_block_size
            
            # Extract packets for this block
            block_packets = [None] * full_block_size
            loss_map = []
            
            for i in range(full_block_size):
                seq = block_start + i
                if seq in packet_map:
                    packet_data, is_parity, _ = packet_map[seq]
                    block_packets[i] = packet_data
                else:
                    loss_map.append(i)
            
            # Try to recover lost packets
            if loss_map:
                recovery_timer = Timer()
                recovery_timer.start()
                
                try:
                    recovered_block = self.fec.decode(block_packets, loss_map)
                    recovery_timer.stop()
                    
                    # Count successful recoveries
                    for idx in loss_map:
                        if idx < block_size:  # Only count data packet recoveries
                            if recovered_block[idx] and len(recovered_block[idx]) > 0:
                                self.metrics.record_recovered(recovery_timer.elapsed_ms())
                    
                    # Concatenate recovered data packets
                    for i in range(block_size):
                        if recovered_block[i]:
                            recovered_data += recovered_block[i]
                
                except Exception as e:
                    logger.error(f"FEC decode error in block {block_idx}: {e}")
                    # Use whatever we received
                    for i in range(block_size):
                        if block_packets[i] is not None:
                            recovered_data += block_packets[i]
            else:
                # No losses in this block - just use received data
                for i in range(block_size):
                    if block_packets[i] is not None:
                        recovered_data += block_packets[i]
        
        return recovered_data
    
    def get_metrics(self) -> TransmissionMetrics:
        """
        Get transmission metrics.
        
        Returns:
            TransmissionMetrics object
        """
        return self.metrics
    
    def close(self):
        """Close the client socket."""
        if self.socket:
            self.socket.close()
            logger.info("UDP Client closed")


def main():
    """Main function for testing the client."""
    import argparse
    from fec import XORSimpleFEC, XORInterleavedFEC, XORDualParityFEC
    
    parser = argparse.ArgumentParser(description='UDP Client with FEC')
    parser.add_argument('--host', default='127.0.0.1', help='Client host')
    parser.add_argument('--port', type=int, default=10000, help='Client port')
    parser.add_argument('--fec', choices=['none', 'xor_simple', 'xor_interleaved', 'xor_dual_parity'],
                       default='none', help='FEC scheme')
    parser.add_argument('--block_size', type=int, default=4, help='FEC block size')
    parser.add_argument('--loss_rate', type=float, default=0.1, help='Packet loss rate')
    
    args = parser.parse_args()
    
    # Create FEC scheme
    fec = None
    if args.fec == 'xor_simple':
        fec = XORSimpleFEC(args.block_size)
    elif args.fec == 'xor_interleaved':
        fec = XORInterleavedFEC(args.block_size)
    elif args.fec == 'xor_dual_parity':
        fec = XORDualParityFEC(args.block_size)
    
    # Create and start client
    client = UDPClient(args.host, args.port, fec, args.loss_rate)
    client.start()
    
    try:
        # Receive data
        data = client.receive_data()
        logger.info(f"Received {len(data)} bytes of data")
        
        # Print metrics
        metrics = client.get_metrics()
        logger.info(f"\n{metrics}")
    finally:
        client.close()


if __name__ == '__main__':
    main()
