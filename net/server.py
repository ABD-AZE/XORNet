"""
UDP Server

This module implements a UDP sender that supports FEC encoding.
It sends data packets with optional FEC protection.
"""

import socket
import time
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fec.base_fec import BaseFEC
from utils import chunk_data, create_packet, get_logger, Timer


logger = get_logger(__name__)


class UDPServer:
    """
    UDP server that sends data with optional FEC protection.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 9999, 
                 fec: Optional[BaseFEC] = None, packet_size: int = 1024):
        """
        Initialize UDP server.
        
        Args:
            host: Server host address
            port: Server port
            fec: FEC scheme to use (None for no FEC)
            packet_size: Size of each data packet in bytes
        """
        self.host = host
        self.port = port
        self.fec = fec
        self.packet_size = packet_size
        self.socket = None
        self.seq_num = 0
    
    def start(self):
        """Start the UDP server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"UDP Server started on {self.host}:{self.port}")
        logger.info(f"FEC: {type(self.fec).__name__ if self.fec else 'None'}")
    
    def send_data(self, data: bytes, client_address: tuple) -> dict:
        """
        Send data to a client with FEC protection.
        
        Args:
            data: Data to send
            client_address: Tuple of (host, port) for the client
            
        Returns:
            Dictionary with transmission statistics
        """
        if not self.socket:
            raise RuntimeError("Server not started. Call start() first.")
        
        timer = Timer()
        timer.start()
        
        # Split data into chunks
        chunks = chunk_data(data, self.packet_size)
        logger.info(f"Sending {len(chunks)} data packets ({len(data)} bytes)")
        
        stats = {
            'total_packets': 0,
            'data_packets': len(chunks),
            'parity_packets': 0,
            'data_bytes': len(data),
            'parity_bytes': 0,
            'duration': 0.0,
        }
        
        # Send data with or without FEC
        if self.fec:
            self._send_with_fec(chunks, client_address, stats)
        else:
            self._send_without_fec(chunks, client_address, stats)
        
        timer.stop()
        stats['duration'] = timer.elapsed()
        
        logger.info(f"Transmission complete: {stats['total_packets']} packets "
                   f"in {stats['duration']:.2f}s")
        
        return stats
    
    def _send_without_fec(self, chunks: list, client_address: tuple, stats: dict):
        """
        Send data chunks without FEC protection.
        
        Args:
            chunks: List of data chunks
            client_address: Client address
            stats: Statistics dictionary to update
        """
        for chunk in chunks:
            packet = create_packet(self.seq_num, chunk, is_parity=False)
            self.socket.sendto(packet, client_address)
            self.seq_num += 1
            stats['total_packets'] += 1
            
            # Small delay to avoid overwhelming the receiver
            time.sleep(0.001)
    
    def _send_with_fec(self, chunks: list, client_address: tuple, stats: dict):
        """
        Send data chunks with FEC protection.
        
        Args:
            chunks: List of data chunks
            client_address: Client address
            stats: Statistics dictionary to update
        """
        block_size = self.fec.block_size
        
        # Process data in blocks
        for i in range(0, len(chunks), block_size):
            block = chunks[i:i + block_size]
            
            # Pad block if necessary
            while len(block) < block_size:
                block.append(b'')
            
            # Encode block with FEC
            encoded_block = self.fec.encode(block)
            
            # Send all packets in the block (data + parity)
            for j, packet_data in enumerate(encoded_block):
                is_parity = j >= block_size
                packet = create_packet(self.seq_num, packet_data, is_parity=is_parity)
                self.socket.sendto(packet, client_address)
                
                if is_parity:
                    stats['parity_packets'] += 1
                    stats['parity_bytes'] += len(packet_data)
                
                self.seq_num += 1
                stats['total_packets'] += 1
                
                # Small delay to avoid overwhelming the receiver
                time.sleep(0.001)
        
        logger.info(f"Sent {stats['data_packets']} data packets + "
                   f"{stats['parity_packets']} parity packets")
    
    def send_end_marker(self, client_address: tuple):
        """
        Send an end-of-transmission marker.
        
        Args:
            client_address: Client address
        """
        # Send empty packet with special sequence number
        packet = create_packet(0xFFFFFFFF, b'END', is_parity=False)
        self.socket.sendto(packet, client_address)
        logger.info("Sent end-of-transmission marker")
    
    def close(self):
        """Close the server socket."""
        if self.socket:
            self.socket.close()
            logger.info("UDP Server closed")


def main():
    """Main function for testing the server."""
    import argparse
    from fec import XORSimpleFEC, XORInterleavedFEC, XORDualParityFEC
    from utils import generate_random_data
    
    parser = argparse.ArgumentParser(description='UDP Server with FEC')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=9999, help='Server port')
    parser.add_argument('--fec', choices=['none', 'xor_simple', 'xor_interleaved', 'xor_dual_parity'],
                       default='none', help='FEC scheme')
    parser.add_argument('--block_size', type=int, default=4, help='FEC block size')
    parser.add_argument('--data_size', type=int, default=10240, help='Data size in bytes')
    parser.add_argument('--client_host', default='127.0.0.1', help='Client host')
    parser.add_argument('--client_port', type=int, default=10000, help='Client port')
    
    args = parser.parse_args()
    
    # Create FEC scheme
    fec = None
    if args.fec == 'xor_simple':
        fec = XORSimpleFEC(args.block_size)
    elif args.fec == 'xor_interleaved':
        fec = XORInterleavedFEC(args.block_size)
    elif args.fec == 'xor_dual_parity':
        fec = XORDualParityFEC(args.block_size)
    
    # Create and start server
    server = UDPServer(args.host, args.port, fec)
    server.start()
    
    try:
        # Generate and send random data
        data = generate_random_data(args.data_size)
        client_addr = (args.client_host, args.client_port)
        
        logger.info(f"Sending data to {client_addr}")
        stats = server.send_data(data, client_addr)
        server.send_end_marker(client_addr)
        
        logger.info(f"Statistics: {stats}")
    finally:
        server.close()


if __name__ == '__main__':
    main()
