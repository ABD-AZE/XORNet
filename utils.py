"""
Utility Functions

This module provides helper functions for packet handling, timing,
logging, and other common operations.
"""

import time
import logging
from typing import List, Tuple
import struct


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def chunk_data(data: bytes, chunk_size: int) -> List[bytes]:
    """
    Split data into fixed-size chunks.
    
    Args:
        data: Data to split
        chunk_size: Size of each chunk in bytes
        
    Returns:
        List of data chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def create_packet(seq_num: int, data: bytes, is_parity: bool = False) -> bytes:
    """
    Create a packet with header.
    
    Packet format:
    - 4 bytes: sequence number
    - 1 byte: flags (bit 0: is_parity)
    - 4 bytes: data length
    - N bytes: data
    
    Args:
        seq_num: Sequence number
        data: Packet payload
        is_parity: Whether this is a parity packet
        
    Returns:
        Complete packet with header
    """
    flags = 1 if is_parity else 0
    header = struct.pack('!IBI', seq_num, flags, len(data))
    return header + data


def parse_packet(packet: bytes) -> Tuple[int, bool, bytes]:
    """
    Parse a packet and extract header information.
    
    Args:
        packet: Raw packet data
        
    Returns:
        Tuple of (seq_num, is_parity, data)
    """
    if len(packet) < 9:
        raise ValueError("Packet too short")
    
    seq_num, flags, data_len = struct.unpack('!IBI', packet[:9])
    is_parity = bool(flags & 1)
    data = packet[9:9 + data_len]
    
    return seq_num, is_parity, data


class Timer:
    """
    Simple timer for measuring elapsed time.
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        return self.elapsed() * 1000


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def calculate_bandwidth(data_size: int, duration: float) -> float:
    """
    Calculate bandwidth in Mbps.
    
    Args:
        data_size: Amount of data in bytes
        duration: Time duration in seconds
        
    Returns:
        Bandwidth in Mbps
    """
    if duration <= 0:
        return 0.0
    
    # Convert to bits and then to Mbps
    bits = data_size * 8
    return bits / (duration * 1_000_000)


def generate_random_data(size: int) -> bytes:
    """
    Generate random data for testing.
    
    Args:
        size: Size of data in bytes
        
    Returns:
        Random bytes
    """
    import random
    return bytes(random.getrandbits(8) for _ in range(size))
