"""
Metrics Module

This module provides functions for calculating and tracking various
performance metrics for FEC-protected UDP transmission.
"""

from typing import Dict, List
import statistics


class TransmissionMetrics:
    """
    Tracks metrics for a UDP transmission session.
    """
    
    def __init__(self):
        self.total_sent = 0
        self.total_received = 0
        self.total_lost = 0
        self.total_recovered = 0
        self.data_bytes_sent = 0
        self.parity_bytes_sent = 0
        self.recovery_times = []  # Time to recover packets in ms
        self.transmission_start = None
        self.transmission_end = None
    
    def record_sent(self, is_parity: bool, size: int):
        """
        Record a sent packet.
        
        Args:
            is_parity: Whether the packet is a parity packet
            size: Size of the packet in bytes
        """
        self.total_sent += 1
        if is_parity:
            self.parity_bytes_sent += size
        else:
            self.data_bytes_sent += size
    
    def record_received(self):
        """Record a received packet."""
        self.total_received += 1
    
    def record_lost(self):
        """Record a lost packet."""
        self.total_lost += 1
    
    def record_recovered(self, recovery_time_ms: float = 0):
        """
        Record a recovered packet.
        
        Args:
            recovery_time_ms: Time taken to recover in milliseconds
        """
        self.total_recovered += 1
        if recovery_time_ms > 0:
            self.recovery_times.append(recovery_time_ms)
    

    
    def get_loss_rate(self) -> float:
        """
        Calculate packet loss rate.
        
        Returns:
            Loss rate as a fraction (0.0 to 1.0)
        """
        if self.total_sent == 0:
            return 0.0
        return self.total_lost / self.total_sent
    
    def get_recovery_ratio(self) -> float:
        """
        Calculate recovery ratio.
        
        Returns:
            Ratio of recovered packets to lost packets (0.0 to 1.0)
        """
        if self.total_lost == 0:
            return 1.0
        return min(1.0, self.total_recovered / self.total_lost)
    
    def get_fec_overhead(self) -> float:
        """
        Calculate FEC overhead.
        
        Returns:
            Overhead ratio (parity_bytes / data_bytes)
        """
        if self.data_bytes_sent == 0:
            return 0.0
        return self.parity_bytes_sent / self.data_bytes_sent
    
    def get_bandwidth_mbps(self, duration: float = None) -> float:
        """
        Calculate bandwidth in Mbps.
        
        Args:
            duration: Duration in seconds (if None, use recorded transmission time)
            
        Returns:
            Bandwidth in Mbps
        """
        if duration is None:
            if self.transmission_start is None or self.transmission_end is None:
                return 0.0
            duration = self.transmission_end - self.transmission_start
        
        if duration <= 0:
            return 0.0
        
        total_bytes = self.data_bytes_sent + self.parity_bytes_sent
        bits = total_bytes * 8
        return bits / (duration * 1_000_000)
    
    def get_goodput_mbps(self, duration: float = None) -> float:
        """
        Calculate goodput (useful data throughput) in Mbps.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Goodput in Mbps
        """
        if duration is None:
            if self.transmission_start is None or self.transmission_end is None:
                return 0.0
            duration = self.transmission_end - self.transmission_start
        
        if duration <= 0:
            return 0.0
        
        # Only count successfully received data bytes
        received_data_bytes = self.data_bytes_sent * (self.total_received / max(1, self.total_sent))
        bits = received_data_bytes * 8
        return bits / (duration * 1_000_000)
    

    
    def get_average_recovery_time_ms(self) -> float:
        """
        Get average recovery time.
        
        Returns:
            Average recovery time in milliseconds
        """
        if not self.recovery_times:
            return 0.0
        return statistics.mean(self.recovery_times)
    
    def to_dict(self) -> Dict:
        """
        Convert metrics to dictionary.
        
        Returns:
            Dictionary containing all metrics
        """
        duration = 0.0
        if self.transmission_start and self.transmission_end:
            duration = self.transmission_end - self.transmission_start
        
        return {
            'total_sent': self.total_sent,
            'total_received': self.total_received,
            'total_lost': self.total_lost,
            'total_recovered': self.total_recovered,
            'loss_rate': self.get_loss_rate(),
            'recovery_ratio': self.get_recovery_ratio(),
            'fec_overhead': self.get_fec_overhead(),
            'bandwidth_mbps': self.get_bandwidth_mbps(duration),
            'goodput_mbps': self.get_goodput_mbps(duration),
            'recovery_time_avg_ms': self.get_average_recovery_time_ms(),
            'duration_seconds': duration,
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.to_dict()
        return (
            f"Transmission Metrics:\n"
            f"  Packets: {metrics['total_sent']} sent, "
            f"{metrics['total_received']} received, "
            f"{metrics['total_lost']} lost, "
            f"{metrics['total_recovered']} recovered\n"
            f"  Loss Rate: {metrics['loss_rate']:.2%}\n"
            f"  Recovery Ratio: {metrics['recovery_ratio']:.2%}\n"
            f"  FEC Overhead: {metrics['fec_overhead']:.2%}\n"
            f"  Bandwidth: {metrics['bandwidth_mbps']:.2f} Mbps\n"
            f"  Goodput: {metrics['goodput_mbps']:.2f} Mbps\n"
            f"  Avg Recovery Time: {metrics['recovery_time_avg_ms']:.2f} ms\n"
            f"  Duration: {metrics['duration_seconds']:.2f} seconds"
        )
