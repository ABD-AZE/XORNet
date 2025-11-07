"""
Simulation Module

This module runs complete UDP transmission simulations with various
FEC schemes and loss rates, collecting metrics for analysis.
"""

import argparse
import json
import os
import sys
import time
import threading
from typing import Optional

from fec import XORSimpleFEC, XORInterleavedFEC, XORDualParityFEC
from net import UDPServer, UDPClient
from utils import generate_random_data, get_logger


logger = get_logger(__name__)


class Simulation:
    """
    Manages a complete UDP transmission simulation.
    """
    
    def __init__(self, fec_type: str, loss_rate: float, block_size: int = 4,
                 data_size: int = 10240, packet_size: int = 1024):
        """
        Initialize simulation.
        
        Args:
            fec_type: FEC scheme ('none', 'xor_simple', 'xor_interleaved', 'xor_dual_parity')
            loss_rate: Packet loss rate (0.0 to 1.0)
            block_size: FEC block size
            data_size: Total data size in bytes
            packet_size: Size of each packet in bytes
        """
        self.fec_type = fec_type
        self.loss_rate = loss_rate
        self.block_size = block_size
        self.data_size = data_size
        self.packet_size = packet_size
        
        # Create FEC scheme
        self.fec = self._create_fec()
        
        # Network configuration
        self.server_host = '127.0.0.1'
        self.server_port = 9999
        self.client_host = '127.0.0.1'
        self.client_port = 10000
    
    def _create_fec(self) -> Optional[object]:
        """
        Create FEC scheme based on type.
        
        Returns:
            FEC instance or None
        """
        if self.fec_type == 'none':
            return None
        elif self.fec_type == 'xor_simple':
            return XORSimpleFEC(self.block_size)
        elif self.fec_type == 'xor_interleaved':
            return XORInterleavedFEC(self.block_size)
        elif self.fec_type == 'xor_dual_parity':
            return XORDualParityFEC(self.block_size)
        else:
            raise ValueError(f"Unknown FEC type: {self.fec_type}")
    
    def run(self) -> dict:
        """
        Run the simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        logger.info("=" * 60)
        logger.info(f"Starting simulation")
        logger.info(f"FEC: {self.fec_type}, Loss Rate: {self.loss_rate:.2%}, "
                   f"Block Size: {self.block_size}")
        logger.info("=" * 60)
        
        # Generate test data
        data = generate_random_data(self.data_size)
        logger.info(f"Generated {self.data_size} bytes of test data")
        
        # Create server and client
        server = UDPServer(
            self.server_host,
            self.server_port,
            self.fec,
            self.packet_size
        )
        
        client = UDPClient(
            self.client_host,
            self.client_port,
            self.fec,
            self.loss_rate
        )
        
        # Start client in a separate thread
        client_thread = threading.Thread(
            target=self._run_client,
            args=(client,)
        )
        
        try:
            # Start client
            client.start()
            client_thread.start()
            
            # Give client time to start
            time.sleep(0.5)
            
            # Start server and send data
            server.start()
            server_stats = server.send_data(
                data,
                (self.client_host, self.client_port)
            )
            server.send_end_marker((self.client_host, self.client_port))
            
            # Wait for client to finish
            client_thread.join(timeout=10.0)
            
            # Get metrics from client
            metrics = client.get_metrics()
            
            # Compile results
            results = self._compile_results(server_stats, metrics)
            
            logger.info("=" * 60)
            logger.info("Simulation complete")
            logger.info("=" * 60)
            logger.info(f"\n{metrics}")
            
            return results
        
        finally:
            server.close()
            client.close()
    
    def _run_client(self, client):
        """
        Run client in a separate thread.
        
        Args:
            client: UDPClient instance
        """
        try:
            received_data = client.receive_data()
            logger.info(f"Client received {len(received_data)} bytes")
        except Exception as e:
            logger.error(f"Client error: {e}")
    
    def _compile_results(self, server_stats: dict, metrics) -> dict:
        """
        Compile simulation results.
        
        Args:
            server_stats: Server statistics
            metrics: TransmissionMetrics object
            
        Returns:
            Dictionary with compiled results
        """
        metrics_dict = metrics.to_dict()
        
        results = {
            'fec': self.fec_type,
            'loss_rate': self.loss_rate,
            'block_size': self.block_size,
            'data_size': self.data_size,
            'packet_size': self.packet_size,
            
            # Packet statistics
            'packets_sent': server_stats['total_packets'],
            'packets_received': metrics_dict['total_received'],
            'packets_lost': metrics_dict['total_lost'],
            'packets_recovered': metrics_dict['total_recovered'],
            
            # Performance metrics
            'loss_rate_actual': metrics_dict['loss_rate'],
            'recovery_ratio': metrics_dict['recovery_ratio'],
            'fec_overhead': metrics_dict['fec_overhead'],
            'bandwidth_mbps': metrics_dict['bandwidth_mbps'],
            'goodput_mbps': metrics_dict['goodput_mbps'],
            'latency_avg_ms': metrics_dict['latency_avg_ms'],
            'latency_stddev_ms': metrics_dict['latency_stddev_ms'],
            'recovery_time_avg_ms': metrics_dict['recovery_time_avg_ms'],
            'duration_seconds': metrics_dict['duration_seconds'],
        }
        
        return results


def save_results(results: dict, output_file: str = 'results.json'):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    # Load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = [all_results]
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []
    
    # Append new results
    all_results.append(results)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run UDP transmission simulation with FEC'
    )
    parser.add_argument(
        '--fec',
        choices=['none', 'xor_simple', 'xor_interleaved', 'xor_dual_parity'],
        default='none',
        help='FEC scheme to use'
    )
    parser.add_argument(
        '--loss_rate',
        type=float,
        default=0.1,
        help='Packet loss rate (0.0 to 1.0)'
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=4,
        help='FEC block size'
    )
    parser.add_argument(
        '--data_size',
        type=int,
        default=10240,
        help='Data size in bytes'
    )
    parser.add_argument(
        '--packet_size',
        type=int,
        default=1024,
        help='Packet size in bytes'
    )
    parser.add_argument(
        '--output',
        default='results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Validate loss rate
    if not 0.0 <= args.loss_rate <= 1.0:
        logger.error("Loss rate must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Run simulation
    sim = Simulation(
        args.fec,
        args.loss_rate,
        args.block_size,
        args.data_size,
        args.packet_size
    )
    
    results = sim.run()
    
    # Save results
    save_results(results, args.output)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
