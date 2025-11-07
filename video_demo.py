"""
Video Demo Module

This module demonstrates FEC-protected video transmission over UDP
with a side-by-side comparison of vanilla UDP vs FEC-protected streams.
"""

import cv2
import numpy as np
import argparse
import threading
import time
import socket
from typing import Optional, Tuple
import sys
import warnings
import os

from fec import XORSimpleFEC, XORInterleavedFEC, XORDualParityFEC
from utils import chunk_data, create_packet, parse_packet, get_logger

# Suppress JPEG decode warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

logger = get_logger(__name__)


class VideoSender:
    """
    Sends video frames over UDP with optional FEC protection.
    """
    
    def __init__(self, video_path: str, host: str, port: int,
                 fec=None, fps: int = 30, quality: int = 50):
        """
        Initialize video sender.
        
        Args:
            video_path: Path to video file
            host: Destination host
            port: Destination port
            fec: FEC scheme (None for no FEC)
            fps: Target frames per second
            quality: JPEG quality (1-100)
        """
        self.video_path = video_path
        self.host = host
        self.port = port
        self.fec = fec
        self.fps = fps
        self.quality = quality
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.chunk_size = 1024
        self.running = False
        self.frame_delay = 1.0 / fps
    
    def send_frame(self, frame: np.ndarray, frame_num: int):
        """
        Send a single frame over UDP.
        
        Args:
            frame: Frame as numpy array
            frame_num: Frame number
        """
        # Encode frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_data = buffer.tobytes()
        
        # Add frame header (frame number + data length)
        import struct
        header = struct.pack('!II', frame_num, len(frame_data))
        frame_data = header + frame_data
        
        # Split into chunks
        chunks = chunk_data(frame_data, self.chunk_size)
        
        # Apply FEC if available
        if self.fec:
            # Process in blocks
            block_size = self.fec.block_size
            all_packets = []
            
            for i in range(0, len(chunks), block_size):
                block = chunks[i:i + block_size]
                while len(block) < block_size:
                    block.append(b'')
                
                encoded_block = self.fec.encode(block)
                all_packets.extend(encoded_block)
            
            chunks = all_packets
        
        # Send all chunks
        for i, chunk in enumerate(chunks):
            packet = create_packet(frame_num * 10000 + i, chunk,
                                 is_parity=(self.fec and i >= len(chunks) - self.fec.get_parity_count()))
            self.socket.sendto(packet, (self.host, self.port))
    
    def start(self):
        """Start sending video."""
        self.running = True
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return
        
        frame_num = 0
        logger.info(f"Sending video to {self.host}:{self.port}")
        
        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_num = 0
                    continue
                
                # Resize frame for faster transmission
                frame = cv2.resize(frame, (320, 240))
                
                # Send frame
                self.send_frame(frame, frame_num)
                frame_num += 1
                
                # Control frame rate
                time.sleep(self.frame_delay)
        
        finally:
            cap.release()
            self.socket.close()
    
    def stop(self):
        """Stop sending video."""
        self.running = False


class VideoReceiver:
    """
    Receives video frames over UDP with optional FEC recovery.
    """
    
    def __init__(self, port: int, fec=None, loss_rate: float = 0.0,
                 window_name: str = "Received"):
        """
        Initialize video receiver.
        
        Args:
            port: Listen port
            fec: FEC scheme (None for no FEC)
            loss_rate: Simulated packet loss rate
            window_name: OpenCV window name
        """
        self.port = port
        self.fec = fec
        self.loss_rate = loss_rate
        self.window_name = window_name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', port))
        self.socket.settimeout(0.1)
        self.running = False
        self.frame_buffer = {}
        self.current_frame = None
    
    def receive_and_display(self):
        """Receive and display video frames."""
        self.running = True
        logger.info(f"Receiving video on port {self.port}")
        
        import random
        
        try:
            while self.running:
                try:
                    packet, addr = self.socket.recvfrom(65535)
                    
                    # Simulate packet loss
                    if random.random() < self.loss_rate:
                        continue
                    
                    # Parse packet
                    seq_num, is_parity, data = parse_packet(packet)
                    
                    # Extract frame number
                    frame_num = seq_num // 10000
                    chunk_idx = seq_num % 10000
                    
                    # Store in buffer
                    if frame_num not in self.frame_buffer:
                        self.frame_buffer[frame_num] = {}
                    
                    self.frame_buffer[frame_num][chunk_idx] = (data, is_parity)
                    
                    # Try to reconstruct frame
                    if len(self.frame_buffer[frame_num]) > 5:  # Enough chunks
                        frame = self._reconstruct_frame(frame_num)
                        if frame is not None:
                            self.current_frame = frame
                            # Clean old frames
                            if frame_num in self.frame_buffer:
                                del self.frame_buffer[frame_num]
                
                except socket.timeout:
                    pass
                
                # Display current frame
                if self.current_frame is not None:
                    cv2.imshow(self.window_name, self.current_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            self.socket.close()
            cv2.destroyWindow(self.window_name)
    
    def _reconstruct_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Reconstruct a frame from received chunks.
        
        Args:
            frame_num: Frame number
            
        Returns:
            Reconstructed frame or None
        """
        if frame_num not in self.frame_buffer:
            return None
        
        chunks = self.frame_buffer[frame_num]
        sorted_chunks = sorted(chunks.items())
        
        # Extract data
        frame_data = b''
        for _, (data, is_parity) in sorted_chunks:
            if not is_parity:
                frame_data += data
        
        if len(frame_data) < 8:
            return None
        
        # Parse frame header
        import struct
        try:
            _, data_len = struct.unpack('!II', frame_data[:8])
            frame_bytes = frame_data[8:8 + data_len]
            
            # Check if we have enough data
            if len(frame_bytes) < data_len * 0.5:  # Less than 50% of expected data
                return self.current_frame  # Return previous frame instead
            
            # Decode JPEG with error suppression
            nparr = np.frombuffer(frame_bytes, np.uint8)
            
            # Suppress stderr temporarily to hide JPEG warnings
            old_stderr = sys.stderr
            try:
                sys.stderr = open(os.devnull, 'w')
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            finally:
                sys.stderr.close()
                sys.stderr = old_stderr
            
            # If decode failed, return previous frame
            if frame is None:
                return self.current_frame
            
            return frame
        except Exception:
            # Silently return previous frame on any error
            return self.current_frame
    
    def stop(self):
        """Stop receiving video."""
        self.running = False


def run_demo(video_path: str, fec_type: str, loss_rate: float, block_size: int):
    """
    Run side-by-side video demo comparing vanilla UDP vs FEC.
    
    Args:
        video_path: Path to video file
        fec_type: FEC scheme type
        loss_rate: Packet loss rate
        block_size: FEC block size
    """
    logger.info("=" * 60)
    logger.info("Starting Video Demo")
    logger.info(f"Video: {video_path}")
    logger.info(f"FEC: {fec_type}, Loss Rate: {loss_rate:.2%}")
    logger.info("=" * 60)
    
    # Create FEC schemes
    fec_vanilla = None
    
    if fec_type == 'xor_simple':
        fec_protected = XORSimpleFEC(block_size)
    elif fec_type == 'xor_interleaved':
        fec_protected = XORInterleavedFEC(block_size)
    elif fec_type == 'xor_dual_parity':
        fec_protected = XORDualParityFEC(block_size)
    else:
        fec_protected = None
    
    # Create senders and receivers
    sender_vanilla = VideoSender(video_path, '127.0.0.1', 11000, fec_vanilla)
    sender_protected = VideoSender(video_path, '127.0.0.1', 11001, fec_protected)
    
    receiver_vanilla = VideoReceiver(11000, fec_vanilla, loss_rate, "Vanilla UDP")
    receiver_protected = VideoReceiver(11001, fec_protected, loss_rate, "FEC Protected")
    
    # Start receivers in threads
    thread_recv_vanilla = threading.Thread(target=receiver_vanilla.receive_and_display)
    thread_recv_protected = threading.Thread(target=receiver_protected.receive_and_display)
    
    thread_recv_vanilla.start()
    thread_recv_protected.start()
    
    time.sleep(1)  # Give receivers time to start
    
    # Start senders in threads
    thread_send_vanilla = threading.Thread(target=sender_vanilla.start)
    thread_send_protected = threading.Thread(target=sender_protected.start)
    
    thread_send_vanilla.start()
    thread_send_protected.start()
    
    logger.info("Demo running. Press 'q' in video windows to quit.")
    
    try:
        # Wait for threads
        thread_recv_vanilla.join()
        thread_recv_protected.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop all
        sender_vanilla.stop()
        sender_protected.stop()
        receiver_vanilla.stop()
        receiver_protected.stop()
        
        cv2.destroyAllWindows()
        logger.info("Demo stopped")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Video Demo: UDP vs FEC-protected transmission'
    )
    parser.add_argument(
        '--video',
        default='sample.mp4',
        help='Path to video file'
    )
    parser.add_argument(
        '--fec',
        choices=['xor_simple', 'xor_interleaved', 'xor_dual_parity'],
        default='xor_simple',
        help='FEC scheme for protected stream'
    )
    parser.add_argument(
        '--loss_rate',
        type=float,
        default=0.2,
        help='Packet loss rate (0.0 to 1.0)'
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=4,
        help='FEC block size'
    )
    
    args = parser.parse_args()
    
    # Check if video file exists
    import os
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        logger.info("You can download a sample video or use your webcam (0) as input")
        sys.exit(1)
    
    run_demo(args.video, args.fec, args.loss_rate, args.block_size)


if __name__ == '__main__':
    main()
