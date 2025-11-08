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
import contextlib
import pygame

from fec import XORSimpleFEC, XORInterleavedFEC, XORDualParityFEC
from utils import chunk_data, create_packet, parse_packet, get_logger

# Suppress all warnings and OpenCV messages
warnings.filterwarnings('ignore')
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'  # Only show fatal errors
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'  # Suppress Qt warnings
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11 backend instead of Wayland

# Redirect stderr to suppress JPEG warnings from C library
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)

logger = get_logger(__name__)


class VideoSender:
    """
    Sends video frames over UDP with optional FEC protection.
    """
    
    def __init__(self, video_path: str, host: str, port: int,
                 fec=None, fps: int = 30, quality: int = 50, loop: bool = True):
        """
        Initialize video sender.
        
        Args:
            video_path: Path to video file
            host: Destination host
            port: Destination port
            fec: FEC scheme (None for no FEC)
            fps: Target frames per second
            quality: JPEG quality (1-100)
            loop: Whether to loop the video
        """
        self.video_path = video_path
        self.host = host
        self.port = port
        self.fec = fec
        self.fps = fps
        self.quality = quality
        self.loop = loop
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.chunk_size = 1024
        self.running = False
        self.frame_delay = 1.0 / fps
        self.completed = False
    
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
        parity_flags = []  # Track which packets are parity
        if self.fec:
            # Process in blocks
            block_size = self.fec.block_size
            parity_count = self.fec.get_parity_count()
            all_packets = []
            
            for i in range(0, len(chunks), block_size):
                block = chunks[i:i + block_size]
                while len(block) < block_size:
                    block.append(b'')
                
                encoded_block = self.fec.encode(block)
                all_packets.extend(encoded_block)
                
                # Mark data packets as False, parity packets as True
                parity_flags.extend([False] * block_size)  # Data packets
                parity_flags.extend([True] * parity_count)  # Parity packets
            
            chunks = all_packets
        else:
            # No FEC - all packets are data
            parity_flags = [False] * len(chunks)
        
        # Send all chunks
        for i, chunk in enumerate(chunks):
            packet = create_packet(frame_num * 10000 + i, chunk,
                                 is_parity=parity_flags[i])
            self.socket.sendto(packet, (self.host, self.port))
    
    def start(self):
        """Start sending video."""
        self.running = True
        self.completed = False
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        logger.info(f"Sending video to {self.host}:{self.port} ({total_frames} frames)")
        
        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    if self.loop:
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_num = 0
                        logger.info(f"Video loop restarting (port {self.port})")
                        continue
                    else:
                        # Video completed
                        logger.info(f"Video transmission completed (port {self.port})")
                        self.completed = True
                        break
                
                # Resize frame for faster transmission
                frame = cv2.resize(frame, (320, 240))
                
                # Send frame
                self.send_frame(frame, frame_num)
                frame_num += 1
                
                # Log progress
                if frame_num % 30 == 0:  # Every second at 30 FPS
                    progress = (frame_num % total_frames) / total_frames * 100
                    logger.debug(f"Port {self.port}: Frame {frame_num % total_frames}/{total_frames} ({progress:.1f}%)")
                
                # Control frame rate with slightly longer delay for stability
                time.sleep(self.frame_delay * 1.2)
        
        finally:
            cap.release()
            self.socket.close()
            if not self.loop:
                logger.info(f"Sender stopped (port {self.port})")
    
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
        
        # Create window with proper flags - must be in main thread
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 320, 240)
        
        # Create a placeholder frame
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for video...", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.current_frame = placeholder
        
        try:
            while self.running:
                # Process packets in batches
                packets_processed = 0
                while packets_processed < 50:  # Process up to 50 packets before display update
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
                        
                        packets_processed += 1
                    
                    except socket.timeout:
                        break
                
                # Display current frame - CRITICAL: this must be called regularly
                if self.current_frame is not None:
                    cv2.imshow(self.window_name, self.current_frame)
                
                # Process GUI events - CRITICAL for window responsiveness
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            self.socket.close()
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass  # Ignore errors when destroying window
    
    def receive_only(self):
        """Receive video frames without display (for threading)."""
        self.running = True
        logger.info(f"Receiving video on port {self.port}")
        
        import random
        
        # Create a placeholder frame
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for video...", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.current_frame = placeholder
        
        # Track statistics
        packets_received = 0
        packets_dropped = 0
        
        try:
            while self.running:
                # Process packets
                try:
                    packet, addr = self.socket.recvfrom(65535)
                    packets_received += 1
                    
                    # Parse packet FIRST
                    seq_num, is_parity, data = parse_packet(packet)
                    
                    # Simulate packet loss - drop ANY packet (data or parity) randomly
                    # This simulates realistic network conditions
                    if random.random() < self.loss_rate:
                        packets_dropped += 1
                        # Still track that we expected this packet but didn't get it
                        # This is important for FEC to know which packets are missing
                        continue  # Drop this packet
                    
                    # Packet survived! Store it
                    # Extract frame number
                    frame_num = seq_num // 10000
                    chunk_idx = seq_num % 10000
                    
                    # Store in buffer
                    if frame_num not in self.frame_buffer:
                        self.frame_buffer[frame_num] = {}
                    
                    self.frame_buffer[frame_num][chunk_idx] = (data, is_parity)
                    
                    # Try to reconstruct frame
                    # For FEC, we need at least block_size packets out of block_size + parity
                    min_chunks_needed = 5 if not self.fec else self.fec.block_size
                    if len(self.frame_buffer[frame_num]) >= min_chunks_needed:
                        frame = self._reconstruct_frame(frame_num)
                        if frame is not None:
                            self.current_frame = frame
                            # Clean old frames
                            if frame_num in self.frame_buffer:
                                del self.frame_buffer[frame_num]
                
                except socket.timeout:
                    pass
        
        finally:
            if packets_received > 0:
                loss_pct = (packets_dropped / packets_received) * 100
                logger.info(f"Port {self.port} stats: {packets_dropped}/{packets_received} packets dropped ({loss_pct:.1f}%)")
            self.socket.close()
    
    def _reconstruct_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Reconstruct a frame from received chunks with FEC recovery.
        
        Args:
            frame_num: Frame number
            
        Returns:
            Reconstructed frame or None
        """
        if frame_num not in self.frame_buffer:
            return None
        
        chunks_dict = self.frame_buffer[frame_num]
        
        # If we have FEC, try to decode with FEC recovery
        if self.fec:
            # Group chunks into blocks for FEC decoding
            block_size = self.fec.block_size
            parity_count = self.fec.get_parity_count()
            total_per_block = block_size + parity_count
            
            # Determine how many blocks we need (estimate from max chunk index)
            if not chunks_dict:
                return None
            
            max_chunk_idx = max(chunks_dict.keys())
            num_blocks = (max_chunk_idx // total_per_block) + 1
            
            # Reconstruct data using FEC block by block
            recovered_chunks = []
            
            for block_idx in range(num_blocks):
                base_idx = block_idx * total_per_block
                
                # Collect chunks for this block
                block_chunks = []
                loss_map = []
                
                for i in range(total_per_block):
                    chunk_idx = base_idx + i
                    if chunk_idx in chunks_dict:
                        block_chunks.append(chunks_dict[chunk_idx][0])  # [0] is data, [1] is is_parity flag
                    else:
                        block_chunks.append(None)
                        loss_map.append(i)
                
                # Try to decode this block with FEC
                try:
                    decoded_block = self.fec.decode(block_chunks, loss_map)
                    # Ensure all chunks are bytes (not None)
                    for chunk in decoded_block:
                        if chunk is None:
                            recovered_chunks.append(b'')
                        else:
                            recovered_chunks.append(chunk)
                except Exception as e:
                    # FEC decode failed, use only available data chunks (skip parity)
                    for i in range(block_size):
                        if block_chunks[i] is not None:
                            recovered_chunks.append(block_chunks[i])
                        else:
                            # Missing chunk, can't recover
                            recovered_chunks.append(b'')
            
            # Combine recovered chunks (ensure all are bytes)
            frame_data = b''.join([c if c is not None else b'' for c in recovered_chunks])
        else:
            # No FEC - just concatenate data chunks in order
            sorted_chunks = sorted(chunks_dict.items())
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
            
            # Suppress stderr to hide JPEG warnings from C library
            try:
                with suppress_stderr():
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                frame = None
            
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


def run_demo(video_path: str, fec_type: str, loss_rate: float, block_size: int, loop: bool = True):
    """
    Run side-by-side video demo comparing vanilla UDP vs FEC.
    
    Args:
        video_path: Path to video file
        fec_type: FEC scheme type
        loss_rate: Packet loss rate
        block_size: FEC block size
        loop: Whether to loop the video
    """
    logger.info("=" * 60)
    logger.info("Starting Video Demo")
    logger.info(f"Video: {video_path}")
    logger.info(f"FEC: {fec_type}, Loss Rate: {loss_rate:.2%}")
    logger.info(f"Loop: {'Yes' if loop else 'No (will exit after one playthrough)'}")
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
    sender_vanilla = VideoSender(video_path, '127.0.0.1', 11000, fec_vanilla, loop=loop)
    sender_protected = VideoSender(video_path, '127.0.0.1', 11001, fec_protected, loop=loop)
    
    receiver_vanilla = VideoReceiver(11000, fec_vanilla, loss_rate, "Vanilla UDP")
    receiver_protected = VideoReceiver(11001, fec_protected, loss_rate, "FEC Protected")
    
    # Start receivers in threads (receive only, no display)
    thread_recv_vanilla = threading.Thread(target=receiver_vanilla.receive_only)
    thread_recv_protected = threading.Thread(target=receiver_protected.receive_only)
    
    thread_recv_vanilla.start()
    thread_recv_protected.start()
    
    time.sleep(1)  # Give receivers time to start
    
    # Start senders in threads
    thread_send_vanilla = threading.Thread(target=sender_vanilla.start)
    thread_send_protected = threading.Thread(target=sender_protected.start)
    
    thread_send_vanilla.start()
    thread_send_protected.start()
    
    logger.info("Demo running. Press 'q' or close the window to quit.")

    # Initialize Pygame
    pygame.init()
    screen_width = 640  # 320 * 2
    screen_height = 240
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("FEC Video Demo (Vanilla UDP vs FEC Protected)")
    
    # Create placeholder surfaces
    font = pygame.font.SysFont(None, 30)
    vanilla_placeholder_text = font.render('Waiting for Vanilla...', True, (255, 255, 255))
    fec_placeholder_text = font.render('Waiting for FEC...', True, (255, 255, 255))
    
    try:
        # Display loop in main thread
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            # Display frames from receivers
            # Vanilla frame
            if receiver_vanilla.current_frame is not None:
                frame_bgr = receiver_vanilla.current_frame
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                screen.blit(pygame_surface, (0, 0))
            else:
                screen.blit(vanilla_placeholder_text, (50, 100))

            # FEC protected frame
            if receiver_protected.current_frame is not None:
                frame_bgr = receiver_protected.current_frame
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                screen.blit(pygame_surface, (320, 0))
            else:
                screen.blit(fec_placeholder_text, (370, 100))

            pygame.display.flip()
            
            # Auto-exit if not looping and both senders completed
            if not loop and sender_vanilla.completed and sender_protected.completed:
                logger.info("✅ Both video streams completed transmission!")
                logger.info("Auto-exiting in 2 seconds...")
                time.sleep(2)
                running = False
            
            clock.tick(30)  # Limit display FPS to 30

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop all
        sender_vanilla.stop()
        sender_protected.stop()
        receiver_vanilla.stop()
        receiver_protected.stop()
        
        # Wait for threads to finish
        thread_recv_vanilla.join(timeout=2)
        thread_recv_protected.join(timeout=2)
        thread_send_vanilla.join(timeout=2)
        thread_send_protected.join(timeout=2)
        
        pygame.quit()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Demo Summary:")
        logger.info(f"  Vanilla sender completed: {sender_vanilla.completed}")
        logger.info(f"  FEC sender completed: {sender_protected.completed}")
        logger.info("=" * 60)
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
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Loop video continuously (default: play once and exit)'
    )
    parser.add_argument(
        '--no-loop',
        dest='loop',
        action='store_false',
        help='Play video once and exit'
    )
    parser.set_defaults(loop=False)  # Default is to play once
    
    args = parser.parse_args()
    
    # Check if video file exists
    import os
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        logger.info("You can download a sample video or use your webcam (0) as input")
        sys.exit(1)
    
    run_demo(args.video, args.fec, args.loss_rate, args.block_size, args.loop)


if __name__ == '__main__':
    main()
