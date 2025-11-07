"""
Generate a simple test video for FEC video demo testing.
"""

import cv2
import numpy as np
import argparse


def generate_test_video(output_path: str, duration: int = 10, fps: int = 30):
    """
    Generate a test video with moving patterns.
    
    Args:
        output_path: Output video file path
        duration: Duration in seconds
        fps: Frames per second
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    print(f"Generating {total_frames} frames at {fps} FPS...")
    
    for frame_num in range(total_frames):
        # Create a frame with moving patterns
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add colorful background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(128 + 127 * np.sin(2 * np.pi * (x + frame_num) / width)),
                    int(128 + 127 * np.sin(2 * np.pi * (y + frame_num) / height)),
                    int(128 + 127 * np.sin(2 * np.pi * (x + y + frame_num) / (width + height)))
                ]
        
        # Add moving circle
        center_x = int(width / 2 + width / 4 * np.sin(2 * np.pi * frame_num / total_frames))
        center_y = int(height / 2 + height / 4 * np.cos(2 * np.pi * frame_num / total_frames))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add frame counter text
        text = f"Frame {frame_num + 1}/{total_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add test pattern label
        cv2.putText(frame, "FEC Test Video", (width // 2 - 60, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        
        if (frame_num + 1) % 30 == 0:
            print(f"  Generated {frame_num + 1}/{total_frames} frames...")
    
    out.release()
    print(f"Test video saved to: {output_path}")
    print(f"Duration: {duration}s, Resolution: {width}x{height}, FPS: {fps}")


def main():
    parser = argparse.ArgumentParser(description='Generate test video for FEC demo')
    parser.add_argument('--output', default='test_video.mp4', help='Output video path')
    parser.add_argument('--duration', type=int, default=10, help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    
    args = parser.parse_args()
    
    generate_test_video(args.output, args.duration, args.fps)
    
    print("\nNow you can test the video demo with:")
    print(f"  python video_demo.py --video {args.output} --fec xor_simple --loss_rate 0.2")


if __name__ == '__main__':
    main()
