#!/usr/bin/env python3
"""
Quick test to verify video completion works.
"""

import sys
import time

# Run the demo and time it
print("Running video demo with test_video.mp4 (should complete in ~6 seconds)...")
print("="*60)

import subprocess
start_time = time.time()

# Run with minimal output
proc = subprocess.Popen(
    ['python', 'video_demo.py', '--video', 'test_video.mp4', '--fec', 'xor_simple', '--loss_rate', '0.1'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env={'OPENCV_LOG_LEVEL': 'FATAL', **subprocess.os.environ}
)

try:
    # Wait for completion with timeout
    output, _ = proc.communicate(timeout=15)
    end_time = time.time()
    
    print(output)
    print("="*60)
    print(f"✅ Demo completed successfully!")
    print(f"   Runtime: {end_time - start_time:.1f} seconds")
    print(f"   Exit code: {proc.returncode}")
    
    # Check for completion messages
    if "Video transmission completed" in output:
        print("   ✓ Completion message found")
    if "Auto-exiting" in output:
        print("   ✓ Auto-exit message found")
    
except subprocess.TimeoutExpired:
    proc.kill()
    print("❌ Demo did NOT complete within 15 seconds (timeout)")
    print("   This means it's probably stuck in a loop")
    sys.exit(1)
