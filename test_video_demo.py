"""
Video Demo Validation Script

Tests the video demo components without requiring a graphical display.
"""

import sys
import os

# Test imports
print("=" * 60)
print("Testing Video Demo Components")
print("=" * 60)

print("\n1. Testing imports...")
try:
    import cv2
    print("   ✓ OpenCV (cv2) imported successfully")
except ImportError as e:
    print(f"   ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ NumPy imported successfully")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    from video_demo import VideoSender, VideoReceiver
    print("   ✓ VideoSender and VideoReceiver imported successfully")
except ImportError as e:
    print(f"   ✗ Video demo import failed: {e}")
    sys.exit(1)

try:
    from fec import XORSimpleFEC, XORInterleavedFEC, XORDualParityFEC
    print("   ✓ FEC modules imported successfully")
except ImportError as e:
    print(f"   ✗ FEC import failed: {e}")
    sys.exit(1)

print("\n2. Testing video file...")
video_path = "test_video.mp4"
if not os.path.exists(video_path):
    print(f"   ✗ Test video not found: {video_path}")
    print("   Run: python generate_test_video.py")
    sys.exit(1)
else:
    print(f"   ✓ Test video found: {video_path}")

print("\n3. Testing video capture...")
try:
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape
            print(f"   ✓ Video opened successfully")
            print(f"     Resolution: {width}x{height}")
            print(f"     Channels: {channels}")
            print(f"     Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            print(f"     FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
        else:
            print("   ✗ Could not read first frame")
            sys.exit(1)
        cap.release()
    else:
        print(f"   ✗ Could not open video: {video_path}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Video capture test failed: {e}")
    sys.exit(1)

print("\n4. Testing FEC schemes...")
try:
    fec_simple = XORSimpleFEC(4)
    print(f"   ✓ Simple XOR FEC created (block_size=4)")
    
    fec_interleaved = XORInterleavedFEC(4)
    print(f"   ✓ Interleaved XOR FEC created (block_size=4)")
    
    fec_dual = XORDualParityFEC(4)
    print(f"   ✓ Dual Parity XOR FEC created (block_size=4)")
except Exception as e:
    print(f"   ✗ FEC creation failed: {e}")
    sys.exit(1)

print("\n5. Testing VideoSender initialization...")
try:
    sender = VideoSender(
        video_path=video_path,
        host='127.0.0.1',
        port=11000,
        fec=fec_simple,
        fps=30,
        quality=50
    )
    print(f"   ✓ VideoSender created successfully")
    print(f"     Host: {sender.host}:{sender.port}")
    print(f"     FEC: {type(sender.fec).__name__}")
    print(f"     FPS: {sender.fps}")
    print(f"     Quality: {sender.quality}")
except Exception as e:
    print(f"   ✗ VideoSender creation failed: {e}")
    sys.exit(1)

print("\n6. Testing VideoReceiver initialization...")
try:
    receiver = VideoReceiver(
        port=11001,
        fec=fec_simple,
        loss_rate=0.2,
        window_name="Test Receiver"
    )
    print(f"   ✓ VideoReceiver created successfully")
    print(f"     Port: {receiver.port}")
    print(f"     FEC: {type(receiver.fec).__name__}")
    print(f"     Loss rate: {receiver.loss_rate:.1%}")
    receiver.socket.close()
except Exception as e:
    print(f"   ✗ VideoReceiver creation failed: {e}")
    sys.exit(1)

print("\n7. Testing frame encoding...")
try:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Resize frame
        frame_resized = cv2.resize(frame, (320, 240))
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        result, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
        
        if result:
            encoded_size = len(buffer.tobytes())
            print(f"   ✓ Frame encoded successfully")
            print(f"     Original size: {frame.shape[1]}x{frame.shape[0]}")
            print(f"     Resized: 320x240")
            print(f"     Encoded size: {encoded_size} bytes")
        else:
            print("   ✗ Frame encoding failed")
            sys.exit(1)
    else:
        print("   ✗ Could not read frame for encoding test")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Frame encoding test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All video demo components validated successfully!")
print("=" * 60)

print("\n📝 To run the video demo (requires graphical display):")
print(f"   python video_demo.py --video {video_path} --fec xor_simple --loss_rate 0.2")

print("\n💡 Notes:")
print("   - The video demo requires X11/display server for OpenCV windows")
print("   - Press 'q' in the video windows to exit")
print("   - Left window: Vanilla UDP (with simulated loss)")
print("   - Right window: FEC-protected stream")
print("   - Packet loss is simulated to demonstrate FEC effectiveness")

print("\n✨ Video Demo is ready to use!")
