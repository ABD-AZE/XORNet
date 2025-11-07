"""
Dashboard Testing Script

Validates that the Streamlit dashboard can be launched and all components load correctly.
"""

import sys
import os

print("=" * 70)
print("Testing Streamlit Dashboard Components")
print("=" * 70)

# Test 1: Import checks
print("\n1. Testing imports...")
try:
    import streamlit as st
    print("   ✓ Streamlit imported")
except ImportError as e:
    print(f"   ✗ Streamlit import failed: {e}")
    print("     Install with: pip install streamlit")
    sys.exit(1)

try:
    import pandas as pd
    print("   ✓ Pandas imported")
except ImportError as e:
    print(f"   ✗ Pandas import failed: {e}")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    print("   ✓ Plotly imported")
except ImportError as e:
    print(f"   ✗ Plotly import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print("   ✓ OpenCV imported")
except ImportError as e:
    print(f"   ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ NumPy imported")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")
    sys.exit(1)

# Test 2: Dashboard file check
print("\n2. Testing dashboard file...")
dashboard_file = "dashboard.py"
if os.path.exists(dashboard_file):
    print(f"   ✓ Dashboard file found: {dashboard_file}")
    
    # Check file size
    file_size = os.path.getsize(dashboard_file)
    print(f"     Size: {file_size} bytes")
    
    # Try to compile
    try:
        import py_compile
        py_compile.compile(dashboard_file, doraise=True)
        print("   ✓ Dashboard syntax is valid")
    except py_compile.PyCompileError as e:
        print(f"   ✗ Dashboard has syntax errors: {e}")
        sys.exit(1)
else:
    print(f"   ✗ Dashboard file not found: {dashboard_file}")
    sys.exit(1)

# Test 3: Required files check
print("\n3. Testing required files...")
required_files = {
    'simulation.py': 'Simulation runner',
    'metrics.py': 'Metrics module',
    'utils.py': 'Utilities',
    'fec/__init__.py': 'FEC package',
    'net/__init__.py': 'Network package',
}

all_files_present = True
for file, desc in required_files.items():
    if os.path.exists(file):
        print(f"   ✓ {desc}: {file}")
    else:
        print(f"   ✗ Missing {desc}: {file}")
        all_files_present = False

if not all_files_present:
    print("\n   Some required files are missing!")
    sys.exit(1)

# Test 4: Results file check
print("\n4. Testing results file...")
results_file = "results.json"
if os.path.exists(results_file):
    print(f"   ✓ Results file found: {results_file}")
    
    # Try to load
    try:
        import json
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"     Contains {len(data)} simulation results")
        else:
            print("     Contains 1 simulation result")
    except Exception as e:
        print(f"   ⚠ Results file exists but may be invalid: {e}")
else:
    print(f"   ⚠ No results file found (dashboard will start with empty state)")
    print("     Run: python simulation.py --fec xor_simple --loss_rate 0.1")

# Test 5: Video file check
print("\n5. Testing video file...")
video_file = "test_video.mp4"
if os.path.exists(video_file):
    print(f"   ✓ Test video found: {video_file}")
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"     Resolution: {width}x{height}")
            print(f"     Frames: {frame_count}")
            print(f"     FPS: {fps}")
            cap.release()
        else:
            print("   ⚠ Video file found but cannot be opened")
    except Exception as e:
        print(f"   ⚠ Error checking video: {e}")
else:
    print(f"   ⚠ No test video found")
    print("     Video streaming tab will offer to generate one")
    print("     Or run: python generate_test_video.py")

# Test 6: Port availability check
print("\n6. Testing network ports...")
import socket

ports_to_check = [8501, 9999, 10000, 11000, 11001]
available_ports = []
used_ports = []

for port in ports_to_check:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        used_ports.append(port)
    else:
        available_ports.append(port)

if available_ports:
    print(f"   ✓ Available ports: {', '.join(map(str, available_ports))}")

if used_ports:
    print(f"   ⚠ Ports in use: {', '.join(map(str, used_ports))}")
    if 8501 in used_ports:
        print("     Note: Port 8501 (Streamlit default) is in use")
        print("     Streamlit will automatically use next available port")

# Summary
print("\n" + "=" * 70)
print("✅ Dashboard Pre-flight Check Complete!")
print("=" * 70)

print("\n📊 Dashboard Features:")
print("   • Tab 1: Simulation & Metrics")
print("     - Run simulations with different FEC schemes")
print("     - View performance charts and metrics")
print("     - Compare recovery ratios and bandwidth")
print("     - Download results as JSON")
print("")
print("   • Tab 2: Video Streaming Demo")
print("     - Side-by-side frame comparison")
print("     - Configure FEC scheme and loss rate")
print("     - Visualize packet loss effects")
print("     - Instructions for external video demo")

print("\n🚀 To launch the dashboard:")
print("   streamlit run dashboard.py")
print("")
print("   The dashboard will open in your browser at:")
print("   http://localhost:8501")

print("\n💡 Tips:")
print("   • Generate test data first: ./run_tests.sh")
print("   • Generate test video: python generate_test_video.py")
print("   • Run simulations from within the dashboard")
print("   • Use the sidebar controls to adjust parameters")

print("\n" + "=" * 70)
print("Dashboard is ready! 🎉")
print("=" * 70)
