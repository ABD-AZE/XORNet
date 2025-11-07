"""
Streamlit Dashboard

Interactive dashboard for visualizing FEC performance metrics
and running simulations with different parameters.
"""

import streamlit as st
import json
import os
import subprocess
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import threading
import time
import cv2
import numpy as np
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="XOR FEC Benchmarking Dashboard",
    page_icon="📊",
    layout="wide"
)


def load_results(filename: str = 'results.json') -> List[Dict]:
    """
    Load results from JSON file.
    
    Args:
        filename: Path to results file
        
    Returns:
        List of result dictionaries
    """
    if not os.path.exists(filename):
        return []
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except json.JSONDecodeError:
        return []


def run_simulation(fec_type: str, loss_rate: float, block_size: int,
                   data_size: int) -> bool:
    """
    Run a simulation with given parameters.
    
    Args:
        fec_type: FEC scheme
        loss_rate: Packet loss rate
        block_size: FEC block size
        data_size: Data size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'python', 'simulation.py',
            '--fec', fec_type,
            '--loss_rate', str(loss_rate),
            '--block_size', str(block_size),
            '--data_size', str(data_size),
            '--output', 'results.json'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return result.returncode == 0
    except Exception as e:
        st.error(f"Error running simulation: {e}")
        return False


def plot_recovery_ratio_vs_loss(df: pd.DataFrame):
    """
    Plot recovery ratio vs loss rate for different FEC schemes.
    
    Args:
        df: DataFrame with results
    """
    fig = go.Figure()
    
    for fec_type in df['fec'].unique():
        fec_data = df[df['fec'] == fec_type].sort_values('loss_rate')
        
        fig.add_trace(go.Scatter(
            x=fec_data['loss_rate'] * 100,
            y=fec_data['recovery_ratio'] * 100,
            mode='lines+markers',
            name=fec_type,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Recovery Ratio vs Loss Rate',
        xaxis_title='Packet Loss Rate (%)',
        yaxis_title='Recovery Ratio (%)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def plot_bandwidth_comparison(df: pd.DataFrame):
    """
    Plot bandwidth and overhead comparison.
    
    Args:
        df: DataFrame with results
    """
    # Group by FEC type and calculate averages
    grouped = df.groupby('fec').agg({
        'bandwidth_mbps': 'mean',
        'goodput_mbps': 'mean',
        'fec_overhead': 'mean'
    }).reset_index()
    
    # Create subplots
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Bandwidth',
        x=grouped['fec'],
        y=grouped['bandwidth_mbps'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Goodput',
        x=grouped['fec'],
        y=grouped['goodput_mbps'],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title='Bandwidth & Goodput Comparison',
        xaxis_title='FEC Scheme',
        yaxis_title='Throughput (Mbps)',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def plot_overhead_comparison(df: pd.DataFrame):
    """
    Plot FEC overhead comparison.
    
    Args:
        df: DataFrame with results
    """
    grouped = df.groupby('fec')['fec_overhead'].mean().reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=grouped['fec'],
            y=grouped['fec_overhead'] * 100,
            marker_color='coral'
        )
    ])
    
    fig.update_layout(
        title='FEC Overhead Comparison',
        xaxis_title='FEC Scheme',
        yaxis_title='Overhead (%)',
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def plot_recovery_time_distribution(df: pd.DataFrame):
    """
    Plot recovery time distribution.
    
    Args:
        df: DataFrame with results
    """
    # Filter only results with recovery data
    df_with_recovery = df[df['recovery_time_avg_ms'] > 0]
    
    if len(df_with_recovery) == 0:
        st.info("No recovery time data available")
        return
    
    fig = go.Figure()
    
    for fec_type in df_with_recovery['fec'].unique():
        fec_data = df_with_recovery[df_with_recovery['fec'] == fec_type]
        
        fig.add_trace(go.Box(
            y=fec_data['recovery_time_avg_ms'],
            name=fec_type,
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='FEC Recovery Time Distribution',
        xaxis_title='FEC Scheme',
        yaxis_title='Recovery Time (ms)',
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")


def run_video_demo():
    """Video streaming demo tab."""
    st.header("🎬 Live Video Streaming Demo")
    st.markdown("Compare **Vanilla UDP** vs **FEC-protected** video transmission with simulated packet loss.")
    
    # Create sub-tabs for different demo modes
    subtab1, subtab2 = st.tabs(["📤 Upload Your Video", "🎥 Test Video Demo"])
    
    with subtab1:
        run_upload_video_demo()
    
    with subtab2:
        run_test_video_demo()


def run_upload_video_demo():
    """Handle custom video upload and demo."""
    st.subheader("📤 Upload Your Own Video")
    st.markdown("Upload a video file to test FEC protection with your own content!")
    
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV, MKV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to test with FEC"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_video_path = f"temp_uploaded_{uploaded_file.name}"
        
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Show video info
        try:
            cap = cv2.VideoCapture(temp_video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Resolution", f"{width}x{height}")
                col2.metric("FPS", fps)
                col3.metric("Frames", frame_count)
                col4.metric("Duration", f"{duration:.1f}s")
                
                # Show preview frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Video Preview", width="stretch")
                
                cap.release()
            else:
                st.error("❌ Could not open video file")
                return
        except Exception as e:
            st.error(f"❌ Error reading video: {e}")
            return
        
        st.markdown("---")
        
        # Configuration
        st.subheader("⚙️ Demo Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fec_scheme = st.selectbox(
                "FEC Scheme",
                options=['xor_simple', 'xor_interleaved', 'xor_dual_parity'],
                help="FEC algorithm for the protected stream",
                key="upload_fec"
            )
        
        with col2:
            loss_rate = st.slider(
                "Packet Loss Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Simulated packet loss rate",
                key="upload_loss"
            )
        
        with col3:
            block_size = st.slider(
                "FEC Block Size",
                min_value=2,
                max_value=16,
                value=4,
                step=1,
                help="Packets per FEC block",
                key="upload_block"
            )
        
        loop_video = st.checkbox("Loop video continuously", value=False, help="Loop video or play once", key="upload_loop")
        
        st.markdown("---")
        
        # Build and show command
        st.subheader("🚀 Run Video Demo")
        
        cmd_parts = [
            'python', 'video_demo.py',
            '--video', temp_video_path,
            '--fec', fec_scheme,
            '--loss_rate', str(loss_rate),
            '--block_size', str(block_size)
        ]
        
        if loop_video:
            cmd_parts.append('--loop')
        
        st.code(' '.join(cmd_parts), language='bash')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Demo", type="primary"):
                try:
                    import subprocess
                    # Run in background
                    process = subprocess.Popen(
                        cmd_parts,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    st.success("✅ Video demo started! Check the OpenCV windows.")
                    st.info("💡 Press 'q' in the video windows to stop the demo.")
                    
                except Exception as e:
                    st.error(f"❌ Failed to start demo: {e}")
        
        with col2:
            if st.button("🗑️ Delete Uploaded Video"):
                try:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        st.success("✅ Uploaded video deleted")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete: {e}")
        
        # Instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            ### Steps:
            1. **Upload** your video file above
            2. **Configure** FEC parameters (scheme, loss rate, block size)
            3. **Click** "▶️ Start Demo" to launch the video streaming
            4. **Watch** two windows appear:
               - **Left window:** Vanilla UDP (with packet loss artifacts)
               - **Right window:** FEC Protected (with recovery)
            5. **Press 'q'** in either window to exit
            
            ### What You'll See:
            - Real-time side-by-side comparison
            - Vanilla UDP shows visible glitches from packet loss
            - FEC-protected stream shows smoother playback
            - Demonstrates FEC effectiveness in recovering lost packets
            
            ### Requirements:
            - X11/Display server for OpenCV windows
            - Ports 11000 and 11001 available
            - Video file in supported format
            """)
    
    else:
        st.info("👆 Upload a video file to get started!")
        
        # Show example
        st.markdown("### 📝 Or use the test video")
        st.markdown("If you don't have a video, switch to the **'Test Video Demo'** tab or generate a test video:")
        
        if st.button("🎬 Generate Test Video"):
            with st.spinner("Generating test video..."):
                result = subprocess.run(
                    ['python', 'generate_test_video.py', '--output', 'test_video.mp4', '--duration', '10'],
                    capture_output=True
                )
                if result.returncode == 0:
                    st.success("✅ Test video generated! Switch to the 'Test Video Demo' tab.")
                else:
                    st.error("❌ Failed to generate video")


def run_test_video_demo():
    """Run demo with the test video."""
    # Check if video file exists
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        st.warning("⚠️ Test video not found. Generate it first!")
        if st.button("🎬 Generate Test Video", key="gen_test_video"):
            with st.spinner("Generating test video..."):
                result = subprocess.run(
                    ['python', 'generate_test_video.py', '--output', video_path, '--duration', '5'],
                    capture_output=True
                )
                if result.returncode == 0:
                    st.success("✅ Test video generated!")
                    st.rerun()
                else:
                    st.error("❌ Failed to generate video")
        return
    
    st.info("ℹ️ **Note:** This demo uses the test video. The side-by-side comparison shows frame differences under packet loss.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        fec_scheme = st.selectbox(
            "FEC Scheme",
            options=['xor_simple', 'xor_interleaved', 'xor_dual_parity'],
            help="FEC algorithm for the protected stream",
            key="test_fec"
        )
        
        loss_rate = st.slider(
            "Packet Loss Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Simulated packet loss rate for both streams",
            key="test_loss"
        )
        
        block_size = st.slider(
            "FEC Block Size",
            min_value=2,
            max_value=16,
            value=4,
            step=1,
            help="Number of packets per FEC block",
            key="test_block"
        )
    
    with col2:
        st.subheader("📊 Comparison")
        
        # Ensure fec_scheme is a string
        fec_name = fec_scheme if isinstance(fec_scheme, str) else ['xor_simple', 'xor_interleaved', 'xor_dual_parity'][fec_scheme]
        
        st.markdown(
            f"""
            **Loss Rate:** {loss_rate:.0%}  
            **FEC Scheme:** {fec_name.replace('_', ' ').title()}  
            **Block Size:** {block_size}
            
            **Left Stream:** Vanilla UDP (no protection)  
            **Right Stream:** FEC-protected (with recovery)
            """
        )
    
    st.markdown("---")
    
    # Video display section
    st.subheader("📺 Video Frames Comparison")
    
    col_left, col_right = st.columns(2)
    
    # Show sample frames from the video
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get a middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with col_left:
                st.markdown("**Vanilla UDP** (packet loss affects video)")
                # Simulate packet loss effect - add noise/artifacts
                corrupted_frame = frame_rgb.copy()
                if loss_rate > 0:
                    # Add random blocks of corruption
                    h, w = corrupted_frame.shape[:2]
                    num_blocks = int(loss_rate * 20)
                    for _ in range(num_blocks):
                        x = np.random.randint(0, w - 20)
                        y = np.random.randint(0, h - 20)
                        corrupted_frame[y:y+20, x:x+20] = 0
                st.image(corrupted_frame, caption=f"With {loss_rate:.0%} packet loss", width="stretch")
            
            with col_right:
                st.markdown(f"**FEC Protected** ({fec_scheme})")
                # Show cleaner frame (simulating FEC recovery)
                recovered_frame = frame_rgb.copy()
                if loss_rate > 0:
                    # Show minimal artifacts (simulating partial recovery)
                    h, w = recovered_frame.shape[:2]
                    num_blocks = int(loss_rate * 5)  # Fewer artifacts due to FEC
                    for _ in range(num_blocks):
                        x = np.random.randint(0, w - 10)
                        y = np.random.randint(0, h - 10)
                        recovered_frame[y:y+10, x:x+10] = recovered_frame[y:y+10, x:x+10] * 0.7
                st.image(recovered_frame, caption="FEC recovers most lost data", width="stretch")
    
    except Exception as e:
        st.error(f"Error loading video: {e}")
    
    st.markdown("---")
    
    # Expected results explanation
    st.subheader("📈 Expected Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Vanilla UDP",
            "Significant degradation",
            delta="No recovery",
            delta_color="inverse"
        )
        st.caption("Lost packets cause visible artifacts and frame corruption")
    
    with col2:
        # Ensure fec_scheme is a string
        fec_name = fec_scheme if isinstance(fec_scheme, str) else ['xor_simple', 'xor_interleaved', 'xor_dual_parity'][fec_scheme]
        
        st.metric(
            f"{fec_name.replace('_', ' ').title()}",
            "Better quality",
            delta="Partial/full recovery",
            delta_color="normal"
        )
        st.caption("FEC recovers lost packets, reducing artifacts")
    
    with col3:
        recovery_estimate = min(100, int((1 - loss_rate) * 100 + loss_rate * 50))
        st.metric(
            "Estimated Recovery",
            f"{recovery_estimate}%",
            delta=f"+{int(loss_rate * 50)}% vs vanilla"
        )
        st.caption("Approximate improvement with FEC")
    
    st.markdown("---")
    
    # Instructions for full demo
    with st.expander("🚀 Run Full Video Demo (External)"):
        st.markdown(
            f"""
            To run the **real-time side-by-side video streaming** with actual packet transmission:
            
            ```bash
            python video_demo.py \\
                --video {video_path} \\
                --fec {fec_scheme} \\
                --loss_rate {loss_rate} \\
                --block_size {block_size}
            ```
            
            **Requirements:**
            - Graphical display (X11) for OpenCV windows
            - Two UDP ports (11000, 11001) available
            - Press 'q' in video windows to exit
            
            **What you'll see:**
            - Left window: Vanilla UDP with visible packet loss
            - Right window: FEC-protected stream with recovery
            - Real-time demonstration of FEC effectiveness
            """
        )


def main():
    """Main dashboard function."""
    
    # Title
    st.title("📊 XOR-based FEC over UDP: Benchmarking Dashboard")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Simulation & Metrics", "🎬 Video Streaming Demo"])
    
    with tab1:
        show_simulation_tab()
    
    with tab2:
        run_video_demo()


def show_simulation_tab():
    """Show the simulation and metrics tab."""
    
    # Initialize session state for simulation status
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'last_simulation' not in st.session_state:
        st.session_state.last_simulation = None
    
    st.markdown("---")
    
    # Sidebar for controls
    st.sidebar.header("⚙️ Simulation Controls")
    
    # FEC scheme selection
    fec_type = st.sidebar.selectbox(
        "FEC Algorithm",
        options=['none', 'xor_simple', 'xor_interleaved', 'xor_dual_parity'],
        index=0,
        help="Select the Forward Error Correction scheme"
    )
    
    # Loss rate slider
    loss_rate = st.sidebar.slider(
        "Packet Loss Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        format="%.2f",
        help="Simulated packet loss rate (0.0 to 0.5)"
    )
    
    # Block size
    block_size = st.sidebar.slider(
        "Block Size",
        min_value=2,
        max_value=16,
        value=4,
        step=1,
        help="Number of packets per FEC block"
    )
    
    # Data size
    data_size = st.sidebar.selectbox(
        "Data Size",
        options=[1024, 5120, 10240, 20480, 51200],
        index=2,
        format_func=lambda x: f"{x} bytes ({x/1024:.1f} KB)",
        help="Total data size to transmit"
    )
    
    st.sidebar.markdown("---")
    
    # Run simulation button
    run_button = st.sidebar.button("▶️ Run Simulation", type="primary", disabled=st.session_state.simulation_running)
    
    if run_button and not st.session_state.simulation_running:
        st.session_state.simulation_running = True
        with st.spinner(f"Running simulation with {fec_type} at {loss_rate:.0%} loss..."):
            success = run_simulation(fec_type, loss_rate, block_size, data_size)
            
            if success:
                st.session_state.last_simulation = f"{fec_type} @ {loss_rate:.0%} loss"
                st.sidebar.success("✅ Simulation completed!")
                time.sleep(0.5)  # Brief pause to show success message
            else:
                st.sidebar.error("❌ Simulation failed!")
        
        st.session_state.simulation_running = False
    
    # Show last simulation info
    if st.session_state.last_simulation:
        st.sidebar.info(f"Last run: {st.session_state.last_simulation}")
    
    # Clear results button
    if st.sidebar.button("🗑️ Clear Results"):
        if os.path.exists('results.json'):
            os.remove('results.json')
            st.session_state.last_simulation = None
            st.sidebar.success("Results cleared! Refresh page to see changes.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        "This dashboard allows you to test and compare different XOR-based "
        "Forward Error Correction schemes for UDP packet transmission under "
        "various loss conditions."
    )
    
    # Main content area
    results = load_results()
    
    if not results:
        st.info("👆 Configure simulation parameters and click **Run Simulation** to get started!")
        
        # Show example information
        st.markdown("### 🎯 FEC Schemes Available")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Simple XOR")
            st.markdown("""
            - 1 parity packet per block
            - Can recover 1 lost packet
            - Low overhead
            - Formula: `p = pkt1 ⊕ pkt2 ⊕ ... ⊕ pktN`
            """)
        
        with col2:
            st.markdown("#### Interleaved XOR")
            st.markdown("""
            - Cross-block parity
            - Better burst loss protection
            - Medium overhead
            - Interleaves packets across blocks
            """)
        
        with col3:
            st.markdown("#### Dual Parity XOR")
            st.markdown("""
            - 2 parity packets per block
            - Even/odd index protection
            - Can recover up to 2 losses
            - Higher overhead
            """)
        
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.header("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Simulations", len(results))
    
    with col2:
        avg_recovery = df['recovery_ratio'].mean() * 100
        st.metric("Avg Recovery Ratio", f"{avg_recovery:.1f}%")
    
    with col3:
        avg_bandwidth = df['bandwidth_mbps'].mean()
        st.metric("Avg Bandwidth", f"{avg_bandwidth:.2f} Mbps")
    
    with col4:
        avg_overhead = df['fec_overhead'].mean() * 100
        st.metric("Avg FEC Overhead", f"{avg_overhead:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    st.header("📊 Performance Visualizations")
    
    # Row 1: Recovery and Bandwidth
    col1, col2 = st.columns(2)
    
    with col1:
        plot_recovery_ratio_vs_loss(df)
    
    with col2:
        plot_bandwidth_comparison(df)
    
    # Row 2: Overhead and Recovery Time
    col1, col2 = st.columns(2)
    
    with col1:
        plot_overhead_comparison(df)
    
    with col2:
        plot_recovery_time_distribution(df)
    
    st.markdown("---")
    
    # Detailed results table
    st.header("📋 Detailed Results")
    
    # Select columns to display
    display_cols = [
        'fec', 'loss_rate', 'block_size',
        'packets_sent', 'packets_lost', 'packets_recovered',
        'recovery_ratio', 'fec_overhead', 'bandwidth_mbps',
        'goodput_mbps', 'recovery_time_avg_ms'
    ]
    
    # Format the dataframe
    display_df = df[display_cols].copy()
    display_df['loss_rate'] = display_df['loss_rate'].apply(lambda x: f"{x:.2%}")
    display_df['recovery_ratio'] = display_df['recovery_ratio'].apply(lambda x: f"{x:.2%}")
    display_df['fec_overhead'] = display_df['fec_overhead'].apply(lambda x: f"{x:.2%}")
    display_df['bandwidth_mbps'] = display_df['bandwidth_mbps'].apply(lambda x: f"{x:.2f}")
    display_df['goodput_mbps'] = display_df['goodput_mbps'].apply(lambda x: f"{x:.2f}")
    display_df['recovery_time_avg_ms'] = display_df['recovery_time_avg_ms'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    
    # Rename columns for better display
    display_df.columns = [
        'FEC', 'Loss Rate', 'Block Size',
        'Sent', 'Lost', 'Recovered',
        'Recovery Ratio', 'FEC Overhead', 'Bandwidth (Mbps)',
        'Goodput (Mbps)', 'Avg Recovery Time (ms)'
    ]
    
    st.dataframe(display_df)
    
    # Download results button
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps(results, indent=2),
        file_name="fec_results.json",
        mime="application/json"
    )


if __name__ == '__main__':
    main()
