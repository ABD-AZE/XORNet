# XOR-based FEC over UDP: Benchmarking and Visualization

A complete and modular Python project that simulates UDP packet transmission under configurable loss conditions, applies multiple XOR-based Forward Error Correction (FEC) schemes, and benchmarks throughput, recovery, and bandwidth overhead.

## 🌟 Features

- **Multiple XOR-based FEC Schemes**:
  - Simple XOR (1 parity per block)
  - Interleaved XOR (cross-block resilience)
  - Dual Parity XOR (2 independent parities)

- **Comprehensive Metrics**:
  - Packet loss, recovery ratio, and FEC overhead
  - Bandwidth, goodput, and latency measurements
  - Statistical analysis and visualization

- **Interactive Dashboard**:
  - Streamlit-based web interface
  - Real-time simulation control
  - Interactive charts and graphs

- **Video Demo Mode**:
  - Side-by-side comparison of vanilla UDP vs FEC-protected streams
  - Visual demonstration of packet loss effects

## 📁 Project Structure

```
xor_fec_udp/
├── fec/
│   ├── __init__.py
│   ├── base_fec.py           # Abstract base class for FEC schemes
│   ├── xor_simple.py         # Simple XOR (1 parity per block)
│   ├── xor_interleaved.py    # Interleaved XOR (cross-block)
│   └── xor_dual_parity.py    # Dual Parity XOR (2 parities)
├── net/
│   ├── __init__.py
│   ├── server.py             # UDP sender with FEC support
│   └── client.py             # UDP receiver with loss simulation
├── simulation.py             # Simulation runner and orchestrator
├── metrics.py                # Metrics calculation and tracking
├── utils.py                  # Utility functions
├── dashboard.py              # Streamlit dashboard
├── video_demo.py             # Video transmission demo
├── requirements.txt          # Python dependencies
├── results.json              # Simulation results (auto-generated)
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install streamlit==1.28.0
pip install pandas==2.1.0
pip install plotly==5.17.0
pip install opencv-python==4.8.1.78
pip install "pygame>=2.1.3"
pip install numpy==1.24.3
```

The required packages are:
- `streamlit` - For the interactive dashboard
- `pandas` - For data manipulation
- `plotly` - For interactive visualizations
- `opencv-python` - For video demo 
- `numpy` - For numerical operations
 - `pygame` - For video/demo display or input handling 

## 📊 Usage

### 1. Run Individual Simulations

Run a simulation with specific parameters:

```bash
# Without FEC (vanilla UDP)
python simulation.py --fec none --loss_rate 0.1

# With Simple XOR FEC
python simulation.py --fec xor_simple --loss_rate 0.2 --block_size 4

# With Interleaved XOR FEC
python simulation.py --fec xor_interleaved --loss_rate 0.2 --block_size 4

# With Dual Parity XOR FEC
python simulation.py --fec xor_dual_parity --loss_rate 0.3 --block_size 4
```

**Available Options:**
- `--fec`: FEC scheme (`none`, `xor_simple`, `xor_interleaved`, `xor_dual_parity`)
- `--loss_rate`: Packet loss rate (0.0 to 1.0, default: 0.1)
- `--block_size`: FEC block size (default: 4)
- `--data_size`: Data size in bytes (default: 10240)
- `--packet_size`: Packet size in bytes (default: 1024)
- `--output`: Output file for results (default: results.json)

### 2. Interactive Dashboard

Launch the Streamlit dashboard for interactive visualization:

```bash
streamlit run dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`.

**Dashboard Features:**
- Configure FEC algorithm, loss rate, block size, and data size
- Run simulations with a single click
- View real-time performance metrics
- Interactive charts:
  - Recovery Ratio vs Loss Rate
  - Bandwidth & Goodput Comparison
  - FEC Overhead Analysis
  - Latency Distribution
- Download results as JSON
- Clear results history

### 3. Video Demo Mode (Optional)

Compare vanilla UDP vs FEC-protected video transmission:

```bash
# First, ensure you have a video file (e.g., sample.mp4)
python video_demo.py --video test_video.mp4 --fec xor_simple --loss_rate 0.2 --block_size 4
```

**Video Demo Options:**
- `--video`: Path to video file (required)
- `--fec`: FEC scheme for protected stream
- `--loss_rate`: Packet loss rate
- `--block_size`: FEC block size

**Note**: Press 'q' in the video windows to quit the demo.

## 🧠 FEC Schemes Explained

### 1. Simple XOR FEC

**How it works:**
- Creates 1 parity packet per block by XORing all data packets
- Formula: `p = pkt1 ⊕ pkt2 ⊕ ... ⊕ pktN`

**Characteristics:**
- ✅ Low overhead (1/N ratio)
- ✅ Can recover from 1 packet loss per block
- ❌ Cannot recover from multiple losses in same block

**Best for:** Low loss rates, bandwidth-constrained scenarios

### 2. Interleaved XOR FEC

**How it works:**
- Creates parity packets across multiple blocks
- Combines packets from different blocks in an interleaved pattern

**Characteristics:**
- ✅ Better protection against burst losses
- ✅ Cross-block resilience
- ⚠️ Medium overhead (N parity packets per N data packets)
- ✅ Can recover scattered losses across blocks

**Best for:** Bursty loss patterns, streaming applications

### 3. Dual Parity XOR FEC

**How it works:**
- Creates 2 independent parity packets per block:
  - Parity 1: XOR of even-indexed packets (0, 2, 4, ...)
  - Parity 2: XOR of odd-indexed packets (1, 3, 5, ...)

**Characteristics:**
- ✅ Can recover up to 2 packet losses per block (in some cases)
- ✅ Higher reliability
- ⚠️ Higher overhead (2/N ratio)

**Best for:** High loss rates, critical data transmission

## 📈 Metrics Explained

### Packet Statistics
- **Total Sent**: Total number of packets transmitted (data + parity)
- **Total Received**: Number of packets successfully received
- **Total Lost**: Number of packets lost during transmission
- **Total Recovered**: Number of lost packets recovered using FEC

### Performance Metrics
- **Loss Rate**: Actual packet loss rate (lost / sent)
- **Recovery Ratio**: Ratio of recovered packets to lost packets (0.0 to 1.0)
- **FEC Overhead**: Ratio of parity bytes to data bytes
- **Bandwidth**: Total throughput including parity (Mbps)
- **Goodput**: Useful data throughput (Mbps)
- **Latency**: End-to-end packet delay (milliseconds)

## 🔬 Example Workflows

### Workflow 1: Compare FEC Schemes at Different Loss Rates

```bash
# Run simulations for each FEC scheme at various loss rates
for fec in none xor_simple xor_interleaved xor_dual_parity; do
  for loss in 0.05 0.1 0.15 0.2 0.25 0.3; do
    python simulation.py --fec $fec --loss_rate $loss --block_size 4
  done
done

# View results in dashboard
streamlit run dashboard.py
```

### Workflow 2: Find Optimal Block Size

```bash
# Test different block sizes with Simple XOR
for block_size in 2 4 8 16; do
  python simulation.py --fec xor_simple --loss_rate 0.2 --block_size $block_size
done

# Analyze results
streamlit run dashboard.py
```

### Workflow 3: Benchmark High-Loss Scenarios

```bash
# Test FEC performance under extreme conditions
python simulation.py --fec xor_dual_parity --loss_rate 0.4 --block_size 8
python simulation.py --fec xor_interleaved --loss_rate 0.4 --block_size 8

streamlit run dashboard.py
```

## 📊 Results Format

Simulation results are stored in `results.json`:

```json
{
  "fec": "xor_simple",
  "loss_rate": 0.2,
  "block_size": 4,
  "packets_sent": 50,
  "packets_received": 40,
  "packets_lost": 10,
  "packets_recovered": 8,
  "recovery_ratio": 0.8,
  "fec_overhead": 0.25,
  "bandwidth_mbps": 3.5,
  "goodput_mbps": 2.8,
  "latency_avg_ms": 42.7,
  "duration_seconds": 2.34
}
```

## 🛠️ Architecture

### FEC Layer
All FEC schemes inherit from `BaseFEC`:
```python
class BaseFEC(ABC):
    def encode(self, packets: List[bytes]) -> List[bytes]:
        """Add parity packets"""
        
    def decode(self, packets: List[bytes], loss_map: List[int]) -> List[bytes]:
        """Recover lost packets"""
```

### Network Layer
- **UDPServer**: Sends data with optional FEC encoding
- **UDPClient**: Receives data, simulates loss, applies FEC decoding

### Simulation Layer
- Orchestrates server and client
- Collects metrics
- Saves results to JSON

## 🧪 Testing

Test individual components:

```bash
# Test FEC encoding/decoding
python -c "from fec import XORSimpleFEC; fec = XORSimpleFEC(4); print('FEC OK')"

# Test server (in one terminal)
python net/server.py --fec xor_simple --block_size 4

# Test client (in another terminal)
python net/client.py --fec xor_simple --block_size 4 --loss_rate 0.1
```

---

**Happy Benchmarking! 🚀**
