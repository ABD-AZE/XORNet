# Quick Start Guide

Get up and running with the XOR FEC Benchmarking project in minutes!

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Test

Run a single simulation:

```bash
python simulation.py --fec xor_simple --loss_rate 0.2 --block_size 4
```

## 📊 Launch Dashboard

Start the interactive dashboard:

```bash
streamlit run dashboard.py
```

Then open your browser to `http://localhost:8501`

## 🧪 Run Comprehensive Tests

Generate diverse results data:

```bash
./run_tests.sh
```

This will run ~30 simulations with various FEC schemes and loss rates.
Results are saved to `results.json`.

## 🎬 Video Demo (Optional)

If you have a video file (e.g., `sample.mp4`):

```bash
python video_demo.py --video sample.mp4 --fec xor_simple --loss_rate 0.2
```

**Note:** Press 'q' in the video windows to exit.

## 📈 Common Use Cases

### Compare FEC schemes at 20% loss

```bash
python simulation.py --fec none --loss_rate 0.2
python simulation.py --fec xor_simple --loss_rate 0.2
python simulation.py --fec xor_interleaved --loss_rate 0.2
python simulation.py --fec xor_dual_parity --loss_rate 0.2

# View results in dashboard
streamlit run dashboard.py
```

### Test different loss rates with Simple XOR

```bash
for loss in 0.05 0.1 0.15 0.2 0.25 0.3; do
    python simulation.py --fec xor_simple --loss_rate $loss
done

# Visualize in dashboard
streamlit run dashboard.py
```

### Find optimal block size

```bash
for size in 2 4 8 16; do
    python simulation.py --fec xor_simple --loss_rate 0.2 --block_size $size
done

# Analyze in dashboard
streamlit run dashboard.py
```

## 🎯 Key Parameters

- `--fec`: FEC scheme (`none`, `xor_simple`, `xor_interleaved`, `xor_dual_parity`)
- `--loss_rate`: Packet loss rate (0.0 to 0.5)
- `--block_size`: Number of packets per FEC block (2-16)
- `--data_size`: Total data to transmit in bytes (default: 10240)

## 💡 Tips

1. **Start with the dashboard** - It's the easiest way to experiment
2. **Use run_tests.sh** - Generates comprehensive comparison data
3. **Compare recovery ratios** - Shows FEC effectiveness under loss
4. **Monitor FEC overhead** - Trade-off between protection and bandwidth
5. **Check results.json** - All simulation data is stored here

## 🐛 Troubleshooting

### Port already in use
If you see "Address already in use" errors:
```bash
# Kill existing processes using the ports
pkill -f "python.*simulation"
```

### Dashboard not loading results
Make sure `results.json` exists:
```bash
# Run at least one simulation first
python simulation.py --fec xor_simple --loss_rate 0.1
```

### Video demo not working
Ensure OpenCV is installed and you have a valid video file:
```bash
pip install opencv-python
# Use a small test video or your webcam (--video 0)
```

## 📚 Learn More

See [README.md](README.md) for detailed documentation.

---

**Happy benchmarking! 🎉**
