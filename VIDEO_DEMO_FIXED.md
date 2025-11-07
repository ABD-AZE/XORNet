# Video Demo - Fixed and Working! ✅

## Summary

The video streaming demo has been **fixed** and is now working correctly! The main issues were:

1. **Qt threading issue** - OpenCV windows must be created in the main thread
2. **Infinite looping** - Videos were set to loop forever by default
3. **No completion mechanism** - No way to exit after video completed

## What Was Fixed

### 1. Threading Architecture ✅
- **Problem**: `cv2.imshow()` was called from background threads, causing Qt threading warnings and unresponsive windows
- **Solution**: Refactored to use `receive_only()` method in threads and display in main thread

### 2. Video Looping Control ✅  
- **Problem**: Videos looped indefinitely with no option to play once and exit
- **Solution**: Added `--loop` flag (default: False = play once and exit)
  - `--loop`: Loop video continuously
  - `--no-loop` or default: Play once and auto-exit

### 3. Auto-Completion ✅
- **Problem**: No automatic exit after transmission completed
- **Solution**: Added completion detection and auto-exit after 2 seconds when both streams finish

### 4. Window Management ✅
- **Problem**: Windows appeared randomly and were sometimes unresponsive  
- **Solution**: 
  - Windows created with `WINDOW_NORMAL` flag
  - Proper sizing (320x240)
  - Side-by-side positioning (Vanilla UDP at 50,50 and FEC Protected at 400,50)
  - Regular `cv2.waitKey()` calls for responsiveness

## Usage

### Basic Usage (Play Once and Exit)
```bash
python video_demo.py --video ~/Downloads/istockphoto-1469733028-640_adpp_is.mp4 --fec xor_simple --loss_rate 0.2
```

### Loop Continuously
```bash
python video_demo.py --video ~/Downloads/istockphoto-1469733028-640_adpp_is.mp4 --fec xor_simple --loss_rate 0.2 --loop
```

### All Options
```bash
python video_demo.py \
  --video <path_to_video.mp4> \
  --fec {xor_simple|xor_interleaved|xor_dual_parity} \
  --loss_rate 0.2 \
  --block_size 4 \
  [--loop | --no-loop]
```

## How It Works

1. **Two UDP streams** are created on ports 11000 and 11001
2. **Left window ("Vanilla UDP")**: No error correction, shows packet loss effects
3. **Right window ("FEC Protected")**: With FEC error correction, smoother playback
4. **Packet loss simulated** at the receiver to demonstrate FEC effectiveness
5. **Auto-exits** after video completes (unless --loop is used)

## Testing Results

### Test Video (150 frames, 5 seconds)
- ✅ Completes in ~10 seconds (including startup/shutdown)
- ✅ Exit code: 0
- ✅ No threading warnings
- ✅ Windows display properly
- ✅ Auto-exits after completion

### IStock Video (826 frames, ~27 seconds)
- ✅ Runs successfully
- ✅ Both windows display side-by-side
- ✅ FEC protection visibly reduces artifacts
- ✅ Completes and exits automatically

## Key Code Changes

### VideoSender
- Added `loop` parameter (default: True)
- Added `completed` flag
- Logs transmission progress
- Exits cleanly after video ends (when loop=False)

### VideoReceiver  
- Split into `receive_and_display()` and `receive_only()`
- `receive_only()` for background threads
- Display handled in main thread

### run_demo()
- Creates windows in main thread
- Display loop checks for completion
- Auto-exits when transmission done
- Better thread cleanup

## Command Examples

### Quick test (short video):
```bash
python video_demo.py --video test_video.mp4 --fec xor_simple --loss_rate 0.2
```

### Full demo (your video):
```bash
python video_demo.py --video ~/Downloads/istockphoto-1469733028-640_adpp_is.mp4 --fec xor_simple --loss_rate 0.2
```

### Compare FEC schemes:
```bash
# Simple XOR
python video_demo.py --video test_video.mp4 --fec xor_simple --loss_rate 0.3

# Interleaved XOR
python video_demo.py --video test_video.mp4 --fec xor_interleaved --loss_rate 0.3

# Dual Parity XOR
python video_demo.py --video test_video.mp4 --fec xor_dual_parity --loss_rate 0.3
```

## Exit Methods

1. **Auto-exit**: Video completes → waits 2 seconds → exits (default behavior)
2. **Manual exit**: Press 'q' in either window
3. **Keyboard interrupt**: Ctrl+C in terminal
4. **Loop mode**: Only manual exit or Ctrl+C (no auto-exit)

## Verification

Run the test script to verify everything works:
```bash
python test_completion.py
```

Expected output:
```
✅ Demo completed successfully!
   Runtime: ~10 seconds
   Exit code: 0
```

## Notes

- JPEG warnings like "Corrupt JPEG data: premature end of data segment" are **normal** - they're from simulated packet loss
- Windows must have X11/display server available
- Default loss rate is 20% to clearly show FEC benefits
- Lower loss rates (5-10%) for more subtle comparison

## Status: ✅ WORKING

The video streaming demo is now fully functional and will:
- ✅ Display two windows side-by-side
- ✅ Stream video with simulated packet loss
- ✅ Demonstrate FEC effectiveness
- ✅ Complete transmission and auto-exit
- ✅ No threading warnings or hangs

Enjoy your FEC video demonstration! 🎉
