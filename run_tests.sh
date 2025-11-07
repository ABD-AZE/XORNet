#!/bin/bash
# Comprehensive test script to generate diverse simulation results

echo "=========================================="
echo "XOR FEC Benchmarking - Comprehensive Tests"
echo "=========================================="
echo ""

# Clear previous results
echo "Clearing previous results..."
rm -f results.json

# Test 1: Compare all FEC schemes at moderate loss rate
echo ""
echo "Test 1: Comparing all FEC schemes at 10% loss rate"
echo "--------------------------------------------------"
python simulation.py --fec none --loss_rate 0.1 --data_size 10240 --block_size 4
python simulation.py --fec xor_simple --loss_rate 0.1 --data_size 10240 --block_size 4
python simulation.py --fec xor_interleaved --loss_rate 0.1 --data_size 10240 --block_size 4
python simulation.py --fec xor_dual_parity --loss_rate 0.1 --data_size 10240 --block_size 4

# Test 2: Simple XOR at varying loss rates
echo ""
echo "Test 2: Simple XOR at varying loss rates"
echo "-----------------------------------------"
for loss in 0.05 0.15 0.2 0.25 0.3; do
    python simulation.py --fec xor_simple --loss_rate $loss --data_size 10240 --block_size 4
done

# Test 3: Interleaved XOR at varying loss rates
echo ""
echo "Test 3: Interleaved XOR at varying loss rates"
echo "----------------------------------------------"
for loss in 0.05 0.15 0.2 0.25 0.3; do
    python simulation.py --fec xor_interleaved --loss_rate $loss --data_size 10240 --block_size 4
done

# Test 4: Dual Parity XOR at varying loss rates
echo ""
echo "Test 4: Dual Parity XOR at varying loss rates"
echo "----------------------------------------------"
for loss in 0.05 0.15 0.2 0.25 0.3; do
    python simulation.py --fec xor_dual_parity --loss_rate $loss --data_size 10240 --block_size 4
done

# Test 5: Compare all schemes at high loss rate
echo ""
echo "Test 5: All FEC schemes at 30% loss rate"
echo "-----------------------------------------"
python simulation.py --fec none --loss_rate 0.3 --data_size 10240 --block_size 4
python simulation.py --fec xor_simple --loss_rate 0.3 --data_size 10240 --block_size 4
python simulation.py --fec xor_interleaved --loss_rate 0.3 --data_size 10240 --block_size 4
python simulation.py --fec xor_dual_parity --loss_rate 0.3 --data_size 10240 --block_size 4

# Test 6: Different block sizes with Simple XOR
echo ""
echo "Test 6: Simple XOR with different block sizes"
echo "----------------------------------------------"
for block_size in 2 4 8 16; do
    python simulation.py --fec xor_simple --loss_rate 0.2 --data_size 10240 --block_size $block_size
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved to results.json"
echo ""
echo "Launch dashboard with: streamlit run dashboard.py"
echo "=========================================="
