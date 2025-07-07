#!/bin/bash
# quick_fix_and_run.sh
# ëîvÐLÏ„,

echo "=== MASt3R-SLAM TensorRTÏëî ==="

# 1. –HÐLî,
echo "Step 1: ÐLî,..."
python fix_quantization.py

# 2. ‚œî,ŸÕŸe„Ï
echo "Step 2: Õî„Ï..."
python run_quantization.py --precision int8

# ‚œ
b1%(€H,
if [ $? -ne 0 ]; then
    echo "Step 3: ŸÏ1%(€H,..."
    python simple_quantization.py --precision int8 --dataset datasets/tum/rgbd_dataset_freiburg1_desk
fi

echo "=== ÏŒ ==="
