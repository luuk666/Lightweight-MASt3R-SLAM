#!/bin/bash
# quick_fix_and_run.sh
# ��v�L��,

echo "=== MASt3R-SLAM TensorRT��� ==="

# 1. �H�L�,
echo "Step 1: �L�,..."
python fix_quantization.py

# 2. ���,�՟e��
echo "Step 2: ����..."
python run_quantization.py --precision int8

# ��
b1%(�H,
if [ $? -ne 0 ]; then
    echo "Step 3: ��1%(�H,..."
    python simple_quantization.py --precision int8 --dataset datasets/tum/rgbd_dataset_freiburg1_desk
fi

echo "=== �� ==="
