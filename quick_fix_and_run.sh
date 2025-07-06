#!/bin/bash
# quick_fix_and_run.sh
# 快速修复并运行量化的脚本

echo "=== MASt3R-SLAM TensorRT量化快速修复 ==="

# 1. 首先运行修复脚本
echo "Step 1: 运行修复脚本..."
python fix_quantization.py

# 2. 如果修复脚本成功，尝试原来的量化
echo "Step 2: 尝试修复后的量化..."
python run_quantization.py --precision int8

# 如果上面失败，使用简化版本
if [ $? -ne 0 ]; then
    echo "Step 3: 原量化失败，使用简化版本..."
    python simple_quantization.py --precision int8 --dataset datasets/tum/rgbd_dataset_freiburg1_desk
fi

echo "=== 量化完成！ ==="
