# test_single_thread_fixed.py
"""修复的单线程环境测试脚本"""

import torch
import os
import sys
sys.path.append('.')

# 应用优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_single_thread_slam():
    """运行单线程SLAM测试"""
    
    # 先加载完整配置
    from mast3r_slam.config import load_config, set_global_config
    
    # 加载基础配置
    load_config("config/calib.yaml")
    
    # 确保所有必要的配置都存在
    additional_config = {
        'single_thread': True,
        'use_calib': True,
        'dataset': {
            'subsample': 4,
            'img_downsample': 1,
            'center_principle_point': True  # 添加缺失的配置
        }
    }
    set_global_config(additional_config)
    
    # 加载数据和模型
    from mast3r_slam.dataloader import load_dataset
    from optimized_mast3r_loader import load_optimized_mast3r
    
    print("加载数据集...")
    dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_room/")
    dataset.subsample(4)  # 只处理1/4的数据
    
    print("加载优化模型...")
    model = load_optimized_mast3r()
    
    print(f"将处理 {len(dataset)} 帧")
    
    # 简单的性能测试
    import time
    
    # 预热
    print("预热...")
    timestamp, img = dataset[0]
    from mast3r_slam.mast3r_utils import resize_img
    img_dict = resize_img(img, 512)
    img_tensor = img_dict["img"].cuda()
    
    for _ in range(5):
        with torch.no_grad():
            feat, pos, _ = model._encode_image(img_tensor, torch.tensor([[512, 512]]))
    
    # 正式测试
    print("开始性能测试...")
    start_time = time.time()
    
    for i in range(min(50, len(dataset))):  # 只测试前50帧
        timestamp, img = dataset[i]
        
        # 模拟编码过程
        img_dict = resize_img(img, 512)
        img_tensor = img_dict["img"].cuda()
        
        with torch.no_grad():
            feat, pos, _ = model._encode_image(img_tensor, torch.tensor([[512, 512]]))
    
    total_time = time.time() - start_time
    print(f"\n=== 单线程性能测试结果 ===")
    print(f"处理50帧用时: {total_time:.2f}s")
    print(f"平均每帧编码: {total_time/50:.4f}s")
    print(f"等效编码FPS: {50/total_time:.2f}")
    
    # 与理论值对比
    theoretical_time = 50 * 0.0288  # 基于优化测试的结果
    actual_vs_theory = total_time / theoretical_time
    print(f"理论时间: {theoretical_time:.2f}s")
    print(f"实际/理论比: {actual_vs_theory:.2f}x")
    
    if actual_vs_theory < 1.5:
        print("✅ 单线程优化效果良好!")
    else:
        print("⚠️  仍有优化空间")

if __name__ == "__main__":
    run_single_thread_slam()
