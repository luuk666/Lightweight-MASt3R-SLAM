# advanced_optimization.py
"""进阶优化方案 - 解决多进程环境下的性能问题"""

import torch
import os

def apply_aggressive_optimizations():
    """应用更激进的系统优化"""
    print("=== 应用进阶优化设置 ===")
    
    # 1. PyTorch性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    
    # 2. 内存优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.8'
    
    # 3. 多进程优化
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # 4. CUDA优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    print("✓ 进阶优化设置已应用")

def create_optimized_main():
    """创建优化版本的main.py修改"""
    
    optimized_code = '''
# 在main.py文件开头添加（在import之前）:

import os
import torch

# 应用系统优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

# 内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# 多进程优化
os.environ['OMP_NUM_THREADS'] = '4'

print("✓ 应用了进阶优化设置")

# 然后在模型加载部分（找到model = load_mast3r这一行）替换为:

try:
    from optimized_mast3r_loader import load_optimized_mast3r
    model = load_optimized_mast3r(device=device)
    
    # 启用更激进的优化
    model.config['use_half_precision'] = True  # 启用半精度
    print("✓ 使用优化模型 + 半精度")
    
except Exception as e:
    print(f"优化模型加载失败: {e}")
    model = load_mast3r(device=device)
'''
    
    print("=== 进阶main.py优化代码 ===")
    print(optimized_code)
    
    return optimized_code

def create_single_thread_test():
    """创建单线程测试"""
    test_code = '''# test_single_thread.py
"""单线程环境下测试优化效果"""

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
    
    # 修改配置使用单线程
    from mast3r_slam.config import load_config, set_global_config
    
    # 加载配置
    load_config("config/calib.yaml")
    
    # 强制单线程
    config_override = {
        'single_thread': True,
        'use_calib': True,
        'dataset': {
            'subsample': 4  # 减少数据量以快速测试
        }
    }
    set_global_config(config_override)
    
    # 加载数据和模型
    from mast3r_slam.dataloader import load_dataset
    from optimized_mast3r_loader import load_optimized_mast3r
    
    dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_room/")
    dataset.subsample(4)  # 只处理1/4的数据
    
    model = load_optimized_mast3r()
    
    print(f"将处理 {len(dataset)} 帧")
    
    # 简单的性能测试
    import time
    start_time = time.time()
    
    for i in range(min(50, len(dataset))):  # 只测试前50帧
        timestamp, img = dataset[i]
        
        # 模拟编码过程
        from mast3r_slam.mast3r_utils import resize_img
        img_dict = resize_img(img, 512)
        img_tensor = img_dict["img"].cuda()
        
        with torch.no_grad():
            feat, pos, _ = model._encode_image(img_tensor, torch.tensor([[512, 512]]))
    
    total_time = time.time() - start_time
    print(f"\\n处理50帧用时: {total_time:.2f}s")
    print(f"平均每帧: {total_time/50:.4f}s")
    print(f"等效FPS: {50/total_time:.2f}")

if __name__ == "__main__":
    run_single_thread_slam()
'''
    
    with open("test_single_thread.py", "w") as f:
        f.write(test_code)
    
    print("✓ 单线程测试脚本已创建: test_single_thread.py")

def suggest_next_steps():
    """建议下一步操作"""
    print("\n=== 建议的优化步骤 ===")
    
    steps = [
        "1. 运行单线程测试: python test_single_thread.py",
        "2. 如果单线程效果好，说明是多进程问题",
        "3. 在main.py中添加单线程模式: 'single_thread': True", 
        "4. 尝试启用半精度优化",
        "5. 考虑减少subsample来处理更少帧"
    ]
    
    for step in steps:
        print(step)
    
    print("\n=== 可能的性能瓶颈 ===")
    bottlenecks = [
        "• 多进程开销影响GPU优化效果",
        "• 检索模型(retrieval)没有优化",  
        "• 后端优化(BA)占用了额外资源",
        "• 内存碎片化影响性能"
    ]
    
    for bottleneck in bottlenecks:
        print(bottleneck)

if __name__ == "__main__":
    apply_aggressive_optimizations()
    create_optimized_main()
    create_single_thread_test()
    suggest_next_steps()
