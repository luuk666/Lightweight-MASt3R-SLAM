# test_optimization.py
"""测试优化效果的脚本"""

import torch
import time
import sys
sys.path.append('.')

def benchmark_models():
    """对比原始和优化模型"""
    
    # 加载原始模型
    print("加载原始模型...")
    from mast3r_slam.mast3r_utils import load_mast3r
    original_model = load_mast3r().cuda().eval()
    
    # 加载优化模型
    print("加载优化模型...")
    from optimized_mast3r_loader import load_optimized_mast3r
    optimized_model = load_optimized_mast3r()
    
    # 测试数据
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    # 预热
    for _ in range(5):
        _ = original_model._encode_image(test_input, test_shape)
        _ = optimized_model._encode_image(test_input, test_shape)
    
    # 测试原始模型
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = original_model._encode_image(test_input, test_shape)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # 测试优化模型
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = optimized_model._encode_image(test_input, test_shape)
    torch.cuda.synchronize()
    optimized_time = time.time() - start
    
    print(f"\n=== 性能对比 ===")
    print(f"原始模型: {original_time:.4f}s")
    print(f"优化模型: {optimized_time:.4f}s")
    print(f"加速比: {original_time/optimized_time:.2f}x")
    
    if optimized_time < original_time:
        print("✓ 优化生效!")
    else:
        print("❌ 优化未生效")

if __name__ == "__main__":
    benchmark_models()
