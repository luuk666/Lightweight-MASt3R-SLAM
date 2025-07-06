# debug_optimization.py
"""调试优化是否生效"""

import torch
import sys
sys.path.append('.')

def check_optimization_status():
    """检查优化是否正确加载"""
    print("=== 调试优化状态 ===")
    
    # 1. 检查是否能导入优化模块
    try:
        from optimized_mast3r_loader import load_optimized_mast3r
        print("✓ 能够导入优化模块")
    except ImportError as e:
        print(f"❌ 无法导入优化模块: {e}")
        return False
    
    # 2. 检查模型加载
    try:
        model = load_optimized_mast3r()
        print("✓ 优化模型加载成功")
        
        # 检查是否是包装器
        if hasattr(model, 'config') and hasattr(model, 'original_model'):
            print("✓ 确认使用了优化包装器")
            print(f"优化配置: {model.config}")
        else:
            print("❌ 没有使用优化包装器")
            return False
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 3. 测试AMP是否生效
    try:
        test_input = torch.randn(1, 3, 512, 512).cuda()
        test_shape = torch.tensor([[512, 512]])
        
        print("测试AMP优化...")
        with torch.amp.autocast('cuda'):
            # 监控autocast状态
            print(f"AMP状态: {torch.is_autocast_enabled('cuda')}")
            _ = model._encode_image(test_input, test_shape)
        
        print("✓ AMP测试成功")
        return True
        
    except Exception as e:
        print(f"❌ AMP测试失败: {e}")
        return False

def fix_main_py():
    """生成正确的main.py修改"""
    print("\n=== main.py修改指南 ===")
    
    fix_code = '''
# 在main.py中找到这一行（大约在第57行）:
model = load_mast3r(device=device)

# 替换为:
try:
    from optimized_mast3r_loader import load_optimized_mast3r
    model = load_optimized_mast3r(device=device)
    print("✓ 使用优化模型")
except Exception as e:
    print(f"优化模型加载失败，使用原始模型: {e}")
    model = load_mast3r(device=device)
'''
    
    print(fix_code)

def create_test_script():
    """创建测试脚本"""
    test_code = '''# test_optimization.py
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
    
    print(f"\\n=== 性能对比 ===")
    print(f"原始模型: {original_time:.4f}s")
    print(f"优化模型: {optimized_time:.4f}s")
    print(f"加速比: {original_time/optimized_time:.2f}x")
    
    if optimized_time < original_time:
        print("✓ 优化生效!")
    else:
        print("❌ 优化未生效")

if __name__ == "__main__":
    benchmark_models()
'''
    
    with open("test_optimization.py", "w") as f:
        f.write(test_code)
    
    print("✓ 测试脚本已创建: test_optimization.py")

if __name__ == "__main__":
    # 检查优化状态
    is_working = check_optimization_status()
    
    if not is_working:
        print("\n❌ 优化未正确设置")
        fix_main_py()
        create_test_script()
        print("\n请执行以下步骤:")
        print("1. 修改main.py（按照上面的指南）")
        print("2. 运行: python test_optimization.py")
        print("3. 确认优化生效后再运行SLAM")
    else:
        print("\n✓ 优化设置正确")
        create_test_script()
        print("运行: python test_optimization.py 来验证性能")
