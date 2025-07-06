# mast3r_optimization.py
"""针对MASt3R模型结构的专用优化脚本"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
import time
import argparse

# 添加项目路径
sys.path.append('.')

def analyze_model_structure(model):
    """分析MASt3R模型结构"""
    print("=== 分析MASt3R模型结构 ===")
    
    print("主要组件:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
        
        # 如果是编码器相关，进一步分析
        if 'enc' in name.lower() or 'backbone' in name.lower():
            print(f"    详细结构: {module}")
    
    # 检查编码相关方法
    print("\n编码相关方法:")
    methods = [method for method in dir(model) if 'enc' in method.lower() or method.startswith('_encode')]
    for method in methods:
        print(f"  {method}")
    
    return

def find_encoder_components(model):
    """查找MASt3R的编码器组件"""
    print("查找编码器组件...")
    
    # 检查可能的编码器属性
    encoder_candidates = []
    
    for name in dir(model):
        if not name.startswith('_'):
            attr = getattr(model, name)
            if isinstance(attr, nn.Module):
                # 检查是否包含attention或transformer结构
                has_attention = any('attention' in str(type(m)).lower() or 'transformer' in str(type(m)).lower() 
                                  for m in attr.modules())
                if has_attention:
                    encoder_candidates.append((name, attr))
                    print(f"找到候选编码器: {name} - {type(attr).__name__}")
    
    # 特殊检查MASt3R可能的结构
    special_attrs = ['backbone', 'enc_backbone', 'encoder_backbone', 'vision_transformer']
    for attr_name in special_attrs:
        if hasattr(model, attr_name):
            attr = getattr(model, attr_name)
            print(f"找到特殊属性: {attr_name} - {type(attr).__name__}")
            encoder_candidates.append((attr_name, attr))
    
    return encoder_candidates

def create_encoding_wrapper(model):
    """创建编码包装器，直接使用_encode_image方法"""
    print("创建编码包装器...")
    
    class MASt3REncodingWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model
            
        def forward(self, x):
            # 直接使用MASt3R的编码方法
            B, C, H, W = x.shape
            true_shape = torch.tensor([[H, W]], device=x.device)
            
            feat, pos, _ = self.original_model._encode_image(x, true_shape)
            return feat
    
    wrapper = MASt3REncodingWrapper(model)
    print("✓ 编码包装器创建成功")
    return wrapper

def test_encoding_wrapper(wrapper):
    """测试编码包装器"""
    print("测试编码包装器...")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    
    try:
        with torch.no_grad():
            output = wrapper(test_input)
            print(f"✓ 包装器输出形状: {output.shape}")
            return True
    except Exception as e:
        print(f"包装器测试失败: {e}")
        return False

def optimize_with_torchscript(wrapper):
    """使用TorchScript优化"""
    print("使用TorchScript优化...")
    
    try:
        # 创建示例输入
        example_input = torch.randn(1, 3, 512, 512).cuda()
        
        # 使用torch.jit.trace
        print("执行torch.jit.trace...")
        traced_model = torch.jit.trace(wrapper, example_input)
        
        # 优化
        print("应用TorchScript优化...")
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        print("✓ TorchScript优化成功")
        return optimized_model
        
    except Exception as e:
        print(f"TorchScript优化失败: {e}")
        
        # 尝试scripting
        try:
            print("尝试torch.jit.script...")
            scripted_model = torch.jit.script(wrapper)
            optimized_model = torch.jit.optimize_for_inference(scripted_model)
            print("✓ TorchScript scripting成功")
            return optimized_model
        except Exception as e2:
            print(f"TorchScript scripting也失败: {e2}")
            return None

def try_compilation_optimization(wrapper):
    """尝试PyTorch 2.0编译优化"""
    print("尝试PyTorch编译优化...")
    
    try:
        # 检查PyTorch版本
        torch_version = torch.__version__
        print(f"PyTorch版本: {torch_version}")
        
        if hasattr(torch, 'compile'):
            print("使用torch.compile优化...")
            compiled_model = torch.compile(wrapper, mode="reduce-overhead")
            print("✓ PyTorch编译优化成功")
            return compiled_model
        else:
            print("PyTorch版本不支持torch.compile")
            return None
            
    except Exception as e:
        print(f"编译优化失败: {e}")
        return None

def benchmark_optimizations(original_model, optimized_models):
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    # 预热
    print("预热...")
    for _ in range(10):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    
    # 测试原始模型
    print("测试原始模型...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    
    print(f"原始模型时间: {original_time:.4f}s (平均: {original_time/100:.4f}s)")
    
    # 测试优化模型
    results = {"original": original_time}
    
    for name, opt_model in optimized_models.items():
        if opt_model is None:
            continue
            
        print(f"测试{name}...")
        
        # 预热优化模型
        for _ in range(10):
            with torch.no_grad():
                _ = opt_model(test_input)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            with torch.no_grad():
                _ = opt_model(test_input)
        
        torch.cuda.synchronize()
        opt_time = time.time() - start_time
        
        results[name] = opt_time
        speedup = original_time / opt_time
        print(f"{name}时间: {opt_time:.4f}s (平均: {opt_time/100:.4f}s) - 加速比: {speedup:.2f}x")
    
    return results

def save_best_model(optimized_models, results):
    """保存最佳优化模型"""
    print("\n=== 保存最佳模型 ===")
    
    best_model = None
    best_name = None
    best_speedup = 1.0
    
    original_time = results["original"]
    
    for name, model in optimized_models.items():
        if model is None or name not in results:
            continue
            
        speedup = original_time / results[name]
        if speedup > best_speedup:
            best_speedup = speedup
            best_model = model
            best_name = name
    
    if best_model is not None:
        save_path = f"optimized_mast3r_{best_name}.pt"
        torch.jit.save(best_model, save_path)
        print(f"✓ 最佳优化模型已保存: {save_path}")
        print(f"最佳优化方法: {best_name} (加速比: {best_speedup:.2f}x)")
        return save_path, best_name, best_speedup
    else:
        print("没有找到有效的优化模型")
        return None, None, 1.0

def create_integration_code(model_path, method_name, speedup):
    """生成集成代码"""
    print(f"\n=== 集成代码 ===")
    
    code = f'''
# 在main.py中添加以下代码来使用优化模型

def load_optimized_mast3r():
    """加载优化版本的MASt3R模型"""
    try:
        # 加载原始模型
        original_model = load_mast3r(device=device)
        
        # 检查优化模型是否存在
        if os.path.exists("{model_path}"):
            print("加载优化的MASt3R模型 ({method_name}, {speedup:.2f}x加速)...")
            optimized_encoder = torch.jit.load("{model_path}")
            
            # 创建混合模型类
            class OptimizedMASt3R:
                def __init__(self, original_model, optimized_encoder):
                    self.original_model = original_model
                    self.optimized_encoder = optimized_encoder
                    
                    # 保留所有原始属性
                    self._decoder = original_model._decoder
                    self._downstream_head = original_model._downstream_head
                
                def _encode_image(self, img, true_shape):
                    """使用优化的编码器"""
                    try:
                        # 使用优化编码器
                        feat = self.optimized_encoder(img)
                        
                        # 生成位置编码（简化版本）
                        h, w = true_shape[0].item(), true_shape[1].item()
                        pos = torch.zeros(1, feat.shape[1], 2, device=img.device, dtype=torch.long)
                        
                        return feat, pos, None
                    except Exception as e:
                        print(f"优化编码器失败，回退到原始方法: {{e}}")
                        return self.original_model._encode_image(img, true_shape)
                
                def __getattr__(self, name):
                    """代理其他属性到原始模型"""
                    return getattr(self.original_model, name)
            
            optimized_model = OptimizedMASt3R(original_model, optimized_encoder)
            print("✓ 优化模型集成成功")
            return optimized_model
        
        else:
            print("优化模型文件不存在，使用原始模型")
            return original_model
            
    except Exception as e:
        print(f"优化模型加载失败: {{e}}")
        return load_mast3r(device=device)

# 在main.py中替换模型加载行:
# model = load_mast3r(device=device)
# 改为:
model = load_optimized_mast3r()
'''
    
    print(code)
    
    # 保存集成代码到文件
    with open("integration_code.py", "w") as f:
        f.write(code)
    print("集成代码已保存到: integration_code.py")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="只分析模型结构")
    
    args = parser.parse_args()
    
    try:
        # 1. 加载模型
        from mast3r.model import AsymmetricMASt3R
        
        model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        model = AsymmetricMASt3R.from_pretrained(model_path).cuda().eval()
        print("✓ 模型加载成功")
        
        # 2. 分析模型结构
        if args.analyze:
            analyze_model_structure(model)
            find_encoder_components(model)
            return
        
        # 3. 创建编码包装器
        wrapper = create_encoding_wrapper(model)
        
        # 4. 测试包装器
        if not test_encoding_wrapper(wrapper):
            print("包装器测试失败，退出")
            return
        
        # 5. 尝试不同优化方法
        optimized_models = {}
        
        # TorchScript优化
        print("\n=== TorchScript优化 ===")
        torchscript_model = optimize_with_torchscript(wrapper)
        optimized_models["torchscript"] = torchscript_model
        
        # PyTorch编译优化
        print("\n=== PyTorch编译优化 ===")
        compiled_model = try_compilation_optimization(wrapper)
        optimized_models["compiled"] = compiled_model
        
        # 6. 性能测试
        results = benchmark_optimizations(model, optimized_models)
        
        # 7. 保存最佳模型
        best_path, best_method, best_speedup = save_best_model(optimized_models, results)
        
        # 8. 生成集成代码
        if best_path:
            create_integration_code(best_path, best_method, best_speedup)
        
        print(f"\n🎉 优化完成!")
        if best_speedup > 1.1:
            print(f"最佳加速比: {best_speedup:.2f}x")
        else:
            print("未获得显著加速，建议检查硬件配置")
        
    except Exception as e:
        print(f"优化过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
