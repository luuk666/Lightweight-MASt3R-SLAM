# practical_optimization.py
"""实用的MASt3R优化方案 - 通过内存和计算优化获得加速"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
import time
import argparse
import contextlib

# 添加项目路径
sys.path.append('.')

class OptimizedMASt3RWrapper:
    """优化的MASt3R包装器"""
    
    def __init__(self, original_model, optimization_config):
        self.original_model = original_model
        self.config = optimization_config
        
        # 应用优化设置
        self._apply_optimizations()
        
        # 保留原始接口
        self._decoder = original_model._decoder
        self._downstream_head = original_model._downstream_head
        
        # 性能统计
        self.inference_times = []
        self.memory_usage = []
        
    def _apply_optimizations(self):
        """应用各种优化技术"""
        
        # 1. 设置模型为eval模式并禁用梯度
        self.original_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        # 2. 使用inference mode
        self.inference_mode = True
        
        # 3. 内存优化
        if self.config.get('use_half_precision', False):
            self.original_model = self.original_model.half()
            print("✓ 启用半精度推理")
        
        # 4. 编译关键模块（如果可能）
        if self.config.get('try_compile_modules', False):
            self._try_compile_modules()
    
    def _try_compile_modules(self):
        """尝试编译单个模块"""
        try:
            # 尝试编译patch_embed
            if hasattr(self.original_model, 'patch_embed'):
                try:
                    self.original_model.patch_embed = torch.compile(
                        self.original_model.patch_embed, 
                        mode="reduce-overhead",
                        dynamic=False
                    )
                    print("✓ patch_embed编译成功")
                except:
                    print("patch_embed编译失败")
            
            # 尝试编译部分enc_blocks
            if hasattr(self.original_model, 'enc_blocks'):
                try:
                    for i, block in enumerate(self.original_model.enc_blocks[:4]):  # 只编译前几层
                        self.original_model.enc_blocks[i] = torch.compile(
                            block, 
                            mode="reduce-overhead",
                            dynamic=False
                        )
                    print("✓ 前4个编码器块编译成功")
                except:
                    print("编码器块编译失败")
                    
        except Exception as e:
            print(f"模块编译失败: {e}")
    
    def _encode_image(self, img, true_shape):
        """优化的图像编码"""
        
        # 选择优化策略
        if self.config.get('use_amp', False):
            return self._encode_with_amp(img, true_shape)
        elif self.config.get('use_inference_mode', True):
            return self._encode_with_inference_mode(img, true_shape)
        else:
            return self.original_model._encode_image(img, true_shape)
    
    def _encode_with_amp(self, img, true_shape):
        """使用自动混合精度"""
        with torch.cuda.amp.autocast():
            return self.original_model._encode_image(img, true_shape)
    
    def _encode_with_inference_mode(self, img, true_shape):
        """使用推理模式"""
        with torch.inference_mode():
            start_time = time.time()
            result = self.original_model._encode_image(img, true_shape)
            
            # 记录性能
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return result
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            "num_inferences": len(times),
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times)
        }
    
    def __getattr__(self, name):
        """代理其他属性"""
        return getattr(self.original_model, name)

def create_optimization_configs():
    """创建不同的优化配置"""
    
    configs = {
        "baseline": {
            "use_inference_mode": True,
            "use_half_precision": False,
            "use_amp": False,
            "try_compile_modules": False
        },
        
        "memory_optimized": {
            "use_inference_mode": True,
            "use_half_precision": True,
            "use_amp": False,
            "try_compile_modules": False
        },
        
        "amp_optimized": {
            "use_inference_mode": True,
            "use_half_precision": False,
            "use_amp": True,
            "try_compile_modules": False
        },
        
        "aggressive": {
            "use_inference_mode": True,
            "use_half_precision": True,
            "use_amp": True,
            "try_compile_modules": True
        }
    }
    
    return configs

def benchmark_configurations(model, configs, num_iterations=50):
    """测试不同配置的性能"""
    print("\n=== 配置性能测试 ===")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n测试配置: {config_name}")
        
        try:
            # 创建优化包装器
            optimized_model = OptimizedMASt3RWrapper(model, config)
            
            # 预热
            for _ in range(10):
                _ = optimized_model._encode_image(test_input, test_shape)
            
            # 清空之前的统计
            optimized_model.inference_times.clear()
            
            # 性能测试
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = optimized_model._encode_image(test_input, test_shape)
            
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            # 获取详细统计
            stats = optimized_model.get_performance_stats()
            results[config_name] = {
                'total_time': total_time,
                'avg_time': total_time / num_iterations,
                'stats': stats
            }
            
            print(f"  总时间: {total_time:.4f}s")
            print(f"  平均时间: {total_time/num_iterations:.4f}s")
            
        except Exception as e:
            print(f"  配置 {config_name} 失败: {e}")
            results[config_name] = None
    
    return results

def find_best_configuration(results):
    """找到最佳配置"""
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return None, None
    
    # 按平均时间排序
    best_config = min(valid_results.items(), key=lambda x: x[1]['avg_time'])
    return best_config

def apply_system_optimizations():
    """应用系统级优化"""
    print("应用系统级优化...")
    
    # PyTorch优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 禁用调试功能
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    
    print("✓ 系统优化已应用")

def create_production_model(model, best_config_name, best_config):
    """创建生产就绪的优化模型"""
    print(f"\n=== 创建生产模型 (配置: {best_config_name}) ===")
    
    optimized_model = OptimizedMASt3RWrapper(model, best_config)
    
    # 保存配置信息
    model_info = {
        'config_name': best_config_name,
        'config': best_config,
        'optimization_applied': True
    }
    
    return optimized_model, model_info

def generate_integration_code(model_info):
    """生成集成代码"""
    
    config_str = str(model_info['config']).replace("'", '"')
    
    code = f'''
# optimized_mast3r_loader.py
"""优化的MASt3R模型加载器"""

import torch
import torch.nn as nn
import time
import numpy as np

class OptimizedMASt3RWrapper:
    """优化的MASt3R包装器"""
    
    def __init__(self, original_model, optimization_config):
        self.original_model = original_model
        self.config = optimization_config
        
        # 应用优化设置
        self._apply_optimizations()
        
        # 保留原始接口
        self._decoder = original_model._decoder
        self._downstream_head = original_model._downstream_head
        
    def _apply_optimizations(self):
        """应用优化技术"""
        self.original_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        if self.config.get('use_half_precision', False):
            self.original_model = self.original_model.half()
    
    def _encode_image(self, img, true_shape):
        """优化的图像编码"""
        if self.config.get('use_amp', False):
            with torch.cuda.amp.autocast():
                return self.original_model._encode_image(img, true_shape)
        elif self.config.get('use_inference_mode', True):
            with torch.inference_mode():
                return self.original_model._encode_image(img, true_shape)
        else:
            return self.original_model._encode_image(img, true_shape)
    
    def __getattr__(self, name):
        return getattr(self.original_model, name)

def load_optimized_mast3r(device="cuda"):
    """加载优化的MASt3R模型"""
    
    # 应用系统优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 加载原始模型
    from mast3r_slam.mast3r_utils import load_mast3r
    original_model = load_mast3r(device=device)
    
    # 最佳配置
    best_config = {config_str}
    
    # 创建优化包装器
    optimized_model = OptimizedMASt3RWrapper(original_model, best_config)
    
    print("✓ 优化MASt3R模型加载完成 (配置: {model_info['config_name']})")
    return optimized_model

# 在main.py中使用:
# from optimized_mast3r_loader import load_optimized_mast3r
# model = load_optimized_mast3r(device=device)
'''
    
    print("=== 集成代码 ===")
    print(code)
    
    # 保存到文件
    with open("optimized_mast3r_loader.py", "w") as f:
        f.write(code)
    
    print("\n✓ 集成代码已保存到: optimized_mast3r_loader.py")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50, help="测试迭代次数")
    
    args = parser.parse_args()
    
    try:
        print("=== MASt3R实用优化方案 ===")
        
        # 1. 应用系统优化
        apply_system_optimizations()
        
        # 2. 加载模型
        print("\n加载MASt3R模型...")
        from mast3r.model import AsymmetricMASt3R
        
        model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        model = AsymmetricMASt3R.from_pretrained(model_path).cuda().eval()
        print("✓ 模型加载成功")
        
        # 3. 测试原始性能
        print("\n测试原始模型性能...")
        test_input = torch.randn(1, 3, 512, 512).cuda()
        test_shape = torch.tensor([[512, 512]])
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model._encode_image(test_input, test_shape)
        
        # 测试
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(args.iterations):
            with torch.no_grad():
                _ = model._encode_image(test_input, test_shape)
        torch.cuda.synchronize()
        baseline_time = time.time() - start_time
        
        print(f"原始模型时间: {baseline_time:.4f}s (平均: {baseline_time/args.iterations:.4f}s)")
        
        # 4. 测试不同优化配置
        configs = create_optimization_configs()
        results = benchmark_configurations(model, configs, args.iterations)
        
        # 5. 找到最佳配置
        best_result = find_best_configuration(results)
        
        if best_result:
            best_name, best_data = best_result
            speedup = baseline_time / best_data['total_time']
            
            print(f"\n🎉 最佳配置: {best_name}")
            print(f"加速比: {speedup:.2f}x")
            print(f"时间减少: {(1-1/speedup)*100:.1f}%")
            
            # 6. 创建生产模型
            config = configs[best_name]
            optimized_model, model_info = create_production_model(model, best_name, config)
            
            # 7. 生成集成代码
            generate_integration_code(model_info)
            
            print(f"\n✅ 优化完成!")
            print(f"最佳性能提升: {speedup:.2f}x")
            print("请使用生成的 optimized_mast3r_loader.py 来加载优化模型")
            
        else:
            print("❌ 未找到有效的优化配置")
        
    except Exception as e:
        print(f"优化过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
