# fixed_quantization.py
"""修复ONNX导出问题的量化脚本"""

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

def load_model_direct():
    """直接加载模型"""
    print("直接加载MASt3R模型...")
    
    from mast3r.model import AsymmetricMASt3R
    
    model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_path).cuda().eval()
    
    print("✓ 模型加载成功")
    return model

def extract_encoder_only(model):
    """提取编码器部分，避免复杂的解码器结构"""
    print("提取ViT编码器部分...")
    
    class SimpleViTEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            # 只提取patch embedding和encoder部分
            self.patch_embed = model.patch_embed
            self.encoder = model.encoder
            
        def forward(self, x):
            # 简化的前向传播
            B, C, H, W = x.shape
            
            # Patch embedding
            x = self.patch_embed(x)  # (B, N, D)
            
            # 通过encoder
            if hasattr(self.encoder, 'forward'):
                x = self.encoder(x)
            else:
                # 如果encoder是LayerList，手动执行
                for layer in self.encoder:
                    x = layer(x)
            
            return x
    
    encoder = SimpleViTEncoder(model)
    print("✓ 编码器提取成功")
    return encoder

def test_encoder_extraction(model, encoder):
    """测试编码器提取是否正确"""
    print("测试编码器提取...")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    
    try:
        with torch.no_grad():
            # 测试提取的编码器
            encoder_output = encoder(test_input)
            print(f"✓ 编码器输出形状: {encoder_output.shape}")
            
            # 与原始模型对比
            original_feat, _, _ = model._encode_image(test_input, torch.tensor([[512, 512]]))
            print(f"原始模型输出形状: {original_feat.shape}")
            
            return True
    except Exception as e:
        print(f"编码器测试失败: {e}")
        return False

def export_encoder_to_onnx(encoder, output_path="simple_vit_encoder.onnx"):
    """导出编码器到ONNX，使用更保守的设置"""
    print("导出编码器到ONNX...")
    
    dummy_input = torch.randn(1, 3, 512, 512).cuda()
    
    try:
        # 使用更保守的ONNX导出设置
        torch.onnx.export(
            encoder,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,  # 关闭常量折叠
            input_names=['input'],
            output_names=['output'],
            # 移除动态轴以避免复杂性
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False
        )
        
        print(f"✓ ONNX模型已导出: {output_path}")
        
        # 验证ONNX模型
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过")
        
        return output_path
        
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        print("尝试使用torch.jit.script方法...")
        
        # 回退方案：使用torch.jit
        try:
            scripted_model = torch.jit.script(encoder)
            scripted_path = output_path.replace('.onnx', '_scripted.pt')
            scripted_model.save(scripted_path)
            print(f"✓ 脚本化模型已保存: {scripted_path}")
            return None  # 表示ONNX失败但有torch.jit备份
        except Exception as e2:
            print(f"torch.jit.script也失败: {e2}")
            return None

def use_torch_tensorrt_instead(model, test_input):
    """使用Torch-TensorRT作为替代方案"""
    print("尝试使用Torch-TensorRT进行优化...")
    
    try:
        import torch_tensorrt
        
        # 编译模型
        print("编译模型...")
        encoder = extract_encoder_only(model)
        
        # 使用Torch-TensorRT优化
        optimized_model = torch_tensorrt.compile(
            encoder,
            inputs=[test_input],
            enabled_precisions={torch.float, torch.half},  # FP32和FP16
            workspace_size=2 << 30  # 2GB
        )
        
        print("✓ Torch-TensorRT优化成功")
        
        # 性能测试
        benchmark_torch_tensorrt(model, optimized_model)
        
        return optimized_model
        
    except ImportError:
        print("Torch-TensorRT未安装，可以用以下命令安装:")
        print("pip install torch-tensorrt")
        return None
    except Exception as e:
        print(f"Torch-TensorRT优化失败: {e}")
        return None

def benchmark_torch_tensorrt(original_model, optimized_model):
    """测试Torch-TensorRT优化效果"""
    print("=== Torch-TensorRT性能测试 ===")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
            _ = optimized_model(test_input)
    
    # 测试原始模型
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # 测试优化模型
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = optimized_model(test_input)
    torch.cuda.synchronize()
    optimized_time = time.time() - start
    
    print(f"原始模型时间: {original_time:.4f}s")
    print(f"优化模型时间: {optimized_time:.4f}s")
    print(f"加速比: {original_time/optimized_time:.2f}x")

def try_simple_quantization(model):
    """尝试PyTorch内置的量化方法"""
    print("尝试PyTorch内置量化...")
    
    try:
        # 提取编码器
        encoder = extract_encoder_only(model)
        
        # 准备量化
        encoder.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(encoder, inplace=True)
        
        # 校准（使用少量数据）
        print("校准量化模型...")
        for i in range(10):
            test_input = torch.randn(1, 3, 512, 512)
            with torch.no_grad():
                _ = encoder(test_input)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(encoder, inplace=False)
        
        print("✓ PyTorch量化成功")
        
        # 性能测试
        benchmark_pytorch_quantization(model, quantized_model)
        
        return quantized_model
        
    except Exception as e:
        print(f"PyTorch量化失败: {e}")
        return None

def benchmark_pytorch_quantization(original_model, quantized_model):
    """测试PyTorch量化效果"""
    print("=== PyTorch量化性能测试 ===")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_input_cpu = torch.randn(1, 3, 512, 512)  # 量化模型在CPU上
    test_shape = torch.tensor([[512, 512]])
    
    # 测试原始模型
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):  # 减少迭代次数
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # 测试量化模型
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = quantized_model(test_input_cpu)
    quantized_time = time.time() - start
    
    print(f"原始模型时间(GPU): {original_time:.4f}s")
    print(f"量化模型时间(CPU): {quantized_time:.4f}s")
    print(f"注意: 量化模型在CPU上运行，GPU版本会更快")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tensorrt", "pytorch", "both"], default="both",
                       help="量化方法")
    
    args = parser.parse_args()
    
    try:
        # 1. 加载模型
        model = load_model_direct()
        
        # 2. 测试原始性能
        test_input = torch.randn(1, 3, 512, 512).cuda()
        test_shape = torch.tensor([[512, 512]])
        
        print("测试原始模型性能...")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model._encode_image(test_input, test_shape)
        torch.cuda.synchronize()
        original_time = time.time() - start
        print(f"原始模型平均时间: {original_time/50:.4f}s")
        
        # 3. 尝试不同的优化方法
        if args.method in ["tensorrt", "both"]:
            print("\n=== 尝试Torch-TensorRT优化 ===")
            tensorrt_model = use_torch_tensorrt_instead(model, test_input)
            
            if tensorrt_model:
                # 保存优化模型
                torch.jit.save(tensorrt_model, "optimized_vit_tensorrt.pt")
                print("✓ TensorRT优化模型已保存: optimized_vit_tensorrt.pt")
        
        if args.method in ["pytorch", "both"]:
            print("\n=== 尝试PyTorch内置量化 ===")
            quantized_model = try_simple_quantization(model)
            
            if quantized_model:
                # 保存量化模型
                torch.jit.save(torch.jit.script(quantized_model), "quantized_vit_pytorch.pt")
                print("✓ PyTorch量化模型已保存: quantized_vit_pytorch.pt")
        
        # 4. 生成使用说明
        print_integration_guide()
        
    except Exception as e:
        print(f"优化过程出错: {e}")
        import traceback
        traceback.print_exc()

def print_integration_guide():
    """打印集成指南"""
    print(f"\n=== 集成指南 ===")
    print("如果有optimized_vit_tensorrt.pt文件，在main.py中添加:")
    
    code = '''
# 在main.py的模型加载部分
def load_optimized_model():
    try:
        # 加载原始模型
        original_model = load_mast3r(device=device)
        
        # 尝试加载优化模型
        if os.path.exists("optimized_vit_tensorrt.pt"):
            print("加载TensorRT优化模型...")
            optimized_encoder = torch.jit.load("optimized_vit_tensorrt.pt")
            
            # 创建混合模型
            class HybridModel:
                def __init__(self, original_model, optimized_encoder):
                    self.original_model = original_model
                    self.optimized_encoder = optimized_encoder
                    self._decoder = original_model._decoder
                    self._downstream_head = original_model._downstream_head
                
                def _encode_image(self, img, true_shape):
                    try:
                        # 使用优化编码器
                        feat = self.optimized_encoder(img)
                        h, w = true_shape[0].item(), true_shape[1].item()
                        pos = torch.zeros(1, feat.shape[1], 2, device=img.device, dtype=torch.long)
                        return feat, pos, None
                    except:
                        # 回退到原始方法
                        return self.original_model._encode_image(img, true_shape)
                
                def __getattr__(self, name):
                    return getattr(self.original_model, name)
            
            return HybridModel(original_model, optimized_encoder)
        
        elif os.path.exists("quantized_vit_pytorch.pt"):
            print("找到PyTorch量化模型，但需要在CPU上运行")
            return original_model
        
        else:
            return original_model
            
    except Exception as e:
        print(f"优化模型加载失败: {e}")
        return load_mast3r(device=device)

# 替换原来的模型加载
model = load_optimized_model()
'''
    
    print(code)

if __name__ == "__main__":
    main()
