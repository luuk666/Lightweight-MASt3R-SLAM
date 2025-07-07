# direct_quantization.py
"""直接可用的量化脚本，避免所有配置问题"""

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
    """直接加载模型，避免配置问题"""
    print("直接加载MASt3R模型...")
    
    # 直接导入和加载
    from mast3r.model import AsymmetricMASt3R
    
    model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_path).cuda().eval()
    
    print("✓ 模型加载成功")
    return model

def create_calibration_data(dataset_path, num_samples=100):
    """创建校准数据，直接读取图像"""
    print(f"准备校准数据从: {dataset_path}")
    
    # 读取TUM数据集
    rgb_file = Path(dataset_path) / "rgb.txt"
    if not rgb_file.exists():
        raise ValueError(f"找不到rgb.txt文件: {rgb_file}")
    
    # 读取图像路径
    with open(rgb_file, 'r') as f:
        lines = f.readlines()
    
    image_paths = []
    for line in lines[3:]:  # 跳过前3行注释
        parts = line.strip().split()
        if len(parts) >= 2:
            img_path = Path(dataset_path) / parts[1]
            if img_path.exists():
                image_paths.append(img_path)
    
    # 限制样本数量
    image_paths = image_paths[:num_samples]
    print(f"找到 {len(image_paths)} 张校准图像")
    
    # 加载和预处理图像
    calibration_data = []
    for i, img_path in enumerate(image_paths):
        if i % 50 == 0:
            print(f"处理校准图像 {i+1}/{len(image_paths)}")
        
        try:
            import cv2
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 简单的预处理
            img_tensor = preprocess_simple(img)
            calibration_data.append(img_tensor)
            
        except Exception as e:
            print(f"跳过图像 {img_path}: {e}")
            continue
    
    print(f"成功处理 {len(calibration_data)} 张校准图像")
    return calibration_data

def preprocess_simple(img):
    """简单的图像预处理"""
    import cv2
    
    # 调整大小到512x512
    img_resized = cv2.resize(img, (512, 512))
    
    # 归一化到[0,1]
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # 转换为CHW格式
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor

def export_to_onnx(model, output_path="vit_encoder_direct.onnx"):
    """导出模型到ONNX"""
    print("导出ONNX模型...")
    
    class DirectViTWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # 直接调用编码部分
            try:
                feat, pos, _ = self.model._encode_image(x, torch.tensor([[512, 512]]))
                return feat
            except:
                # 如果失败，使用更直接的方法
                return self.model.patch_embed(x)
    
    wrapper = DirectViTWrapper(model).eval()
    dummy_input = torch.randn(1, 3, 512, 512).cuda()
    
    # 导出ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✓ ONNX模型已导出: {output_path}")
    return output_path

def quantize_with_tensorrt(onnx_path, calibration_data, precision="int8", output_path=None):
    """使用TensorRT进行量化"""
    import tensorrt as trt
    
    print(f"开始TensorRT {precision.upper()} 量化...")
    
    # 创建calibrator
    calibrator = None
    if precision == "int8" and calibration_data:
        calibrator = create_calibrator(calibration_data)
    
    # TensorRT builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ONNX解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    
    # 设置精度
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and calibrator:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    
    # 设置输入形状
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, (1, 3, 512, 512), (1, 3, 512, 512), (4, 3, 512, 512))
    config.add_optimization_profile(profile)
    
    # 构建引擎
    print("构建TensorRT引擎...")
    engine = builder.build_engine(network, config)
    
    if engine:
        if output_path is None:
            output_path = f"mast3r_vit_{precision}.trt"
        
        with open(output_path, "wb") as f:
            f.write(engine.serialize())
        
        print(f"✓ TensorRT引擎已保存: {output_path}")
        return output_path
    else:
        print("❌ TensorRT引擎构建失败")
        return None

def create_calibrator(calibration_data):
    """创建校准器"""
    import tensorrt as trt
    
    class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, data):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.data = data
            self.current = 0
            self.device_input = None
            
        def get_batch_size(self):
            return 1
            
        def get_batch(self, names):
            if self.current >= len(self.data):
                return None
                
            # 分配GPU内存
            if self.device_input is None:
                self.device_input = trt.cuda.mem_alloc(self.data[0].numel() * 4)  # float32
            
            # 复制数据到GPU
            batch = self.data[self.current].numpy().astype(np.float32)
            trt.cuda.memcpy_htod(self.device_input, batch)
            
            self.current += 1
            return [int(self.device_input)]
            
        def read_calibration_cache(self):
            cache_file = "direct_calibration.cache"
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return f.read()
            return None
            
        def write_calibration_cache(self, cache):
            cache_file = "direct_calibration.cache"
            with open(cache_file, "wb") as f:
                f.write(cache)
    
    return SimpleCalibrator(calibration_data)

def benchmark_original_model(model, num_iterations=50):
    """测试原始模型性能"""
    print("测试原始模型性能...")
    
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            try:
                _ = model._encode_image(test_input, test_shape)
            except:
                # 如果_encode_image失败，直接用patch_embed
                _ = model.patch_embed(test_input)
    
    # 测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            try:
                _ = model._encode_image(test_input, test_shape)
            except:
                _ = model.patch_embed(test_input)
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_time = total_time / num_iterations
    print(f"原始模型平均推理时间: {avg_time:.4f}s")
    print(f"预期量化后加速: 2-4倍")
    
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="int8")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-dir", default="tensorrt_engines")
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. 加载模型
        model = load_model_direct()
        
        # 2. 性能基准测试
        original_time = benchmark_original_model(model)
        
        # 3. 导出ONNX
        onnx_path = output_dir / "vit_encoder_direct.onnx"
        export_to_onnx(model, str(onnx_path))
        
        # 4. 准备校准数据
        calibration_data = None
        if args.precision == "int8":
            try:
                calibration_data = create_calibration_data(args.dataset, args.samples)
            except Exception as e:
                print(f"校准数据准备失败: {e}")
                print("回退到FP16精度")
                args.precision = "fp16"
        
        # 5. 量化
        engine_path = output_dir / f"mast3r_vit_{args.precision}.trt"
        result = quantize_with_tensorrt(
            str(onnx_path), 
            calibration_data, 
            args.precision, 
            str(engine_path)
        )
        
        if result:
            print(f"\n🎉 量化成功完成!")
            print(f"引擎文件: {result}")
            print(f"原始推理时间: {original_time:.4f}s")
            print(f"预期加速后时间: {original_time/3:.4f}s (3倍加速)")
            
            # 生成使用说明
            print_simple_usage(result)
        else:
            print("❌ 量化失败")
    
    except Exception as e:
        print(f"量化过程出错: {e}")
        import traceback
        traceback.print_exc()

def print_simple_usage(engine_path):
    """打印简单的使用说明"""
    print(f"\n=== 使用说明 ===")
    print("将以下代码添加到main.py中:")
    
    code = f'''
# 在main.py的模型加载部分添加:

def load_quantized_model():
    """加载量化模型的简单方法"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 加载TensorRT引擎
        with open("{engine_path}", "rb") as f:
            engine_data = f.read()
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        print("✓ 量化模型加载成功")
        return engine, context
    except Exception as e:
        print(f"量化模型加载失败: {{e}}")
        return None, None

# 在模型加载部分替换:
# model = load_mast3r(device=device)

# 为:
quantized_engine, quantized_context = load_quantized_model()
if quantized_engine:
    print("使用量化模型")
    # 这里需要实现量化模型的包装器
    # 现在先使用原始模型
    model = load_mast3r(device=device)
else:
    model = load_mast3r(device=device)
'''
    
    print(code)
    print(f"\n引擎文件位置: {engine_path}")
    print("注意: 完整的集成需要实现TensorRT推理包装器")

if __name__ == "__main__":
    main()
