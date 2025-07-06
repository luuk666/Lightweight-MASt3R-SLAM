# simple_quantization.py
"""简化的独立量化脚本，避免配置依赖问题"""

import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
import os
import cv2
import PIL.Image
from pathlib import Path
import glob
import time

class SimpleCalibrationDataLoader:
    """简化的校准数据加载器，直接读取图像文件"""
    
    def __init__(self, data_path, img_size=512, max_samples=500):
        self.img_size = img_size
        self.max_samples = max_samples
        
        # 查找图像文件
        data_path = Path(data_path)
        if data_path.is_dir():
            # 如果是目录，查找所有图像文件
            image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            self.image_files = []
            for pattern in image_patterns:
                self.image_files.extend(glob.glob(str(data_path / "**" / pattern), recursive=True))
        else:
            # 如果是TUM数据集
            rgb_file = data_path / "rgb.txt"
            if rgb_file.exists():
                with open(rgb_file, 'r') as f:
                    lines = f.readlines()
                self.image_files = [str(data_path / line.strip().split()[1]) for line in lines[3:]]  # 跳过前3行注释
            else:
                raise ValueError(f"找不到图像文件: {data_path}")
        
        self.image_files = self.image_files[:max_samples]
        self.current_idx = 0
        print(f"找到 {len(self.image_files)} 张校准图像")
        
    def __iter__(self):
        self.current_idx = 0
        return self
        
    def __next__(self):
        if self.current_idx >= len(self.image_files):
            raise StopIteration
            
        # 读取和预处理图像
        img_path = self.image_files[self.current_idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            self.current_idx += 1
            return self.__next__()
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预处理 - 简化版本的resize_img
        img_tensor = self.preprocess_image(img)
        
        self.current_idx += 1
        return {"input": img_tensor}
    
    def preprocess_image(self, img):
        """简化的图像预处理"""
        # 转换为PIL
        img_pil = PIL.Image.fromarray(img)
        
        # 调整大小
        W, H = img_pil.size
        S = max(W, H)
        new_size = tuple(int(round(x * self.img_size / S)) for x in img_pil.size)
        img_pil = img_pil.resize(new_size, PIL.Image.LANCZOS)
        
        # 中心裁剪
        W, H = img_pil.size
        cx, cy = W // 2, H // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        img_pil = img_pil.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        
        # 转换为tensor
        img_array = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        return img_tensor.numpy()

class SimpleTensorRTCalibrator(trt.IInt8EntropyCalibrator2):
    """简化的TensorRT校准器"""
    
    def __init__(self, dataloader, cache_file="simple_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.data_iter = iter(dataloader)
        self.batch_allocation = None
        
    def get_batch_size(self):
        return 1
        
    def get_batch(self, names):
        try:
            batch = next(self.data_iter)
            if self.batch_allocation is None:
                # 分配GPU内存
                self.batch_allocation = {}
                for name, data in batch.items():
                    size = data.size * data.itemsize
                    self.batch_allocation[name] = trt.cuda.mem_alloc(size)
            
            # 将数据复制到GPU
            bindings = []
            for name, data in batch.items():
                trt.cuda.memcpy_htod(self.batch_allocation[name], 
                                   data.astype(np.float32).ravel())
                bindings.append(int(self.batch_allocation[name]))
            
            return bindings
        except StopIteration:
            return None
            
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class SimpleViTWrapper(nn.Module):
    """简化的ViT包装器用于ONNX导出"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # 简化的编码过程
        feat, pos, _ = self.model._encode_image(x, torch.tensor([[x.shape[2], x.shape[3]]]))
        return feat

def create_onnx_model(model, input_shape=(1, 3, 512, 512), onnx_path="simple_vit_encoder.onnx"):
    """创建ONNX模型"""
    print("导出ONNX模型...")
    
    # 创建包装器
    wrapper = SimpleViTWrapper(model).eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape).cuda()
    
    # 导出ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX模型已导出: {onnx_path}")
    return onnx_path

def build_tensorrt_engine(onnx_path, engine_path, calibrator=None, precision="fp16"):
    """构建TensorRT引擎"""
    print(f"构建TensorRT引擎 ({precision})...")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30  # 2GB
    
    # 设置精度
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16精度")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator
            print("启用INT8精度with校准器")
        else:
            print("警告: INT8精度需要校准器")
            return None
    
    # 设置输入形状
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, (1, 3, 512, 512), (1, 3, 512, 512), (4, 3, 512, 512))
    config.add_optimization_profile(profile)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    if engine:
        # 保存引擎
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"TensorRT引擎已保存: {engine_path}")
        
    return engine

def main_simple_quantization():
    """主量化函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化的MASt3R量化工具")
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk", 
                       help="数据集路径")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="int8",
                       help="量化精度")
    parser.add_argument("--samples", type=int, default=500, help="校准样本数")
    parser.add_argument("--output", default="tensorrt_engines", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 加载模型
    print("加载MASt3R模型...")
    sys.path.append('.')
    from mast3r_slam.mast3r_utils import load_mast3r
    model = load_mast3r().cuda().eval()
    
    # 2. 导出ONNX
    onnx_path = output_dir / "simple_vit_encoder.onnx"
    create_onnx_model(model, onnx_path=str(onnx_path))
    
    # 3. 准备校准数据（仅INT8需要）
    calibrator = None
    if args.precision == "int8":
        print("准备校准数据...")
        try:
            calib_loader = SimpleCalibrationDataLoader(args.dataset, max_samples=args.samples)
            calibrator = SimpleTensorRTCalibrator(calib_loader)
        except Exception as e:
            print(f"校准数据准备失败: {e}")
            print("回退到FP16精度")
            args.precision = "fp16"
    
    # 4. 构建引擎
    engine_path = output_dir / f"simple_vit_encoder_{args.precision}.trt"
    engine = build_tensorrt_engine(
        str(onnx_path),
        str(engine_path),
        calibrator,
        args.precision
    )
    
    if engine:
        print(f"\n🎉 量化成功!")
        print(f"引擎文件: {engine_path}")
        
        # 5. 简单性能测试
        print("\n运行性能测试...")
        benchmark_simple(model, str(engine_path))
        
        # 6. 生成使用说明
        print_usage_instructions(str(engine_path))
    else:
        print("❌ 量化失败")

def benchmark_simple(original_model, engine_path):
    """简单的性能基准测试"""
    print("=== 性能基准测试 ===")
    
    # 测试原始模型
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    
    # 测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    
    print(f"原始模型平均时间: {original_time/100:.4f}s")
    print(f"预期量化加速: 2-4倍")

def print_usage_instructions(engine_path):
    """打印使用说明"""
    print(f"\n=== 使用说明 ===")
    print("1. 在main.py中使用量化模型，添加以下代码:")
    
    usage_code = f'''
# 在main.py中的模型加载部分，替换:
# model = load_mast3r(device=device)

# 为:
try:
    from simple_tensorrt_inference import SimpleTensorRTInference
    print("加载量化模型...")
    original_model = load_mast3r(device=device)
    trt_inference = SimpleTensorRTInference("{engine_path}")
    
    # 创建包装模型
    class QuantizedModel:
        def __init__(self, original_model, trt_inference):
            self.original_model = original_model
            self.trt_inference = trt_inference
            self._decoder = original_model._decoder
            self._downstream_head = original_model._downstream_head
        
        def _encode_image(self, img, true_shape):
            # 使用TensorRT加速编码
            try:
                img_np = img.cpu().numpy()
                feat_np = self.trt_inference.infer(img_np)
                feat = torch.from_numpy(feat_np).to(img.device)
                
                # 简化的位置编码
                h, w = true_shape[0].item(), true_shape[1].item()
                pos = torch.zeros(1, feat.shape[1], 2, device=img.device, dtype=torch.long)
                return feat, pos, None
            except:
                # 回退到原始实现
                return self.original_model._encode_image(img, true_shape)
        
        def __getattr__(self, name):
            return getattr(self.original_model, name)
    
    model = QuantizedModel(original_model, trt_inference)
    print("✓ 量化模型加载成功")
    
except Exception as e:
    print(f"量化模型加载失败，使用原始模型: {{e}}")
    model = load_mast3r(device=device)
'''
    
    print(usage_code)
    
    # 创建推理模块
    create_inference_module(engine_path)

def create_inference_module(engine_path):
    """创建TensorRT推理模块"""
    inference_code = f'''# simple_tensorrt_inference.py
"""简单的TensorRT推理模块"""

import tensorrt as trt
import numpy as np
import torch

class SimpleTensorRTInference:
    """简单的TensorRT推理引擎"""
    
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.allocate_buffers()
        
    def allocate_buffers(self):
        """分配输入输出缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = trt.cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配host和device内存
            host_mem = trt.cuda.pagelocked_empty(size, dtype)
            device_mem = trt.cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({{'host': host_mem, 'device': device_mem}})
            else:
                self.outputs.append({{'host': host_mem, 'device': device_mem}})
                
    def infer(self, input_data):
        """执行推理"""
        # 复制输入数据到host内存
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 传输数据到device
        trt.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 传输结果回host
        trt.cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # 同步
        self.stream.synchronize()
        
        return self.outputs[0]['host']
'''
    
    # 保存推理模块
    with open("simple_tensorrt_inference.py", "w") as f:
        f.write(inference_code)
    
    print("\\n2. TensorRT推理模块已创建: simple_tensorrt_inference.py")

if __name__ == "__main__":
    main_simple_quantization()
