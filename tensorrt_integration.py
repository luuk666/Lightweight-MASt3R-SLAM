# tensorrt_integration.py
"""TensorRT INT8量化模型集成到MASt3R-SLAM"""

import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import time
from pathlib import Path
import sys

class TensorRTInference:
    """TensorRT推理引擎包装器"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError(f"无法加载TensorRT引擎: {engine_path}")
            
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 分配内存缓冲区
        self._allocate_buffers()
        print(f"✓ TensorRT引擎加载成功: {engine_path}")
        
    def _allocate_buffers(self):
        """分配输入输出缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # 创建CUDA流
        import pycuda.driver as cuda
        import pycuda.autoinit
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_bindings):
            binding = self.engine.get_binding_name(i)
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配host和device内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'size': size})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'size': size})
                
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行TensorRT推理"""
        import pycuda.driver as cuda
        
        # 确保输入在CPU上
        if input_tensor.is_cuda:
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = input_tensor.detach().numpy()
            
        # 展平输入数据
        input_flat = input_np.astype(np.float32).ravel()
        
        # 复制输入数据到host内存
        np.copyto(self.inputs[0]['host'], input_flat)
        
        # 传输数据到GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 传输结果回CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # 同步等待完成
        self.stream.synchronize()
        
        # 将结果转换为torch tensor并reshape
        output_data = self.outputs[0]['host']
        
        # 估算输出形状 (需要根据实际模型调整)
        # ViT-Large通常输出 [batch_size, num_patches + 1, hidden_dim]
        # 对于512x512输入，patch_size=16，num_patches = (512/16)^2 = 1024
        batch_size = input_tensor.shape[0]
        num_patches = 1024 + 1  # +1 for CLS token
        hidden_dim = 1024  # ViT-Large hidden dimension
        
        try:
            output_tensor = torch.from_numpy(output_data).reshape(batch_size, num_patches, hidden_dim)
            return output_tensor.to(input_tensor.device)
        except:
            # 如果reshape失败，使用动态推断
            total_elements = len(output_data)
            # 假设batch_size=1，计算其他维度
            remaining = total_elements // batch_size
            output_tensor = torch.from_numpy(output_data).reshape(batch_size, -1)
            return output_tensor.to(input_tensor.device)

class QuantizedMASt3RWrapper:
    """量化MASt3R模型包装器"""
    
    def __init__(self, original_model, engine_path: str):
        self.original_model = original_model
        self.trt_engine = TensorRTInference(engine_path)
        
        # 保留原始模型的其他组件
        self._decoder = original_model._decoder
        self._downstream_head = original_model._downstream_head
        
        # 性能统计
        self.inference_times = []
        self.total_calls = 0
        
        print("✓ 量化MASt3R模型初始化完成")
        
    def _encode_image(self, img, true_shape):
        """使用TensorRT加速的图像编码"""
        start_time = time.time()
        
        try:
            # 使用TensorRT编码器
            feat = self.trt_engine.infer(img)
            
            # 生成位置编码
            pos = self._generate_position_encoding(feat, img.device)
            
            # 记录性能
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_calls += 1
            
            return feat, pos, None
            
        except Exception as e:
            print(f"TensorRT推理失败，回退到原始实现: {e}")
            return self.original_model._encode_image(img, true_shape)
    
    def _generate_position_encoding(self, feat, device):
        """生成位置编码（简化版本）"""
        batch_size, seq_len, _ = feat.shape
        
        # 创建2D位置编码
        # 假设是正方形patch grid
        sqrt_patches = int(np.sqrt(seq_len - 1))  # -1 for CLS token
        
        pos = torch.zeros(batch_size, seq_len, 2, device=device, dtype=torch.long)
        
        # CLS token position = (0, 0)
        pos[:, 0, :] = 0
        
        # Patch positions
        for i in range(sqrt_patches):
            for j in range(sqrt_patches):
                idx = i * sqrt_patches + j + 1  # +1 for CLS token
                if idx < seq_len:
                    pos[:, idx, 0] = j  # x coordinate
                    pos[:, idx, 1] = i  # y coordinate
        
        return pos
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            "total_calls": self.total_calls,
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_time": np.std(times),
            "total_time": np.sum(times)
        }
    
    def __getattr__(self, name):
        """代理其他属性到原始模型"""
        return getattr(self.original_model, name)

def load_quantized_mast3r(engine_path: str = "mast3r_encoder_int8.trt", device: str = "cuda"):
    """加载量化版本的MASt3R模型"""
    
    # 检查引擎文件是否存在
    if not Path(engine_path).exists():
        raise FileNotFoundError(f"TensorRT引擎文件不存在: {engine_path}")
    
    print(f"加载量化MASt3R模型...")
    print(f"TensorRT引擎: {engine_path}")
    
    # 加载原始模型
    sys.path.append('.')
    from mast3r_slam.mast3r_utils import load_mast3r
    original_model = load_mast3r(device=device)
    
    # 创建量化包装器
    quantized_model = QuantizedMASt3RWrapper(original_model, engine_path)
    
    return quantized_model

def benchmark_quantized_model(model_path: str = "mast3r_encoder_int8.trt"):
    """测试量化模型性能"""
    print("=== 量化模型性能测试 ===")
    
    # 加载原始和量化模型
    from mast3r_slam.mast3r_utils import load_mast3r
    original_model = load_mast3r().cuda().eval()
    quantized_model = load_quantized_mast3r(model_path)
    
    # 创建测试数据
    test_input = torch.randn(1, 3, 512, 512).cuda()
    test_shape = torch.tensor([[512, 512]])
    
    # 预热
    print("预热中...")
    for _ in range(10):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
            _ = quantized_model._encode_image(test_input, test_shape)
    
    # 测试原始模型
    print("测试原始模型...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = original_model._encode_image(test_input, test_shape)
    
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    
    # 测试量化模型
    print("测试量化模型...")
    quantized_model.inference_times.clear()  # 重置统计
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = quantized_model._encode_image(test_input, test_shape)
    
    torch.cuda.synchronize()
    quantized_time = time.time() - start_time
    
    # 打印结果
    print(f"\n=== 性能对比结果 ===")
    print(f"原始模型总时间: {original_time:.4f}s")
    print(f"量化模型总时间: {quantized_time:.4f}s")
    print(f"原始模型平均时间: {original_time/100:.4f}s")
    print(f"量化模型平均时间: {quantized_time/100:.4f}s")
    print(f"加速比: {original_time/quantized_time:.2f}x")
    print(f"性能提升: {(1 - quantized_time/original_time)*100:.1f}%")
    
    # 详细统计
    stats = quantized_model.get_performance_stats()
    if stats:
        print(f"\n=== 量化模型详细统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")

def create_integration_script():
    """创建main.py集成脚本"""
    
    integration_code = '''
# 在main.py中集成量化模型的修改

# 1. 在文件开头导入量化模块
from pathlib import Path

# 2. 在模型加载部分替换为以下代码:
def load_model_with_quantization(device="cuda", engine_path="mast3r_encoder_int8.trt"):
    """加载模型，支持量化版本"""
    try:
        # 检查量化引擎是否存在
        if Path(engine_path).exists():
            print(f"发现量化引擎: {engine_path}")
            from tensorrt_integration import load_quantized_mast3r
            model = load_quantized_mast3r(engine_path, device)
            print("✓ 量化模型加载成功")
            return model
        else:
            print(f"量化引擎不存在: {engine_path}，使用原始模型")
    except Exception as e:
        print(f"量化模型加载失败: {e}")
        print("回退到原始模型")
    
    # 回退到原始模型
    from mast3r_slam.mast3r_utils import load_mast3r
    model = load_mast3r(device=device)
    return model

# 3. 在main.py中替换模型加载行:
# 将: model = load_mast3r(device=device)
# 改为: model = load_model_with_quantization(device=device)

# 4. 或者直接修改现有代码:
try:
    from tensorrt_integration import load_quantized_mast3r
    engine_path = "mast3r_encoder_int8.trt"
    if Path(engine_path).exists():
        model = load_quantized_mast3r(engine_path, device=device)
        print("✓ 使用量化模型")
    else:
        model = load_mast3r(device=device)
        print("使用原始模型")
except Exception as e:
    print(f"量化模型加载失败: {e}")
    model = load_mast3r(device=device)
'''
    
    print("=== main.py集成指南 ===")
    print(integration_code)
    
    # 保存集成代码
    with open("tensorrt_integration_guide.py", "w") as f:
        f.write(integration_code)
    print("\n集成指南已保存到: tensorrt_integration_guide.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorRT量化模型集成工具")
    parser.add_argument("--engine-path", default="mast3r_encoder_int8.trt", 
                       help="TensorRT引擎文件路径")
    parser.add_argument("--benchmark", action="store_true", 
                       help="运行性能基准测试")
    parser.add_argument("--test-load", action="store_true", 
                       help="测试量化模型加载")
    parser.add_argument("--create-guide", action="store_true", 
                       help="创建集成指南")
    
    args = parser.parse_args()
    
    try:
        if args.create_guide:
            create_integration_script()
        
        if args.test_load:
            print("测试量化模型加载...")
            model = load_quantized_mast3r(args.engine_path)
            print("✓ 量化模型加载测试成功")
        
        if args.benchmark:
            benchmark_quantized_model(args.engine_path)
            
        if not any([args.benchmark, args.test_load, args.create_guide]):
            print("使用 --help 查看可用选项")
            print("推荐先运行: python tensorrt_integration.py --test-load --benchmark")
            
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
