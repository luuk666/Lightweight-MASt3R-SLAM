"""
优化的MASt3R模型加载器，支持TensorRT INT8量化模型
这个文件应该保存为 optimized_mast3r_loader.py
"""

import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import json
import time
from typing import Optional, Tuple, Dict, Any

# 原项目导入
import sys
sys.path.append('thirdparty/mast3r')
sys.path.append('thirdparty/mast3r/dust3r')

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.mast3r_utils import load_mast3r
except ImportError:
    print("Warning: Could not import MASt3R modules")


class TensorRTInference:
    """TensorRT推理引擎包装器"""
    
    def __init__(self, engine_path: str, max_batch_size: int = 1):
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        
        # TensorRT组件
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None
        
        # CUDA内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """加载TensorRT引擎"""
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        self.runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        print(f"✓ TensorRT engine loaded: {self.engine_path}")
    
    def _allocate_buffers(self):
        """分配GPU内存缓冲区"""
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        # 确保输入数据是连续的
        input_data = np.ascontiguousarray(input_data)
        
        # 拷贝输入数据到GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 拷贝输出数据到CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host'].copy()
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
        except:
            pass


class OptimizedMASt3R(nn.Module):
    """优化的MASt3R模型，结合原模型和TensorRT加速"""
    
    def __init__(self, original_model: AsymmetricMASt3R, 
                 tensorrt_engine_path: Optional[str] = None,
                 device: str = "cuda:0"):
        super().__init__()
        self.original_model = original_model
        self.device = device
        self.tensorrt_engine = None
        self.use_tensorrt = False
        
        # 尝试加载TensorRT引擎
        if tensorrt_engine_path and Path(tensorrt_engine_path).exists():
            try:
                self.tensorrt_engine = TensorRTInference(tensorrt_engine_path)
                self.use_tensorrt = True
                print("✓ TensorRT acceleration enabled")
            except Exception as e:
                print(f"Warning: Failed to load TensorRT engine: {e}")
                print("Falling back to PyTorch model")
        
        # 模型属性代理
        self._copy_attributes()
    
    def _copy_attributes(self):
        """复制原模型的属性"""
        important_attrs = [
            'patch_embed', 'pos_embed', 'enc_blocks', 'decoder',
            'head1', 'head2', 'downstream_head1', 'downstream_head2',
            'conf_mode', 'depth_mode', 'output_mode'
        ]
        
        for attr in important_attrs:
            if hasattr(self.original_model, attr):
                setattr(self, attr, getattr(self.original_model, attr))
    
    def _encode_image_tensorrt(self, img: torch.Tensor, true_shape: torch.Tensor) -> torch.Tensor:
        """使用TensorRT加速的图像编码"""
        # 转换为numpy
        img_np = img.detach().cpu().numpy()
        
        # TensorRT推理
        features_np = self.tensorrt_engine.infer(img_np)
        
        # 转换回torch tensor
        # 注意：这里需要根据实际的输出形状调整
        features = torch.from_numpy(features_np).to(self.device)
        
        # 重新整形为正确的特征维度
        # 这个需要根据你的具体模型输出调整
        batch_size = img.shape[0]
        # 假设输出是 [batch, seq_len, hidden_dim] 格式
        features = features.view(batch_size, -1, features.shape[-1])
        
        return features
    
    def _encode_image(self, img: torch.Tensor, true_shape: torch.Tensor):
        """图像编码（自动选择加速方式）"""
        if self.use_tensorrt and img.shape[0] == 1:  # TensorRT通常适合batch_size=1
            try:
                return [self._encode_image_tensorrt(img, true_shape)]
            except Exception as e:
                print(f"TensorRT inference failed: {e}, falling back to PyTorch")
                self.use_tensorrt = False
        
        # 使用原始PyTorch模型
        return self.original_model._encode_image(img, true_shape)
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.original_model.forward(*args, **kwargs)
    
    def _encode_image_pairs(self, *args, **kwargs):
        """编码图像对"""
        return self.original_model._encode_image_pairs(*args, **kwargs)
    
    def _decoder(self, *args, **kwargs):
        """解码器"""
        return self.original_model._decoder(*args, **kwargs)
    
    def _downstream_head(self, *args, **kwargs):
        """下游头部"""
        return self.original_model._downstream_head(*args, **kwargs)
    
    def share_memory(self):
        """共享内存"""
        return self.original_model.share_memory()
    
    def benchmark(self, input_shape: Tuple[int, ...] = (1, 3, 512, 512), 
                 num_iterations: int = 100) -> Dict[str, float]:
        """基准测试原模型vs优化模型"""
        print(f"Benchmarking with input shape: {input_shape}")
        
        # 创建测试数据
        test_img = torch.randn(input_shape).to(self.device)
        test_shape = torch.tensor([[input_shape[2], input_shape[3]]]).to(self.device)
        
        results = {}
        
        # 测试原模型
        print("Testing original PyTorch model...")
        torch.cuda.synchronize()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = self.original_model._encode_image(test_img, test_shape)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.original_model._encode_image(test_img, test_shape)
        
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / num_iterations
        results['pytorch_time_ms'] = pytorch_time * 1000
        
        # 测试TensorRT模型（如果可用）
        if self.use_tensorrt:
            print("Testing TensorRT optimized model...")
            torch.cuda.synchronize()
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = self._encode_image(test_img, test_shape)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self._encode_image(test_img, test_shape)
            
            torch.cuda.synchronize()
            tensorrt_time = (time.time() - start_time) / num_iterations
            results['tensorrt_time_ms'] = tensorrt_time * 1000
            results['speedup'] = pytorch_time / tensorrt_time
        else:
            results['tensorrt_time_ms'] = None
            results['speedup'] = None
        
        return results


def load_optimized_mast3r(device: str = "cuda:0", 
                         quantized_models_dir: str = "quantized_models",
                         checkpoint_dir: str = "checkpoints") -> OptimizedMASt3R:
    """加载优化的MASt3R模型"""
    print("Loading optimized MASt3R model...")
    
    # 1. 加载原始模型
    try:
        model = load_mast3r(device=device)
        print("✓ Original MASt3R model loaded")
    except Exception as e:
        print(f"Failed to load with load_mast3r: {e}")
        # 备用加载方式
        checkpoint_path = Path(checkpoint_dir) / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        model = AsymmetricMASt3R.from_pretrained(str(checkpoint_path)).to(device)
        print("✓ MASt3R model loaded from checkpoint")
    
    # 2. 查找TensorRT引擎
    tensorrt_engine_path = None
    quantized_dir = Path(quantized_models_dir)
    
    if quantized_dir.exists():
        engine_files = list(quantized_dir.glob("*.trt"))
        if engine_files:
            tensorrt_engine_path = str(engine_files[0])
            print(f"✓ Found TensorRT engine: {tensorrt_engine_path}")
        else:
            print("No TensorRT engine found in quantized_models directory")
    else:
        print(f"Quantized models directory not found: {quantized_models_dir}")
    
    # 3. 创建优化模型
    optimized_model = OptimizedMASt3R(
        original_model=model,
        tensorrt_engine_path=tensorrt_engine_path,
        device=device
    )
    
    return optimized_model


class QuantizationConfig:
    """量化配置类"""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        self.data_dir = "datasets/tum/rgbd_dataset_freiburg1_room"  # 默认校准数据
        self.output_dir = "quantized_models"
        self.num_calibration_samples = 50
        self.max_batch_size = 1
        self.workspace_size = 2 << 30  # 2GB
        self.device = "cuda:0"
        
        # 模型文件
        self.model_files = {
            'main': "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            'retrieval': "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            'codebook': "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
        }
    
    def validate(self) -> bool:
        """验证配置"""
        # 检查checkpoint目录
        checkpoint_path = Path(self.checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint directory not found: {self.checkpoint_dir}")
            return False
        
        # 检查模型文件
        for name, filename in self.model_files.items():
            file_path = checkpoint_path / filename
            if not file_path.exists():
                print(f"Error: Model file not found: {file_path}")
                return False
        
        # 检查数据目录
        if not Path(self.data_dir).exists():
            print(f"Warning: Data directory not found: {self.data_dir}")
            print("You may need to download calibration data or specify a different path")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'checkpoint_dir': self.checkpoint_dir,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'num_calibration_samples': self.num_calibration_samples,
            'max_batch_size': self.max_batch_size,
            'workspace_size': self.workspace_size,
            'device': self.device,
            'model_files': self.model_files
        }


def main():
    """主函数示例"""
    print("=== MASt3R TensorRT INT8 Quantization ===\n")
    
    # 创建配置
    config = QuantizationConfig()
    
    # 验证配置
    if not config.validate():
        print("Configuration validation failed. Please check your setup.")
        return
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # 导入量化器（这里需要从主量化脚本导入）
        from mast3r_tensorrt_quantizer import MASt3RQuantizer
        
        # 创建量化器
        quantizer = MASt3RQuantizer(
            checkpoint_dir=config.checkpoint_dir,
            device=config.device
        )
        
        # 执行量化
        print("Starting quantization process...")
        results = quantizer.quantize_model(
            data_dir=config.data_dir,
            output_dir=config.output_dir,
            num_calibration_samples=config.num_calibration_samples
        )
        
        print("\n=== Quantization Completed ===")
        print(f"✓ ONNX model: {results['onnx_path']}")
        print(f"✓ TensorRT engine: {results['engine_path']}")
        print(f"✓ Calibration samples: {results['calibration_samples']}")
        
        # 测试优化模型
        print("\n=== Testing Optimized Model ===")
        optimized_model = load_optimized_mast3r(
            device=config.device,
            quantized_models_dir=config.output_dir,
            checkpoint_dir=config.checkpoint_dir
        )
        
        # 运行基准测试
        benchmark_results = optimized_model.benchmark()
        
        print("\nBenchmark Results:")
        for key, value in benchmark_results.items():
            if value is not None:
                if 'time' in key:
                    print(f"  {key}: {value:.2f} ms")
                elif 'speedup' in key:
                    print(f"  {key}: {value:.2f}x")
        
        # 保存完整结果
        final_results = {
            'config': config.to_dict(),
            'quantization_results': results,
            'benchmark_results': benchmark_results
        }
        
        results_file = Path(config.output_dir) / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✓ Complete results saved to: {results_file}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install tensorrt pycuda onnx")
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()