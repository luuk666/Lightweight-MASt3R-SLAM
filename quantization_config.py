# quantization_config.py
"""MASt3R-SLAM TensorRT量化配置"""

import yaml
import sys
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 初始化基本配置
def init_basic_config():
    """初始化基本配置以避免KeyError"""
    from mast3r_slam.config import set_global_config
    basic_config = {
        "use_calib": False,
        "single_thread": False,
        "dataset": {
            "subsample": 1,
            "img_downsample": 1,
            "center_principle_point": True
        }
    }
    set_global_config(basic_config)

@dataclass
class QuantizationConfig:
    """量化配置类"""
    
    # 基本设置
    model_path: str = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    device: str = "cuda:0"
    precision: str = "int8"  # fp16, int8
    
    # 校准设置
    calibration_dataset: str = "datasets/tum/rgbd_dataset_freiburg1_desk"
    calibration_samples: int = 500
    calibration_cache: str = "calibration.cache"
    
    # 模型设置
    img_size: int = 512
    batch_size: int = 1
    
    # TensorRT设置
    max_workspace_size: int = 2 << 30  # 2GB
    onnx_opset_version: int = 11
    
    # 输出设置
    output_dir: str = "tensorrt_engines"
    engine_name: str = "mast3r_vit_encoder"
    
    # 性能测试
    benchmark_iterations: int = 100
    warmup_iterations: int = 10
    
    def save_config(self, path: str):
        """保存配置到文件"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, path: str):
        """从文件加载配置"""
        with open(path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config_dict)

class FixedCalibrationDataLoader:
    """修复的校准数据加载器"""
    
    def __init__(self, dataset_path, img_size=512, max_samples=500):
        # 确保配置已初始化
        init_basic_config()
        
        from mast3r_slam.dataloader import load_dataset
        self.dataset = load_dataset(dataset_path)
        self.img_size = img_size
        self.max_samples = min(max_samples, len(self.dataset))
        self.current_idx = 0
        
    def __iter__(self):
        self.current_idx = 0
        return self
        
    def __next__(self):
        if self.current_idx >= self.max_samples:
            raise StopIteration
            
        # 获取校准图像
        timestamp, img = self.dataset[self.current_idx]
        
        # 预处理图像
        from mast3r_slam.mast3r_utils import resize_img
        img_dict = resize_img(img, self.img_size)
        img_tensor = img_dict["img"].numpy()  # (1, 3, H, W)
        
        self.current_idx += 1
        return {"input": img_tensor}

# integrate_tensorrt.py
"""将TensorRT量化集成到MASt3R-SLAM中"""

import torch
import torch.nn as nn
from pathlib import Path
import time
import numpy as np
from typing import Dict, Any

class QuantizedMASt3RModel(nn.Module):
    """量化版本的MASt3R模型"""
    
    def __init__(self, original_model, engine_path: str, config: QuantizationConfig):
        super().__init__()
        self.original_model = original_model
        self.config = config
        
        # 加载TensorRT引擎
        self.trt_encoder = self._load_tensorrt_engine(engine_path)
        
        # 保留原始模型的其他组件
        self._decoder = original_model._decoder
        self._downstream_head = original_model._downstream_head
        
        # 性能统计
        self.encoding_times = []
        self.total_inferences = 0
        
    def _load_tensorrt_engine(self, engine_path: str):
        """加载TensorRT引擎"""
        try:
            from tensorrt_inference import TensorRTViTInference
            return TensorRTViTInference(engine_path)
        except ImportError:
            print("TensorRT推理模块未找到，回退到原始实现")
            return None
        except Exception as e:
            print(f"加载TensorRT引擎失败: {e}")
            return None
    
    def _encode_image(self, img, true_shape):
        """使用TensorRT加速的图像编码"""
        start_time = time.time()
        
        if self.trt_encoder is not None:
            try:
                # 使用TensorRT推理
                img_np = img.cpu().numpy()
                encoded = self.trt_encoder.infer(img_np)
                
                # 转换回PyTorch tensor
                feat = torch.from_numpy(encoded).to(img.device)
                
                # 生成位置编码（需要根据实际模型调整）
                h, w = true_shape[0].item(), true_shape[1].item()
                num_patches = feat.shape[1] - 1  # 减去CLS token
                pos = self._generate_position_encoding(num_patches, img.device)
                
                encoding_time = time.time() - start_time
                self.encoding_times.append(encoding_time)
                self.total_inferences += 1
                
                return feat, pos, None
                
            except Exception as e:
                print(f"TensorRT推理失败，回退到原始实现: {e}")
        
        # 回退到原始实现
        return self.original_model._encode_image(img, true_shape)
    
    def _generate_position_encoding(self, num_patches: int, device: torch.device):
        """生成位置编码"""
        # 简化的位置编码生成，需要根据实际模型调整
        sqrt_num = int(np.sqrt(num_patches))
        pos_embed = torch.zeros(1, num_patches + 1, 2, device=device, dtype=torch.long)
        
        for i in range(sqrt_num):
            for j in range(sqrt_num):
                idx = i * sqrt_num + j + 1  # +1 for CLS token
                if idx <= num_patches:
                    pos_embed[0, idx, 0] = j
                    pos_embed[0, idx, 1] = i
        
        return pos_embed
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if not self.encoding_times:
            return {}
        
        times = np.array(self.encoding_times)
        return {
            "total_inferences": self.total_inferences,
            "avg_encoding_time": np.mean(times),
            "min_encoding_time": np.min(times),
            "max_encoding_time": np.max(times),
            "std_encoding_time": np.std(times),
            "total_encoding_time": np.sum(times)
        }
    
    def reset_stats(self):
        """重置性能统计"""
        self.encoding_times.clear()
        self.total_inferences = 0
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.original_model.forward(*args, **kwargs)

class MASt3RQuantizationManager:
    """MASt3R量化管理器"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def quantize_model(self):
        """执行模型量化"""
        print("=== MASt3R模型量化开始 ===")
        
        # 确保配置已初始化
        init_basic_config()
        
        # 1. 加载原始模型
        print("加载原始模型...")
        from mast3r_slam.mast3r_utils import load_mast3r
        original_model = load_mast3r(self.config.model_path, device=self.config.device)
        
        # 2. 准备校准数据
        print("准备校准数据...")
        calib_loader = FixedCalibrationDataLoader(
            self.config.calibration_dataset, 
            img_size=self.config.img_size,
            max_samples=self.config.calibration_samples
        )
        
        # 3. 执行量化
        print(f"开始{self.config.precision.upper()}量化...")
        from tensorrt_quantization import MASt3RTensorRTQuantizer
        
        quantizer = MASt3RTensorRTQuantizer(original_model, device=self.config.device)
        engine_path = quantizer.quantize_model(calib_loader, precision=self.config.precision)
        
        if engine_path:
            # 移动引擎文件到输出目录
            engine_name = f"{self.config.engine_name}_{self.config.precision}.trt"
            final_engine_path = self.output_dir / engine_name
            Path(engine_path).rename(final_engine_path)
            
            print(f"量化完成! 引擎保存在: {final_engine_path}")
            
            # 4. 创建量化模型
            quantized_model = QuantizedMASt3RModel(
                original_model, 
                str(final_engine_path), 
                self.config
            )
            
            # 5. 性能测试
            self.benchmark_models(original_model, quantized_model)
            
            return quantized_model, str(final_engine_path)
        else:
            print("量化失败!")
            return None, None
    
    def benchmark_models(self, original_model, quantized_model):
        """性能基准测试"""
        print("\n=== 性能基准测试 ===")
        
        # 创建测试数据
        test_img = torch.randn(
            self.config.batch_size, 3, 
            self.config.img_size, self.config.img_size
        ).to(self.config.device)
        test_shape = torch.tensor([[self.config.img_size, self.config.img_size]])
        
        # 预热
        print("预热中...")
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = original_model._encode_image(test_img, test_shape)
                if hasattr(quantized_model, 'trt_encoder') and quantized_model.trt_encoder:
                    _ = quantized_model._encode_image(test_img, test_shape)
        
        # 测试原始模型
        print("测试原始模型...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.config.benchmark_iterations):
            with torch.no_grad():
                _ = original_model._encode_image(test_img, test_shape)
        
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # 测试量化模型
        print("测试量化模型...")
        quantized_model.reset_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(self.config.benchmark_iterations):
            with torch.no_grad():
                _ = quantized_model._encode_image(test_img, test_shape)
        
        torch.cuda.synchronize()
        quantized_time = time.time() - start_time
        
        # 打印结果
        print(f"\n=== 性能对比结果 ===")
        print(f"原始模型总时间: {original_time:.4f}s")
        print(f"量化模型总时间: {quantized_time:.4f}s")
        print(f"原始模型平均时间: {original_time/self.config.benchmark_iterations:.4f}s")
        print(f"量化模型平均时间: {quantized_time/self.config.benchmark_iterations:.4f}s")
        print(f"加速比: {original_time/quantized_time:.2f}x")
        print(f"性能提升: {(1 - quantized_time/original_time)*100:.1f}%")
        
        # 获取详细统计
        stats = quantized_model.get_performance_stats()
        if stats:
            print(f"\n=== 量化模型详细统计 ===")
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")

def update_mast3r_utils(engine_path: str):
    """更新mast3r_utils.py以使用量化模型"""
    
    update_code = f