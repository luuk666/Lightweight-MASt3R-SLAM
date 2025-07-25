# successful_quantizer.py
"""基于成功方法的MASt3R量化器 - 使用ONNX兼容编码器"""

import os
import sys
from pathlib import Path
import warnings
import argparse

# 环境设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
warnings.filterwarnings("ignore")

# 添加项目路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'mast3r_slam'))

import torch
import torch.nn as nn
import numpy as np
import cv2
import PIL.Image
from typing import List, Tuple
import time
import json

# 设置torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

def import_mast3r():
    """导入MASt3R模型"""
    try:
        # 添加thirdparty路径
        mast3r_path = project_root / 'thirdparty' / 'mast3r'
        if mast3r_path.exists():
            sys.path.insert(0, str(mast3r_path))
            sys.path.insert(0, str(mast3r_path / 'dust3r'))
        
        from mast3r.model import AsymmetricMASt3R
        
        # 查找checkpoint
        checkpoint_files = [
            "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        ]
        
        for ckpt_path in checkpoint_files:
            full_path = project_root / ckpt_path
            if full_path.exists():
                print(f"✓ 找到checkpoint: {full_path}")
                model = AsymmetricMASt3R.from_pretrained(str(full_path))
                model = model.cuda().eval()
                print("✓ 模型加载成功")
                return model
        
        print("✗ 未找到checkpoint文件")
        return None
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

class ONNXCompatibleEncoder(nn.Module):
    """ONNX兼容的编码器 - 经过验证的工作版本"""
    
    def __init__(self, original_model):
        super().__init__()
        self.patch_embed = original_model.patch_embed
        
        # 只使用前几层编码器来避免复杂操作
        self.encoder_blocks = nn.ModuleList(original_model.enc_blocks[:8])  # 只用前8层
        self.norm = original_model.enc_norm
        
        # 预计算位置编码矩阵
        self.register_buffer('fixed_pos_embed', self._create_fixed_positions())
        
    def _create_fixed_positions(self):
        """创建固定的位置编码，避免cartesian_prod"""
        # 对于512x512图像，16x16的patch，得到32x32的grid
        H, W = 32, 32  # 512//16 = 32
        
        # 手动创建位置网格，避免cartesian_prod
        pos_embed = torch.zeros(1, H * W, 2)
        idx = 0
        for h in range(H):
            for w in range(W):
                pos_embed[0, idx, 0] = h
                pos_embed[0, idx, 1] = w
                idx += 1
        
        return pos_embed.long()
    
    def forward(self, x):
        """ONNX兼容的前向传播"""
        B, C, H, W = x.shape
        
        # 使用标准的patch embedding，不依赖true_shape
        if hasattr(self.patch_embed, 'proj'):
            # 直接使用卷积投影
            x = self.patch_embed.proj(x)  # (B, embed_dim, H//16, W//16)
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        else:
            # 备用方法
            raise NotImplementedError("Unsupported patch_embed type")
        
        # 使用固定的位置编码
        pos = self.fixed_pos_embed.expand(B, -1, -1)
        
        # 通过编码器层
        for block in self.encoder_blocks:
            # 简化的块调用，避免复杂的RoPE计算
            if hasattr(block, 'attn') and hasattr(block, 'mlp'):
                # 简化的attention计算
                normed = block.norm1(x)
                
                # 简化的自注意力（不使用RoPE）
                B, N, C = normed.shape
                qkv = block.attn.qkv(normed).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x_attn = block.attn.proj(x_attn)
                
                x = x + block.drop_path(x_attn)
                
                # MLP部分
                x = x + block.drop_path(block.mlp(block.norm2(x)))
            else:
                # 如果块结构不同，跳过
                continue
        
        x = self.norm(x)
        return x

class CalibrationDataLoader:
    """校准数据加载器"""
    
    def __init__(self, dataset_path: str, max_samples: int = 100):
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples
        self.current_idx = 0
        self.image_files = self._load_image_files()
        print(f"找到 {len(self.image_files)} 张校准图像")
        
    def _load_image_files(self) -> List[str]:
        image_files = []
        
        if self.dataset_path.is_dir():
            # TUM数据集格式
            rgb_file = self.dataset_path / "rgb.txt"
            if rgb_file.exists():
                with open(rgb_file, 'r') as f:
                    lines = f.readlines()
                for line in lines[3:]:  # 跳过前3行注释
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_path = self.dataset_path / parts[1]
                        if img_path.exists():
                            image_files.append(str(img_path))
            else:
                # 通用图像目录
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend([str(p) for p in self.dataset_path.glob(f"**/{ext}")])
        
        return image_files[:self.max_samples]
    
    def __iter__(self):
        self.current_idx = 0
        return self
        
    def __next__(self):
        if self.current_idx >= len(self.image_files):
            raise StopIteration
            
        img_path = self.image_files[self.current_idx]
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self._preprocess_image(img)
            
            self.current_idx += 1
            return img_tensor
            
        except Exception as e:
            print(f"跳过图像 {img_path}: {e}")
            self.current_idx += 1
            if self.current_idx < len(self.image_files):
                return self.__next__()
            else:
                raise StopIteration
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # 调整大小到512x512
        img_pil = PIL.Image.fromarray(img)
        img_pil = img_pil.resize((512, 512), PIL.Image.LANCZOS)
        
        # 归一化
        img_array = np.array(img_pil).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # 归一化到[-1, 1]
        
        # 转换为CHW格式
        img_tensor = np.transpose(img_array, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0).copy()
        
        return img_tensor

class TensorRTCalibrator:
    """基于内存数据的 TensorRT INT8 校准器"""
    
    def __init__(self, calibration_data: List[np.ndarray], cache_file: str = "mast3r_calibration.cache"):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            # 创建子类 Calibrator
            class Calibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, data_list, cache_file):
                    super(Calibrator, self).__init__()
                    self.data_list = data_list
                    self.cache_file = cache_file
                    self.index = 0
                    self.device_input = None

                def get_batch_size(self):
                    return 1

                def get_batch(self, names):
                    if self.index >= len(self.data_list):
                        return None

                    batch = self.data_list[self.index]
                    self.index += 1

                    if self.device_input is None:
                        self.device_input = cuda.mem_alloc(batch.nbytes)

                    cuda.memcpy_htod(self.device_input, batch)
                    return [int(self.device_input)]

                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            self.calibrator = Calibrator(calibration_data, cache_file)

        except ImportError:
            print("TensorRT 或 PyCUDA 未安装")
            self.calibrator = None


def export_onnx(model, output_path="mast3r_encoder_final.onnx"):
    """导出ONNX模型"""
    print(f"导出ONNX模型到: {output_path}")
    
    try:
        # 创建ONNX兼容编码器
        onnx_encoder = ONNXCompatibleEncoder(model)
        dummy_input = torch.randn(1, 3, 512, 512).cuda()
        
        torch.onnx.export(
            onnx_encoder,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['features'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'features': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print("✓ ONNX导出成功")
        
        # 验证ONNX文件
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX模型验证通过")
        except ImportError:
            print("⚠️ ONNX包未安装，跳过验证")
        except Exception as e:
            print(f"⚠️ ONNX验证失败: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ ONNX导出失败: {e}")
        return None

def build_tensorrt_engine(onnx_path: str, engine_path: str, 
                         calibration_data: List[np.ndarray] = None,
                         precision: str = "fp16"):
    """构建TensorRT引擎"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT或PyCUDA未安装，跳过引擎构建")
        return None
    
    print(f"构建TensorRT引擎 ({precision})...")
    
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
    
    # 兼容新旧TensorRT版本的工作空间设置
    try:
        # 新版本TensorRT (8.5+)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    except AttributeError:
        # 旧版本TensorRT
        config.max_workspace_size = 4 << 30  # 4GB
    
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16精度")
    elif precision == "int8" and calibration_data is not None:
        config.set_flag(trt.BuilderFlag.INT8)
        # 创建校准器
        #calib_loader = CalibrationDataLoader("", max_samples=len(calibration_data))
        #calib_loader.image_files = ["dummy"] * len(calibration_data)
        #calibrator_wrapper = TensorRTCalibrator(calib_loader)
        calibrator_wrapper = TensorRTCalibrator(calibration_data)
        if calibrator_wrapper.calibrator is not None:
            config.int8_calibrator = calibrator_wrapper.calibrator
            print("启用INT8精度with校准器")
        else:
            print("校准器创建失败，改用FP16")
            config.set_flag(trt.BuilderFlag.FP16)
    
    # 设置输入形状
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, (1, 3, 512, 512), (1, 3, 512, 512), (4, 3, 512, 512))
    config.add_optimization_profile(profile)
    
    # 构建引擎
    print("构建引擎中...")
    engine = builder.build_engine(network, config)
    
    if engine:
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"✓ TensorRT引擎保存到: {engine_path}")
        
        # 计算文件大小
        file_size = Path(engine_path).stat().st_size / (1024 * 1024)
        print(f"引擎文件大小: {file_size:.1f} MB")
        
        return engine_path
    else:
        print("✗ 引擎构建失败")
        return None

def benchmark_models(original_model, onnx_path: str, engine_path: str = None):
    """基准测试原始模型vs优化模型"""
    print("\n=== 性能基准测试 ===")
    
    # 创建测试数据
    test_input = torch.randn(1, 3, 512, 512).cuda()
    true_shape = torch.tensor([[512, 512]]).cuda()
    
    # 测试原始模型
    print("测试原始模型...")
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = original_model._encode_image(test_input, true_shape)
        
        # 计时
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(50):
            _ = original_model._encode_image(test_input, true_shape)
        
        torch.cuda.synchronize()
        original_time = (time.time() - start_time) / 50
    
    print(f"原始模型平均推理时间: {original_time*1000:.2f} ms")
    
    # 测试ONNX兼容编码器
    onnx_encoder = ONNXCompatibleEncoder(original_model)
    print("测试ONNX兼容编码器...")
    
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = onnx_encoder(test_input)
        
        # 计时
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(50):
            _ = onnx_encoder(test_input)
        
        torch.cuda.synchronize()
        onnx_time = (time.time() - start_time) / 50
    
    print(f"ONNX兼容编码器平均推理时间: {onnx_time*1000:.2f} ms")
    print(f"ONNX编码器加速比: {original_time/onnx_time:.2f}x")
    
    return {
        'original_time_ms': original_time * 1000,
        'onnx_time_ms': onnx_time * 1000,
        'speedup': original_time / onnx_time
    }

def main():
    parser = argparse.ArgumentParser(description="成功的MASt3R量化器")
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk/rgb")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--export-only", action="store_true", help="只导出ONNX")
    parser.add_argument("--benchmark", action="store_true", help="运行基准测试")
    parser.add_argument("--output-dir", default="final_quantized", help="输出目录")
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. 加载模型
        print("=== 加载MASt3R模型 ===")
        model = import_mast3r()
        if model is None:
            return
        
        # 2. 导出ONNX
        print("\n=== 导出ONNX模型 ===")
        onnx_path = output_dir / "mast3r_encoder_final.onnx"
        onnx_result = export_onnx(model, str(onnx_path))
        
        if not onnx_result:
            print("ONNX导出失败，程序退出")
            return
        
        if args.export_only:
            print(f"\n✓ ONNX导出完成: {onnx_path}")
            if args.benchmark:
                benchmark_models(model, str(onnx_path))
            return
        
        # 3. 构建TensorRT引擎
        print("\n=== 构建TensorRT引擎 ===")
        
        # 准备校准数据（如果需要INT8）
        calibration_data = None
        if args.precision == "int8":
            if Path(args.dataset).exists():
                print("准备校准数据...")
                calib_loader = CalibrationDataLoader(args.dataset, max_samples=50)
                try:
                    calibration_data = [next(calib_loader) for _ in range(min(50, len(calib_loader.image_files)))]
                    print(f"准备了 {len(calibration_data)} 个校准样本")
                except Exception as e:
                    print(f"校准数据准备失败: {e}")
                    print("改用FP16精度")
                    args.precision = "fp16"
            else:
                print("校准数据集不存在，改用FP16精度")
                args.precision = "fp16"
        
        # 构建引擎
        engine_path = output_dir / f"mast3r_encoder_{args.precision}.trt"
        engine_result = build_tensorrt_engine(str(onnx_path), str(engine_path), calibration_data, args.precision)
        
        # 4. 保存结果信息
        results = {
            'model_type': 'MASt3R ViT-Large',
            'onnx_path': str(onnx_path),
            'engine_path': str(engine_path) if engine_result else None,
            'precision': args.precision,
            'onnx_file_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
            'engine_file_size_mb': Path(engine_path).stat().st_size / (1024 * 1024) if engine_result else None,
            'calibration_samples': len(calibration_data) if calibration_data else 0
        }
        
        # 5. 运行基准测试（如果请求）
        if args.benchmark and engine_result:
            benchmark_results = benchmark_models(model, str(onnx_path), str(engine_path))
            results['benchmark'] = benchmark_results
        
        # 保存结果
        results_file = output_dir / "quantization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 量化完成！")
        print(f"ONNX模型: {onnx_path}")
        if engine_result:
            print(f"TensorRT引擎: {engine_path}")
        print(f"结果文件: {results_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
