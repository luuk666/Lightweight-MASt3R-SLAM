# run_quantization.py
"""运行MASt3R-SLAM模型量化的主脚本"""

import argparse
import sys
import os
from pathlib import Path
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查依赖项"""
    missing_deps = []
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT版本: {trt.__version__}")
    except ImportError:
        missing_deps.append("tensorrt")
    
    try:
        import onnx
        print(f"✓ ONNX版本: {onnx.__version__}")
    except ImportError:
        missing_deps.append("onnx")
    
    if not torch.cuda.is_available():
        print("✗ CUDA不可用")
        return False
    else:
        print(f"✓ CUDA可用, 设备数量: {torch.cuda.device_count()}")
    
    if missing_deps:
        print(f"✗ 缺少依赖: {', '.join(missing_deps)}")
        print("请安装: pip install tensorrt onnx")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="MASt3R-SLAM模型量化工具")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="int8",
                       help="量化精度 (默认: int8)")
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk",
                       help="校准数据集路径")
    parser.add_argument("--calibration-samples", type=int, default=500,
                       help="校准样本数量")
    parser.add_argument("--output-dir", default="tensorrt_engines",
                       help="输出目录")
    parser.add_argument("--benchmark", action="store_true",
                       help="运行性能基准测试")
    parser.add_argument("--config", default=None,
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 导入量化模块
    try:
        from quantization_config import QuantizationConfig, MASt3RQuantizationManager
    except ImportError as e:
        print(f"导入量化模块失败: {e}")
        print("请确保quantization_config.py在正确位置")
        return
    
    # 加载或创建配置
    if args.config and Path(args.config).exists():
        config = QuantizationConfig.load_config(args.config)
        print(f"已加载配置文件: {args.config}")
    else:
        config = QuantizationConfig(
            precision=args.precision,
            calibration_dataset=args.dataset,
            calibration_samples=args.calibration_samples,
            output_dir=args.output_dir
        )
        print("使用默认配置")
    
    # 显示配置信息
    print(f"\n=== 量化配置 ===")
    print(f"精度: {config.precision}")
    print(f"校准数据集: {config.calibration_dataset}")
    print(f"校准样本数: {config.calibration_samples}")
    print(f"输出目录: {config.output_dir}")
    
    # 检查数据集是否存在
    if not Path(config.calibration_dataset).exists():
        print(f"✗ 数据集不存在: {config.calibration_dataset}")
        print("请下载数据集或指定正确路径")
        return
    
    # 执行量化
    try:
        manager = MASt3RQuantizationManager(config)
        quantized_model, engine_path = manager.quantize_model()
        
        if quantized_model and engine_path:
            print(f"\n🎉 量化成功完成!")
            print(f"引擎文件: {engine_path}")
            
            # 保存配置文件
            config_path = Path(config.output_dir) / "quantization_config.yaml"
            config.save_config(str(config_path))
            print(f"配置已保存: {config_path}")
            
            # 运行基准测试
            if args.benchmark:
                print("\n运行额外基准测试...")
                run_detailed_benchmark(quantized_model, config)
            
            # 生成使用说明
            generate_usage_instructions(engine_path)
        else:
            print("✗ 量化失败")
            
    except Exception as e:
        print(f"量化过程中出错: {e}")
        import traceback
        traceback.print_exc()

def run_detailed_benchmark(quantized_model, config):
    """运行详细的性能基准测试"""
    import time
    import numpy as np
    
    print("=== 详细性能测试 ===")
    
    # 不同批次大小测试
    batch_sizes = [1, 2, 4] if torch.cuda.get_device_properties(0).total_memory > 8e9 else [1, 2]
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        test_img = torch.randn(batch_size, 3, config.img_size, config.img_size).cuda()
        test_shape = torch.tensor([[config.img_size, config.img_size]] * batch_size)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = quantized_model._encode_image(test_img, test_shape)
        
        # 测试
        times = []
        torch.cuda.synchronize()
        
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = quantized_model._encode_image(test_img, test_shape)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        times = np.array(times)
        print(f"  平均时间: {np.mean(times):.4f}s")
        print(f"  标准差: {np.std(times):.4f}s")
        print(f"  吞吐量: {batch_size/np.mean(times):.2f} imgs/s")

def generate_usage_instructions(engine_path):
    """生成使用说明"""
    print(f"\n=== 使用说明 ===")
    print("1. 更新mast3r_slam/mast3r_utils.py，添加以下代码:")
    
    usage_code = f'''
# 在load_mast3r函数后添加:
def load_mast3r_quantized(engine_path="{engine_path}", device="cuda"):
    """加载量化版本的MASt3R模型"""
    from integrate_tensorrt import QuantizedMASt3RModel, QuantizationConfig
    
    # 加载原始模型
    original_model = load_mast3r(device=device)
    
    # 创建量化配置
    config = QuantizationConfig(device=device)
    
    # 创建量化模型
    quantized_model = QuantizedMASt3RModel(original_model, engine_path, config)
    
    return quantized_model

# 修改现有的load_mast3r函数:
def load_mast3r(path=None, device="cuda", use_quantized=False, engine_path=None):
    if use_quantized and engine_path and Path(engine_path).exists():
        return load_mast3r_quantized(engine_path, device)
    
    # 原有代码...
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model
'''
    
    print(usage_code)
    
    print("\n2. 在main.py中使用量化模型:")
    print(f'''
# 在main.py中修改模型加载部分:
model = load_mast3r(device=device, use_quantized=True, engine_path="{engine_path}")
''')
    
    print("\n3. 或者直接替换现有的load_mast3r调用:")
    print(f'''
from mast3r_slam.mast3r_utils import load_mast3r_quantized
model = load_mast3r_quantized("{engine_path}", device=device)
''')

# modify_main_for_quantization.py
"""修改main.py以支持量化模型的脚本"""

def modify_main_py(engine_path: str):
    """修改main.py文件以支持量化"""
    
    main_py_path = Path("main.py")
    if not main_py_path.exists():
        print("main.py文件不存在")
        return
    
    # 读取原文件
    with open(main_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = main_py_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原文件到: {backup_path}")
    
    # 修改内容
    # 1. 添加量化模型支持
    import_addition = '''from pathlib import Path
import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp

# 量化模型支持
try:
    from integrate_tensorrt import QuantizedMASt3RModel, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("量化模块不可用，使用原始模型")
'''
    
    # 2. 修改参数解析器
    parser_addition = '''
    parser.add_argument("--use-quantized", action="store_true", help="使用量化模型")
    parser.add_argument("--engine-path", default="tensorrt_engines/mast3r_vit_encoder_int8.trt", 
                       help="TensorRT引擎路径")'''
    
    # 3. 修改模型加载部分
    model_loading_modification = f'''
    # 加载模型 (支持量化)
    if args.use_quantized and QUANTIZATION_AVAILABLE:
        engine_path = Path(args.engine_path)
        if engine_path.exists():
            print(f"加载量化模型: {{engine_path}}")
            original_model = load_mast3r(device=device)
            quant_config = QuantizationConfig(device=device)
            model = QuantizedMASt3RModel(original_model, str(engine_path), quant_config)
        else:
            print(f"量化引擎不存在: {{engine_path}}, 使用原始模型")
            model = load_mast3r(device=device)
    else:
        model = load_mast3r(device=device)
    '''
    
    # 应用修改
    # 替换导入部分
    content = content.replace(
        "import torch.multiprocessing as mp",
        import_addition
    )
    
    # 添加参数
    content = content.replace(
        'parser.add_argument("--calib", default="")',
        'parser.add_argument("--calib", default="")\n' + parser_addition
    )
    
    # 替换模型加载
    content = content.replace(
        "model = load_mast3r(device=device)",
        model_loading_modification
    )
    
    # 写入修改后的文件
    with open(main_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修改main.py文件，备份保存在: {backup_path}")
    print("现在可以使用 --use-quantized 参数运行量化版本")

if __name__ == "__main__":
    main()
