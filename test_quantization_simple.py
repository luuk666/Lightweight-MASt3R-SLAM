#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的量化测试脚本
用于测试TensorRT INT8量化的基本功能
"""

import os
import sys
import numpy as np
import time
import argparse
from pathlib import Path

def test_basic_imports():
    """测试基本导入"""
    print("=== 测试基本导入 ===")
    
    try:
        import cv2
        print("✓ OpenCV 导入成功")
    except ImportError:
        print("❌ OpenCV 导入失败")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy 导入成功")
    except ImportError:
        print("❌ NumPy 导入失败")
        return False
    
    return True

def test_tensorrt_import():
    """测试TensorRT导入"""
    print("\n=== 测试TensorRT导入 ===")
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT 导入成功，版本: {trt.__version__}")
        return True
    except ImportError:
        print("❌ TensorRT 导入失败")
        print("请安装TensorRT: pip install tensorrt")
        return False

def test_pytorch_import():
    """测试PyTorch导入"""
    print("\n=== 测试PyTorch导入 ===")
    
    try:
        import torch
        print(f"✓ PyTorch 导入成功，版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
        return True
    except ImportError:
        print("❌ PyTorch 导入失败")
        print("请安装PyTorch: pip install torch torchvision")
        return False

def create_dummy_calibration_data(num_samples=10):
    """创建虚拟校准数据"""
    print(f"\n=== 创建虚拟校准数据 ({num_samples} 样本) ===")
    
    calibration_data = []
    for i in range(num_samples):
        # 创建随机图像数据 (1, 3, 512, 512)
        dummy_img = np.random.rand(1, 3, 512, 512).astype(np.float32)
        calibration_data.append(dummy_img)
        if i % 5 == 0:
            print(f"创建样本 {i+1}/{num_samples}")
    
    print(f"✓ 成功创建 {len(calibration_data)} 个虚拟校准样本")
    return calibration_data

def test_tensorrt_basic():
    """测试TensorRT基本功能"""
    print("\n=== 测试TensorRT基本功能 ===")
    
    try:
        import tensorrt as trt
        
        # 创建logger
        logger = trt.Logger(trt.Logger.WARNING)
        print("✓ TensorRT Logger 创建成功")
        
        # 创建builder
        builder = trt.Builder(logger)
        print("✓ TensorRT Builder 创建成功")
        
        # 创建network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        print("✓ TensorRT Network 创建成功")
        
        # 创建config
        config = builder.create_builder_config()
        print("✓ TensorRT Config 创建成功")
        
        # 设置最大工作空间
        config.max_workspace_size = 1 << 30  # 1GB
        print("✓ 工作空间设置成功")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorRT 基本功能测试失败: {e}")
        return False

def test_int8_calibration():
    """测试INT8校准功能"""
    print("\n=== 测试INT8校准功能 ===")
    
    try:
        import tensorrt as trt
        
        # 创建虚拟校准数据
        calibration_data = create_dummy_calibration_data(5)
        
        # 创建校准器
        class DummyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.data = data
                self.current = 0
                
            def get_batch_size(self):
                return 1
                
            def get_batch(self, names):
                if self.current >= len(self.data):
                    return None
                
                batch = self.data[self.current]
                self.current += 1
                return [batch.ctypes.data]
                
            def read_calibration_cache(self):
                return None
                
            def write_calibration_cache(self, cache):
                pass
        
        calibrator = DummyCalibrator(calibration_data)
        print("✓ INT8 校准器创建成功")
        
        # 测试校准器
        batch_size = calibrator.get_batch_size()
        print(f"✓ 校准器批次大小: {batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ INT8 校准功能测试失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化的量化测试脚本")
    parser.add_argument("--test-all", action="store_true", help="运行所有测试")
    parser.add_argument("--test-imports", action="store_true", help="只测试导入")
    parser.add_argument("--test-tensorrt", action="store_true", help="只测试TensorRT")
    
    args = parser.parse_args()
    
    print("🚀 开始量化测试...")
    print(f"Python 版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 运行测试
    if args.test_imports or args.test_all:
        if not test_basic_imports():
            print("❌ 基本导入测试失败")
            return
        
        if not test_pytorch_import():
            print("⚠️  PyTorch导入失败，但可以继续测试TensorRT功能")
        
        if not test_tensorrt_import():
            print("❌ TensorRT导入失败")
            return
    
    if args.test_tensorrt or args.test_all:
        if not test_tensorrt_basic():
            print("❌ TensorRT基本功能测试失败")
            return
        
        if not test_int8_calibration():
            print("❌ INT8校准功能测试失败")
            return
    
    print("\n🎉 所有测试完成!")
    print("\n下一步:")
    print("1. 如果PyTorch导入失败，请安装: pip install torch torchvision")
    print("2. 如果TensorRT导入失败，请安装: pip install tensorrt")
    print("3. 安装完成后，运行: python direct_quantization.py")

if __name__ == "__main__":
    main() 