# env_check_and_fix.py
"""环境检查和修复脚本，解决torch/torchvision兼容性问题"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_and_fix_environment():
    """检查并修复环境问题"""
    
    print("=== 环境检查和修复 ===")
    
    # 1. 检查torch版本
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    
    # 2. 检查torchvision兼容性
    try:
        # 尝试设置环境变量来绕过torchvision问题
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        
        import torchvision
        print(f"✓ TorchVision版本: {torchvision.__version__}")
    except Exception as e:
        print(f"⚠️ TorchVision问题: {e}")
        print("尝试修复...")
        
        try:
            # 降级torchvision到兼容版本
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torchvision==0.15.2", "--force-reinstall"
            ], check=True)
            print("✓ TorchVision已修复")
        except:
            print("✗ 自动修复失败，请手动执行:")
            print("pip install torchvision==0.15.2 --force-reinstall")
    
    # 3. 检查TensorRT
    try:
        import tensorrt as trt
        print(f"✓ TensorRT版本: {trt.__version__}")
    except ImportError:
        print("⚠️ TensorRT未安装")
        print("请安装TensorRT:")
        print("pip install tensorrt")
    
    # 4. 检查PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✓ PyCUDA可用")
    except ImportError:
        print("⚠️ PyCUDA未安装")
        print("请安装PyCUDA:")
        print("pip install pycuda")
    
    # 5. 检查项目结构
    required_paths = [
        "thirdparty/mast3r",
        "mast3r_slam",
        "checkpoints"
    ]
    
    for path in required_paths:
        if Path(path).exists():
            print(f"✓ 找到目录: {path}")
        else:
            print(f"⚠️ 缺少目录: {path}")
    
    return True

def setup_environment():
    """设置环境变量和路径"""
    import os
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    
    # 添加路径
    project_root = Path(__file__).parent
    paths_to_add = [
        str(project_root),
        str(project_root / 'thirdparty' / 'mast3r'),
        str(project_root / 'thirdparty' / 'mast3r' / 'dust3r'),
        str(project_root / 'mast3r_slam')
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print("✓ 环境设置完成")

if __name__ == "__main__":
    check_and_fix_environment()
    setup_environment()
    
    # 测试导入
    print("\n=== 测试导入 ===")
    try:
        import torch
        torch.set_grad_enabled(False)
        print("✓ Torch导入成功")
        
        # 尝试导入MASt3R相关模块
        try:
            from mast3r_slam.mast3r_utils import load_mast3r
            print("✓ 从mast3r_slam导入成功")
        except Exception as e:
            print(f"从mast3r_slam导入失败: {e}")
            
            try:
                from mast3r.model import AsymmetricMASt3R
                print("✓ 直接导入AsymmetricMASt3R成功")
            except Exception as e2:
                print(f"直接导入也失败: {e2}")
        
    except Exception as e:
        print(f"✗ 导入测试失败: {e}")
    
    print("\n环境检查完成")
