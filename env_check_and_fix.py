# env_check_and_fix.py
"""�����,�torch/torchvision|�'�"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_and_fix_environment():
    """��v����"""
    
    print("=== ����� ===")
    
    # 1. ��torchH,
    try:
        import torch
        print(f" PyTorchH,: {torch.__version__}")
        print(f" CUDA�(: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" CUDAH,: {torch.version.cuda}")
    except ImportError:
        print(" PyTorch*��")
        return False
    
    # 2. ��torchvision|�'
    try:
        # վn����e��torchvision�
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        
        import torchvision
        print(f" TorchVisionH,: {torchvision.__version__}")
    except Exception as e:
        print(f"� TorchVision�: {e}")
        print("��...")
        
        try:
            # M�torchvision0|�H,
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torchvision==0.15.2", "--force-reinstall"
            ], check=True)
            print(" TorchVision��")
        except:
            print(" ��1%�K�gL:")
            print("pip install torchvision==0.15.2 --force-reinstall")
    
    # 3. ��TensorRT
    try:
        import tensorrt as trt
        print(f" TensorRTH,: {trt.__version__}")
    except ImportError:
        print("� TensorRT*��")
        print("���TensorRT:")
        print("pip install tensorrt")
    
    # 4. ��PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(" PyCUDA�(")
    except ImportError:
        print("� PyCUDA*��")
        print("���PyCUDA:")
        print("pip install pycuda")
    
    # 5. ��y�ӄ
    required_paths = [
        "thirdparty/mast3r",
        "mast3r_slam",
        "checkpoints"
    ]
    
    for path in required_paths:
        if Path(path).exists():
            print(f" ~0�U: {path}")
        else:
            print(f"� :�U: {path}")
    
    return True

def setup_environment():
    """�n���ό�"""
    import os
    
    # �n����
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    
    # ���
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
    
    print(" ���n�")

if __name__ == "__main__":
    check_and_fix_environment()
    setup_environment()
    
    # K��e
    print("\n=== K��e ===")
    try:
        import torch
        torch.set_grad_enabled(False)
        print(" Torch�e�")
        
        # ��eMASt3R�s!W
        try:
            from mast3r_slam.mast3r_utils import load_mast3r
            print(" �mast3r_slam�e�")
        except Exception as e:
            print(f"�mast3r_slam�e1%: {e}")
            
            try:
                from mast3r.model import AsymmetricMASt3R
                print(" ���eAsymmetricMASt3R�")
            except Exception as e2:
                print(f"���e_1%: {e2}")
        
    except Exception as e:
        print(f" �eK�1%: {e}")
    
    print("\n����")
