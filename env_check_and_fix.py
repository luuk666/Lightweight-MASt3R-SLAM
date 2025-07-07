# env_check_and_fix.py
"""¯ƒÀåŒî,ã³torch/torchvision|¹'î˜"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_and_fix_environment():
    """Àåvî¯ƒî˜"""
    
    print("=== ¯ƒÀåŒî ===")
    
    # 1. ÀåtorchH,
    try:
        import torch
        print(f" PyTorchH,: {torch.__version__}")
        print(f" CUDAï(: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" CUDAH,: {torch.version.cuda}")
    except ImportError:
        print(" PyTorch*‰Å")
        return False
    
    # 2. Àåtorchvision|¹'
    try:
        # Õ¾n¯ƒØÏeÕÇtorchvisionî˜
        import os
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        
        import torchvision
        print(f" TorchVisionH,: {torchvision.__version__}")
    except Exception as e:
        print(f"  TorchVisionî˜: {e}")
        print("Õî...")
        
        try:
            # M§torchvision0|¹H,
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torchvision==0.15.2", "--force-reinstall"
            ], check=True)
            print(" TorchVisionòî")
        except:
            print(" ê¨î1%÷K¨gL:")
            print("pip install torchvision==0.15.2 --force-reinstall")
    
    # 3. ÀåTensorRT
    try:
        import tensorrt as trt
        print(f" TensorRTH,: {trt.__version__}")
    except ImportError:
        print("  TensorRT*‰Å")
        print("÷‰ÅTensorRT:")
        print("pip install tensorrt")
    
    # 4. ÀåPyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(" PyCUDAï(")
    except ImportError:
        print("  PyCUDA*‰Å")
        print("÷‰ÅPyCUDA:")
        print("pip install pycuda")
    
    # 5. ÀåyîÓ„
    required_paths = [
        "thirdparty/mast3r",
        "mast3r_slam",
        "checkpoints"
    ]
    
    for path in required_paths:
        if Path(path).exists():
            print(f" ~0îU: {path}")
        else:
            print(f"  :îU: {path}")
    
    return True

def setup_environment():
    """¾n¯ƒØÏŒï„"""
    import os
    
    # ¾n¯ƒØÏ
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    
    # û ï„
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
    
    print(" ¯ƒ¾nŒ")

if __name__ == "__main__":
    check_and_fix_environment()
    setup_environment()
    
    # KÕüe
    print("\n=== KÕüe ===")
    try:
        import torch
        torch.set_grad_enabled(False)
        print(" TorchüeŸ")
        
        # ÕüeMASt3Røs!W
        try:
            from mast3r_slam.mast3r_utils import load_mast3r
            print(" Îmast3r_slamüeŸ")
        except Exception as e:
            print(f"Îmast3r_slamüe1%: {e}")
            
            try:
                from mast3r.model import AsymmetricMASt3R
                print(" ô¥üeAsymmetricMASt3RŸ")
            except Exception as e2:
                print(f"ô¥üe_1%: {e2}")
        
    except Exception as e:
        print(f" üeKÕ1%: {e}")
    
    print("\n¯ƒÀåŒ")
