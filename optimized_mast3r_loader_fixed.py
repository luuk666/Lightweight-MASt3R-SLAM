# optimized_mast3r_loader_fixed.py
"""修复递归错误的优化MASt3R模型加载器"""

import torch
import torch.nn as nn
import time
import numpy as np

class OptimizedMASt3RWrapper:
    """优化的MASt3R包装器 - 修复版本"""
    
    def __init__(self, original_model, optimization_config):
        self.original_model = original_model
        self.config = optimization_config
        
        # 应用优化设置
        self._apply_optimizations()
        
        # 显式保留原始接口，避免递归
        self._decoder = original_model._decoder
        self._downstream_head = original_model._downstream_head
        
        # 显式保留其他重要方法
        self.forward = original_model.forward
        self.share_memory = original_model.share_memory
        
    def _apply_optimizations(self):
        """应用优化技术"""
        self.original_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        if self.config.get('use_half_precision', False):
            self.original_model = self.original_model.half()
    
    def _encode_image(self, img, true_shape):
        """优化的图像编码"""
        if self.config.get('use_amp', False):
            with torch.amp.autocast('cuda'):  # 修复deprecated警告
                return self.original_model._encode_image(img, true_shape)
        elif self.config.get('use_inference_mode', True):
            with torch.inference_mode():
                return self.original_model._encode_image(img, true_shape)
        else:
            return self.original_model._encode_image(img, true_shape)
    
    def __getattr__(self, name):
        """安全的属性代理，避免递归"""
        # 避免递归调用
        if name in ['original_model', 'config']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 获取原始模型的属性
        try:
            return getattr(self.original_model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def load_optimized_mast3r(device="cuda"):
    """加载优化的MASt3R模型"""
    
    # 应用系统优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 加载原始模型
    from mast3r_slam.mast3r_utils import load_mast3r
    original_model = load_mast3r(device=device)
    
    # 最佳配置
    best_config = {
        "use_inference_mode": True, 
        "use_half_precision": False, 
        "use_amp": True, 
        "try_compile_modules": False
    }
    
    # 创建优化包装器
    optimized_model = OptimizedMASt3RWrapper(original_model, best_config)
    
    print("✓ 优化MASt3R模型加载完成 (配置: amp_optimized)")
    return optimized_model
