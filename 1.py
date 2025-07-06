import tensorrt as trt
import torch
print(f"TensorRT版本: {trt.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
