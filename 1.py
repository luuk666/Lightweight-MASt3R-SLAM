import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import tensorrt as trt
print(f"TensorRT: {trt.__version__}")
