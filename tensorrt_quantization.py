import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

class CalibrationDataLoader:
    """校准数据加载器，用于INT8量化"""
    
    def __init__(self, dataset, img_size=512, max_samples=500):
        self.dataset = dataset
        self.img_size = img_size
        self.max_samples = min(max_samples, len(dataset))
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

class TensorRTCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT INT8校准器"""
    
    def __init__(self, dataloader, cache_file="calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.data_iter = iter(dataloader)
        self.batch_allocation = None
        self.batch_generator = None
        
    def get_batch_size(self):
        return 1
        
    def get_batch(self, names):
        try:
            batch = next(self.data_iter)
            if self.batch_allocation is None:
                # 分配GPU内存
                self.batch_allocation = {}
                for name, data in batch.items():
                    size = trt.volume(data.shape) * trt.float32.itemsize
                    self.batch_allocation[name] = trt.cuda.mem_alloc(size)
            
            # 将数据复制到GPU
            bindings = []
            for name, data in batch.items():
                trt.cuda.memcpy_htod(self.batch_allocation[name], 
                                   data.astype(np.float32).ravel())
                bindings.append(int(self.batch_allocation[name]))
            
            return bindings
        except StopIteration:
            return None
            
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class MASt3RTensorRTQuantizer:
    """MASt3R模型TensorRT量化器"""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def extract_vit_components(self):
        """提取ViT编码器和解码器组件"""
        # 获取编码器
        encoder = self.model.patch_embed, self.model.encoder
        
        # 获取解码器
        decoder = self.model._decoder
        
        return encoder, decoder
        
    def create_encoder_onnx(self, input_shape=(1, 3, 512, 512), onnx_path="vit_encoder.onnx"):
        """将ViT编码器导出为ONNX"""
        
        class ViTEncoderWrapper(nn.Module):
            def __init__(self, patch_embed, encoder):
                super().__init__()
                self.patch_embed = patch_embed
                self.encoder = encoder
                
            def forward(self, x):
                # Patch embedding
                x = self.patch_embed(x)
                
                # Add position embedding
                if hasattr(self.patch_embed, 'pos_embed'):
                    x = x + self.patch_embed.pos_embed
                
                # Encoder forward
                x = self.encoder(x)
                return x
        
        # 创建编码器包装器
        patch_embed = self.model.patch_embed
        encoder = self.model.encoder
        encoder_wrapper = ViTEncoderWrapper(patch_embed, encoder).eval()
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 导出ONNX
        torch.onnx.export(
            encoder_wrapper,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ViT编码器已导出至: {onnx_path}")
        return onnx_path
        
    def build_tensorrt_engine(self, onnx_path, engine_path, calibrator=None, precision="fp16"):
        """构建TensorRT引擎"""
        
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # 解析ONNX模型
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # 配置构建器
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30  # 2GB
        
        # 设置精度
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            print("启用FP16精度")
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator:
                config.int8_calibrator = calibrator
                print("启用INT8精度with校准器")
            else:
                print("警告: INT8精度需要校准器")
                return None
        
        # 设置输入形状
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, (1, 3, 512, 512), (1, 3, 512, 512), (4, 3, 512, 512))
        config.add_optimization_profile(profile)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        
        if engine:
            # 保存引擎
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT引擎已保存至: {engine_path}")
            
        return engine
        
    def quantize_model(self, dataloader, precision="int8"):
        """量化整个模型"""
        
        # 1. 导出编码器ONNX
        encoder_onnx = "vit_encoder.onnx"
        self.create_encoder_onnx(onnx_path=encoder_onnx)
        
        # 2. 创建校准器（仅INT8需要）
        calibrator = None
        if precision == "int8":
            calibrator = TensorRTCalibrator(dataloader)
        
        # 3. 构建TensorRT引擎
        engine_path = f"vit_encoder_{precision}.trt"
        engine = self.build_tensorrt_engine(
            encoder_onnx, 
            engine_path, 
            calibrator, 
            precision
        )
        
        return engine_path if engine else None

class TensorRTViTInference:
    """TensorRT ViT推理引擎"""
    
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.allocate_buffers()
        
    def allocate_buffers(self):
        """分配输入输出缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = trt.cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配host和device内存
            host_mem = trt.cuda.pagelocked_empty(size, dtype)
            device_mem = trt.cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, input_data):
        """执行推理"""
        # 复制输入数据到host内存
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 传输数据到device
        trt.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 传输结果回host
        trt.cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # 同步
        self.stream.synchronize()
        
        return self.outputs[0]['host']

def integrate_tensorrt_mast3r(model, engine_path):
    """将TensorRT引擎集成到MASt3R模型中"""
    
    class TensorRTMASt3R(nn.Module):
        def __init__(self, original_model, engine_path):
            super().__init__()
            self.original_model = original_model
            self.trt_encoder = TensorRTViTInference(engine_path)
            
            # 保留其他组件
            self._decoder = original_model._decoder
            self._downstream_head = original_model._downstream_head
            
        def _encode_image(self, img, true_shape):
            """使用TensorRT加速的编码"""
            # 使用TensorRT编码器
            img_np = img.cpu().numpy()
            encoded = self.trt_encoder.infer(img_np)
            
            # 转换回torch tensor
            feat = torch.from_numpy(encoded).to(img.device)
            
            # 生成位置编码（简化版本）
            h, w = true_shape[0].item(), true_shape[1].item()
            pos = torch.zeros(1, feat.shape[1], 2, device=img.device, dtype=torch.long)
            
            return feat, pos, None
            
        def forward(self, *args, **kwargs):
            return self.original_model.forward(*args, **kwargs)
    
    return TensorRTMASt3R(model, engine_path)

# 使用示例
def quantize_mast3r_model():
    """量化MASt3R模型的完整流程"""
    
    # 1. 加载原始模型
    from mast3r_slam.mast3r_utils import load_mast3r
    model = load_mast3r()
    
    # 2. 准备校准数据
    from mast3r_slam.dataloader import load_dataset
    dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_desk")
    calib_loader = CalibrationDataLoader(dataset, max_samples=100)
    
    # 3. 创建量化器
    quantizer = MASt3RTensorRTQuantizer(model)
    
    # 4. 执行量化
    print("开始INT8量化...")
    engine_path = quantizer.quantize_model(calib_loader, precision="int8")
    
    if engine_path:
        print(f"量化完成! 引擎保存在: {engine_path}")
        
        # 5. 集成到原模型
        quantized_model = integrate_tensorrt_mast3r(model, engine_path)
        return quantized_model
    else:
        print("量化失败!")
        return None

# 性能测试
def benchmark_models(original_model, quantized_model, test_data):
    """对比原始模型和量化模型的性能"""
    import time
    
    # 测试原始模型
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(100):
        with torch.no_grad():
            _ = original_model._encode_image(test_data, torch.tensor([[512, 512]]))
    
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    
    # 测试量化模型
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(100):
        with torch.no_grad():
            _ = quantized_model._encode_image(test_data, torch.tensor([[512, 512]]))
    
    torch.cuda.synchronize()
    quantized_time = time.time() - start_time
    
    print(f"原始模型平均推理时间: {original_time/100:.4f}s")
    print(f"量化模型平均推理时间: {quantized_time/100:.4f}s") 
    print(f"加速比: {original_time/quantized_time:.2f}x")

if __name__ == "__main__":
    # 执行量化
    quantized_model = quantize_mast3r_model()
    
    if quantized_model:
        # 创建测试数据
        test_img = torch.randn(1, 3, 512, 512).cuda()
        
        # 性能对比
        from mast3r_slam.mast3r_utils import load_mast3r
        original_model = load_mast3r()
        benchmark_models(original_model, quantized_model, test_img)
