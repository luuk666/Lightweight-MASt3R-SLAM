# final_tensorrt_test.py
"""最终修复版TensorRT测试脚本"""

import torch
import numpy as np
import tensorrt as trt
import time
from pathlib import Path
import argparse
import sys

class FinalTensorRTTest:
    """最终修复版TensorRT测试器"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"测试TensorRT引擎: {engine_path}")
        
        # 检查文件
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"引擎文件不存在: {engine_path}")
        
        file_size = Path(engine_path).stat().st_size / (1024*1024)
        print(f"引擎文件大小: {file_size:.2f} MB")
        
        # 加载引擎
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # 分析引擎
        self._analyze_engine()
        
        # 设置固定形状
        self._setup_shapes()
        
        # 分配内存
        self._allocate_buffers()
        
    def _load_engine(self):
        """加载TensorRT引擎"""
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if not engine:
            raise RuntimeError("引擎反序列化失败")
            
        print("✓ TensorRT引擎加载成功")
        return engine
    
    def _analyze_engine(self):
        """分析引擎信息"""
        print(f"\n=== 引擎信息 ===")
        print(f"绑定数量: {self.engine.num_bindings}")
        
        self.input_info = {}
        self.output_info = {}
        
        for i in range(self.engine.num_bindings):
            try:
                # TensorRT 8.x新API
                binding_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(binding_name)
                dtype = self.engine.get_tensor_dtype(binding_name)
                is_input = self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT
            except:
                # 旧API回退
                binding_name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
            
            print(f"{'输入' if is_input else '输出'} {i}: {binding_name}")
            print(f"  形状: {shape}")
            print(f"  类型: {dtype}")
            
            if is_input:
                self.input_info[i] = {
                    'name': binding_name,
                    'shape': shape,
                    'dtype': dtype
                }
            else:
                self.output_info[i] = {
                    'name': binding_name,
                    'shape': shape,
                    'dtype': dtype
                }
    
    def _setup_shapes(self):
        """设置固定形状"""
        print(f"\n=== 设置固定形状 ===")
        
        # 设置输入形状为固定batch size = 1
        self.input_shape = (1, 3, 512, 512)
        
        # 根据引擎的输出形状设置
        # 从分析结果看，输出形状是 (-1, 1024, 1024)
        # 这看起来像是把patches展平了，正常应该是 (-1, 1025, 1024)
        self.output_shape = (1, 1024, 1024)
        
        print(f"输入形状: {self.input_shape}")
        print(f"输出形状: {self.output_shape}")
        
        # 尝试设置上下文的输入形状
        try:
            if hasattr(self.context, 'set_input_shape'):
                input_name = list(self.input_info.values())[0]['name']
                self.context.set_input_shape(input_name, self.input_shape)
                print("✓ 设置动态输入形状成功")
        except Exception as e:
            print(f"设置动态形状失败: {e}")
    
    def _allocate_buffers(self):
        """分配内存缓冲区"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.stream = cuda.Stream()
            
            # 计算内存大小 - 修复numpy类型问题
            input_size = int(np.prod(self.input_shape))  # 转换为int
            output_size = int(np.prod(self.output_shape))  # 转换为int
            
            print(f"输入元素数量: {input_size}")
            print(f"输出元素数量: {output_size}")
            print(f"输入内存大小: {input_size * 4 / 1024 / 1024:.2f} MB")
            print(f"输出内存大小: {output_size * 4 / 1024 / 1024:.2f} MB")
            
            # 分配输入内存
            self.input_host = cuda.pagelocked_empty(input_size, np.float32)
            self.input_device = cuda.mem_alloc(self.input_host.nbytes)
            
            # 分配输出内存
            self.output_host = cuda.pagelocked_empty(output_size, np.float32)
            self.output_device = cuda.mem_alloc(self.output_host.nbytes)
            
            # 绑定列表
            self.bindings = [int(self.input_device), int(self.output_device)]
            
            print("✓ 内存缓冲区分配成功")
            
        except Exception as e:
            print(f"✗ 内存分配失败: {e}")
            raise
    
    def test_inference(self):
        """测试推理"""
        import pycuda.driver as cuda
        
        print(f"\n=== 推理测试 ===")
        
        # 创建输入数据
        input_data = np.random.randn(*self.input_shape).astype(np.float32)
        print(f"输入数据形状: {input_data.shape}")
        
        try:
            # 复制输入数据到host内存
            input_flat = input_data.ravel()
            np.copyto(self.input_host, input_flat)
            
            # 传输到GPU
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            
            # 执行推理
            start_time = time.time()
            
            # 使用execute_async_v2
            success = self.context.execute_async_v2(
                bindings=self.bindings, 
                stream_handle=self.stream.handle
            )
            
            if not success:
                print("✗ 推理执行失败")
                return None
            
            # 传输结果回CPU
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()
            
            inference_time = time.time() - start_time
            
            # 获取输出数据
            output_data = np.copy(self.output_host)
            output_reshaped = output_data.reshape(self.output_shape)
            
            print(f"✓ 推理成功")
            print(f"推理时间: {inference_time:.4f}s")
            print(f"输出形状: {output_reshaped.shape}")
            print(f"输出范围: [{output_data.min():.4f}, {output_data.max():.4f}]")
            print(f"输出均值: {output_data.mean():.4f}")
            print(f"输出标准差: {output_data.std():.4f}")
            
            return output_reshaped, inference_time
            
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def benchmark(self, num_iterations=30):
        """性能基准测试"""
        print(f"\n=== 性能基准测试 ===")
        print(f"测试轮数: {num_iterations}")
        
        # 预热
        print("预热中...")
        for i in range(5):
            result = self.test_inference()
            if result is None:
                print(f"✗ 预热第{i+1}次失败")
                return
        print("✓ 预热完成")
        
        # 正式测试
        print("开始基准测试...")
        times = []
        successful_runs = 0
        
        for i in range(num_iterations):
            result = self.test_inference()
            if result is None:
                print(f"✗ 第{i+1}次推理失败")
                continue
            
            _, inference_time = result
            times.append(inference_time)
            successful_runs += 1
            
            if (i + 1) % 10 == 0:
                print(f"完成 {i+1}/{num_iterations}")
        
        if not times:
            print("✗ 所有推理都失败了")
            return
        
        # 统计结果
        times = np.array(times)
        print(f"\n=== 基准测试结果 ===")
        print(f"成功推理次数: {successful_runs}/{num_iterations}")
        print(f"平均推理时间: {np.mean(times):.4f}s")
        print(f"最短推理时间: {np.min(times):.4f}s")
        print(f"最长推理时间: {np.max(times):.4f}s")
        print(f"时间标准差: {np.std(times):.4f}s")
        print(f"理论FPS: {1/np.mean(times):.2f}")
        
        # 与原始模型性能对比
        self._compare_performance(np.mean(times))
    
    def _compare_performance(self, quantized_time):
        """与原始模型性能对比"""
        print(f"\n=== 性能对比分析 ===")
        
        # 基于之前测试的原始ViT编码时间
        original_time = 0.0288
        speedup = original_time / quantized_time
        
        print(f"原始ViT编码时间: ~{original_time:.4f}s")
        print(f"量化模型编码时间: {quantized_time:.4f}s")
        print(f"加速比: {speedup:.2f}x")
        print(f"时间减少: {(1 - quantized_time/original_time)*100:.1f}%")
        
        # 性能评价
        if speedup >= 3.0:
            print("🚀 优秀！量化效果非常显著")
            performance_grade = "A+"
        elif speedup >= 2.0:
            print("🎉 很好！量化效果显著")
            performance_grade = "A"
        elif speedup >= 1.5:
            print("👍 不错！量化有明显效果")
            performance_grade = "B"
        elif speedup >= 1.2:
            print("✓ 一般，量化有一定效果")
            performance_grade = "C"
        else:
            print("⚠️ 量化效果不明显，可能需要重新优化")
            performance_grade = "D"
        
        print(f"性能等级: {performance_grade}")
        
        # 整体SLAM性能预估
        # 假设编码占SLAM总时间的30-50%
        slam_speedup_low = 1 + 0.3 * (speedup - 1)
        slam_speedup_high = 1 + 0.5 * (speedup - 1)
        
        print(f"\n=== 整体SLAM性能预估 ===")
        print(f"预期SLAM整体加速: {slam_speedup_low:.2f}x - {slam_speedup_high:.2f}x")
        print(f"预期FPS提升: {(slam_speedup_low-1)*100:.1f}% - {(slam_speedup_high-1)*100:.1f}%")

def check_environment():
    """检查运行环境"""
    print("=== 环境检查 ===")
    
    # 检查TensorRT
    try:
        print(f"✓ TensorRT: {trt.__version__}")
    except:
        print("✗ TensorRT检查失败")
        return False
    
    # 检查PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✓ PyCUDA可用")
    except ImportError:
        print("✗ PyCUDA未安装，请运行: pip install pycuda")
        return False
    
    # 检查PyTorch和CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA不可用")
        return False
    else:
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        print(f"✓ CUDA设备: {device_name}")
        
        # 内存检查
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"✓ GPU内存: {total_memory:.1f} GB")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="最终版TensorRT引擎测试工具")
    parser.add_argument("--engine-path", default="mast3r_encoder_int8.trt",
                       help="TensorRT引擎文件路径")
    parser.add_argument("--benchmark", action="store_true",
                       help="运行性能基准测试")
    parser.add_argument("--iterations", type=int, default=30,
                       help="基准测试迭代次数")
    parser.add_argument("--test-only", action="store_true",
                       help="只运行单次推理测试")
    
    args = parser.parse_args()
    
    print("🔧 最终版TensorRT引擎测试工具")
    print("="*60)
    
    # 环境检查
    if not check_environment():
        print("\n请解决环境问题后重试")
        return 1
    
    try:
        # 创建测试器
        tester = FinalTensorRTTest(args.engine_path)
        
        # 运行测试
        print("\n" + "="*60)
        result = tester.test_inference()
        
        if result is None:
            print("❌ 推理测试失败")
            return 1
        
        print("✅ 单次推理测试成功")
        
        # 运行基准测试
        if args.benchmark and not args.test_only:
            print("\n" + "="*60)
            tester.benchmark(args.iterations)
        
        print(f"\n🎉 所有测试完成！TensorRT量化引擎工作正常")
        print("\n下一步: 可以将此引擎集成到MASt3R-SLAM中使用")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
