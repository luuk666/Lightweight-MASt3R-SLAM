# main_quantized.py
"""修改后的main.py，支持TensorRT量化模型"""

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
from pathlib import Path
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

from mast3r_slam.profiler import profiler
import queue

# 导入量化模型支持
try:
    from tensorrt_integration import load_quantized_mast3r
    QUANTIZATION_AVAILABLE = True
    print("✓ 量化模块可用")
except ImportError as e:
    QUANTIZATION_AVAILABLE = False
    print(f"量化模块不可用: {e}")

def load_model_with_quantization(device="cuda", engine_path="mast3r_encoder_int8.trt", force_quantized=False):
    """智能加载模型，优先使用量化版本"""
    
    if QUANTIZATION_AVAILABLE and Path(engine_path).exists():
        try:
            print(f"发现量化引擎: {engine_path}")
            model = load_quantized_mast3r(engine_path, device)
            print("✓ 量化模型加载成功，预期获得2-4倍加速")
            return model, True
        except Exception as e:
            print(f"量化模型加载失败: {e}")
            if force_quantized:
                raise e
            print("回退到原始模型")
    elif force_quantized:
        raise FileNotFoundError(f"强制使用量化模型，但引擎文件不存在: {engine_path}")
    else:
        if not Path(engine_path).exists():
            print(f"未找到量化引擎: {engine_path}")
        else:
            print("量化模块不可用")
    
    # 回退到原始模型
    print("使用原始MASt3R模型")
    model = load_mast3r(device=device)
    return model, False

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # 保持原有的relocalization逻辑不变
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure

def run_backend(cfg, model, states, keyframes, K, q):
    # 保持原有的backend逻辑不变
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()
            
        if config["use_calib"]:
            with profiler.timer('ba_calib'):
                factor_graph.solve_GN_calib()
        else:
            with profiler.timer('ba_rays'):
                factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

    backend_stats = profiler.get_stats()
    print("[BACKEND] send stats, iter =", idx)
    q.put(backend_stats) 

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    
    # 量化模型相关参数
    parser.add_argument("--use-quantized", action="store_true", 
                       help="优先使用量化模型")
    parser.add_argument("--force-quantized", action="store_true", 
                       help="强制使用量化模型（如果失败则退出）")
    parser.add_argument("--engine-path", default="mast3r_encoder_int8.trt", 
                       help="TensorRT引擎文件路径")

    args = parser.parse_args()

    load_config(args.config)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {config}")

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)
    backend2main = manager.Queue()

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    # === 模型加载 - 支持量化 ===
    print("\n=== 模型加载 ===")
    
    # 智能选择模型
    use_quantized = args.use_quantized or args.force_quantized
    if use_quantized or Path(args.engine_path).exists():
        model, is_quantized = load_model_with_quantization(
            device=device, 
            engine_path=args.engine_path, 
            force_quantized=args.force_quantized
        )
        if is_quantized:
            print("🚀 启用TensorRT INT8加速，预期编码速度提升2-4倍")
        else:
            print("📝 使用原始模型")
    else:
        print("使用原始MASt3R模型")
        model = load_mast3r(device=device)
        is_quantized = False
    
    model.share_memory()

    # 其余逻辑保持不变
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K, backend2main))
    backend.start()

    i = 0
    fps_timer = time.time()
    frames = []

    print(f"\n🏃 开始SLAM处理 (使用{'量化' if is_quantized else '原始'}模型)")

    while True:
        # 后端统计收集
        try:
            ba_stats = backend2main.get_nowait()
            profiler.merge_stats(ba_stats)
        except queue.Empty:
            pass

        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg

        mode = states.get_mode()
        
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        
        # 性能监控 - 显示量化模型的性能提升
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            status = "🚀 量化加速" if is_quantized else "📝 原始模型"
            print(f"FPS: {FPS:.2f} ({status})")
        i += 1

    # 保存结果和性能统计
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)
    
    backend.join()  
    if not args.no_viz:
        viz.join()    
    
    # 打印性能统计
    print("\n" + "="*60)
    print("SLAM处理完成")
    print("="*60)
    
    if is_quantized and hasattr(model, 'get_performance_stats'):
        stats = model.get_performance_stats()
        if stats:
            print(f"量化模型统计:")
            print(f"  总编码次数: {stats.get('total_calls', 0)}")
            print(f"  平均编码时间: {stats.get('avg_time', 0):.4f}s")
            print(f"  最快编码时间: {stats.get('min_time', 0):.4f}s")
    
    profiler.print_summary()
    print("done")
