import time
import torch
from collections import defaultdict
from contextlib import contextmanager

class TimeProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
    
    def merge_stats(self, other_stats: dict):
        for name, rec in other_stats.items():
            self.total[name] += rec['total']
            self.count[name] += rec['count']    
            
    @contextmanager
    def timer(self, name):
        torch.cuda.synchronize()
        start = time.time()
        yield
        torch.cuda.synchronize()
        elapsed = time.time() - start
        self.timings[name].append(elapsed)
        self.counts[name] += 1
    
    def get_stats(self):
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'total': sum(times),
                'mean': sum(times) / len(times),
                'count': self.counts[name],
                'times': times
            }
        return stats
    
    def print_summary(self):
        print("\n" + "="*50)
        print("TIMING SUMMARY")
        print("="*50)

        stats = self.get_stats()

        # 个别组件时间和调用次数
        t_patch = stats.get('PatchEmbed', {}).get('total', 0.0)
        n_patch = stats.get('PatchEmbed', {}).get('count', 0)
    
        t_enc_attn = stats.get('Encoder_attn', {}).get('total', 0.0)
        t_enc_mlp  = stats.get('Encoder_mlp', {}).get('total', 0.0)
        n_enc = stats.get('Encoder_attn', {}).get('count', 0)
    
        t_dec_attn = stats.get('Decoder_attn', {}).get('total', 0.0)
        t_dec_cross = stats.get('Decoder_cross_attn', {}).get('total', 0.0)
        t_dec_mlp  = stats.get('Decoder_mlp', {}).get('total', 0.0)
        n_dec = stats.get('Decoder_attn', {}).get('count', 0)
        
        ba_total = stats.get('ba_calib', {}).get('total', 0.0)

        # 总计
        t_enc_total = t_enc_attn + t_enc_mlp
        t_dec_total = t_dec_attn + t_dec_cross + t_dec_mlp
        vit_total = t_patch + t_enc_total + t_dec_total

        print(f"\nViT-Large Inference Total: {vit_total:.3f}s")
        if t_patch > 0:
            print(f"  - PatchEmbed: {t_patch:.3f}s ({n_patch} calls)")
        if t_enc_attn > 0:
            print(f"  - Encoder_Attn: {t_enc_attn:.3f}s ({n_enc} calls)")
        if t_enc_mlp > 0:
            print(f"  - Encoder_MLP: {t_enc_mlp:.3f}s ({n_enc} calls)")
        if t_dec_attn > 0:
            print(f"  - Decoder_Attn: {t_dec_attn:.3f}s ({n_dec} calls)")
        if t_dec_cross > 0:
            print(f"  - Decoder_CrossAttn: {t_dec_cross:.3f}s ({n_dec} calls)")
        if t_dec_mlp > 0:
            print(f"  - Decoder_MLP: {t_dec_mlp:.3f}s ({n_dec} calls)")

        # Encoder vs Decoder 总对比
        total_enc_dec = t_enc_total + t_dec_total
        if total_enc_dec > 0:
            pct_enc = (t_enc_total / total_enc_dec) * 100
            pct_dec = (t_dec_total / total_enc_dec) * 100
            print(f"\nEncoder vs Decoder:")
            print(f"  - Encoder Total: {t_enc_total:.3f}s ({pct_enc:.1f}%)")
            print(f"  - Decoder Total: {t_dec_total:.3f}s ({pct_dec:.1f}%)")

        # Encoder 内部细分
        if t_enc_total > 0:
            enc_attn_pct = (t_enc_attn / t_enc_total) * 100
            enc_mlp_pct  = (t_enc_mlp  / t_enc_total) * 100
            print(f"\nEncoder Internal Breakdown:")
            print(f"  - Attention: {t_enc_attn:.3f}s ({enc_attn_pct:.1f}%)")
            print(f"  - MLP: {t_enc_mlp:.3f}s ({enc_mlp_pct:.1f}%)")

        # Decoder 内部细分
        if t_dec_total > 0:
            dec_attn_pct  = (t_dec_attn / t_dec_total) * 100
            dec_cross_pct = (t_dec_cross / t_dec_total) * 100
            dec_mlp_pct   = (t_dec_mlp / t_dec_total) * 100
            print(f"\nDecoder Internal Breakdown:")
            print(f"  - Self-Attention: {t_dec_attn:.3f}s ({dec_attn_pct:.1f}%)")
            print(f"  - Cross-Attention: {t_dec_cross:.3f}s ({dec_cross_pct:.1f}%)")
            print(f"  - MLP: {t_dec_mlp:.3f}s ({dec_mlp_pct:.1f}%)")

        # BA + 其他（总 profiled 时间 - vit 相关）
        total_profiled = sum(v['total'] for v in stats.values())
        #ba_total = total_profiled - vit_total
        print(f"\nBackend Total (ba_calib only): {ba_total:.3f}s")
        print("All profiler keys:", list(stats.keys()))
        print("="*50 + "\n")

        


# Global profiler instance
profiler = TimeProfiler()