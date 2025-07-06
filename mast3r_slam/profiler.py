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
        
        # ViT Inference
        vit_total = stats.get('vit_encode', {}).get('total', 0) + \
                    stats.get('decoder', {}).get('total', 0)
        print(f"\nViT-Large Inference Total: {vit_total:.3f}s")
        if 'vit_encode' in stats:
            print(f"  - Encoding: {stats['vit_encode']['total']:.3f}s ({stats['vit_encode']['count']} calls)")
        if 'decoder' in stats:
            print(f"  - Decoding: {stats['decoder']['total']:.3f}s ({stats['decoder']['count']} calls)")
        
        # Attention vs MLP
        if 'attention' in stats and 'mlp' in stats:
            attn_time = stats['attention']['total']
            mlp_time = stats['mlp']['total']
            total_time = attn_time + mlp_time
            print(f"\nAttention vs MLP:")
            print(f"  - Attention: {attn_time:.3f}s ({attn_time/total_time*100:.1f}%)")
            print(f"  - MLP: {mlp_time:.3f}s ({mlp_time/total_time*100:.1f}%)")
        
        # Backend BA
        #ba_total = stats.get('ba_rays', {}).get('total', 0) + \
        #           stats.get('ba_calib', {}).get('total', 0)
        #print(f"\nBackend BA Total: {ba_total:.3f}s")
        #if 'ba_rays' in stats:
        #    print(f"  - Ray BA: {stats['ba_rays']['total']:.3f}s ({stats['ba_rays']['count']} calls)")
        #if 'ba_calib' in stats:
        #    print(f"  - Calib BA: {stats['ba_calib']['total']:.3f}s ({stats['ba_calib']['count']} calls)")
        
        #print("="*50 + "\n")
            
         
         # ---------- 全部计时减去 ViT = “Backend BA+杂项” ----------
        total_time = sum(rec['total'] for rec in stats.values())
        ba_total   = total_time - vit_total
        print(f"\nBackend BA+Other Total: {ba_total:.3f}s")

        print("="*50 + "\n")
        


# Global profiler instance
profiler = TimeProfiler()
