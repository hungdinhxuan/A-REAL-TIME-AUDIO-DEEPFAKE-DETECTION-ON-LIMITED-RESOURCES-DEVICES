import torch
import time
import psutil
import os
import gc
import tracemalloc
from tqdm import tqdm

def clear_pytorch_caches():
    """Clear PyTorch internal caches safely"""
    gc.collect()
    
    # Try different methods to clear PyTorch caches
    if hasattr(torch, 'jit') and hasattr(torch.jit, '_state'):
        try:
            torch.jit._state._python_cu.clear_cache()
        except AttributeError:
            pass
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection again
    gc.collect()

def debug_memory_usage():
    """Debug memory usage patterns"""
    model = torch.jit.load("./exports/Distil_XLSR_5_Custom_Trans_Layer_ConformerTCM_conf-1-2_best_mdt_lora_mini_wrapper_no_pad3mobile.pt")
    input = torch.randn(1, 33600)
    
    # Enable memory tracking
    tracemalloc.start()
    
    print("=== Memory Debug Session ===")
    print(f"Initial memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Test single inference
    with torch.no_grad():
        _ = model(input)
    
    print(f"After first inference: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Test multiple inferences
    for i in range(10):
        with torch.no_grad():
            _ = model(input)
        
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Run {i+1}: {current_mem:.2f} MB")
        
        # Clear PyTorch caches
        clear_pytorch_caches()
    
    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\n=== Top Memory Allocations ===")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()

def benchmark_with_memory_tracking():
    """Benchmark with detailed memory tracking"""
    model = torch.jit.load("./exports/Distil_XLSR_5_Custom_Trans_Layer_ConformerTCM_conf-1-2_best_mdt_lora_mini_wrapper_no_pad3mobile.pt")
    input = torch.randn(1, 33600)
    
    # Set CPU threads
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    
    times = []
    memory_usage = []
    process = psutil.Process()
    
    print("=== Starting Memory-Tracked Benchmark ===")
    
    with torch.no_grad():
        for i in range(100):
            # Clear memory before each run
            clear_pytorch_caches()
            
            start_mem = process.memory_info().rss
            start_time = time.time()
            
            _ = model(input)
            
            end_time = time.time()
            end_mem = process.memory_info().rss
            
            times.append(end_time - start_time)
            memory_usage.append(end_mem - start_mem)
            
            if (i + 1) % 20 == 0:
                avg_time = sum(times[-20:]) / 20
                avg_mem = sum(memory_usage[-20:]) / 20
                current_total = end_mem / 1024 / 1024
                print(f"Run {i+1}: Avg time={avg_time:.4f}s, Avg mem={avg_mem/1024:.2f}KB, Total={current_total:.2f}MB")
    
    print(f"\nFinal Results:")
    print(f"Average time: {sum(times)/len(times):.4f}s")
    print(f"Peak time: {max(times):.4f}s")
    print(f"Average memory per run: {sum(memory_usage)/len(memory_usage)/1024:.2f}KB")
    print(f"Peak memory per run: {max(memory_usage)/1024:.2f}KB")

if __name__ == "__main__":
    print("Choose debugging option:")
    print("1. Memory debug session")
    print("2. Benchmark with memory tracking")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        debug_memory_usage()
    elif choice == "2":
        benchmark_with_memory_tracking()
    else:
        print("Invalid choice") 