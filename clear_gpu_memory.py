"""
Helper script to clear GPU memory before running VPR evaluation.
Run this script if you're getting CUDA out of memory errors.

Usage:
    python clear_gpu_memory.py
"""
import torch
import gc
import sys

if torch.cuda.is_available():
    print("=" * 60)
    print("GPU Memory Cleanup Tool")
    print("=" * 60)
    
    # Get initial memory status
    initial_allocated = torch.cuda.memory_allocated(0) / 1024**3
    initial_reserved = torch.cuda.memory_reserved(0) / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nInitial GPU Memory Status:")
    print(f"  Total GPU Memory: {total_memory:.2f} GB")
    print(f"  Allocated: {initial_allocated:.2f} GB")
    print(f"  Reserved: {initial_reserved:.2f} GB")
    print(f"  Free: {total_memory - initial_reserved:.2f} GB")
    
    if initial_reserved > 1.0:
        print(f"\n⚠️  WARNING: {initial_reserved:.2f} GB already reserved!")
        print("   This will cause out-of-memory errors.")
        response = input("\n   Do you want to clear it? (y/n): ")
        if response.lower() != 'y':
            print("   Skipping cleanup. Please restart Python/IDE to clear memory.")
            sys.exit(0)
    
    print("\nClearing GPU memory...")
    
    # Aggressive clearing
    for i in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Get final memory info
    final_allocated = torch.cuda.memory_allocated(0) / 1024**3
    final_reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"\nFinal GPU Memory Status:")
    print(f"  Allocated: {final_allocated:.2f} GB")
    print(f"  Reserved: {final_reserved:.2f} GB")
    print(f"  Free: {total_memory - final_reserved:.2f} GB")
    
    if final_reserved > 0.5:
        print(f"\n⚠️  WARNING: {final_reserved:.2f} GB still reserved on GPU.")
        print("   This might be from other processes. Consider:")
        print("   1. Closing other Python processes using GPU")
        print("   2. Restarting your Python kernel/IDE")
        print("   3. Running: nvidia-smi to check for other GPU processes")
        print("   4. Killing processes: taskkill /PID <PID> /F")
    else:
        print("\n✅ GPU memory cleared successfully!")
        print("   You can now run your VPR evaluation script.")
    
    print("=" * 60)
else:
    print("CUDA is not available. No GPU memory to clear.")

