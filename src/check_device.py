"""
Quick script to check if your system will use GPU or CPU for training.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    print("\n" + "="*70)
    print("DEVICE DETECTION")
    print("="*70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is AVAILABLE")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"\n🚀 Training will use GPU (fast)")
    else:
        print(f"\n✗ CUDA is NOT available")
        print(f"\n⚠️  Training will use CPU (slow)")
        print(f"\nTo use GPU, install CUDA-enabled PyTorch:")
        print(f"  Visit: https://pytorch.org/get-started/locally/")
    
    print("="*70 + "\n")
    
except ImportError:
    print("\nERROR: PyTorch not installed")
    print("Install with: pip install torch")
