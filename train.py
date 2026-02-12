#!/usr/bin/env python3
"""
Simple training script for Go2 quadruped robot.

Examples:
    python train.py
    python train.py --latest
    python train.py --steps 50000 --envs 8192
    python train.py --checkpoint 5000
"""

import sys
import os

# Remove local genesis folder from import path to avoid conflicts
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

# Add src to path
sys.path.insert(0, os.path.join(script_dir, 'src'))

# Import and run the main training script
from go2_train import main

if __name__ == "__main__":
    main()
