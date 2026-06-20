#!/usr/bin/env python3
"""
DKI Pipeline — Entry point
==========================
Run from dki_pipeline/ directory.
Automatically detects GPU and runs cross-noise DKI evaluation on Phase2 data.

Usage:
    python run.py                          # Full experiment (200 files, 50 epochs, 3 seeds)
    python run.py --epochs 30 --seeds 42   # Quick test

Results saved to: outputs/
"""

import sys
from pathlib import Path

# Add code/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from run_dki_extension import main as run_main

if __name__ == "__main__":
    # Override defaults to point to this package's structure
    sys.argv = [sys.argv[0]] + [
        "--data-source", "phase2",
        "--phase2-root", str(Path(__file__).resolve().parent.parent / "data" / "03_Phase2_UNet_Synthesis_DKI"),
        "--output-root", str(Path(__file__).resolve().parent.parent / "outputs"),
        "--cross-eval",
    ] + sys.argv[1:]
    run_main()
