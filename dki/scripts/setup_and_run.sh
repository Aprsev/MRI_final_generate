#!/usr/bin/env bash
# ============================================
# DKI Pipeline — Setup and Run (Linux/macOS)
# ============================================
set -e

echo ""
echo "=== DKI Pipeline Setup ==="
echo ""

# 1. Create conda environment
echo "[1/4] Creating conda environment 'dki_pipeline'..."
conda create -n dki_pipeline python=3.12 -y
conda activate dki_pipeline

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib scipy

# 3. Verify CUDA
echo "[3/4] Checking CUDA..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 4. Run experiment
echo "[4/4] Running DKI experiment..."
python run.py --phase2-max-files 200 --seeds 42 123 514 --epochs 50 \
    --voxel-methods supervised pinn_log pinn_log_rician

echo ""
echo "=== Done! Results saved in outputs/ ==="
