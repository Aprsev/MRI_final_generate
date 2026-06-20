@echo off
REM ============================================
REM DKI Pipeline — Setup and Run (Windows)
REM ============================================
echo.
echo === DKI Pipeline Setup ===
echo.

REM 1. Create conda environment
echo [1/4] Creating conda environment 'dki_pipeline' with PyTorch (CUDA)...
call conda create -n dki_pipeline python=3.12 -y
call conda activate dki_pipeline

REM 2. Install dependencies
echo [2/4] Installing dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib scipy

REM 3. Verify CUDA
echo [3/4] Checking CUDA...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if %errorlevel% neq 0 (
    echo WARNING: CUDA not available! Running on CPU will be slow.
)

REM 4. Run experiment
echo [4/4] Running DKI experiment...
python run.py --phase2-max-files 200 --seeds 42 123 514 --epochs 50 --voxel-methods supervised pinn_log pinn_log_rician

echo.
echo === Done! Results saved in outputs/ ===
pause
