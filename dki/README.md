# DKI PINN Pipeline — 便携实验包

## 目录结构

```
dki_pipeline/
├── code/
│   ├── dki_utils.py              # 核心工具库
│   └── run_dki_extension.py      # 实验运行脚本
├── data/
│   └── 03_Phase2_UNet_Synthesis/ # ← 这里放 Phase2 数据（需手动复制）
├── outputs/                      # ← 实验结果输出到这里
├── run.py                        # 入口脚本
├── setup_and_run.bat             # Windows 一键运行
├── setup_and_run.sh              # Linux/macOS 一键运行
└── README.md                     # 本文件
```

## 使用步骤

### 第一步：复制数据

将 Phase2 数据复制到 `data/` 目录下：

```bash
# 在原始项目目录中
cp -r data/simulated/03_Phase2_UNet_Synthesis dki_pipeline/data/
```

### 第二步：复制到新电脑

将整个 `dki_pipeline/` 文件夹复制到目标电脑（有 GPU）。

### 第三步：在新电脑上运行

**Windows：**
```bash
双击 setup_and_run.bat
# 或命令行：
cd dki_pipeline
setup_and_run.bat
```

**Linux/macOS：**
```bash
cd dki_pipeline
chmod +x setup_and_run.sh
./setup_and_run.sh
```

**手动运行（已有 conda 环境）：**
```bash
conda create -n dki_pipeline python=3.12
conda activate dki_pipeline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib scipy
python run.py
```

### 第四步：复制结果回来

实验完成后，将 `outputs/` 目录复制回原始项目的 `task_A/` 下：

```bash
# 在新电脑上
cp -r dki_pipeline/outputs /目标路径/task_A/outputs_dki_full
```

## 自定义运行参数

```bash
# 快速测试（单 seed、少样本、短训练）
python run.py --phase2-max-files 48 --seeds 42 --epochs 10

# 完整实验（3 seeds、200 样本、50 epochs）
python run.py --phase2-max-files 200 --seeds 42 123 514 --epochs 50

# 指定输出目录
python run.py --output-root /自定义路径/outputs
```

## 输出内容

```
outputs/
├── checkpoint/
│   ├── train_sigma_0.01/         # 在 sigma=0.01 上训练的模型
│   ├── train_sigma_0.03/
│   ├── train_sigma_0.05/
│   └── train_sigma_0.08/
├── metrics/
│   ├── per_seed_metrics.csv      # 逐 seed 详细指标
│   ├── summary.csv               # 多 seed 汇总 (mean ± std)
│   └── noise_summary.csv         # 噪声分层汇总
├── figures/
│   ├── dki_method_comparison.png # D/K RMSE 方法对比柱状图
│   ├── cross_noise_*.png         # 交叉噪声热力图 (每方法 × 每指标)
│   └── dki_noise_curves.png      # 噪声曲线
└── reports/
    └── dki_extension_report.md   # 自动实验报告
```

## 依赖

自动安装：
- Python 3.12
- PyTorch (CUDA 12.4)
- NumPy, Matplotlib, SciPy
