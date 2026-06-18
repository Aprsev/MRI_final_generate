# MRI Final Project — 磁共振成像原理及应用 期末项目

> **T1/T2 → ADC 映射 + DKI 前向模拟** 完整管线  
> 利用真实 MRI T1/T2 加权图像，通过物理合理的灰度映射生成 ADC 真值图，再经 DKI 前向模型模拟多 b 值 DWI 数据

---

## 📋 项目概述

本项目实现了一套从 **真实 T1/T2 加权 MRI → ADC 真值图 → 多 b 值 DWI 模拟数据** 的完整生成管线。

**核心目标**：为扩散重建算法（DKI fitting、深度学习重建等）提供具有**真实解剖结构 + 已知参数真值**的模拟训练与验证数据。

### 关键创新
- **连续灰度映射法**：利用 T2 信号与 ADC 的天然正相关性，无需组织分割即可生成像素级连续参数图
- **双 Phase 架构**：Phase 1（零成本 T1 反转法）即插即用；Phase 2（ML 合成法）更高质量
- **多种生成策略**：灰度映射、弹性形变、多维物理变异，覆盖丰富的解剖与信号多样性

---

## 🏗️ 文件夹架构

```
MRI_final/
│
├── README.md                          ← 本文件
├── environment.yml                    ← Conda/Mamba 环境配置
│
├── src/                               ★ 所有源代码集中管理
│   ├── pipelines/                     ← 数据生成管线
│   │   ├── t1_comprehensive_pipeline.py   ★ 主管线：T1→ADC→DWI (Phase 1 + Phase 2)
│   │   ├── t2_grayscale_mapping_pipeline.py  ← T2 灰度映射管线
│   │   ├── t2_deform_pipeline.py          ← 弹性形变生成多样大脑
│   │   └── t2_variation_pipeline_v2.py    ← 强度/对比度/噪声多维变异
│   ├── training/                      ← 机器学习模型训练
│   │   ├── phase2_sklearn_train.py    ← Phase 2 方案A：RandomForest 训练
│   │   └── train_unet_model.py        ← Phase 2 方案B：UNet 训练
│   └── utils/                         ← 工具与辅助脚本
│       ├── verify_actual.py           ← 数据验证脚本
│       ├── gen_comparison_fig.py      ← 对比图生成
│       ├── compare_gan_unet.py        ← GAN vs UNet 对比
│       ├── generate_figures.py        ← 报告用图生成
│       ├── run_pipeline.py            ← 管线运行助手
│       └── train_unet_comparison.py   ← UNet 对比训练
│
├── data/                              ★ 所有输入数据
│   ├── raw/                           ← 原始采集数据
│   │   └── All_Subjects_T1_Raw/       ← 43 例人脑 T1 DICOM 原始数据 (~2.1 GB)
│   ├── templates/                     ← 标准脑模板
│   │   ├── MNI152/                    ← MNI152 T1/T2 标准模板
│   │   └── parcellations/             ← 脑图谱分区文件
│   └── archives/                      ← 压缩归档
│       ├── All_Subjects_T1_Raw.zip
│       └── summary.zip
│
├── results/                           ★ 所有输出结果，按类型分类
│   ├── simulated_dwi/                 ← 主要模拟 DWI 数据 (npz 格式)
│   │   ├── 01_MNI152_Grayscale/       ← MNI152 T2 灰度映射法 (~30 样本)
│   │   ├── 02_Phase1_T1_Inversion/    ← Phase 1 T1 反转法 (~860 样本)
│   │   ├── 03_Phase2_UNet_Synthesis/  ← Phase 2 UNet 合成法 (~860 样本)
│   │   └── 04_ADC_GroundTruth/        ← ADC 真值图样本
│   ├── deformed/                      ← 弹性形变管线输出 (10 全脑)
│   ├── variation/                     ← 多维变异管线输出 (5 全脑)
│   ├── figures/                       ← 可视化结果图
│   │   ├── pipeline_overview/         ← 管线概览图
│   │   ├── training_curves/           ← 训练损失与指标曲线
│   │   ├── subject_previews/          ← 各被试预览图
│   │   └── comparison/                ← 对比分析图 (Phase1 vs Phase2 等)
│   └── training_vis/                  ← 训练过程快照
│
├── models/                            ★ 训练好的模型
│   ├── unet/                          ← UNet 模型
│   │   ├── t1_to_t2_unet_best.pth     ← 最佳验证损失模型
│   │   └── t1_to_t2_unet_final.pth    ← 最终 epoch 模型
│   └── random_forest/                 ← RandomForest 模型
│       └── t1_to_t2_rf_model.pkl      ← 训练好的 RF 模型 (~30 MB)
│
├── docs/                              ★ 技术文档
│   ├── README_2.2.md                  ← 灰度映射法技术说明
│   ├── README_DWI_Simulation_Guide.md ← DWI 模拟数据生成指南
│   ├── T1_to_T2_for_task2.2.md        ← T1→T2 技术路径分析
│   └── 文档_Phase2模拟数据生成流程与原理.md  ← Phase 2 详细原理文档
│
└── external/                          ★ 外部工具
    ├── GAN-MAT/                       ← GAN 合成工具 (MATLAB 实现)
    │   ├── functions/                 ← MATLAB 函数
    │   ├── template/                  ← 模板文件
    │   ├── parcellations/             ← 分区文件
    │   ├── example_data/              ← 示例数据
    │   ├── report/                    ← GAN 分析报告
    │   └── docs/                      ← 文档
    └── GAN-MAT_build/                 ← GAN-MAT 预处理数据
        ├── input/                     ← 预处理输入
        ├── output/                    ← 预处理输出
        ├── figures/                   ← 生成图
        └── functions/                 ← Python 功能函数
```

---

## 🧪 管线详解

### 1. 核心主管线：`src/pipelines/t1_comprehensive_pipeline.py`

**功能**：将 T1 加权 MRI 数据转换为 ADC 真值图，并前向生成多 b 值 DWI 模拟数据。

#### Phase 1 — T1 反转法（零训练成本，即插即用）
```
I_T2 = 1 - I_T1                  # 灰度反转模拟 T2 对比度
ADC  = 2.4×10⁻³ · I_T2 + 0.6×10⁻³  # 线性映射
S0   = 0.85 + 0.35 · I_T2          # 质子密度映射
K    = 1.0 - 0.95 · I_T2           # 峰度映射
DWI  = S₀ · exp(-b·D + ⅙·b²·D²·K)  # DKI 前向模型
```

#### Phase 2 — 机器学习 T1→T2 合成
- **方案A（推荐）**：RandomForest — 无需 GPU，CPU 1-2 分钟完成训练
- **方案B**：UNet — 需要 PyTorch，合成质量更高 (PSNR~31dB, SSIM~0.90)

### 2. T2→ADC 灰度映射管线

| 脚本 | 描述 | 关键特性 |
|------|------|----------|
| `t2_grayscale_mapping_pipeline.py` | 从 MNI152 T2 模板直接映射 ADC | 标准灰度映射，最简路径 |
| `t2_deform_pipeline.py` | 弹性形变生成多形态大脑 | 随机位移场，解剖多样性 |
| `t2_variation_pipeline_v2.py` | 强度/对比度/偏置场/噪声多维变异 | MRI 物理合理的变化 |

### 3. ML 模型训练

| 脚本 | 框架 | 适用场景 | 输出路径 |
|------|------|----------|----------|
| `src/training/phase2_sklearn_train.py` | scikit-learn | 快速原型，无 GPU 需求 | `models/random_forest/t1_to_t2_rf_model.pkl` |
| `src/training/train_unet_model.py` | PyTorch | 高质量合成 | `models/unet/t1_to_t2_unet_best.pth` |

---

## ⚙️ 环境配置

### Mamba（推荐）

```bash
mamba env create -f environment.yml
conda activate MRI_final
```

### 手动安装核心依赖

```bash
pip install numpy scipy matplotlib pydicom nibabel scikit-learn joblib
```

### PyTorch（UNet 需要）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate MRI_final

# 2. Phase 1：T1 反转法（前 3 例测试）
python src/pipelines/t1_comprehensive_pipeline.py --phase 1 --n_subjects 3

# 3. Phase 2 方案A：训练 RandomForest
python src/training/phase2_sklearn_train.py

# 4. Phase 2 推理
python src/pipelines/t1_comprehensive_pipeline.py --phase 2 --inference --all

# 5. 查看结果
python src/utils/gen_comparison_fig.py
```

---

## 📊 数据输出格式

每个 `.npz` 文件包含以下字段：

| 字段 | 含义 | 形状 | 说明 |
|------|------|------|------|
| `adc_gt` | ADC 真值图 | (H, W) | 无扰动纯净值，供评价对比 |
| `d_gt` | 扩散系数 D | (H, W) | 含随机扰动，模拟组织异质性 |
| `k_gt` | 峰度系数 K | (H, W) | 含随机扰动 |
| `s0_gt` | S0 参数图 | (H, W) | 含偏置场和扰动 |
| `dwi_clean` | 无噪声 DWI 序列 | (5, H, W) | ADC 单指数前向生成 |
| `dwi_noisy` | 含 Rician 噪声 DWI | (5, H, W) | 模拟真实 MRI 噪声 |
| `bvals` | b 值列表 | (5,) | [0, 500, 1000, 1500, 2000] |
| `mask` | 脑组织掩膜 | (H, W) | bool 类型 |
| `tissue_label` | 组织标签 | (H, W) | 0=BG, 1=CSF, 2=GM, 3=WM |
| `noise_sigma` | 噪声标准差 | scalar | Rician 噪声水平 |
| `slice_idx` | 切片索引 | scalar | 原始体积中的位置 |
| `subject` | 被试名称 | str | — |
| `source` | 管线标识 | str | 生成方式 |

---

## 📈 已生成数据规模

| 数据集 | 样本数 | 生成方式 | 存储路径 |
|--------|--------|----------|----------|
| MNI152 灰度映射 | 30 个 2D 样本 | `t2_grayscale_mapping_pipeline.py` | `results/simulated_dwi/01_MNI152_Grayscale/` |
| Phase 1 反转法 | ~860 个 2D 样本 | `t1_comprehensive_pipeline.py --phase 1` | `results/simulated_dwi/02_Phase1_T1_Inversion/` |
| Phase 2 UNet | ~860 个 2D 样本 | `t1_comprehensive_pipeline.py --phase 2` | `results/simulated_dwi/03_Phase2_UNet_Synthesis/` |
| 弹性形变 | 10 个 3D 全脑 | `t2_deform_pipeline.py` | `results/deformed/` |
| 多维变异 | 5 个 3D 全脑 | `t2_variation_pipeline_v2.py` | `results/variation/` |

---

## 🔬 关键物理公式

### DKI 前向模型

$$S(b) = S_0 \cdot \exp\left(-b \cdot D + \frac{1}{6} \cdot b^2 \cdot D^2 \cdot K\right)$$

### ADC 灰度映射

$$ADC(x,y) = 2.4 \times 10^{-3} \cdot I_{T2}(x,y) + 0.6 \times 10^{-3}$$

### 组织参数范围

| 组织 | T2 亮度 | ADC (×10⁻³ mm²/s) | S0 | K |
|------|---------|-------------------|---|----|
| 白质 (WM) | 暗 | 0.6 ~ 0.7 | 0.85 | 1.0 |
| 灰质 (GM) | 中 | 0.9 ~ 1.2 | 0.94 | 0.76 |
| 脑脊液 (CSF) | 亮 | 2.5 ~ 3.0 | 1.20 | 0.05 |

---

## 📚 技术文档

| 文档 | 内容 |
|------|------|
| `docs/README_2.2.md` | T2 灰度映射法技术说明 |
| `docs/README_DWI_Simulation_Guide.md` | DWI 模拟数据生成完整指南 |
| `docs/T1_to_T2_for_task2.2.md` | T1→T2 技术路径分析 |
| `docs/文档_Phase2模拟数据生成流程与原理.md` | Phase 2 详细原理与流程（含大量图表） |

---

## 📝 引用与致谢

- MNI152 模板: ICBM 2009a Nonlinear Symmetric
- DKI 模型: Jensen et al., 2005, Magnetic Resonance in Medicine
- GAN-MAT: GitHub - WTCN-computational-anatomy-group/GAN-MAT
- 灰度映射法: 本课程 Task 2.2 技术路径

---

> **浙江大学 · 磁共振成像原理及应用 · 2026 春夏学期**
