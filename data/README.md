# T1 → ADC/DWI 数据生成说明

本目录包含三组由不同管线生成的模拟DWI（Diffusion-Weighted Imaging）数据，
DWI 由 **ADC 单指数模型** `S(b) = S₀·exp(-b·D)` 生成（不含K项），
用于磁共振成像课程实验的教学目的。

---

## 目录概览

```
data/
├── 01_MNI152_Grayscale/        MNI152 T2模板直接灰度映射（金标准参考）
├── 02_Phase1_T1_Inversion/     Phase 1: T1反转法（零训练成本，即插即用）
├── 03_Phase2_UNet_Synthesis/   Phase 2: UNet T1→T2合成（深度学习驱动）
└── README.md                   本说明文件
```

---

## 通用字段说明 (.npz 文件内部结构)

每个 `.npz` 文件包含以下 13 个字段，形状和类型完全一致：

| 字段名 | 类型 | 形状 | 含义 |
|--------|------|------|------|
| `dwi_clean` | float32 | (5, H, W) | ADC单指数前向模拟的无噪声DWI，5个b值各一张 |
| `dwi_noisy` | float32 | (5, H, W) | 添加Rician噪声后的DWI（用于训练/测试） |
| `bvals` | int32 | (5,) | b值序列 = [0, 500, 1000, 1500, 2000] (s/mm²) |
| `s0_gt` | float32 | (H, W) | 基线信号 S0 图（b=0 时的信号强度，已加偏置场和扰动） |
| `d_gt` | float32 | (H, W) | 实际用于DWI前向模型的 ADC 图（= `adc_gt` 加内部扰动后），也称 `adc_noisy` |
| `k_gt` | float32 | (H, W) | 峰度系数 K 图（无单位，已加内部扰动） |
| `adc_gt` | float32 | (H, W) | ADC 真值（纯净的线性映射结果，**无扰动、无偏置**，供对比评价用） |
| `mask` | bool | (H, W) | 脑区二值掩膜，True=脑组织，False=背景 |
| `tissue_label` | int32 | (H, W) | 组织标签：0=背景, 1=CSF, 2=GM, 3=WM |
| `noise_sigma` | float32 | scalar | 添加的Rician噪声标准差 σ |
| `slice_idx` | int32 | scalar | 原始3D体积中的切片索引 |
| `subject` | str | — | 受试者/来源名称 |
| `source` | str | — | 数据来源管线标识 |
| `description` | str | — | 内置数据说明长字符串 |

> **注意 `adc_gt` 与 `d_gt` 的区别**：
> - `adc_gt` = 从归一化 T2 信号直接线性映射的原始 ADC（纯净无扰动）  
>   `ADC_raw = 2.4×10⁻³ × I_T2_norm + 0.6×10⁻³`
> - `d_gt` = `adc_gt` 经过生物扰动（乘性高斯噪声 σ=0.03）后的版本，即实际输入 DKI 公式的 D 值  
> - 评价 ADC 重建质量时应使用 `adc_gt` 作为真值，`d_gt` 更接近反映实际组织异质性的扩散系数

其中 `H = W = 192`（所有样本统一尺寸），`B = 5`（b值个数）。

---

## 各数据集详细生成说明

---

### 数据集 01_MNI152_Grayscale（30个样本）

#### 生成管线文件
`t2_grayscale_mapping_pipeline.py`

#### 数据来源
- **MNI152 T2 模板**（标准脑模板）：`mni_icbm152_t2_tal_nlin_sym_09a.mnc`
- 不依赖任何临床 DICOM，完全基于公开标准模板

#### 核心流程
```
MNI152 T2 3D模板 → 提取2D切片（轴位，自动过滤无效切片）
    → 缩放至192×192 → 高斯平滑（σ=1.0）
    → 脑区内归一化到 [0,1]
    → ADC映射: ADC = 2.4×10⁻³ × I_T2 + 0.6×10⁻³
        I=0 (WM) → 0.6×10⁻³, I=1 (CSF) → 3.0×10⁻³
    → 推导S0图（WM=0.85, CSF=1.20, 线性内插）
    → 推导K图（WM=1.0, CSF=0.05, 线性内插）
    → 添加内部生物扰动（S0: σ=0.05, K: σ=0.10, ADC: σ=0.03）
    → 添加低频偏置场（模拟线圈不均匀性）
    → ADC单指数模型生成5个b值DWI: S(b)=S₀·exp(-b·D) （b=0, 500, 1000, 1500, 2000）
    → 添加Rician噪声
    → 保存为 sample_XXXX.npz
```

#### 样本分布
- 共 **30个** npz 文件
- 从 MNI152 T2 模板轴位中间60%脑区范围内均匀采样切片
- 每个样本使用不同的 **Rician噪声水平**，5个等级各6个样本：

| sigma | 样本数 | 文件范围 |
|-------|--------|----------|
| 0.01  | 6      | sample_0000~0005 |
| 0.02  | 6      | sample_0006~0011 |
| 0.03  | 6      | sample_0012~0017 |
| 0.04  | 6      | sample_0018~0023 |
| 0.05  | 6      | sample_0024~0029 |

- 由于 MNI152 是单个模板，所有样本的 `subject` 字段均为空，`slice_idx` 各不相同
- `source = "mni152_t2_grayscale_mapping"`
- **加噪情况**：全部 30 个样本均为 `dwi_clean ≠ dwi_noisy`（即有噪声版本和无噪声版本均提供）

#### 特点
- T2 对比度来自**真实 T2 模板**，是最接近临床真实T2加权像的参考数据
- 适合作为**金标准**用于评估 Phase 1 和 Phase 2 的合成质量

---

### 数据集 02_Phase1_T1_Inversion（860个样本）

#### 生成管线文件
`t1_comprehensive_pipeline.py` 的 Phase 1 部分（`phase1_process_all`）

#### 数据来源
- **43 个临床 T1 DICOM 数据集**，来自 `All_Subjects_T1_Raw/` 目录
- 受试者名称如 `MRICourse_chenyiwei2_20220515`, `mricourse_child1_20240803` 等

#### 核心流程
```
临床T1 DICOM → 读取3D体积
    → 自动脑提取（阈值+孔洞填充+最大连通域）
    → 高斯平滑归一化到 [0,1]
    → 提取20个2D切片（均匀采样策略，自动过滤颅顶/颅底）
    → 缩放至192×192
    → T1灰度反转 → T2-like: I_T2 = 1 - I_T1
        （利用T1与T2对比度相反的特性：T1中WM亮、CSF暗 → T2中WM暗、CSF亮）
    → 之后与01相同：ADC映射 → S0/K推导 → 扰动 → 偏置场 → ADC单指数 DWI → Rician噪声
```

#### 样本分布
- 共 **860个** npz 文件
- 每个 subject 生成 20 个切片，43 subjects × 20 slices = **860 个样本**
- 文件名连续编号：`sample_0000.npz` ~ `sample_0859.npz`
- 同一 subject 的所有切片连续排列（按 subject 遍历）
- 所有样本使用相同的噪声水平：`noise_sigma = 0.03`（默认值）
- `source = "phase1_t1_inversion"`
- `subject` 字段记录具体的受试者名称
- **加噪情况**：全部 860 个样本均为 `dwi_clean ≠ dwi_noisy`（有噪声版本 + 无噪声纯净版本）

#### 特点
- **零训练成本**，即插即用，无需任何预训练模型
- T1反转法生成的 T2-like 图像是**近似估计**，对比度与真实 T2 存在差距
- 每个 subject 均匀采样 20 个切片，覆盖全脑范围

---

### 数据集 03_Phase2_UNet_Synthesis（860个样本）

#### 生成管线文件
`t1_comprehensive_pipeline.py` 的 Phase 2 推理部分（`phase2_inference_all`）  
依赖预训练模型：`models/t1_to_t2_unet_best.pth`  
训练脚本：`train_unet_model.py`

#### 数据来源
- **相同的 43 个临床 T1 DICOM**，与 02 完全一致
- UNet 模型在 **MNI152 T1-T2 模板对上训练**（T1→T2 图像翻译）

#### 核心流程
```
临床T1 DICOM → 读取3D体积
    → 脑提取 → 归一化
    → 提取20个2D切片（与Phase 1相同的切片索引策略）
    → 缩放至192×192
    → 输入预训练UNet模型 → 输出合成T2图像
        UNet架构: 编码器-解码器 with skip connections
        base_filters=64, 4层下采样
        训练损失: L1 + 0.5×SSIM
        训练数据: MNI152 T1-T2模板对 ~140个切片
    → 之后与01/02相同：ADC映射 → S0/K推导 → 扰动 → 偏置场 → ADC单指数 DWI → Rician噪声
```

#### 样本分布
- 共 **860个** npz 文件
- 与 02 完全相同的 subject 和切片分布：43 subjects × 20 slices = 860
- 文件名连续编号：`sample_0000.npz` ~ `sample_0859.npz`
- 所有样本：`noise_sigma = 0.03`
- `source = "phase2_unet_t1t2"`
- `subject` 字段记录具体的受试者名称
- **加噪情况**：全部 860 个样本均为 `dwi_clean ≠ dwi_noisy`

#### 与 Phase 1 的关键区别
| 方面 | Phase 1 (T1反转) | Phase 2 (UNet) |
|------|-------------------|-----------------|
| T2获取方式 | `1 - T1` (简单反转) | UNet(T1) (深度学习翻译) |
| T2对比度真实性 | 较差（WM/GM对比度不够准确） | **更接近真实T2** |
| 是否需要预训练 | 否 | 是（需在MNI152上预训练） |
| subject/切片索引 | 与Phase 2完全一致 | 与Phase 1完全一致 |
| 后续DKI生成流程 | 完全相同 | 完全相同 |

---

## 三组数据的 ADC 图对比

三个数据集虽然共享相同的 ADC 映射公式（`ADC = 2.4×10⁻³ × I_T2 + 0.6×10⁻³`），但由于输入的 T2-like 图像来源不同，最终得到的 ADC 图也不同：

| 数据集 | T2-like 来源 | ADC 差异来源 |
|--------|-------------|-------------|
| 01_MNI152_Grayscale | MNI152 真实 T2 模板 | 最接近真实 T2 对比度，作为金标准参考 |
| 02_Phase1_T1_Inversion | T1 灰度反转：`I_T2 = 1 - I_T1` | T1→T2 只是近似反转，WM/GM 对比度与真实 T2 有偏差 |
| 03_Phase2_UNet_Synthesis | UNet(T1) 合成 | 深度学习学到 T1→T2 的非线性映射，对比度**更接近**真实 T2 |

虽然 02 和 03 来自**相同的 43 个临床 T1 subjects**，且切片索引完全对应（相同 subject 的相同 slice_idx 代表同一位置），但由于：
- **02** 的 T2-like 是简单的 `1 - T1` 线性反转
- **03** 的 T2-like 是 UNet 在 MNI152 上训练后合成的非线性映射

所以 **02 和 03 的 `adc_gt` / `d_gt` 是不同的**。03 的 ADC 更接近 01（金标准）。

但如果未来用更完美的 T1→T2 映射方法（理想情况下），02 和 03 的 ADC 应该趋于一致。

> 每个 sample 中同时保存了 `adc_gt`（纯净的原始ADC）和 `d_gt`（加扰动后的ADC，即实际用于生成DWI的扩散系数图）。

---

## 噪声说明

所有三个数据集均遵循 `dwi_clean`（无噪声）和 `dwi_noisy`（加噪后）同时保存的格式：

- **dwi_clean**: ADC单指数前向模型 `S(b) = S₀ × exp(-b×D)` 直接生成，完全无噪声
- **dwi_noisy**: 在 dwi_clean 基础上添加 **Rician噪声**（MRI中幅度图像的标准噪声模型）：
  ```
  S_noisy = sqrt((S_clean + n₁)² + n₂²),  n₁,n₂ ~ N(0, σ)
  ```
- **sigma 分布差异**：
  - `01_MNI152_Grayscale`: 5种噪声水平（σ=0.01~0.05），每种6个样本
  - `02_Phase1_T1_Inversion`: 统一 σ=0.03
  - `03_Phase2_UNet_Synthesis`: 统一 σ=0.03

---

## 训练/测试用途建议

| 用途 | 推荐数据集 | 解释 |
|------|-----------|------|
| 训练降噪模型 | 01 (MNI152) 或 02/03 | dwi_clean 作为目标，dwi_noisy 作为输入 |
| 验证T2合成质量 | 03 vs 02 对比 | 同一subject，Phase 1 vs Phase 2 |
| 金标准参考 | 01 (MNI152) | 真实T2模板映射，对比度最准确 |
| 大规模训练 | 02 或 03 (860样本) | 43 subjects，多样本多样性好 |

---

## 使用示例

```python
import numpy as np

# 读取Phase 2数据
data = np.load('data/03_Phase2_UNet_Synthesis/sample_0000.npz')
print(f'subject: {data["subject"]}')
print(f'source: {data["source"]}')
print(f'dwi_noisy shape: {data["dwi_noisy"].shape}')   # (5, 192, 192)
print(f'bvals: {data["bvals"]}')                        # [0, 500, 1000, 1500, 2000]
print(f'noise_sigma: {data["noise_sigma"]:.3f}')       # 0.030

# 获取无噪声的b=1000 DWI
b1000_idx = np.where(data['bvals'] == 1000)[0][0]
dwi_clean_b1000 = data['dwi_clean'][b1000_idx]
dwi_noisy_b1000 = data['dwi_noisy'][b1000_idx]

# 获取ADC图并乘以mask
adc = data['adc_gt']
mask = data['mask']
adc_masked = adc * mask

# 验证ADC单指数模型: S(b=1000) = S0 * exp(-1000 * D)
b = 1000
S0 = data['s0_gt']
D = data['d_gt']
dwi_recon = S0 * np.exp(-b * D)
error = np.abs(dwi_recon - data['dwi_clean'][b1000_idx]).max()
print(f'ADC model reconstruction error: {error:.8f}')  # 应接近 0
```

---

> 生成管线详见:  
> `t2_grayscale_mapping_pipeline.py`  (01_MNI152_Grayscale)  
> `t1_comprehensive_pipeline.py`     (02_Phase1 + 03_Phase2)  
> `train_unet_model.py`              (UNet模型训练)
