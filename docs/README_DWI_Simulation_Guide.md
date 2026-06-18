# README：基于 MNI、T2 的 DWI-ADC 模拟数据生成与使用指南

---

## 1. 核心结论

随机生成矩阵和前向DWI存在一些问题：

1. 解剖结构不真实；
2. 人工设定组织分布比较繁琐，容易被质疑“模拟过于理想化”。





---

## 2. 参考路线分级

---

### 2.1 最低要求：

## 像素级随机矩阵替代路径

### 2.1.1 适用场景与特殊说明

**【极其重要】**：该路径生成的图像**不具备空间拓扑结构和解剖形态**，因此**绝对不能**用于训练包含 2D 空间卷积（如 U-Net, 2D CNN）的图像重建网络。否则网络会发生严重的几何过拟合，失去处理真实大脑的能力。

该路径专门用于**验证和训练像素级的神经网络（如纯粹依赖单像素衰减曲线拟合的 PINN / MLP）或传统拟合算法**。它的核心优势是不需要模拟任何几何形状，直接利用随机数值矩阵进行大批量的高效测试。



### 2.2 推荐：通过T2引导的连续灰度映射法

### 2.2.1 具体原理 

在传统的脑模板仿真中，通常需要将大脑分割成离散的区域（白质 WM、灰质 GM、脑脊液 CSF），然后为每个区域赋予单一的、均匀的 ADC 值。这种做法不仅耗时，且容易导致组织内部过于死板（缺乏真实的微结构纹理）。 **连续灰度映射法**的核心思想是：直接利用现有的高质量 2D 脑部 T2 加权（或 T1）图像的灰度值，通过连续的数学函数直接映射出 ADC 图。 在 T2 图像中，脑脊液（CSF）亮度最高，水分子自由扩散极强（ADC 最高，约 $3.0 \\times 10^{-3} \\text{ mm}^2/\\text{s}$）；白质（WM）相对较暗，水分子扩散受限（ADC 较低，约 $0.6 - 0.7 \\times 10^{-3} \\text{ mm}^2/\\text{s}$）。这种物理对比度的天然相关性使我们可以通过灰度映射，**在完全不需要进行组织分割的前提下，直接获得具有极度真实解剖边界和内部渐变纹理的 ADC 真值图 (Ground Truth)**。

这个ADC Groud truth可以再进行前向传播来得到最真实的模拟DWI图像，同时也容易进行加噪退化。 

### 2.2.2 数学映射公式 
假定输入一张已经归一化到 $[0, 1]$ 区间的 2D T2 图像灰度值 $I_{T2}(x, y)$，其对应的真值 $ADC(x, y)$ 可通过以下线性或分段线性公式映射： $$ADC(x, y) = \\alpha \\cdot I_{T2}(x, y) + \\beta$$ 为了让映射后的数值完美落在人脑临床合理的物理区间（$0.6 \\times 10^{-3} \\sim 3.0 \\times 10^{-3} \\text{ mm}^2/\\text{s}$），我们可以通过边界点求解 $\\alpha$ 和 $\\beta$： - 当 $I_{T2} = 0$ (最暗处，白质下限) 时，$ADC = 0.6 \\times 10^{-3}$ - 当 $I_{T2} = 1$ (最亮处，脑脊液上限) 时，$ADC = 3.0 \\times 10^{-3}$ 由此解得：$\\alpha = 2.4 \\times 10^{-3}$，$\\beta = 0.6 \\times 10^{-3}$。即： $$ADC(x, y) = [2.4 \\times I_{T2}(x, y) + 0.6] \\times 10^{-3}$$

### 2.2.3 T2数据参考和模板

向组长要整理好的采集数据



### 2.3 最后可选：MNI152模板 + 组织分割

用途：

- 提供真实脑形态；
- 提供 WM / GM / CSF tissue masks；
- 用于生成 anatomy-informed 参数图；
- 最适合课程项目。

优点：

- 获取容易；
- 脑形态真实；
- 与 MRI 标准空间一致；
- 可以快速生成 2D 或 3D 模拟 DWI；
- 适合解释给老师。

缺点：

- MNI152 本身是 T1 / anatomical template，不是 DWI；
- 需要自己根据组织类别赋予 D / K 参数；
- 不能直接代表真实个体的扩散参数异质性。

推荐用途：

```text
主模拟数据来源
```

更推荐采用下面的路线：

```text
MNI152 / BrainWeb 真实脑模板或组织分割
        ↓
得到 WM / GM / CSF / brain mask
        ↓
根据组织类别赋予 D / K / S0 参数范围
        ↓
用 ADC 或 DKI 信号模型生成多 b 值 DWI
        ↓
加入 Rician noise、bias field、方向平均退化
        ↓
得到具有真实脑形态 + 已知参数真值的模拟数据
```

---

## 3. 推荐数据组成

最终每个模拟样本保存为：

```text
sample_xxxx.npz
```

内部包括：

```text
dwi_clean: [B, H, W] 或 [B, X, Y, Z]
dwi_noisy: [B, H, W] 或 [B, X, Y, Z]
bvals: [B]
s0_gt: [H, W] 或 [X, Y, Z]
d_gt: [H, W] 或 [X, Y, Z]
k_gt: [H, W] 或 [X, Y, Z]
adc_gt: [H, W] 或 [X, Y, Z]，可选
mask: [H, W] 或 [X, Y, Z]
tissue_label: [H, W] 或 [X, Y, Z]
noise_sigma: scalar
```

其中：

- `d_gt` 是 DKI 中的 diffusion coefficient；
- `k_gt` 是 kurtosis；
- `adc_gt` 可以等于 `d_gt`，作为 ADC baseline 参考；
- `tissue_label` 用于 ROI 评价。

---

## 7. MNI152  模拟数据生成流程

### Step 1：获取脑模板和组织 mask

可选来源：

1. MNI152 / ICBM152 template；
2. MNI152 tissue probability maps；
3. FSL FAST segmentation；
4. BrainWeb tissue model；
5. Nilearn MNI152 GM / WM mask；
6. MNITemplate 包中的 FAST segmentation。

目标是得到：

```text
brain_mask
wm_mask
gm_mask
csf_mask
```

或者：

```text
tissue_label:
0 = background
1 = CSF
2 = GM
3 = WM
```

---

### Step 2：选择 2D slice 或 3D volume

先做 2D：

```text
middle axial slice
```

或者选择多张 slice：

```text
slice z = 45, 50, 55, 60
```

原因：

1. 调试快；
2. 训练快；
3. 可视化清晰；
4. 足够展示方法逻辑。

---

### Step 3：根据组织类型赋予 S0 / D / K 

推荐参数表：

| Tissue | S0 | D 或 ADC | K |
|---|---:|---:|---:|
| WM | 0.85 | 0.70e-3 | 1.0 |
| GM | 1.00 | 0.90e-3 | 0.7 |
| CSF | 1.20 | 2.50e-3 | 0.05 |
| Background | 0 | 0 | 0 |

如果需要模拟病灶，可以在 MNI / BrainWeb 上额外加一个 lesion mask：

| Lesion type | D | K |
|---|---:|---:|
| Low diffusion lesion | 0.45e-3 | 1.5 |
| High diffusion lesion / edema | 1.50e-3 | 0.4 |

注意：病灶可以作为可选增强，不是必须。

---

### Step 4：加入组织内部随机变化

为了避免所有 WM / GM / CSF 内部完全均匀，可以加入轻微随机扰动：

```python
D = D_base * (1 + normal_noise)
K = K_base * (1 + normal_noise)
S0 = S0_base * (1 + normal_noise)
```

推荐扰动：

```text
D: ±5% 到 ±10%
K: ±10% 到 ±20%
S0: ±5% 到 ±10%
```

示例：

```python
D[mask] *= np.random.normal(1.0, 0.05, size=D[mask].shape)
K[mask] *= np.random.normal(1.0, 0.10, size=K[mask].shape)
S0[mask] *= np.random.normal(1.0, 0.05, size=S0[mask].shape)
```

---

### Step 5：加入 bias field

真实 MRI 存在低频强度不均匀。可以加入一个平滑的 bias field：

```python
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
xx, yy = np.meshgrid(x, y)

bias = 1 + 0.10 * xx + 0.08 * yy + 0.05 * xx * yy
S0 = S0 * bias
```

这样生成的 b=0 图像更像真实 MRI。

---

### Step 6：设定 b 值

DKI 推荐：

```python
bvals = np.array([0, 500, 1000, 1500, 2000])
```

可选：

```python
bvals = np.array([0, 250, 500, 1000, 1500, 2000])
```

ADC baseline 可以从其中取：

```python
[0, 1000]
```

---

### Step 7：生成 DKI clean DWI

DKI 信号模型：

$$
S(b)=S_0\exp\left(-bD+\frac{1}{6}b^2D^2K\right)
$$

代码形式：

```python
dwi_clean = []

for b in bvals:
    signal = S0 * np.exp(-b * D + (1.0 / 6.0) * (b ** 2) * (D ** 2) * K)
    dwi_clean.append(signal)

dwi_clean = np.stack(dwi_clean, axis=0)
```

---

### Step 8：加入 Rician noise

```python
sigma = 0.03

n1 = np.random.normal(0, sigma, size=dwi_clean.shape)
n2 = np.random.normal(0, sigma, size=dwi_clean.shape)

dwi_noisy = np.sqrt((dwi_clean + n1) ** 2 + n2 ** 2)
```

推荐噪声水平：

```text
sigma = 0.01, 0.03, 0.05, 0.08
```

---

### Step 9：保存为 npz

```python
np.savez(
    "sample_0001.npz",
    dwi_clean=dwi_clean,
    dwi_noisy=dwi_noisy,
    bvals=bvals,
    s0_gt=S0,
    d_gt=D,
    k_gt=K,
    adc_gt=D,
    mask=brain_mask,
    tissue_label=tissue_label,
    noise_sigma=sigma
)
```

## 8. 推荐DKI实验设计

### 实验 1：T2灰度映射法模拟主实验

比较：

1. traditional DKI fitting；
2. pure CNN；
3. supervised physics-informed network；
4. self-supervised physics-informed network。

指标：

1. D RMSE；
2. K RMSE；
3. D bias；
4. K bias；
5. ROI mean error；
6. ROI CV；
7. signal reconstruction error；
8. invalid parameter ratio。

---

### 实验 2：噪声鲁棒性

设置：

```text
sigma = 0.01, 0.03, 0.05, 0.08
```

比较各方法随噪声增加的性能变化。

---

### 实验 3：方向数 / 平均次数退化

模拟多个 noisy repetition：

```text
N = 30, 12, 6, 3
```

对每个 b 值平均后得到 mean DWI，再拟合或输入网络。

---

## 9. 与真实自采数据的衔接

最终项目可以分成三部分：

```text
Part 1：MNI / BrainWeb 模拟数据
        有 D / K ground truth，用于严格定量评价

Part 2：自采人脑 DWI / DTI 数据
        无 ground truth，用于真实性展示和自监督 fine-tuning
```

这样逻辑非常完整：

```text
模拟数据证明方法有效
人脑数据展示真实应用潜力
```

## 10. 项目目录建议（仅供参考）

```text
ADC_PINN_Project/
│
├── data/
│   ├── templates/
│   │   ├── mni152/
│   ├── reference/T2   
│   │
│   ├── simulated/
│   │   ├── brain_mni/
│   │   └── random_mri/
│   │
│   ├── real/
│   │   ├── human_brain/
│   │   └── random_mri/
│   │
│   └── processed/
│
├── src/
│   ├── load_template.py
│   ├── generate_param_maps.py
│   ├── simulate_dki_signal.py
│   ├── add_noise.py
│   ├── fit_adc.py
│   ├── fit_dki.py
│   ├── models.py
│   ├── losses.py
│   ├── train_supervised.py
│   ├── train_self_supervised.py
│   ├── evaluate.py
│   └── visualize.py
│
├── experiments/
│   ├── exp_noise_robustness/
│   ├── exp_direction_reduction/
│   ├── exp_solution_phantom/
│   └── exp_real_brain_demo/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── checkpoints/
│
└── README.md
```

---

### 参数要限制范围

查阅相关资料查看参数范围需要在真实值范围内，要求相对合理即可
