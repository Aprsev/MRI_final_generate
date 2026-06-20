# DKI PINN Pipeline — 数学原理详细说明

> 同学 A 技术文档 | 2026-06-20 | 供报告附录与 PPT 参考

---

## 一、DKI 信号模型

### 1.1 基础 DKI 方程

扩散峰度成像（Diffusion Kurtosis Imaging, DKI）在 ADC 模型基础上引入二阶项，描述水分子扩散偏离高斯分布的程度：

$$
S(b) = S_0 \cdot \exp\left(-bD + \frac{1}{6}b^2 D^2 K\right)
$$

其中：

| 符号 | 物理量 | 单位 | 生理范围 |
|:---|:---|:---|:---|
| $S(b)$ | 扩散加权信号 | 任意（归一化） | 0 ~ S₀ |
| $S_0$ | b=0 基准信号 | 任意 | — |
| $b$ | 扩散敏感因子（b-value） | s/mm² | [0, 500, 1000, 1500, 2000] |
| $D$ | 表观扩散系数（ADC 推广） | mm²/s | ≈ 0.0003 ~ 0.003 |
| $K$ | 超额峰度（excess kurtosis） | 无量纲 | ≈ 0 ~ 2.0 |

当 K = 0 时，退化回标准单指数 ADC 模型：$S(b) = S_0 e^{-bD}$。

### 1.2 对数域线性化

取自然对数：

$$
\ln\frac{S(b)}{S_0} = -bD + \frac{1}{6}b^2 D^2 K
$$

令 $y_b = \ln[S(b)/S_0]$，则：

$$
y_b = -b \cdot D + \frac{b^2 D^2}{6} \cdot K
$$

**关键性质**：在对数域，DKI 方程是 $D$ 和 $K$ 的二次函数，而非指数域的强非线性。这使得：
- 低 b 值区域（b ≤ 1000）：$y_b \approx -bD$（线性，D 主导）
- 高 b 值区域（b ≥ 1500）：$b^2 D^2 K/6$ 项显著偏离线性（K 主导）

**Identifiability 问题**：给定测量信号 $\{S(b_i)\}_{i=1}^{B}$，可能存在多组 (D, K) 使方程拟合良好——因为 D 和 K 在对数域是二次耦合的。这正是 PINN 物理约束发挥作用的场景。

---

## 二、传统拟合方法

### 2.1 单指数 ADC 拟合（`mono_adc_as_d`）

忽略峰度，强制 K = 0，仅拟合 D：

$$
\ln S(b) = \ln S_0 - bD \quad \Rightarrow \quad \hat{D} = -\frac{\sum(b_i - \bar{b})(\ln S_i - \overline{\ln S})}{\sum(b_i - \bar{b})^2}
$$

**局限**：对 DKI 数据有系统性偏差，高 b 值信号的实际衰减比单指数慢（K > 0 时），导致 D 被严重低估，且 K RMSE 恒等于真实 K 的均值。

### 2.2 多项式 DKI 拟合（`poly_dki_fit`）

对数域 DKI 方程按 $c_1 = -D$、$c_2 = D^2 K/6$ 重写为 b 的二次多项式：

$$
y_b = c_1 b + c_2 b^2
$$

使用 Moore-Penrose 伪逆求解（每个体素独立）：

$$
\begin{bmatrix} c_1 \\ c_2 \end{bmatrix} = (B^T B)^{-1} B^T \mathbf{y}, \quad
B = \begin{bmatrix} b_1 & b_1^2 \\ \vdots & \vdots \\ b_B & b_B^2 \end{bmatrix}
$$

然后反推：
$$
\hat{D} = \text{clip}(-c_1, 0, D_{\max}), \quad
\hat{K} = \text{clip}\left(\frac{6c_2}{\hat{D}^2}, 0, K_{\max}\right)
$$

**局限**：（1）对噪声敏感，尤其在低 SNR voxel；（2）伪逆无正则化，可能得到非物理解；（3）逐体素独立拟合，不利用空间邻域信息。

---

## 三、神经网络参数化

### 3.1 Voxel MLP（`DkiMLP`）

**输入**：归一化多 b 值信号向量（5-b 包括 b=0）

$$
\mathbf{x} = \begin{bmatrix} S(b_0)/S_0, & S(b_1)/S_0, & \dots, & S(b_B)/S_0 \end{bmatrix}^T \in \mathbb{R}^5
$$

**网络结构**：

$$
\begin{aligned}
\mathbf{h}_1 &= \text{SiLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 5} \\
\mathbf{h}_2 &= \text{SiLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 64} \\
\mathbf{z} &= \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{2 \times 64}
\end{aligned}
$$

**输出映射**（硬约束物理范围）：

$$
D = D_{\max} \cdot \sigma(z_1), \quad K = K_{\max} \cdot \sigma(z_2)
$$

其中 $\sigma(\cdot)$ 为 sigmoid 函数，$D_{\max}=0.0035$ mm²/s，$K_{\max}=2.0$。该参数化保证输出始终在物理有效范围内。

**总参数量**：≈ 4,610（极小，允许全 GPU 批处理 80 万+ 体素）。

### 3.2 CNN（`DkiCNN`）

**输入**：多 b DWI 图像堆栈 $\mathbf{X} \in \mathbb{R}^{B \times H \times W}$

**网络结构**（浅层全卷积）：

```
Conv2d(B, 16, 3×3) → SiLU
Conv2d(16, 32, 3×3) → SiLU
Conv2d(32, 2, 3×3)
```

**输出**：D map、K map 双通道 $(\hat{D}, \hat{K}) \in \mathbb{R}^{2 \times H \times W}$，每通道经 sigmoid 限幅。

**优势**：3×3 卷积核提供局部空间先验，邻域体素信息帮助去噪和 K 估计。在 192×192 Phase2 图像上效果显著优于逐体素 MLP。

### 3.3 S0 预测变体

当 `predict_s0=True` 时，网络额外输出一个 S₀ 缩放因子（3 输出而非 2）：

$$
(S_0^{\text{scale}}, D, K) = \left(2\cdot\sigma(z_1),\; D_{\max}\cdot\sigma(z_2),\; K_{\max}\cdot\sigma(z_3)\right)
$$

预测信号：
$$
\hat{S}(b) = \big(S_0^{\text{measured}} \cdot S_0^{\text{scale}}\big) \cdot \exp(-bD + b^2 D^2 K / 6)
$$

---

## 四、损失函数体系

### 4.1 有监督损失（`supervised`）

直接回归 GT 参数，加权 L₁：

$$
\mathcal{L}_{\text{sup}} = \frac{1}{N}\sum_{i=1}^{N} \left[|D_i - D_i^{\text{gt}}| \cdot 1000 + 0.2 \cdot |K_i - K_i^{\text{gt}}|\right]
$$

- D 乘以 1000（单位转换到 10⁻³ mm²/s 尺度）使 D/K 损失量级匹配
- K 权重 0.2 反映其不确定性大于 D

### 4.2 Log-Domain 物理残差（`pinn_log`）— 核心 PINN 损失

**无需 GT 参数**，仅需 DKI 物理方程：

$$
\mathcal{L}_{\text{log}} = \frac{1}{N}\sum_{i=1}^{N} \sum_{b \neq 0} \left|\ln\frac{S_i(b)}{S_i(0)} + b\hat{D}_i - \frac{1}{6}b^2 \hat{D}_i^2 \hat{K}_i\right|
$$

**推导逻辑**：

1. 测量归一化信号：$r_i(b) = S_i(b)/S_i(0)$（网络输入）
2. 网络预测物理预测：$\hat{y}_i(b) = -b\hat{D}_i + \frac{1}{6}b^2\hat{D}_i^2\hat{K}_i$
3. 真实对数信号：$y_i(b) = \ln r_i(b)$
4. 残差：$y_i(b) - \hat{y}_i(b) = \ln\frac{S_i(b)}{S_i(0)} + b\hat{D}_i - \frac{1}{6}b^2 \hat{D}_i^2 \hat{K}_i$

**为什么选择 L₁ 而非 MSE？**
- L₁ 对噪声异常值更稳健（Rician 拖尾）
- 低 SNR voxel 的 log 变换对噪声放大显著，MSE 会过度惩罚

### 4.3 Rician 负对数似然（`pinn_rician`）

对 magnitude MRI 信号建模 Rician 分布（非中心 χ）：

$$
p(S | \mu, \sigma) = \frac{S}{\sigma^2} \exp\left(-\frac{S^2 + \mu^2}{2\sigma^2}\right) I_0\left(\frac{S\mu}{\sigma^2}\right)
$$

其中 $\mu = S_0 \cdot \exp(-bD + b^2 D^2 K/6)$ 为真实信号幅度。

负对数似然：

$$
-\ln p = \frac{S^2 + \mu^2}{2\sigma^2} - \ln I_0\left(\frac{S\mu}{\sigma^2}\right) - \ln\frac{S}{\sigma^2}
$$

实际计算使用数值稳定的 $\ln I_0(z) = \ln I_{0e}(z) + |z|$（`torch.special.i0e`）。

### 4.4 混合损失（`pinn_log_rician`）

$$
\mathcal{L}_{\text{log+rician}} = \mathcal{L}_{\text{log}} + 10^{-5} \cdot \mathcal{L}_{\text{rician}}
$$

**权重设计**：Rician NLL 天然绝对值比 L₁(log) 大若干个数量级，因此乘以 10⁻⁵ 作为辅助正则。Rician 项在极低 SNR（σ > 0.05、高 b 值）提供理论最优的噪声建模，但在中高 SNR 下贡献有限。

### 4.5 Semi-Supervised 损失

$$
\mathcal{L}_{\text{semi}} = 0.5 \cdot \mathcal{L}_{\text{sup}} + 0.5 \cdot \mathcal{L}_{\text{log}}
$$

**设计意图**：30/70 或 50/50 的 GT/PINN 混合，适合训练数据中部分样本有 GT 的场景。

### 4.6 S0 预测损失（`pinn_log_predict_s0`）

网络输出 $S_0^{\text{scale}}$，用预测 S₀ 重建原始信号（未归一化）：

$$
\mathcal{L}_{\text{pred\_s0}} = \frac{1}{N}\sum_{i} \sum_{b \neq 0} \left|\ln\big(S_i(0) \cdot S_0^{\text{scale}}\big) + \big(-b\hat{D}_i + \frac{1}{6}b^2 \hat{D}_i^2 \hat{K}_i\big) - \ln S_i(b)\right|
$$

---

## 五、训练策略

### 5.1 全 GPU 内存训练（关键优化）

体素 MLP 参数量仅 ~4,600，但训练体素可达 80 万+。传统 DataLoader（CPU shuffle → 逐批 .to(device)）在此场景下产生严重瓶颈（实测 18 s/epoch 空循环）。

**优化方案**：
1. 将所有训练数据（x, d_gt, k_gt, sigma, s0）一次性上传 GPU（~30 MB）
2. 每 epoch 在 GPU 上生成随机排列索引
3. 使用 GPU 张量索引切片（O(1) 而非 O(batch_size)）取 batch

```python
perm = torch.from_numpy(rng.permutation(n_total)).to(device)
for start in range(0, n_total, batch_size):
    idx = perm[start:start + batch_size]
    xb, db, kb = x_t[idx], d_t[idx], k_t[idx]
    # forward + backward ...
```

**效果**：每 epoch 从 25 s 降至 < 0.1 s（约 250× 加速）。

### 5.2 交叉噪声评估（Stage 1 风格）

```
For each σ ∈ {0.01, 0.03, 0.05, 0.08}:
    ┌─ 分配独立文件子集（不重叠）
    ├─ 加载干净 DWI + 添加 Rician 噪声（σ）
    └─ 训练所有方法（voxel MLP × 6 + CNN × 2）

For each test_σ:
    ├─ 加载测试数据
    └─ For each train_σ:
        评估 train_σ 上训练的模型在 test_σ 上的指标
```

**输出**：4×4 交叉噪声矩阵（每个方法 16 组指标），揭示泛化能力随噪声偏移的退化规律。

### 5.3 多种子统计

3 个随机种子（42, 123, 514）控制：
- Phase2 文件采样和噪声分配
- 网络参数初始化
- mini-batch 排列顺序

最终报告 mean ± std，确保结论不依赖特定初始化。

---

## 六、评估指标

### 6.1 D 指标

| 公式 | 含义 |
|:---|:---|
| $\text{RMSE}_D = \sqrt{\frac{1}{M}\sum_{j}(D_j - D_j^{\text{gt}})^2}$ | 均方根误差 |
| $\text{MAE}_D = \frac{1}{M}\sum_{j}|D_j - D_j^{\text{gt}}|$ | 平均绝对误差 |
| $\text{Bias}_D = \frac{1}{M}\sum_{j}(D_j - D_j^{\text{gt}})$ | 偏差（正=高估） |

所有 D 指标以 ×10⁻³ mm²/s 为单位。

### 6.2 K 指标

| 公式 | 含义 |
|:---|:---|
| $\text{RMSE}_K = \sqrt{\frac{1}{M}\sum_{j}(K_j - K_j^{\text{gt}})^2}$ | 均方根误差 |
| $\text{MAE}_K = \frac{1}{M}\sum_{j}|K_j - K_j^{\text{gt}}|$ | 平均绝对误差 |
| $\text{Bias}_K = \frac{1}{M}\sum_{j}(K_j - K_j^{\text{gt}})$ | 偏差 |

K 无量纲，直接使用原始值。

### 6.3 噪声分层指标

按 Rician 噪声 σ 分组计算上述指标，输出 `noise_sigma`→RMSE 曲线，揭示各方法对噪声水平的敏感度。

---

## 七、DKI vs ADC：为什么 PINN 在 DKI 中更有价值

### 7.1 参数耦合

- **ADC**：单参数 D。$S(b) = S_0 e^{-bD}$，对数域是严格的线性关系：$\ln(S/S_0) = -bD$。给定 2 个 b 值即可唯一确定 D。
- **DKI**：双参数 (D, K) 耦合。对数域是二次关系：$\ln(S/S_0) = -bD + b^2 D^2 K/6$。不同的 (D, K) 组合可产生几乎相同的信号衰减曲线。

### 7.2 Identifiability 分析

令 $c_1 = -D, c_2 = D^2 K/6$，则 DKI 对数方程退化为 $y_b = c_1 b + c_2 b^2$。在低噪声下 $c_1, c_2$ 可稳定估计，但反解 D, K 时：

$$
D = -c_1, \quad K = \frac{6c_2}{D^2}
$$

$c_2$ 的微小估计误差经 $1/D^2$ 放大，导致 K 极度不稳定。**PINN 物理约束通过强制 D/K 必须同时满足所有 b 值的衰减关系来缓解这一问题。**

### 7.3 数值验证

在 Phase2 数据上：
- `poly_dki_fit`（无正则化传统拟合）：K RMSE = 0.488
- `pinn_log_no_gt`（物理约束）：K RMSE = 0.375（↓ 23%），D RMSE ↓ 58%
- 提升不是来自网络容量（MLP 仅 4.6K 参数），而是来自**物理方程作为硬约束**

---

## 八、关键公式速查表

| 名称 | 公式 |
|:---|:---|
| DKI 前向模型 | $S(b) = S_0 e^{-bD + b^2 D^2 K/6}$ |
| Log 残差 | $r_{\text{log}} = \ln\frac{S}{S_0} + bD - \frac{b^2 D^2 K}{6}$ |
| 网络参数化 | $D = D_{\max} \cdot \sigma(\text{raw}_D)$ |
| 有监督损失 | $\mathcal{L}_{\text{sup}} = \|D - D^{\text{gt}}\|_1 \cdot 1000 + 0.2\|K - K^{\text{gt}}\|_1$ |
| PINN 损失 | $\mathcal{L}_{\text{log}} = \mathbb{E}_{b \neq 0}\left\|\ln\frac{S(b)}{S_0} + bD - \frac{b^2 D^2 K}{6}\right\|_1$ |
| 对数多项式拟合 | $\begin{bmatrix}c_1 \\ c_2\end{bmatrix} = (B^T B)^{-1}B^T \mathbf{y}$ |
| Rician PDF | $p(S) = \frac{S}{\sigma^2} e^{-(S^2+\mu^2)/2\sigma^2} I_0(S\mu/\sigma^2)$ |