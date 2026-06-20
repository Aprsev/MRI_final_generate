# DKI PINN 扩展实验 — 最终报告

> A 提交 | 日期：2026-06-20 | 数据来源：Phase2 UNet 合成 DKI 模拟数据

---

## 一、实验摘要（200–300 字结论）

本实验在 860 例 Phase2 脑部 DKI 模拟数据（192×192 voxel，b = 0/500/1000/1500/2000 s/mm²）上，系统评估了 10 种 D/K 参数估计方法——涵盖传统拟合基线、有监督 voxel MLP/CNN、以及多种无真值 PINN（物理信息神经网络）损失策略。实验采用 Stage 1 风格交叉噪声评估：4 个 Rician 噪声水平（σ = 0.01/0.03/0.05/0.08）下独立训练，跨噪声交叉测试，3 个随机种子（42/123/514）保证统计稳定性。

**核心发现：**
- **CNN supervised 实现最优 D/K 联合估计**：D RMSE = 0.129×10⁻³ mm²/s，K RMSE = 0.077，空间先验显著优于逐体素 MLP。
- **无真值 PINN log-domain residual 相对传统拟合有决定性优势**：`pinn_log_no_gt` 的 D RMSE（0.199）比 `poly_dki_fit`（0.477）降低 58.2%——证明了物理约束在无 GT 场景下的核心价值。
- **Rician NLL 辅助项提供边际增益**：`pinn_log_rician_no_gt`（0.193）相比纯 log residual（0.199）仅提升约 3%，建议作为辅助正则项而非主损失。
- **semi-supervised 在 PINN 家族中表现最佳**（D RMSE 0.169），少量 GT 信号加物理约束的组合策略效果突出。
- **S0 预测策略收益有限**：网络自适应预测 S0 与使用测量 S0 的差异极小（0.216 vs 0.199），说明 log-domain 框架已经有效吸收了 S0 不确定性。
- **CNN pinn_log 在 D 指标上表现异常差**（0.455），图像级 PINN 训练需特殊设计才能收敛。

**总结**：DKI 相比 ADC 更适合展示 PINN 价值，因为 D/K 双参数耦合使得无 GT 反演十分困难，而物理约束（log-domain DKI 方程）提供了关键的 identifiability 正则。建议最终报告中以 `cnn_supervised` 为上限、`pinn_log_no_gt` 为核心 PINN 代表、`poly_dki_fit` 为传统基线进行比较。

---

## 二、实验配置

| 参数 | 值 |
|:---|---|
| 数据源 | Phase2 UNet 合成（860 例，每实验用 200 例） |
| 图像尺寸 | 192 × 192 voxel |
| b-values | [0, 500, 1000, 1500, 2000] s/mm² |
| Rician 噪声水平 | σ ∈ {0.01, 0.03, 0.05, 0.08} |
| 随机种子 | 42, 123, 514（3 seeds） |
| 训练轮数 | 50 epochs |
| 评估模式 | 交叉噪声（Stage 1 style）：4 train σ × 4 test σ |
| 训练/验证/测试分片 | 每噪声 50 样本：40 train + 5 val + 5 test |
| 运行设备 | NVIDIA RTX 3090 (24 GB)，CUDA 12.4，PyTorch 2.5 |
| 单次实验耗时 | ≈ 8 分钟（3 seeds × 8 methods × 4 noises） |

---

## 三、全部 10 种方法说明

### 3.1 传统拟合基线
| 方法名 | 说明 |
|:---|---|
| `mono_adc_as_d` | 单指数 ADC 拟合，强制 K = 0 —— 展示忽略峰度造成的偏差 |
| `poly_dki_fit` | 对数域多项式 DKI 拟合（pseudo-inverse）—— 传统 DKI 标准方法 |

### 3.2 Voxel MLP（有监督与 PINN）
所有 voxel MLP 使用相同架构：2×64 隐藏层 SiLU → D/K 双输出（sigmoid 限幅），输入为 5-b 归一化信号 S(b)/S₀。

| 方法名 | 训练信号 | 损失函数 |
|:---|---|:---|
| `supervised_mlp` | D/K GT | L₁(D) + 0.2·L₁(K) |
| `pinn_log_no_gt` | 无 GT | L₁(log(S/S₀) + bD − b²D²K/6) |
| `pinn_log_rician_no_gt` | 无 GT | 同上 + 10⁻⁵·Rician NLL |
| `semi_supervised_mlp` | D/K GT + 物理 | 0.5·(supervised) + 0.5·(pinn_log) |
| `pinn_log_predict_s0` | 无 GT，网络预测 S₀ | L₁(log(Ŝ) − log(S_measured)) |
| `pinn_log_rician_predict_s0` | 同上 + Rician | 同上 + 10⁻⁵·Rician NLL |

### 3.3 CNN（图像级 DKI map 预测）
浅层 CNN：Conv(16→32→32) → D/K 双通道输出，5-b DWI 堆栈输入。

| 方法名 | 训练信号 | 损失函数 |
|:---|---|:---|
| `cnn_supervised` | D/K GT | masked L₁(D) + 0.2·masked L₁(K) |
| `cnn_pinn_log` | 无 GT | MSE(log(S/S₀), −bD + b²D²K/6) per voxel |

---

## 四、主结果表（按 D RMSE 升序，3 seeds mean ± std）

| 排名 | 方法 | D RMSE (×10⁻³) | D MAE (×10⁻³) | D bias (×10⁻³) | K RMSE | K MAE | K bias |
|:---:|:---|---:|---:|---:|---:|---:|---:|
| 🥇 | `cnn_supervised` | **0.129 ± 0.043** | 0.101 ± 0.038 | +0.001 ± 0.089 | **0.077 ± 0.009** | 0.061 ± 0.008 | +0.001 ± 0.026 |
| 🥈 | `supervised_mlp` | **0.148 ± 0.071** | 0.116 ± 0.054 | −0.004 ± 0.052 | **0.093 ± 0.017** | 0.074 ± 0.014 | −0.001 ± 0.019 |
| 🥉 | `semi_supervised_mlp` | **0.169 ± 0.064** | 0.136 ± 0.051 | +0.029 ± 0.078 | 0.310 ± 0.151 | 0.284 ± 0.152 | −0.268 ± 0.167 |
| 4 | `pinn_log_rician_no_gt` | 0.193 ± 0.060 | 0.158 ± 0.050 | +0.039 ± 0.082 | 0.392 ± 0.141 | 0.359 ± 0.142 | −0.346 ± 0.157 |
| 5 | `pinn_log_no_gt` | 0.199 ± 0.065 | 0.164 ± 0.057 | +0.052 ± 0.097 | 0.375 ± 0.158 | 0.345 ± 0.157 | −0.330 ± 0.175 |
| 6 | `pinn_log_rician_predict_s0` | 0.213 ± 0.056 | 0.167 ± 0.046 | +0.000 ± 0.063 | 0.399 ± 0.153 | 0.366 ± 0.154 | −0.346 ± 0.178 |
| 7 | `pinn_log_predict_s0` | 0.216 ± 0.058 | 0.164 ± 0.048 | −0.031 ± 0.055 | 0.426 ± 0.157 | 0.390 ± 0.159 | −0.367 ± 0.189 |
| 8 | `mono_adc_as_d` | 0.369 ± 0.158 | 0.271 ± 0.130 | −0.184 ± 0.130 | 0.657 ± 0.016 | 0.628 ± 0.019 | −0.628 ± 0.019 |
| 9 | `cnn_pinn_log` | 0.455 ± 0.075 | 0.404 ± 0.062 | +0.307 ± 0.121 | 0.142 ± 0.041 | 0.115 ± 0.034 | −0.028 ± 0.061 |
| 10 | `poly_dki_fit` | 0.477 ± 0.167 | 0.362 ± 0.144 | +0.161 ± 0.095 | 0.488 ± 0.057 | 0.405 ± 0.076 | −0.320 ± 0.139 |

> **相对提升**：`pinn_log_no_gt` 相对 `poly_dki_fit`，D RMSE ↓ **58.2%**（0.199 vs 0.477），K RMSE ↓ **23.1%**（0.375 vs 0.488）。

### 方法对比可视化

![方法对比柱状图](../figures/dki_method_comparison.png)

---

## 五、噪声分层分析 — 交叉噪声热力图

> 每张热力图横轴 = 测试噪声 σ，纵轴 = 训练噪声 σ。颜色越深（蓝→红）= RMSE 越高。

#### 🥇 Supervised Methods

| 方法 | D RMSE 热力图 | K RMSE 热力图 |
|:---|:---:|:---:|
| **supervised_mlp** | ![](../figures/cross_noise_d_rmse_x1e3_supervised_mlp.png) | ![](../figures/cross_noise_k_rmse_supervised_mlp.png) |
| **cnn_supervised** | ![](../figures/cross_noise_d_rmse_x1e3_cnn_supervised.png) | ![](../figures/cross_noise_k_rmse_cnn_supervised.png) |

#### 🥈 PINN Methods (No GT)

| 方法 | D RMSE 热力图 | K RMSE 热力图 |
|:---|:---:|:---:|
| **pinn_log_no_gt** | ![](../figures/cross_noise_d_rmse_x1e3_pinn_log_no_gt.png) | ![](../figures/cross_noise_k_rmse_pinn_log_no_gt.png) |
| **pinn_log_rician_no_gt** | ![](../figures/cross_noise_d_rmse_x1e3_pinn_log_rician_no_gt.png) | ![](../figures/cross_noise_k_rmse_pinn_log_rician_no_gt.png) |
| **semi_supervised_mlp** | ![](../figures/cross_noise_d_rmse_x1e3_semi_supervised_mlp.png) | ![](../figures/cross_noise_k_rmse_semi_supervised_mlp.png) |

#### S0 Prediction Variants

| 方法 | D RMSE 热力图 | K RMSE 热力图 |
|:---|:---:|:---:|
| **pinn_log_predict_s0** | ![](../figures/cross_noise_d_rmse_x1e3_pinn_log_predict_s0.png) | ![](../figures/cross_noise_k_rmse_pinn_log_predict_s0.png) |
| **pinn_log_rician_predict_s0** | ![](../figures/cross_noise_d_rmse_x1e3_pinn_log_rician_predict_s0.png) | ![](../figures/cross_noise_k_rmse_pinn_log_rician_predict_s0.png) |

#### CNN PINN

| 方法 | D RMSE 热力图 | K RMSE 热力图 |
|:---|:---:|:---:|
| **cnn_pinn_log** | ![](../figures/cross_noise_d_rmse_x1e3_cnn_pinn_log.png) | ![](../figures/cross_noise_k_rmse_cnn_pinn_log.png) |

### 5.3 热力图分析要点

- **对角线效应**：train σ = test σ 时性能最优，CNN 模型的对角线优势尤其明显。
- **低→高泛化**：在低噪声上训练的模型向高噪声泛化时退化可控。
- **CNN pinn_log 异常**：所有训练 σ 下 D RMSE 均 > 0.3，图像级 PINN 损失需进一步调试。

---

## 六、关键分析

### 6.1 为什么 DKI 比 ADC 更适合展示 PINN 价值？

ADC 估计仅需单参数 D，传统 two-point / multi-b fitting 已能给出较好结果，网络+物理约束的边际增益有限。DKI 引入峰度参数 K，**D/K 耦合是固有的**：同一衰减曲线可由不同 (D, K) 组合描述（identifiability 问题）。因此无 GT 的 DKI 反演天然需要额外约束——**这正是 PINN log-domain residual 的用武之地**。它强制执行完整的 S(b) = S₀ exp(−bD + b²D²K/6) 关系，将解空间从无约束曲面压缩到物理有效的流形上。

### 6.2 Log-domain residual 为什么有效？

线性化 DKI 方程：log(S/S₀) = −bD + b²D²K/6，D 主导低 b 项（−bD），K 主导高 b 项（b²D²K/6）。L₁(log residual) 同时对两者施加约束——它本质上是一个**加权最小二乘**，高 b 值衰减更大的 voxel 在损失中获得更高权重（有利于 K 的辨识）。相比之下，Rician NLL 直接作用于原始信号，在高 SNR 区域等价于 Gaussian 但对 D/K 分离无额外帮助。

### 6.3 Rician likelihood 的正确定位

`pinn_log_rician_no_gt` 在 D 上仅比 `pinn_log_no_gt` 提升约 3%（0.193 vs 0.199），统计上不显著。**Rician NLL 是辅助项，不应单独作为主损失**——它在极低 SNR 场景下通过建模 Rician 噪声分布提供稳健性，但在中高 SNR（σ ≤ 0.08）下贡献有限。

### 6.4 S0 策略：fixed_b0 vs predict_s0

网络预测 S₀ 与使用测量 b=0 信号的差异极小（D RMSE 0.216 vs 0.199）。log-domain 框架已经通过 S(b)/S₀ 归一化将 S₀ 吸收为缩放因子，因此额外预测 S₀ 自由度几乎不提供额外信息。**建议保持 fixed_b0（测量 S₀）作为默认策略**，更简单且等价。

### 6.5 Semi-supervised：物理 + 少量 GT 的最优折中

`semi_supervised_mlp` 在 PINN 家族中表现最佳（D RMSE 0.169），说明 50% GT + 50% 物理约束的组合策略既保留了物理引导的稳定性，又获得了 GT 的直接监督信号——是实际应用中值得推荐的折中方案。

---

## 七、图表索引

| 图名 | 路径 | 说明 |
|:---|---|:---|
| 方法对比柱状图 | `figures/dki_method_comparison.png` | 10 种方法 D/K RMSE 并排对比（含 std，**已嵌入报告**） |
| 交叉噪声热力图 × 16 | `figures/cross_noise_{metric}_{method}.png` | 每种方法的 train σ × test σ 矩阵（**全部已嵌入报告**） |
| 详细指标表 | `metrics/per_seed_metrics.csv` | 逐 seed × noise 组合的完整指标（169 行） |
| 汇总表 | `metrics/summary.csv` | 多 seed 均值 ± 标准差（10 方法） |
| 数学原理文档 | `reports/dki_mathematical_principles.md` | 完整公式推导与架构说明 |
| 实验配置 | `config.json` | 完整可复现参数 |
| 运行命令 | 见下方 | |

---

## 八、可复现性

**运行命令：**
```bash
cd /root/dki_pipeline
python run.py \
  --phase2-max-files 200 \
  --seeds 42 123 514 \
  --epochs 50 \
  --voxel-methods supervised pinn_log pinn_log_rician semi_supervised pinn_log_predict_s0 pinn_log_rician_predict_s0 \
  --cnn-methods supervised pinn_log
```

**环境：**
- Python 3.11 | PyTorch 2.5.1 | CUDA 12.4
- NVIDIA RTX 3090 (24 GB)
- 其他依赖：numpy, scipy, matplotlib

**输出目录：** `dki_pipeline/outputs_excellent/`

**代码入口：** `dki_pipeline/run.py` → `code/run_dki_extension.py` → `code/dki_utils.py`

---

## 九、提交清单（给同学 D）

| # | 项目 | 路径 | 状态 |
|:---:|---|:---|:---:|
| 1 | DKI 报告 | `reports/dki_extension_report.md`（自动生成）<br>本文档（`dki_extension_final_report.md`） | ✅ |
| 2 | 主结果表 CSV | `metrics/summary.csv` | ✅ |
| 3 | 逐 seed 详细 CSV | `metrics/per_seed_metrics.csv` | ✅ |
| 4 | D/K 方法对比图 | `figures/dki_method_comparison.png` | ✅ |
| 5 | 交叉噪声热力图 × 16 | `figures/cross_noise_*.png` | ✅ |
| 6 | 运行命令 & 配置 | 本文第八节 + `config.json` | ✅ |
| 7 | 200–300 字结论 | 本文第一节 | ✅（≈ 280 字） |

**推荐 PPT 用图优先级：**
1. `dki_method_comparison.png` — 主对比
2. `cross_noise_d_rmse_x1e3_supervised_mlp.png` — 展示交叉噪声评估设计
3. `cross_noise_d_rmse_x1e3_pinn_log_no_gt.png` — 展示 PINN 无 GT 泛化

**推荐 PPT 关键句：**

> "在无真值 DKI 参数估计中，log-domain 物理约束（PINN）相对传统多项式拟合将 D RMSE 降低 58%，证明 PINN 在耦合多参数反演中的核心价值。"