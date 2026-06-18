#!/usr/bin/env python3
"""
t2_grayscale_mapping_pipeline.py
=================================
对应 README 2.2 节 —— T2引导的连续灰度映射法

数据来源：MNI152 T2 模板 (方案A)
  mni_icbm152_t2_tal_nlin_sym_09a.mnc

流程：
  1. 从MNI152 T2模板提取2D切片
  2. 灰度归一化 → 线性映射 → ADC Ground Truth
  3. 从T2推导 S0、K 参数图
  4. ADC单指数模型生成多b值DWI (S(b)=S0*exp(-b*D)，不含K项)
  5. 添加Rician噪声
  6. 保存为标准 npz 格式
  7. 可视化结果
"""

import numpy as np
import os, sys, glob, argparse
from scipy.ndimage import gaussian_filter, zoom

# ============================================================
# 配置
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MNI152模板位于: final/MRI_analog_data/MRI_analog_data/data/templates/...
MRI_ANALOG_DIR = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data')
TEMPLATE_DIR = os.path.join(MRI_ANALOG_DIR, 'data', 'templates',
                            'mni_icbm152_nlin_sym_09a_minc1')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output_grayscale')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 0: 加载MNI152 T2 模板
# ============================================================

def load_mni152_t2():
    """加载MNI152 T2模板，从T2灰度自身生成mask（不依赖外部mask文件）"""
    import nibabel as nib
    from scipy.ndimage import binary_fill_holes, label as ndimage_label

    t2_file = os.path.join(TEMPLATE_DIR, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')
    if not os.path.exists(t2_file):
        raise FileNotFoundError(f'T2模板文件不存在: {t2_file}')

    print(f'加载MNI152 T2模板: {os.path.basename(t2_file)}')
    t2_img = nib.load(t2_file)
    t2_data = t2_img.get_fdata().astype(np.float64)

    # 从T2灰度阈值自动生成mask
    threshold = t2_data.max() * 0.15
    mask_data = t2_data > threshold
    mask_data = binary_fill_holes(mask_data)
    # 保留最大连通域
    labeled, nf = ndimage_label(mask_data)
    if nf > 0:
        sizes = np.bincount(labeled.ravel())
        mask_data = labeled == np.argmax(sizes[1:]) + 1

    print(f'  形状: {t2_data.shape}')
    print(f'  T2范围: [{t2_data.min():.2f}, {t2_data.max():.2f}]')
    print(f'  mask阈值: >{threshold:.1f}, 脑区: {mask_data.sum()} / {mask_data.size}')

    # mask外的T2值置零
    t2_data[~mask_data] = 0.0

    return t2_data, mask_data

def extract_slice(volume, axis=2, idx=None):
    """从3D体积中提取2D切片"""
    if idx is None:
        idx = volume.shape[axis] // 2
    idx = int(idx)
    if axis == 0:
        slice_2d = volume[idx, :, :]
    elif axis == 1:
        slice_2d = volume[:, idx, :]
    else:
        slice_2d = volume[:, :, idx]
    return slice_2d, idx

def extract_slice_safe(t2_vol, mask_vol, axis=2, idx=None):
    """同时提取T2和mask的切片，索引越界时自动裁剪"""
    max_idx = t2_vol.shape[axis] - 1
    if idx is None:
        idx = t2_vol.shape[axis] // 2
    idx = max(0, min(int(idx), max_idx))
    t2_slice, _ = extract_slice(t2_vol, axis, idx)
    mask_slice, _ = extract_slice(mask_vol, axis, idx)
    return t2_slice, mask_slice

def resize_to(slice_2d, target_size=(192, 192)):
    """缩放图像到目标尺寸"""
    if slice_2d.shape == target_size:
        return slice_2d
    factors = (target_size[0] / slice_2d.shape[0],
               target_size[1] / slice_2d.shape[1])
    return zoom(slice_2d, factors, order=1)

# ============================================================
# Step 1: T2灰度 → ADC 映射（核心公式）
# ============================================================

def normalize_image(img, mask=None):
    """归一化到[0,1]，如果给mask则仅在mask内归一化"""
    if mask is None:
        mask = img > img.max() * 0.01  # 自动阈值去背景
    img_masked = img.copy()
    vmin = img[mask].min()
    vmax = img[mask].max()
    img_masked[~mask] = vmin
    img_norm = (img_masked - vmin) / (vmax - vmin + 1e-10)
    img_norm[~mask] = 0.0
    return np.clip(img_norm, 0, 1), mask

def t2_to_adc(t2_norm, alpha=2.4e-3, beta=0.6e-3):
    """
    [核心公式] ADC = alpha * I_T2 + beta
    边界: I=0(WM) -> 0.6e-3, I=1(CSF) -> 3.0e-3
    """
    return alpha * t2_norm + beta

# ============================================================
# Step 2: 从T2推导 S0, K 参数图
# ============================================================

def derive_s0_from_t2(t2_norm, mask, s0_wm=0.85, s0_gm=1.00, s0_csf=1.20):
    """
    T2亮度与S0有一定相关性：CSF最亮->S0最高，WM最暗->S0最低
    用线性映射近似
    """
    s0 = s0_wm + (s0_csf - s0_wm) * t2_norm
    s0[~mask] = 0.0
    return s0

def derive_k_from_t2(t2_norm, mask, k_wm=1.0, k_gm=0.7, k_csf=0.05):
    """
    T2亮度与K成反比：WM暗->K高，CSF亮->K低
    映射: K = k_wm + (k_csf - k_wm) * t2_norm = k_wm - (k_wm - k_csf) * t2_norm
    """
    k = k_wm - (k_wm - k_csf) * t2_norm  # 反向映射
    k[~mask] = 0.0
    return k

def add_internal_variation(param_map, mask, noise_std=0.05):
    """在mask区域内添加轻微随机扰动"""
    noise = np.random.normal(1.0, noise_std, size=param_map.shape)
    param_map[mask] *= noise[mask]
    return param_map

def add_bias_field(img, mask, coeffs=None):
    """添加低频偏置场"""
    if coeffs is None:
        coeffs = [0.10, 0.08, 0.05]
    H, W = img.shape
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    bias = 1 + coeffs[0]*xx + coeffs[1]*yy + coeffs[2]*xx*yy
    result = img * bias
    result[~mask] = 0.0
    return result, bias

# ============================================================
# Step 3: ADC 单指数前向生成（简单模型，不使用K项）
# ============================================================

def generate_adc_dwi(s0_map, d_map, bvals, mask):
    """
    ADC单指数信号模型: S(b) = S0 * exp(-b*D)
    返回: dwi_clean: [B, H, W]
    （注意：K参数图k_map虽然保留在npz中供后续DKI使用，但不参与此处DWI生成）
    """
    B = len(bvals)
    H, W = s0_map.shape
    dwi = np.zeros((B, H, W), dtype=np.float32)

    for i, b in enumerate(bvals):
        if b == 0:
            dwi[i] = s0_map
        else:
            dwi[i] = s0_map * np.exp(-b * d_map)
        dwi[i, ~mask] = 0.0

    return dwi

def add_rician_noise(dwi_clean, sigma=0.03):
    """Rician噪声: S_noisy = sqrt((S_clean+n1)^2 + n2^2)"""
    n1 = np.random.normal(0, sigma, size=dwi_clean.shape)
    n2 = np.random.normal(0, sigma, size=dwi_clean.shape)
    return np.sqrt((dwi_clean + n1)**2 + n2**2).astype(np.float32)

# ============================================================
# Step 4: 生成近似组织标签（用于兼容npz格式）
# ============================================================

def generate_tissue_labels(t2_norm, mask):
    """
    从T2灰度阈值生成近似组织标签（仅用于格式兼容）
    0=BG, 1=CSF, 2=GM, 3=WM
    T2: CSF最亮(>0.7), GM中等(0.35~0.7), WM最暗(<0.35)
    """
    label = np.zeros(t2_norm.shape, dtype=np.int32)
    label[mask & (t2_norm > 0.70)] = 1  # CSF
    label[mask & (t2_norm > 0.35) & (t2_norm <= 0.70)] = 2  # GM
    label[mask & (t2_norm <= 0.35)] = 3  # WM
    return label

# ============================================================
# Step 5: 可视化
# ============================================================

def visualize_sample_multi_b(t2_slice, adc_gt, s0_map, k_map,
                              dwi_clean, dwi_noisy, bvals, mask,
                              save_path):
    """多b值可视化：每行显示一个b值的clean和noisy对比"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_b = len(bvals)
    fig, axes = plt.subplots(2, n_b, figsize=(n_b * 2.5, 5))

    for i, b in enumerate(bvals):
        clean = dwi_clean[i] * mask
        noisy = dwi_noisy[i] * mask

        axes[0, i].imshow(clean, cmap='gray')
        axes[0, i].set_title(f'Clean b={b}')
        axes[0, i].axis('off')

        axes[1, i].imshow(noisy, cmap='gray')
        axes[1, i].set_title(f'Noisy b={b}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 主流水线
# ============================================================

def generate_sample(t2_slice, brain_mask_slice, slice_idx, axis_name, target_size=(192, 192),
                    bvals=None, noise_sigma=0.03, add_perturbation=True,
                    add_bias=True, seed=None):
    """从单个T2切片生成一个完整样本"""
    if bvals is None:
        bvals = np.array([0, 500, 1000, 1500, 2000], dtype=np.int32)
    if seed is not None:
        np.random.seed(seed)

    # 1. 预处理：缩放并对齐
    t2_resized = resize_to(t2_slice, target_size)
    mask_resized = resize_to(brain_mask_slice.astype(float), target_size) > 0.5
    # 轻微平滑去噪（仅在mask内）
    t2_smoothed = gaussian_filter(t2_resized, sigma=1.0)
    t2_smoothed[~mask_resized] = 0.0

    brain_mask = mask_resized

    # 2. 归一化（仅在脑区内）
    t2_norm, _ = normalize_image(t2_smoothed, brain_mask)

    # 3. T2 -> ADC (核心!)
    adc_gt = t2_to_adc(t2_norm)
    adc_gt[~brain_mask] = 0.0

    # 4. 推导S0和K
    s0_map = derive_s0_from_t2(t2_norm, brain_mask)
    k_map = derive_k_from_t2(t2_norm, brain_mask)

    # 5. 添加组织内部扰动
    if add_perturbation:
        s0_map = add_internal_variation(s0_map, brain_mask, noise_std=0.05)
        k_map = add_internal_variation(k_map, brain_mask, noise_std=0.10)
        adc_noisy = add_internal_variation(adc_gt.copy(), brain_mask, noise_std=0.03)
    else:
        adc_noisy = adc_gt.copy()

    # 6. 添加偏置场
    if add_bias:
        s0_map, _ = add_bias_field(s0_map, brain_mask)

    # 7. ADC单指数前向生成（简单模型，不使用K项）
    dwi_clean = generate_adc_dwi(s0_map, adc_noisy, bvals, brain_mask)

    # 8. Rician噪声
    dwi_noisy = add_rician_noise(dwi_clean, sigma=noise_sigma)

    # 9. 生成组织标签（格式兼容）
    tissue_label = generate_tissue_labels(t2_norm, brain_mask)

    # 10. 组装结果
    description = (
        '=== DWI模拟数据 npz 字段说明 ===\n'
        '  dwi_clean  [float32, shape=(B,H,W)]  ADC单指数前向无噪声DWI, B=b值个数\n'
        '  dwi_noisy  [float32, shape=(B,H,W)]  添加Rician噪声后的DWI\n'
        '  bvals      [int32,   shape=(B,)]      b值序列 (s/mm^2)\n'
        '  s0_gt      [float32, shape=(H,W)]     S0基图 (b=0信号), 含偏置场/扰动\n'
        '  d_gt       [float32, shape=(H,W)]     扩散系数图 D (mm^2/s), ADC加扰动后\n'
        '  k_gt       [float32, shape=(H,W)]     峰度系数图 K (无单位), 含扰动 (保留供后续DKI使用)\n'
        '  adc_gt     [float32, shape=(H,W)]     ADC真值 (无扰动), 用于对比\n'
        '  mask       [bool,    shape=(H,W)]     脑区mask (True=脑组织)\n'
        '  tissue_label [int32, shape=(H,W)]     组织标签: 0=BG 1=CSF 2=GM 3=WM\n'
        '  noise_sigma [float32]                 Rician噪声标准差\n'
        '  slice_idx   [int32]                   MNI152原始切片索引\n'
        '  source      [str]                     数据来源标识\n'
        '  description [str]                     本说明字段\n'
        '---\n'
        '生成管线: t2_grayscale_mapping_pipeline.py (MNI152 T2 灰度映射法)\n'
        'ADC模型: S(b)=S0*exp(-b*D)  (单指数模型，不含K项)\n'
        'ADC映射: ADC = 2.4e-3 * I_T2 + 0.6e-3,  I=0(WM)->0.6e-3, I=1(CSF)->3.0e-3\n'
        f'目标尺寸: {target_size}, 噪声水平: {noise_sigma:.3f}\n'
    )
    sample = {
        'dwi_clean': dwi_clean.astype(np.float32),
        'dwi_noisy': dwi_noisy.astype(np.float32),
        'bvals': bvals.astype(np.int32),
        's0_gt': s0_map.astype(np.float32),
        'd_gt': adc_noisy.astype(np.float32),
        'k_gt': k_map.astype(np.float32),
        'adc_gt': adc_gt.astype(np.float32),
        'mask': brain_mask,
        'tissue_label': tissue_label,
        'noise_sigma': np.float32(noise_sigma),
        'slice_idx': np.int32(slice_idx),
        'source': 'mni152_t2_grayscale_mapping',
        'description': description,
    }

    return sample

def main():
    parser = argparse.ArgumentParser(description='T2灰度映射法生成DKI模拟数据')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='生成的样本数量')
    parser.add_argument('--slice_axis', type=int, default=2,
                        help='切片轴: 0=矢状, 1=冠状, 2=轴位')
    parser.add_argument('--slice_start', type=int, default=None,
                        help='起始切片索引 (默认自动选择脑区范围)')
    parser.add_argument('--slice_end', type=int, default=None,
                        help='结束切片索引')
    parser.add_argument('--target_size', type=int, nargs=2, default=[192, 192],
                        help='输出图像尺寸 (H W)')
    parser.add_argument('--noise_sigma', type=float, default=0.03,
                        help='Rician噪声标准差')
    parser.add_argument('--no_perturb', action='store_true',
                        help='关闭组织内部扰动')
    parser.add_argument('--no_bias', action='store_true',
                        help='关闭偏置场')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    target_size = tuple(args.target_size)
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('  T2灰度映射法 DKI 模拟数据生成 (方案A: MNI152 T2)')
    print('=' * 60)
    print(f'  输出目录: {output_dir}')
    print(f'  目标尺寸: {target_size}')
    print(f'  切片轴: {args.slice_axis}')
    print(f'  样本数: {args.n_samples}')

    # 加载T2模板（mask从T2灰度自身生成）
    t2_volume, mask_volume = load_mni152_t2()
    n_slices = t2_volume.shape[args.slice_axis]

    # 自动选择有脑组织的切片范围（面积阈值过滤，去掉两头只有零星像素的切片）
    axis_name = ['矢状', '冠状', '轴位'][args.slice_axis]
    print(f'\n自动检测脑组织范围...')
    # 先遍历所有切片，计算每层脑区面积
    slice_areas = []
    for s in range(n_slices):
        _, sl_mask = extract_slice_safe(t2_volume, mask_volume, args.slice_axis, s)
        if sl_mask is not None:
            area = np.sum(sl_mask)
            slice_areas.append((s, area))
    # 找到最大脑区面积，用其 15% 作为有效阈值（确保只取脑组织丰富的中间段）
    if slice_areas:
        max_area = max(a for _, a in slice_areas)
        area_threshold = max_area * 0.15
        slice_range = [s for s, a in slice_areas if a >= area_threshold]
        print(f'  最大脑区面积: {max_area} 像素, 面积阈值: {area_threshold:.0f} 像素')
    else:
        slice_range = list(range(n_slices // 4, n_slices * 3 // 4))
    if len(slice_range) == 0:
        slice_range = list(range(n_slices // 4, n_slices * 3 // 4))
    print(f'  有效切片范围: [{slice_range[0]}, {slice_range[-1]}] ({len(slice_range)}层)')

    # 确定要使用的切片索引 — 进一步裁剪，只取中间 60% 区域，彻底排除两端
    if args.slice_start is not None and args.slice_end is not None:
        valid = [s for s in slice_range if args.slice_start <= s <= args.slice_end]
        if len(valid) == 0:
            print(f'  警告: 指定范围无有效切片，使用全范围')
            available = slice_range
        else:
            available = valid
    else:
        # 默认只取中间 60% 的切片，避开头顶和颅底
        mid_start = int(len(slice_range) * 0.20)
        mid_end = int(len(slice_range) * 0.80)
        available = slice_range[mid_start:mid_end]
        print(f'  采样范围（中间60%）: [{available[0]}, {available[-1]}] ({len(available)}层)')

    # 选取切片（带随机性，确保多样性）
    rng = np.random.RandomState(args.seed)
    if args.n_samples <= len(available):
        # 均匀选取
        indices = rng.choice(available, size=args.n_samples, replace=False)
        indices.sort()
    else:
        # 允许重复选取不同噪声水平
        indices = rng.choice(available, size=args.n_samples, replace=True)

    print(f'\n生成样本:')
    print('-' * 60)

    bvals = np.array([0, 500, 1000, 1500, 2000], dtype=np.int32)
    sigma_values = np.linspace(max(0.01, args.noise_sigma - 0.02),
                               min(0.08, args.noise_sigma + 0.02), 5)

    for i in range(args.n_samples):
        slice_idx = int(indices[i])
        sigma = sigma_values[i % len(sigma_values)]

        # 提取T2切片和对应的脑mask
        t2_slice, mask_slice = extract_slice_safe(t2_volume, mask_volume, args.slice_axis, slice_idx)

        # 跳过没有脑组织的切片（安全措施）
        if np.sum(mask_slice) < 100:
            print(f'  [跳过] 切片 {slice_idx} 无脑组织')
            continue

        seed_i = args.seed + i * 7
        sample = generate_sample(
            t2_slice=t2_slice,
            brain_mask_slice=mask_slice,
            slice_idx=slice_idx,
            axis_name=axis_name,
            target_size=target_size,
            bvals=bvals,
            noise_sigma=sigma,
            add_perturbation=not args.no_perturb,
            add_bias=not args.no_bias,
            seed=seed_i
        )

        # 保存
        filename = os.path.join(output_dir, f'sample_{i:04d}.npz')
        np.savez(filename, **sample)
        print(f'  [{i+1:3d}/{args.n_samples}] sample_{i:04d}.npz  '
              f'切片={slice_idx}  sigma={sigma:.3f}  '
              f'DWI={sample["dwi_clean"].shape}')

        # 每个样本保存多b值可视化
        t2_vis = resize_to(t2_slice, target_size)
        t2_vis[~sample['mask']] = 0
        vis_path = os.path.join(output_dir, f'sample_{i:04d}_bvals.png')
        visualize_sample_multi_b(
            t2_vis, sample['adc_gt'], sample['s0_gt'], sample['k_gt'],
            sample['dwi_clean'], sample['dwi_noisy'],
            bvals, sample['mask'], vis_path
        )

    print('-' * 60)
    print(f'完成! 共生成 {args.n_samples} 个样本')
    print(f'输出目录: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
