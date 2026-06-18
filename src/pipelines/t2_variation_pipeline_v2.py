"""
t2_variation_pipeline.py (v2)
从MNI152 T2模板出发，通过强度/对比度/噪声/偏置场等多维度变化，
生成多颗表观不同但解剖合理的完整3D脑模拟DWI数据。

和v1的区别：不依赖弹性形变（形变容易失真），改用MRI物理合理的变化。
"""

import numpy as np
import os, argparse
from scipy.ndimage import gaussian_filter, zoom, binary_fill_holes, label as ndimage_label

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data',
                            'data', 'templates', 'mni_icbm152_nlin_sym_09a_minc1')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载T2 + 组织概率图
# ============================================================
def load_template():
    import nibabel as nib
    base = TEMPLATE_DIR
    t2 = nib.load(os.path.join(base, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)

    # mask从T2阈值生成
    thresh = t2.max() * 0.15
    mask = binary_fill_holes(t2 > thresh)
    labeled, nf = ndimage_label(mask)
    if nf > 0:
        sizes = np.bincount(labeled.ravel())
        mask = labeled == np.argmax(sizes[1:]) + 1

    print(f'  T2 shape: {t2.shape}, range: [{t2.min():.1f}, {t2.max():.1f}]')
    print(f'  mask: {mask.sum()}/{mask.size} voxels')
    return t2, mask

# ============================================================
# 2. 随机生成多种物理上合理的变异
# ============================================================
def random_bias_field(shape, seed=None):
    """随机3D偏置场（模拟不同线圈灵敏度）"""
    rng = np.random.RandomState(seed)
    cx, cy, cz = shape
    x = np.linspace(-1, 1, cx)[:, None, None]
    y = np.linspace(-1, 1, cy)[None, :, None]
    z = np.linspace(-1, 1, cz)[None, None, :]
    # 随机2阶多项式系数
    coeffs = rng.uniform(-0.15, 0.15, size=4)
    bias = 1.0 + coeffs[0]*x + coeffs[1]*y + coeffs[2]*z + coeffs[3]*x*y
    return np.clip(bias, 0.6, 1.4)

def random_contrast_mod(t2, mask, seed=None):
    """随机T2对比度修改（模拟不同TE/TR效应）"""
    rng = np.random.RandomState(seed)
    # 在mask内做gamma变换 + 亮度偏移
    masked_vals = t2[mask]
    vmin, vmax = masked_vals.min(), masked_vals.max()
    norm = (t2 - vmin) / (vmax - vmin + 1e-10)
    gamma = rng.uniform(0.7, 1.5)   # <1 提高CSF对比, >1 压低
    norm_mod = norm ** gamma
    # 随机线性拉伸
    a = rng.uniform(0.85, 1.15)
    b = rng.uniform(-0.05, 0.05)
    norm_mod = np.clip(a * norm_mod + b, 0, 1)
    # 映射回原始值范围
    t2_mod = norm_mod * (vmax - vmin) + vmin
    t2_mod[~mask] = t2[~mask]
    return t2_mod

def add_synthetic_lesions(t2, mask, seed=None):
    """在随机位置添加模拟病灶（高信号或低信号）"""
    rng = np.random.RandomState(seed)
    t2_out = t2.copy()
    n_lesions = rng.randint(0, 3)  # 0~2个病灶
    for _ in range(n_lesions):
        # 随机位置（在mask内）
        z = rng.randint(30, t2.shape[2]-30)
        y = rng.randint(30, t2.shape[1]-30)
        x = rng.randint(30, t2.shape[0]-30)
        # 随机椭球半径
        rx, ry, rz = rng.randint(4, 12, size=3)
        # 随机强度偏移 (+20%~-30% of max)
        delta = rng.uniform(-0.3, 0.25) * t2.max()
        # 生成椭球mask
        xx, yy, zz = np.ogrid[:t2.shape[0], :t2.shape[1], :t2.shape[2]]
        lesion_mask = ((xx-x)/rx)**2 + ((yy-y)/ry)**2 + ((zz-z)/rz)**2 <= 1
        lesion_mask = lesion_mask & mask
        t2_out[lesion_mask] = t2_out[lesion_mask] + delta
    return t2_out

# ============================================================
# 3. 灰度映射 + DKI前向
# ============================================================
def generate_brain(t2_3d, mask_3d, bvals, noise_sigma=0.03, seed=None):
    """完整2.2灰度映射法"""
    if seed is not None:
        np.random.seed(seed)

    bg = ~mask_3d
    vmin, vmax = t2_3d[mask_3d].min(), t2_3d[mask_3d].max()
    t2_norm = np.clip((t2_3d - vmin) / (vmax - vmin + 1e-10), 0, 1)
    t2_norm[bg] = 0.0

    # ADC = 2.4e-3 * I_T2 + 0.6e-3
    adc_gt = 2.4e-3 * t2_norm + 0.6e-3
    adc_gt[bg] = 0.0

    # S0, K
    s0 = 0.85 + 0.35 * t2_norm; s0[bg] = 0.0
    k = 1.0 - 0.95 * t2_norm; k[bg] = 0.0

    # 扰动
    adc = adc_gt * np.random.normal(1.0, 0.03, adc_gt.shape)
    k = k * np.random.normal(1.0, 0.10, k.shape)
    s0 = s0 * np.random.normal(1.0, 0.05, s0.shape)
    adc[bg] = 0.0; k[bg] = 0.0; s0[bg] = 0.0

    # DKI前向
    B = len(bvals)
    dwi_clean = np.zeros((B, *adc.shape), dtype=np.float32)
    for i, b in enumerate(bvals):
        if b == 0:
            dwi_clean[i] = s0.astype(np.float32)
        else:
            exp = -b * adc + (1./6.) * (b**2) * (adc**2) * k
            dwi_clean[i] = (s0 * np.exp(exp)).astype(np.float32)
        dwi_clean[i, bg] = 0.0

    # Rician
    n1 = np.random.normal(0, noise_sigma, dwi_clean.shape)
    n2 = np.random.normal(0, noise_sigma, dwi_clean.shape)
    dwi_noisy = np.sqrt((dwi_clean + n1)**2 + n2**2).astype(np.float32)

    # tissue label
    label = np.zeros(t2_norm.shape, dtype=np.int32)
    label[mask_3d & (t2_norm > 0.70)] = 1
    label[mask_3d & (t2_norm > 0.35) & (t2_norm <= 0.70)] = 2
    label[mask_3d & (t2_norm <= 0.35)] = 3

    description = (
        '=== DKI模拟数据 npz 字段说明 ===\n'
        '  dwi_clean  [float32, shape=(B,H,W,D)]  DKI前向无噪声DWI (3D体积)\n'
        '  dwi_noisy  [float32, shape=(B,H,W,D)]  添加Rician噪声后的DWI\n'
        '  bvals      [int32,   shape=(B,)]        b值序列 (s/mm^2)\n'
        '  s0_gt      [float32, shape=(H,W,D)]     S0基图, 含扰动/偏置场\n'
        '  d_gt       [float32, shape=(H,W,D)]     扩散系数图 D (mm^2/s), 加扰动后\n'
        '  k_gt       [float32, shape=(H,W,D)]     峰度系数图 K (无单位), 含扰动\n'
        '  adc_gt     [float32, shape=(H,W,D)]     ADC真值 (无扰动), 用于对比\n'
        '  mask       [bool,    shape=(H,W,D)]     脑区mask (True=脑组织)\n'
        '  tissue_label [int32, shape=(H,W,D)]     组织标签: 0=BG 1=CSF 2=GM 3=WM\n'
        '  noise_sigma [float32]                   Rician噪声标准差\n'
        '  source      [str]                       数据来源标识\n'
        '  description [str]                       本说明字段\n'
        '---\n'
        '生成管线: t2_variation_pipeline_v2.py (多维物理变异 + 灰度映射)\n'
        'DKI模型: S(b)=S0*exp(-b*D + (1/6)*b^2*D^2*K)\n'
        'ADC映射: ADC = 2.4e-3 * I_T2 + 0.6e-3\n'
        f'噪声水平: {noise_sigma:.3f}\n'
    )
    return {
        'dwi_clean': dwi_clean, 'dwi_noisy': dwi_noisy,
        'bvals': bvals.astype(np.int32),
        's0_gt': s0.astype(np.float32), 'd_gt': adc.astype(np.float32),
        'k_gt': k.astype(np.float32), 'adc_gt': adc_gt.astype(np.float32),
        'mask': mask_3d, 'tissue_label': label,
        'noise_sigma': np.float32(noise_sigma),
        'source': 't2_variation_v2',
        'description': description,
    }

# ============================================================
# 4. 多切片预览
# ============================================================
def save_preview(sample, idx, save_dir, n_slices=8):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mask = sample['mask']
    z_idx = np.where(np.any(mask, axis=(0,1)))[0]
    selected = np.linspace(z_idx[0], z_idx[-1], n_slices, dtype=int)

    fig, axes = plt.subplots(4, n_slices, figsize=(n_slices*2.5, 10))
    for j, z in enumerate(selected):
        for row, data, cmap, vmin, vmax in [
            (0, sample['adc_gt'], 'viridis', 0, 3e-3),
            (1, sample['k_gt'], 'plasma', 0, 1.2),
            (2, sample['s0_gt'], 'gray', None, None),
            (3, sample['dwi_noisy'][3], 'gray', None, None),
        ]:
            d = data[:,:,z] * mask[:,:,z]
            im = axes[row, j].imshow(d, cmap=cmap, vmin=vmin, vmax=vmax) if vmin is not None else axes[row, j].imshow(d, cmap=cmap)
            axes[row, j].axis('off')
            if j == 0:
                axes[row, j].set_ylabel(['ADC','K','S0','DWI b1000'][row])

    sigma = float(sample['noise_sigma'])
    fig.suptitle(f'Brain {idx:04d} | σ={sigma:.3f} | mask={mask.sum()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'brain_{idx:04d}_preview.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_brains', type=int, default=5)
    parser.add_argument('--noise_base', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--lesions', action='store_true', help='添加模拟病灶')
    args = parser.parse_args()

    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    bvals = np.array([0, 500, 1000, 1500, 2000])

    print('='*60)
    print('  v2: T2变异法生成多颗不同大脑')
    print('  (对比度/偏置场/噪声/扰动多维变化)')
    print('='*60)

    t2_vol, mask_vol = load_template()
    print(f'  输出目录: {out_dir}')

    for i in range(args.n_brains):
        seed_i = args.seed + i * 17
        rng = np.random.RandomState(seed_i)

        # --- 生成变异 ---
        # 1) 对比度修改（gamma拉伸）
        t2_var = random_contrast_mod(t2_vol, mask_vol, seed=seed_i)

        # 2) 偏置场（空间不均匀）
        bias = random_bias_field(t2_vol.shape, seed=seed_i+1)

        # 3) 可选：病灶
        if args.lesions:
            t2_var = add_synthetic_lesions(t2_var, mask_vol, seed=seed_i+2)

        # 4) 随机噪声水平
        sigma_i = args.noise_base * rng.uniform(0.6, 1.4)

        # 走灰度映射
        sample = generate_brain(t2_var * bias, mask_vol, bvals,
                                noise_sigma=sigma_i, seed=seed_i+3)

        # 保存
        fname = os.path.join(out_dir, f'brain_{i:04d}.npz')
        np.savez(fname, **sample)
        print(f'  [{i+1}/{args.n_brains}] brain_{i:04d}.npz  '
              f'σ={sigma_i:.3f}  gamma变化  bias幅度={bias.std():.3f}')
        save_preview(sample, i, out_dir)

    print('-'*60)
    print(f'完成! 输出: {out_dir}')
    print(f'每颗脑: DWI (5, {t2_vol.shape[0]}, {t2_vol.shape[1]}, {t2_vol.shape[2]})')
    print('='*60)

if __name__ == '__main__':
    main()
