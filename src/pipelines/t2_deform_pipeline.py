"""
t2_deform_pipeline.py
从MNI152 T2模板出发，通过随机弹性形变生成多颗形态不同的大脑，
再走2.2灰度映射法生成完整的3D模拟DWI数据。

Mask直接从T2灰度阈值生成，不依赖外部mask文件。
"""

import numpy as np
import os, argparse
from scipy.ndimage import gaussian_filter, zoom, map_coordinates, binary_fill_holes

# ============================================================
# 配置
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MRI_ANALOG_DIR = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data')
TEMPLATE_DIR = os.path.join(MRI_ANALOG_DIR, 'data', 'templates',
                            'mni_icbm152_nlin_sym_09a_minc1')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output_deformed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载模板 + T2阈值生成mask
# ============================================================
def load_template():
    import nibabel as nib
    t2 = nib.load(os.path.join(TEMPLATE_DIR, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
    # 直接从T2灰度阈值生成mask
    # MNI152 T2中脑组织约>15%最大值，颅骨/背景更低
    threshold = t2.max() * 0.15
    mask = t2 > threshold
    # 填洞 + 保留最大连通域（去噪）
    from scipy.ndimage import label
    mask = binary_fill_holes(mask)
    labeled, n_features = label(mask)
    if n_features > 0:
        sizes = np.bincount(labeled.ravel())
        largest = np.argmax(sizes[1:]) + 1
        mask = labeled == largest
    print(f"  T2范围: [{t2.min():.2f}, {t2.max():.2f}], 阈值={threshold:.1f}")
    print(f"  mask: {mask.sum()}/{mask.size} voxels ({mask.sum()/mask.size*100:.1f}%)")
    return t2, mask

# ============================================================
# 2. 随机弹性形变
# ============================================================
def random_displacement_field(shape, sigma=15, amplitude=12, seed=None):
    """
    生成一个光滑的随机位移场（3D）
    sigma: 控制形变的光滑程度（越大越平滑）
    amplitude: 控制形变幅度（越大形变越剧烈）
    """
    rng = np.random.RandomState(seed)
    # 在粗网格上生成随机偏移
    coarse_shape = tuple(max(2, s // 4) for s in shape)
    field_coarse = [rng.randn(*coarse_shape) for _ in range(3)]
    # 插值到原始分辨率并高斯平滑
    field = np.zeros((3, *shape), dtype=np.float64)
    factors = [shape[i] / coarse_shape[i] for i in range(3)]
    for i in range(3):
        f = zoom(field_coarse[i], factors, order=1)
        f = gaussian_filter(f, sigma=sigma, mode='reflect')
        f = f / f.std() * amplitude  # 标准化到指定幅度
        field[i] = f
    return field

def apply_deformation(volume, field):
    """用位移场对volume进行弹性形变"""
    shape = volume.shape
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    coords = [grid[i] + field[i] for i in range(3)]
    return map_coordinates(volume, coords, order=1, mode='nearest').reshape(shape)

# ============================================================
# 3. 灰度映射 + DKI前向
# ============================================================
def generate_brain(t2_3d, mask_3d, bvals, noise_sigma=0.03, seed=None):
    """对一颗完整的3D脑走2.2灰度映射法"""
    if seed is not None:
        np.random.seed(seed)

    # 只在mask内做归一化
    t2_masked = t2_3d.copy()
    bg = ~mask_3d
    vmin = t2_3d[mask_3d].min()
    vmax = t2_3d[mask_3d].max()
    t2_norm = (t2_masked - vmin) / (vmax - vmin + 1e-10)
    t2_norm[bg] = 0.0
    t2_norm = np.clip(t2_norm, 0, 1)

    # ADC = 2.4e-3 * I_T2 + 0.6e-3
    adc_gt = 2.4e-3 * t2_norm + 0.6e-3
    adc_gt[bg] = 0.0

    # S0 = 0.85 + 0.35 * I_T2
    s0 = 0.85 + 0.35 * t2_norm
    s0[bg] = 0.0

    # K = 1.0 - 0.95 * I_T2 (WM高K, CSF低K)
    k = 1.0 - 0.95 * t2_norm
    k[bg] = 0.0

    # 组织内扰动
    perturb_d = np.random.normal(1.0, 0.03, size=adc_gt.shape)
    perturb_k = np.random.normal(1.0, 0.10, size=k.shape)
    perturb_s0 = np.random.normal(1.0, 0.05, size=s0.shape)
    adc = adc_gt * perturb_d
    k = k * perturb_k
    s0 = s0 * perturb_s0
    adc[bg] = 0.0; k[bg] = 0.0; s0[bg] = 0.0

    # 偏置场（3D）
    bias = 1.0 + 0.08 * np.linspace(-1, 1, s0.shape[0])[:,None,None] \
                + 0.06 * np.linspace(-1, 1, s0.shape[1])[None,:,None] \
                + 0.04 * np.linspace(-1, 1, s0.shape[2])[None,None,:]
    s0 = s0 * bias
    s0[bg] = 0.0

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

    # Rician噪声
    n1 = np.random.normal(0, noise_sigma, size=dwi_clean.shape)
    n2 = np.random.normal(0, noise_sigma, size=dwi_clean.shape)
    dwi_noisy = np.sqrt((dwi_clean + n1)**2 + n2**2).astype(np.float32)

    # tissue_label (格式兼容,从T2阈值)
    label = np.zeros(t2_norm.shape, dtype=np.int32)
    label[mask_3d & (t2_norm > 0.70)] = 1
    label[mask_3d & (t2_norm > 0.35) & (t2_norm <= 0.70)] = 2
    label[mask_3d & (t2_norm <= 0.35)] = 3

    description = (
        '=== DKI模拟数据 npz 字段说明 ===\n'
        '  dwi_clean  [float32, shape=(B,H,W,D)]  DKI前向无噪声DWI (3D体积)\n'
        '  dwi_noisy  [float32, shape=(B,H,W,D)]  添加Rician噪声后的DWI\n'
        '  bvals      [int32,   shape=(B,)]        b值序列 (s/mm^2)\n'
        '  s0_gt      [float32, shape=(H,W,D)]     S0基图, 含扰动\n'
        '  d_gt       [float32, shape=(H,W,D)]     扩散系数图 D (mm^2/s), 加扰动后\n'
        '  k_gt       [float32, shape=(H,W,D)]     峰度系数图 K (无单位), 含扰动\n'
        '  adc_gt     [float32, shape=(H,W,D)]     ADC真值 (无扰动), 用于对比\n'
        '  mask       [bool,    shape=(H,W,D)]     脑区mask (True=脑组织)\n'
        '  tissue_label [int32, shape=(H,W,D)]     组织标签: 0=BG 1=CSF 2=GM 3=WM\n'
        '  noise_sigma [float32]                   Rician噪声标准差\n'
        '  source      [str]                       数据来源标识\n'
        '  description [str]                       本说明字段\n'
        '---\n'
        '生成管线: t2_deform_pipeline.py (弹性形变 + 灰度映射)\n'
        'DKI模型: S(b)=S0*exp(-b*D + (1/6)*b^2*D^2*K)\n'
        'ADC映射: ADC = 2.4e-3 * I_T2 + 0.6e-3\n'
        f'噪声水平: {noise_sigma:.3f}\n'
    )
    return {
        'dwi_clean': dwi_clean,
        'dwi_noisy': dwi_noisy,
        'bvals': bvals.astype(np.int32),
        's0_gt': s0.astype(np.float32),
        'd_gt': adc.astype(np.float32),
        'k_gt': k.astype(np.float32),
        'adc_gt': adc_gt.astype(np.float32),
        'mask': mask_3d,
        'tissue_label': label,
        'noise_sigma': np.float32(noise_sigma),
        'source': 'deformed_mni152_t2_grayscale',
        'description': description,
    }

# ============================================================
# 4. 多切片可视化
# ============================================================
def save_slice_preview(sample, brain_idx, save_dir, n_slices=8):
    """保存一颗大脑的多张轴位切片预览图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mask = sample['mask']
    adc = sample['adc_gt']
    k = sample['k_gt']
    s0 = sample['s0_gt']
    dwi = sample['dwi_noisy']

    # 找到有脑组织的切片范围
    z_indices = np.where(np.any(mask, axis=(0, 1)))[0]
    if len(z_indices) < n_slices:
        z_indices = np.arange(mask.shape[-1])
    # 均匀选取n_slices张
    selected = np.linspace(z_indices[0], z_indices[-1], n_slices, dtype=int)

    fig, axes = plt.subplots(4, n_slices, figsize=(n_slices * 2.5, 10))

    for j, z in enumerate(selected):
        # 第1行: ADC
        im = axes[0, j].imshow(adc[:, :, z] * mask[:, :, z], cmap='viridis', vmin=0, vmax=3e-3)
        axes[0, j].set_title(f'z={z}')
        axes[0, j].axis('off')
        if j == 0:
            axes[0, j].set_ylabel('ADC', fontsize=10)

        # 第2行: K
        im = axes[1, j].imshow(k[:, :, z] * mask[:, :, z], cmap='plasma', vmin=0, vmax=1.2)
        axes[1, j].axis('off')
        if j == 0:
            axes[1, j].set_ylabel('K', fontsize=10)

        # 第3行: S0
        axes[2, j].imshow(s0[:, :, z] * mask[:, :, z], cmap='gray')
        axes[2, j].axis('off')
        if j == 0:
            axes[2, j].set_ylabel('S0', fontsize=10)

        # 第4行: DWI b=1000 noisy
        dwi_slice = dwi[3, :, :, z] * mask[:, :, z]  # b=1000是index 3
        axes[3, j].imshow(dwi_slice, cmap='gray')
        axes[3, j].axis('off')
        if j == 0:
            axes[3, j].set_ylabel('DWI\nb=1000', fontsize=10)

    fig.suptitle(f'Brain {brain_idx:04d} | noise σ={sample["noise_sigma"]:.3f} | '
                 f'mask voxels={mask.sum()}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'brain_{brain_idx:04d}_preview.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [预览] brain_{brain_idx:04d}_preview.png ({n_slices} slices)')

# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_brains', type=int, default=5, help='生成几颗完整脑')
    parser.add_argument('--deform_sigma', type=float, default=15, help='形变平滑度')
    parser.add_argument('--deform_amp', type=float, default=12, help='形变幅度')
    parser.add_argument('--noise_sigma', type=float, default=0.03, help='噪声水平')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    bvals = np.array([0, 500, 1000, 1500, 2000])

    print('='*60)
    print(f'  弹性形变生成 {args.n_brains} 颗不同大脑')
    print('='*60)

    t2_vol, mask_vol = load_template()
    print(f'  原始形状: {t2_vol.shape}')

    for i in range(args.n_brains):
        seed_i = args.seed + i * 13

        # 生成随机位移场
        field = random_displacement_field(
            t2_vol.shape, sigma=args.deform_sigma,
            amplitude=args.deform_amp * (0.7 + 0.6 * np.random.random()),
            seed=seed_i
        )

        # 形变T2和mask
        t2_def = apply_deformation(t2_vol, field)
        mask_def = apply_deformation(mask_vol.astype(float), field) > 0.5

        # 形变后mask可能变形，做些清理
        from scipy.ndimage import binary_closing, binary_opening
        mask_def = binary_closing(mask_def, iterations=2)
        mask_def = binary_opening(mask_def, iterations=1)
        mask_def = gaussian_filter(mask_def.astype(float), sigma=1) > 0.5

        # 噪声水平轻微浮动
        sigma_i = args.noise_sigma * (0.8 + 0.4 * np.random.random())

        # 走灰度映射
        sample = generate_brain(t2_def, mask_def, bvals, noise_sigma=sigma_i, seed=seed_i+7)

        # 保存 —— 注意这是3D数据
        fname = os.path.join(output_dir, f'brain_{i:04d}.npz')
        np.savez(fname, **sample)

        vox_vol = sample['mask'].sum()
        print(f'  [{i+1}/{args.n_brains}] brain_{i:04d}.npz  '
              f'脑体积={vox_vol}vox  σ={sigma_i:.3f}  '
              f'形变幅度={field[0].std():.1f}px')

        # 保存多切片预览图
        save_slice_preview(sample, i, output_dir, n_slices=8)

    print('-'*60)
    print(f'完成! 输出: {output_dir}')
    print(f'每个文件是完整3D脑: DWI shape = (5, {t2_vol.shape[0]}, {t2_vol.shape[1]}, {t2_vol.shape[2]})')
    print('='*60)

if __name__ == '__main__':
    main()
