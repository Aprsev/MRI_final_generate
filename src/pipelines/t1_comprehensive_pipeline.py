#!/usr/bin/env python3
"""
t1_comprehensive_pipeline.py
=============================
完整 T1→ADC 映射管线，完成任务2.2（T2引导的连续灰度映射法）

Phase 1: T1反转法（零训练成本，即插即用）
Phase 2: UNet-based T1→T2合成（在MNI152模板对上训练，提升对比度真实度）

流程:
  1. 读取T1 DICOM / MNI152模板
  2. 脑提取 + 归一化
  3. (Phase 1) T1灰度反转 → ADC映射
     (Phase 2) UNet T1→T2合成 → ADC映射
  4. 推导S0、K参数图
  5. ADC单指数模型生成多b值DWI (S(b)=S0*exp(-b*D)，不含K项)
  6. 添加Rician噪声
  7. 保存为标准npz格式

用法:
  # Phase 1: T1反转法
  python t1_comprehensive_pipeline.py --phase 1 --all

  # Phase 2: 训练UNet模型
  python t1_comprehensive_pipeline.py --phase 2 --train
  
  # Phase 2: 用训练好的模型推理
  python t1_comprehensive_pipeline.py --phase 2 --inference --all
"""

import os, sys, glob, argparse, json, time
import warnings
warnings.filterwarnings('ignore')

# OpenMP冲突处理：matplotlib/pytorch/numpy可能使用不同OpenMP运行时
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_fill_holes, label as ndimage_label

# ============================================================
# Configuration
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
T1_DATA_DIR = os.path.join(CURRENT_DIR, 'All_Subjects_T1_Raw', 'All_Subjects_T1_Raw')
MRI_ANALOG_DIR = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data')
TEMPLATE_DIR = os.path.join(MRI_ANALOG_DIR, 'data', 'templates', 'mni_icbm152_nlin_sym_09a_minc1')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
OUTPUT_PHASE1 = os.path.join(CURRENT_DIR, 'output_phase1')
OUTPUT_PHASE2 = os.path.join(CURRENT_DIR, 'output_phase2')
for d in [MODELS_DIR, OUTPUT_PHASE1, OUTPUT_PHASE2]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Part 1: Data Loading Utilities
# ============================================================

def load_dicom_subject(subject_dir):
    """从单个subject的DICOM文件夹加载3D T1体积"""
    import pydicom as dcm
    items = os.listdir(subject_dir)
    dcm_folders = [os.path.join(subject_dir, d) for d in items 
                   if os.path.isdir(os.path.join(subject_dir, d))]
    
    if not dcm_folders:
        dcm_files = sorted([f for f in os.listdir(subject_dir) 
                           if f.lower().endswith('.dcm')])
        if dcm_files:
            slices = []
            for f in dcm_files:
                ds = dcm.dcmread(os.path.join(subject_dir, f))
                slices.append(ds.pixel_array.astype(np.float64))
            if slices:
                volume = np.stack(slices, axis=-1)
                return volume, None
    
    dcm_dir = dcm_folders[0]
    dcm_files = sorted([f for f in os.listdir(dcm_dir) 
                       if f.lower().endswith('.dcm')])
    if not dcm_files:
        raise FileNotFoundError(f'No DICOM files in {dcm_dir}')
    
    slices = []
    instance_nums = []
    for f in dcm_files:
        ds = dcm.dcmread(os.path.join(dcm_dir, f))
        instance_nums.append(int(ds.get('InstanceNumber', 0)))
        slices.append((instance_nums[-1], ds.pixel_array.astype(np.float64)))
    
    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=-1)
    
    print(f'    DICOM: {os.path.basename(dcm_dir)}')
    print(f'    Slices: {volume.shape[-1]}, Size: {volume.shape[0]}x{volume.shape[1]}')
    return volume, None


def auto_brain_mask(volume, threshold_frac=0.1):
    """从T1体积自动生成脑mask"""
    threshold = volume.max() * threshold_frac
    mask = volume > threshold
    for i in range(mask.shape[-1]):
        mask[:,:,i] = binary_fill_holes(mask[:,:,i])
    labeled, nf = ndimage_label(mask)
    if nf > 0:
        sizes = np.bincount(labeled.ravel())
        mask = labeled == np.argmax(sizes[1:]) + 1
    return mask


def normalize_volume(volume, mask, smooth_sigma=1.0):
    """在mask内归一化到[0,1]"""
    vol = volume.copy().astype(np.float64)
    if smooth_sigma > 0:
        vol = gaussian_filter(vol, sigma=smooth_sigma)
    vmin = vol[mask].min()
    vmax = vol[mask].max()
    norm = (vol - vmin) / (vmax - vmin + 1e-10)
    norm = np.clip(norm, 0, 1)
    norm[~mask] = 0.0
    return norm


def extract_slices(volume, mask, axis=2, target_size=(192, 192), max_slices=20,
                   strategy='uniform', min_brain_ratio=0.2):
    """
    从3D体积中提取2D切片，自动过滤颅顶/颅底无效切片

    参数:
        strategy: 'uniform'   — 在全脑范围内均匀采样N张 (默认，推荐)
                  'central'   — 只取脑中间最饱满的N张
                  'sequential' — 顺序取前N张
        min_brain_ratio: 最低脑区面积比例（相对最大切片），默认20%
                         用于自动剔除颅顶/颅底等只有少量脑组织的切片
    """
    slices_2d, mask_slices, slice_indices = [], [], []
    n = volume.shape[axis]

    # 第一步：扫描所有切片，记录每层的脑区面积
    all_areas = []
    for i in range(n):
        if axis == 0:
            ml = mask[i, :, :]
        elif axis == 1:
            ml = mask[:, i, :]
        else:
            ml = mask[:, :, i]
        all_areas.append((i, np.sum(ml)))

    if not all_areas:
        return slices_2d, mask_slices, slice_indices

    max_area = max(a for _, a in all_areas)
    area_threshold = max_area * min_brain_ratio

    # 只保留脑区面积 > 最大面积 * min_brain_ratio 的切片
    valid_slices = [(i, a) for i, a in all_areas if a >= area_threshold]
    print(f'    全脑 {n} 层, 最大脑区 {max_area} px, '
          f'阈值 {area_threshold:.0f} px, 有效 {len(valid_slices)} 层')

    if not valid_slices:
        # 降级：用绝对阈值 500
        valid_slices = [(i, a) for i, a in all_areas if a >= 500]
        print(f'    降级阈值 500 px, 有效 {len(valid_slices)} 层')

    if not valid_slices:
        return slices_2d, mask_slices, slice_indices

    # 第二步：按策略选择切片索引
    if strategy == 'central':
        valid_slices.sort(key=lambda x: -x[1])
        selected = [v[0] for v in valid_slices[:max_slices]]
        selected.sort()

    elif strategy == 'uniform':
        indices = [v[0] for v in valid_slices]
        if len(indices) <= max_slices:
            selected = indices
        else:
            step = len(indices) / max_slices
            selected = [indices[int(step * i)] for i in range(max_slices)]

    else:  # 'sequential'
        selected = [v[0] for v in valid_slices[:max_slices]]

    print(f'    选定 {len(selected)} 层: [{selected[0]}-{selected[-1]}]')

    # 第三步：提取所选切片
    for i in selected:
        if axis == 0:
            sl = volume[i, :, :]; ml = mask[i, :, :]
        elif axis == 1:
            sl = volume[:, i, :]; ml = mask[:, i, :]
        else:
            sl = volume[:, :, i]; ml = mask[:, :, i]

        if sl.shape != target_size:
            factors = (target_size[0] / sl.shape[0], target_size[1] / sl.shape[1])
            sl = zoom(sl, factors, order=1)
            ml = zoom(ml.astype(float), factors, order=1) > 0.5

        slices_2d.append(sl)
        mask_slices.append(ml)
        slice_indices.append(i)

    return slices_2d, mask_slices, slice_indices


# ============================================================
# Part 2: DKI Generation (shared between phases)
# ============================================================

def normalize_image(img, mask=None):
    """归一化到[0,1]"""
    if mask is None:
        mask = img > img.max() * 0.01
    img_masked = img.copy()
    vmin = img[mask].min()
    vmax = img[mask].max()
    img_norm = (img_masked - vmin) / (vmax - vmin + 1e-10)
    img_norm[~mask] = 0.0
    return np.clip(img_norm, 0, 1), mask


def t2_weight_to_adc(t2_weight, alpha=2.4e-3, beta=0.6e-3):
    """[Core] ADC = alpha * I_T2 + beta"""
    return alpha * t2_weight + beta


def derive_s0(t2_norm, mask, s0_wm=0.85, s0_csf=1.20):
    s0 = s0_wm + (s0_csf - s0_wm) * t2_norm
    s0[~mask] = 0.0
    return s0


def derive_k(t2_norm, mask, k_wm=1.0, k_csf=0.05):
    k = k_wm - (k_wm - k_csf) * t2_norm
    k[~mask] = 0.0
    return k


def add_internal_variation(param_map, mask, noise_std=0.05):
    noise = np.random.normal(1.0, noise_std, size=param_map.shape)
    param_map[mask] *= noise[mask]
    return param_map


def add_bias_field(img, mask, coeffs=None):
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


def generate_adc_dwi(s0_map, d_map, bvals, mask):
    """
    ADC单指数信号模型: S(b)=S0*exp(-b*D)
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
    n1 = np.random.normal(0, sigma, size=dwi_clean.shape)
    n2 = np.random.normal(0, sigma, size=dwi_clean.shape)
    return np.sqrt((dwi_clean + n1)**2 + n2**2).astype(np.float32)


def generate_tissue_labels(t2_norm, mask):
    label = np.zeros(t2_norm.shape, dtype=np.int32)
    label[mask & (t2_norm > 0.70)] = 1   # CSF
    label[mask & (t2_norm > 0.35) & (t2_norm <= 0.70)] = 2  # GM
    label[mask & (t2_norm <= 0.35)] = 3   # WM
    return label


def generate_full_sample(t2_like_slice, brain_mask, slice_idx, subject_name,
                          bvals=None, noise_sigma=0.03, add_perturbation=True,
                          add_bias=True, source='t1_inversion', seed=None):
    """从T2-like切片生成完整DKI样本"""
    if bvals is None:
        bvals = np.array([0, 500, 1000, 1500, 2000], dtype=np.int32)
    if seed is not None:
        np.random.seed(seed)
    
    t2_norm, _ = normalize_image(t2_like_slice, brain_mask)
    
    adc_gt = t2_weight_to_adc(t2_norm)
    adc_gt[~brain_mask] = 0.0
    
    s0_map = derive_s0(t2_norm, brain_mask)
    k_map = derive_k(t2_norm, brain_mask)
    
    if add_perturbation:
        s0_map = add_internal_variation(s0_map, brain_mask, noise_std=0.05)
        k_map = add_internal_variation(k_map, brain_mask, noise_std=0.10)
        adc_noisy = add_internal_variation(adc_gt.copy(), brain_mask, noise_std=0.03)
    else:
        adc_noisy = adc_gt.copy()
    
    if add_bias:
        s0_map, _ = add_bias_field(s0_map, brain_mask)
    
    dwi_clean = generate_adc_dwi(s0_map, adc_noisy, bvals, brain_mask)
    dwi_noisy = add_rician_noise(dwi_clean, sigma=noise_sigma)
    
    tissue_label = generate_tissue_labels(t2_norm, brain_mask)
    
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
        'subject': subject_name,
        'source': source,
        'description': (
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
            '  slice_idx   [int32]                   切片索引\n'
            '  source      [str]                     数据来源标识\n'
            '  description [str]                     本说明字段\n'
            '---\n'
            f'生成管线: t1_comprehensive_pipeline.py ({source})\n'
            'ADC模型: S(b)=S0*exp(-b*D)  (单指数模型，不含K项)\n'
            'ADC映射: ADC = 2.4e-3 * I_T2 + 0.6e-3,  I=0(WM)->0.6e-3, I=1(CSF)->3.0e-3\n'
            f'subject: {subject_name}, slice: {slice_idx}, source: {source}\n'
        ),
    }
    return sample

# ============================================================
# Part 3: Phase 1 — T1 Inversion Method
# ============================================================

def save_subject_preview(subject_name, output_dir, phase_label='Phase1',
                          t1_slices=None, t2_slices=None, max_show=6):
    """保存单个subject的完整预览图：T1 / T2-like / ADC / S0 / K / DWI(b=1000)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    npz_files = sorted(glob.glob(os.path.join(output_dir, f'sample_*.npz')))
    # Try to find files belonging to this subject
    subject_files = [f for f in npz_files if subject_name in np.load(f, allow_pickle=True).get('subject', '')]
    if not subject_files:
        subject_files = npz_files  # fallback: use whatever is there

    if not subject_files:
        print(f'  [Preview] No npz files found for {subject_name}')
        return

    n = min(max_show, len(subject_files))
    fig, axes = plt.subplots(6, n, figsize=(n * 2.8, 15))
    fig.suptitle(f'{phase_label}: {subject_name}  (showing {n} slices)', fontsize=14, y=1.01)

    row_labels = ['T1 Input', 'T2-like', 'ADC (×1e-3)', 'S0', 'K', 'DWI b=1000']

    for i in range(n):
        d = np.load(subject_files[i])
        m = d['mask']
        t2 = np.clip((d['adc_gt'] - 0.6e-3) / 2.4e-3, 0, 1) * m
        adc = d['adc_gt'] * m
        s0 = d['s0_gt'] * m
        k = np.clip(d['k_gt'], 0, 2) * m
        # Find b=1000 index
        bvals = d['bvals']
        b1000_idx = np.argmin(np.abs(bvals - 1000))
        dwi = d['dwi_noisy'][b1000_idx] * m

        # T1 (if original provided)
        if t1_slices is not None and i < len(t1_slices):
            t1_show = t1_slices[i]
        else:
            t1_show = np.zeros_like(t2)

        data_list = [t1_show, t2, adc / 1e-3, s0, k, dwi]
        cmaps = ['gray', 'gray', 'viridis', 'viridis', 'plasma', 'gray']
        vmin_list = [0, 0, 0, 0, 0, 0]
        vmax_list = [1, 1, 3, 2, 2, None]

        for j, (data, cmap, vmin, vmax) in enumerate(zip(data_list, cmaps, vmin_list, vmax_list)):
            ax = axes[j, i]
            if i == 0:
                ax.set_ylabel(row_labels[j], fontsize=9)
            if vmax is not None:
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=np.percentile(data[m], 98))
            ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{subject_name}_preview.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Preview saved: {out_path}')


def save_all_subjects_summary(output_dir, phase_label='Phase1', max_subjects=12):
    """对所有subject的首个切片做一张对比总览图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    npz_files = sorted(glob.glob(os.path.join(output_dir, 'sample_*.npz')))
    if not npz_files:
        print(f'  [Summary] No npz files in {output_dir}')
        return

    # Group by subject
    subjects = {}
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        subj = str(d.get('subject', 'unknown'))
        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(f)

    subj_names = sorted(subjects.keys())[:max_subjects]
    n_subj = len(subj_names)
    if n_subj == 0:
        return

    n_cols = min(6, n_subj)
    n_rows = int(np.ceil(n_subj / n_cols))

    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 2.5, n_rows * 6))
    fig.suptitle(f'{phase_label} — All Subjects Overview', fontsize=14, y=1.01)

    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

    for idx, subj in enumerate(subj_names):
        r0 = (idx // n_cols) * 3
        c = idx % n_cols
        first_file = subjects[subj][0]
        d = np.load(first_file)
        m = d['mask']
        t2 = np.clip((d['adc_gt'] - 0.6e-3) / 2.4e-3, 0, 1) * m
        adc = d['adc_gt'] * m

        axes[r0, c].imshow(t2, cmap='gray', vmin=0, vmax=1)
        axes[r0, c].set_title(f'{subj}\nT2-like', fontsize=7); axes[r0, c].axis('off')
        axes[r0 + 1, c].imshow(adc / 1e-3, cmap='viridis', vmin=0, vmax=3)
        axes[r0 + 1, c].set_title('ADC (×1e-3)', fontsize=7); axes[r0 + 1, c].axis('off')
        axes[r0 + 2, c].imshow(d['dwi_noisy'][1] * m, cmap='gray')
        axes[r0 + 2, c].set_title('DWI b=500', fontsize=7); axes[r0 + 2, c].axis('off')

    # Hide unused subplots
    for idx in range(len(subj_names), n_rows * n_cols):
        r0 = (idx // n_cols) * 3
        c = idx % n_cols
        for rr in range(3):
            if r0 + rr < axes.shape[0] and c < axes.shape[1]:
                axes[r0 + rr, c].axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'all_subjects_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Summary: {out_path}')


def save_training_history_plot(history, save_dir):
    """保存训练损失曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)

    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
               label=f'Best @ epoch {best_epoch} ({history["val_loss"][best_epoch-1]:.6f})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (L1 + 0.5×SSIM)')
    ax.set_title('UNet Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss_curves.png'), dpi=150)
    plt.close()
    print(f'  Loss curves: {save_dir}training_loss_curves.png')


def t1_inversion_to_t2_like(t1_norm, mask):
    """I_T2-like = 1 - I_T1_norm (contrast reversal)"""
    t2_like = 1.0 - t1_norm
    t2_like[~mask] = 0.0
    return t2_like


def phase1_process_subject(subject_name, subject_dir, output_dir,
                           target_size=(192, 192), bvals=None,
                           noise_sigma=0.03, max_slices=20,
                           slice_strategy='uniform', sample_counter=0):
    """Phase 1: Process one subject via T1 inversion"""
    print(f'\n[{subject_name}] Phase 1 processing (slice={slice_strategy})...')

    try:
        t1_volume, _ = load_dicom_subject(subject_dir)
    except Exception as e:
        print(f'  ERROR: {e}')
        return 0

    mask_3d = auto_brain_mask(t1_volume)
    print(f'  Brain mask: {mask_3d.sum()} voxels')

    if mask_3d.sum() < 1000:
        print(f'  WARNING: brain region too small')
        return 0

    t1_norm = normalize_volume(t1_volume, mask_3d, smooth_sigma=1.0)

    slices_t1, masks, indices = extract_slices(
        t1_norm, mask_3d, axis=2, target_size=target_size,
        max_slices=max_slices, strategy=slice_strategy
    )
    print(f'  Extracted {len(slices_t1)} slices')

    bvals = bvals or np.array([0, 500, 1000, 1500, 2000], dtype=np.int32)
    n_gen = 0

    for t1_slice, brain_mask, slice_idx in zip(slices_t1, masks, indices):
        t2_like = t1_inversion_to_t2_like(t1_slice, brain_mask)

        seed = abs(hash(f'{subject_name}_{slice_idx}')) % (2**31)
        sample = generate_full_sample(
            t2_like, brain_mask, slice_idx, subject_name,
            bvals=bvals, noise_sigma=noise_sigma,
            add_perturbation=True, add_bias=True,
            source='phase1_t1_inversion', seed=seed
        )

        fname = os.path.join(output_dir, f'sample_{sample_counter + n_gen:04d}.npz')
        np.savez(fname, **sample)
        n_gen += 1

    print(f'  Generated {n_gen} samples (sample_{sample_counter:04d} ~ sample_{sample_counter + n_gen - 1:04d})')

    # Save preview visualization
    try:
        save_subject_preview(subject_name, output_dir, phase_label='Phase 1',
                              t1_slices=slices_t1)
    except Exception as e:
        print(f'  [Preview warning] {e}')

    return n_gen


def phase1_process_all(n_subjects=None, output_dir=None, slice_strategy='uniform'):
    """Process all T1 subjects with Phase 1"""
    output_dir = output_dir or OUTPUT_PHASE1
    os.makedirs(output_dir, exist_ok=True)

    subjects = sorted([d for d in os.listdir(T1_DATA_DIR)
                      if os.path.isdir(os.path.join(T1_DATA_DIR, d))])
    print(f'\n{"="*60}')
    print(f'  Phase 1: T1 Inversion ADC Mapping')
    print(f'  Subjects found: {len(subjects)}')
    print(f'  Slice strategy: {slice_strategy}')
    print(f'  Output: {output_dir}')
    print(f'  File format: sample_0000.npz ... sample_NNNN.npz')
    print(f'{"="*60}')

    if n_subjects:
        subjects = subjects[:n_subjects]

    total_gen = 0
    subjects_ok = 0
    for s in subjects:
        n = phase1_process_subject(s, os.path.join(T1_DATA_DIR, s), output_dir,
                                    slice_strategy=slice_strategy,
                                    sample_counter=total_gen)
        total_gen += n
        if n > 0:
            subjects_ok += 1

    print(f'\n{"="*60}')
    print(f'  Phase 1 done: {subjects_ok}/{len(subjects)} subjects')
    print(f'  Total samples: {total_gen}')
    print(f'  Range: sample_0000.npz ~ sample_{total_gen-1:04d}.npz')
    print(f'{"="*60}')

    # Save cross-subject summary
    try:
        save_all_subjects_summary(output_dir, phase_label='Phase 1')
    except Exception as e:
        print(f'  [Summary warning] {e}')

    return subjects_ok

# ============================================================
# Part 4: Phase 2 — UNet T1→T2 Synthesis
# ============================================================

def load_mni152_templates():
    """Load MNI152 T1 and T2 templates"""
    import nibabel as nib
    t1_file = os.path.join(TEMPLATE_DIR, 'mni_icbm152_t1_tal_nlin_sym_09a.mnc')
    t2_file = os.path.join(TEMPLATE_DIR, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')
    
    t1 = nib.load(t1_file).get_fdata().astype(np.float64)
    t2 = nib.load(t2_file).get_fdata().astype(np.float64)
    print(f'  T1: {t1.shape}, T2: {t2.shape}')
    
    mask = (t1 > t1.max()*0.1) & (t2 > t2.max()*0.1)
    mask = binary_fill_holes(mask)
    labeled, nf = ndimage_label(mask)
    if nf > 0:
        sizes = np.bincount(labeled.ravel())
        mask = labeled == np.argmax(sizes[1:]) + 1
    
    def _norm(v):
        vn = (v - v[mask].min()) / (v[mask].max() - v[mask].min() + 1e-10)
        vn[~mask] = 0; return np.clip(vn, 0, 1)
    
    return _norm(t1), _norm(t2), mask


def prepare_mni152_pairs(t1_norm, t2_norm, mask, axis=2, target_size=(192,192), val_ratio=0.1):
    """Extract T1-T2 2D slice pairs from MNI152 for training"""
    slices_t1, slices_t2, masks = [], [], []
    n = t1_norm.shape[axis]
    
    for i in range(n):
        if axis == 0:
            sl1=t1_norm[i,:,:]; sl2=t2_norm[i,:,:]; ml=mask[i,:,:]
        elif axis == 1:
            sl1=t1_norm[:,i,:]; sl2=t2_norm[:,i,:]; ml=mask[:,i,:]
        else:
            sl1=t1_norm[:,:,i]; sl2=t2_norm[:,:,i]; ml=mask[:,:,i]
        
        if np.sum(ml) < 500:
            continue
        
        if sl1.shape != target_size:
            f = (target_size[0]/sl1.shape[0], target_size[1]/sl1.shape[1])
            sl1 = zoom(sl1, f, order=1); sl2 = zoom(sl2, f, order=1)
            ml = zoom(ml.astype(float), f, order=1) > 0.5
        
        slices_t1.append(sl1); slices_t2.append(sl2); masks.append(ml)
    
    print(f'  Total slice pairs: {len(slices_t1)}')
    
    n_total = len(slices_t1)
    n_val = max(1, int(n_total * val_ratio))
    perm = np.random.RandomState(42).permutation(n_total)
    
    train_i = perm[:-n_val]; val_i = perm[-n_val:]
    train = ([slices_t1[i] for i in train_i], [slices_t2[i] for i in train_i])
    val = ([slices_t1[i] for i in val_i], [slices_t2[i] for i in val_i])
    print(f'  Train: {len(train[0])}, Val: {len(val[0])}')
    return train, val


def define_unet(n_channels=1, n_classes=1, base_filters=64):
    """Define 2D UNet using PyTorch"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.conv(x)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
        def forward(self, x): return self.mpconv(x)

    class Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
        def forward(self, x1, x2):
            x1 = self.up(x1)
            dy, dx = x2.size()[2]-x1.size()[2], x2.size()[3]-x1.size()[3]
            x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            return self.conv(torch.cat([x2, x1], dim=1))

    class UNet(nn.Module):
        def __init__(self, n_channels=1, n_classes=1, base_filters=64):
            super().__init__()
            self.inc = DoubleConv(n_channels, base_filters)
            self.d1 = Down(base_filters, base_filters*2)
            self.d2 = Down(base_filters*2, base_filters*4)
            self.d3 = Down(base_filters*4, base_filters*8)
            self.d4 = Down(base_filters*8, base_filters*16)
            self.u1 = Up(base_filters*16, base_filters*8)
            self.u2 = Up(base_filters*8, base_filters*4)
            self.u3 = Up(base_filters*4, base_filters*2)
            self.u4 = Up(base_filters*2, base_filters)
            self.out = nn.Conv2d(base_filters, n_classes, 1)
        def forward(self, x):
            x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
            x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1)
            return self.out(x)

    return UNet(n_channels=n_channels, n_classes=n_classes, base_filters=base_filters)


def ssim_loss(y_true, y_pred, window_size=11):
    """SSIM loss"""
    import torch
    import torch.nn.functional as F
    
    def _ssim(i1, i2):
        pad = window_size//2
        mu1 = F.avg_pool2d(i1, window_size, 1, pad)
        mu2 = F.avg_pool2d(i2, window_size, 1, pad)
        s1 = F.avg_pool2d(i1*i1, window_size, 1, pad) - mu1**2
        s2 = F.avg_pool2d(i2*i2, window_size, 1, pad) - mu2**2
        s12 = F.avg_pool2d(i1*i2, window_size, 1, pad) - mu1*mu2
        c1, c2 = 0.01**2, 0.03**2
        return ((2*mu1*mu2+c1)*(2*s12+c2))/((mu1**2+mu2**2+c1)*(s1+s2+c2))
    
    return 1 - _ssim(y_pred, y_true).mean()


def train_unet(train_data, val_data, model_dir, n_epochs=50, batch_size=8, lr=1e-4):
    """Train UNet on MNI152 T1-T2 pairs"""
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    train_t1, train_t2 = train_data
    val_t1, val_t2 = val_data
    
    def _to_tensor(slices):
        return torch.from_numpy(np.array(slices)[:, np.newaxis].astype(np.float32))
    
    X_train, y_train = _to_tensor(train_t1), _to_tensor(train_t2)
    X_val, y_val = _to_tensor(val_t1), _to_tensor(val_t2)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    
    model = define_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    l1_loss = nn.L1Loss()
    
    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f'  Training {n_epochs} epochs...')
    print(f'  Train: {len(train_t1)}, Val: {len(val_t1)}, Batch: {batch_size}')
    
    for epoch in range(n_epochs):
        model.train()
        tl = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            yp = model(X)
            loss = l1_loss(yp, y) + 0.5 * ssim_loss(y, yp)
            loss.backward()
            optimizer.step()
            tl += loss.item() * X.size(0)
        tl /= len(train_loader.dataset)
        
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                vl += l1_loss(model(X), y).item() * X.size(0)
        vl /= len(val_loader.dataset)
        
        scheduler.step(vl)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{n_epochs} | Train: {tl:.6f} | Val: {vl:.6f}')
        
        if vl < best_val:
            best_val = vl
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_loss': vl,
                'model_config': {'n_channels': 1, 'n_classes': 1, 'base_filters': 64},
            }, os.path.join(model_dir, 't1_to_t2_unet_best.pth'))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'n_channels': 1, 'n_classes': 1, 'base_filters': 64},
    }, os.path.join(model_dir, 't1_to_t2_unet_final.pth'))
    
    print(f'  Best val loss: {best_val:.6f}')
    print(f'  Model saved to {model_dir}')
    return model, history


def visualize_prediction(model, val_data, save_dir):
    """Visualize UNet predictions on validation samples"""
    import torch, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    
    val_t1, val_t2 = val_data
    n = min(5, len(val_t1))
    
    fig, axes = plt.subplots(3, n, figsize=(n*4, 12))
    with torch.no_grad():
        for i in range(n):
            x = torch.from_numpy(val_t1[i][np.newaxis, np.newaxis].astype(np.float32)).to(device)
            yp = np.clip(model(x).cpu().numpy()[0,0], 0, 1)
            axes[0,i].imshow(val_t1[i], cmap='gray', vmin=0, vmax=1)
            axes[0,i].set_title(f'T1 {i+1}'); axes[0,i].axis('off')
            axes[1,i].imshow(yp, cmap='gray', vmin=0, vmax=1)
            axes[1,i].set_title('Synth T2'); axes[1,i].axis('off')
            axes[2,i].imshow(val_t2[i], cmap='gray', vmin=0, vmax=1)
            axes[2,i].set_title('Real T2'); axes[2,i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase2_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Validation preview: {save_dir}phase2_validation.png')

def phase2_train(n_epochs=50, batch_size=8, lr=1e-4):
    """Phase 2: Train UNet on MNI152"""
    print(f'\n{"="*60}')
    print('  Phase 2: Training UNet T1->T2')
    print(f'{"="*60}')
    
    print('\n[1/3] Loading MNI152 templates...')
    t1_norm, t2_norm, mask = load_mni152_templates()
    
    print('\n[2/3] Preparing slice pairs...')
    train_data, val_data = prepare_mni152_pairs(t1_norm, t2_norm, mask)
    
    print('\n[3/3] Training...')
    try:
        model, history = train_unet(train_data, val_data, MODELS_DIR,
                                     n_epochs=n_epochs, batch_size=batch_size, lr=lr)
        visualize_prediction(model, val_data, MODELS_DIR)

        # Save loss curves
        try:
            save_training_history_plot(history, MODELS_DIR)
        except Exception as e:
            print(f'  [Loss curves warning] {e}')

        import json
        hist_s = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(os.path.join(MODELS_DIR, 'training_history.json'), 'w') as f:
            json.dump(hist_s, f, indent=2)
    except ImportError as e:
        print(f'ERROR: PyTorch required - {e}')
        return False

    print(f'Done! Models in {MODELS_DIR}')
    return True


def phase2_inference_subject(subject_name, subject_dir, output_dir,
                              model_path=None, target_size=(192,192),
                              bvals=None, noise_sigma=0.03, max_slices=20,
                              slice_strategy='uniform', sample_counter=0):
    """Phase 2: Apply trained UNet to one subject"""
    import torch

    if model_path is None:
        for p in ['t1_to_t2_unet_best.pth', 't1_to_t2_unet_final.pth']:
            mp = os.path.join(MODELS_DIR, p)
            if os.path.exists(mp):
                model_path = mp; break
        if model_path is None:
            print('  ERROR: No model found. Run --train first.')
            return 0

    print(f'\n[{subject_name}] Phase 2 UNet inference (slice={slice_strategy})...')

    try:
        t1_volume, _ = load_dicom_subject(subject_dir)
    except Exception as e:
        print(f'  ERROR: {e}')
        return 0

    mask_3d = auto_brain_mask(t1_volume)
    if mask_3d.sum() < 1000:
        print('  WARNING: brain too small'); return 0

    t1_norm = normalize_volume(t1_volume, mask_3d, smooth_sigma=1.0)
    slices_t1, masks, indices = extract_slices(
        t1_norm, mask_3d, axis=2, target_size=target_size,
        max_slices=max_slices, strategy=slice_strategy
    )
    print(f'  Slices: {len(slices_t1)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get('model_config', {'n_channels':1,'n_classes':1,'base_filters':64})
    model = define_unet(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Model: {os.path.basename(model_path)}')

    bvals = bvals or np.array([0,500,1000,1500,2000], dtype=np.int32)
    n_gen = 0

    with torch.no_grad():
        for t1_sl, bm, sidx in zip(slices_t1, masks, indices):
            x = torch.from_numpy(t1_sl[np.newaxis, np.newaxis].astype(np.float32)).to(device)
            t2_syn = np.clip(model(x).cpu().numpy()[0,0], 0, 1)
            t2_syn[~bm] = 0.0

            seed = abs(hash(f'{subject_name}_{sidx}_p2')) % (2**31)
            sample = generate_full_sample(
                t2_syn, bm, sidx, subject_name, bvals=bvals,
                noise_sigma=noise_sigma, add_perturbation=True, add_bias=True,
                source='phase2_unet_t1t2', seed=seed
            )
            fname = os.path.join(output_dir, f'sample_{sample_counter + n_gen:04d}.npz')
            np.savez(fname, **sample)
            n_gen += 1

    print(f'  Generated {n_gen} samples (sample_{sample_counter:04d} ~ sample_{sample_counter + n_gen - 1:04d})')

    # Save preview visualization
    try:
        save_subject_preview(subject_name, output_dir, phase_label='Phase 2 UNet',
                              t1_slices=slices_t1)
    except Exception as e:
        print(f'  [Preview warning] {e}')

    return n_gen


def phase2_inference_all(n_subjects=None, output_dir=None, slice_strategy='uniform'):
    """Phase 2: Run UNet inference on all subjects"""
    output_dir = output_dir or OUTPUT_PHASE2
    os.makedirs(output_dir, exist_ok=True)

    subjects = sorted([d for d in os.listdir(T1_DATA_DIR)
                      if os.path.isdir(os.path.join(T1_DATA_DIR, d))])
    print(f'\n{"="*60}')
    print(f'  Phase 2: UNet Inference')
    print(f'  Slice strategy: {slice_strategy}')
    print(f'  File format: sample_0000.npz ... sample_NNNN.npz')
    print(f'  Subjects: {len(subjects)}')
    print(f'{"="*60}')

    if n_subjects: subjects = subjects[:n_subjects]

    total_gen = 0
    subjects_ok = 0
    for s in subjects:
        n = phase2_inference_subject(s, os.path.join(T1_DATA_DIR, s), output_dir,
                                      slice_strategy=slice_strategy,
                                      sample_counter=total_gen)
        total_gen += n
        if n > 0:
            subjects_ok += 1

    print(f'\nPhase 2 done: {subjects_ok}/{len(subjects)} subjects, {total_gen} samples')

    # Save cross-subject summary
    try:
        save_all_subjects_summary(output_dir, phase_label='Phase 2 UNet')
    except Exception as e:
        print(f'  [Summary warning] {e}')

    return subjects_ok


# ============================================================
# Part 5: Visualization
# ============================================================

def visualize_subject(subject_name, output_dir, title='Phase 1'):
    """Visualize results for a subject"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    npz_files = sorted(glob.glob(os.path.join(output_dir, f'{subject_name}_*.npz')))
    if not npz_files:
        print(f'No results for {subject_name} in {output_dir}')
        return
    
    n = min(5, len(npz_files))
    fig, axes = plt.subplots(3, n, figsize=(n*4, 12))
    
    for i in range(n):
        d = np.load(npz_files[i]); m = d['mask']
        t2 = (d['adc_gt'] - 0.6e-3) / 2.4e-3
        axes[0,i].imshow(t2*m, cmap='gray', vmin=0, vmax=1)
        axes[0,i].set_title(f'Slice {i+1}: T2-like'); axes[0,i].axis('off')
        axes[1,i].imshow(d['adc_gt']*m, cmap='viridis', vmin=0, vmax=3e-3)
        axes[1,i].set_title('ADC'); axes[1,i].axis('off')
        axes[2,i].imshow(d['dwi_noisy'][3]*m, cmap='gray')
        axes[2,i].set_title('DWI b=1000'); axes[2,i].axis('off')
    
    plt.suptitle(f'{title}: {subject_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{subject_name}_preview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Preview: {output_dir}{subject_name}_preview.png')


def compare_phases(subject_name):
    """Compare Phase 1 vs Phase 2 results"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    p1 = sorted(glob.glob(os.path.join(OUTPUT_PHASE1, f'{subject_name}_*.npz')))[:3]
    p2 = sorted(glob.glob(os.path.join(OUTPUT_PHASE2, f'{subject_name}_*.npz')))[:3]
    
    if not p1 and not p2:
        print(f'No results for {subject_name}'); return
    
    n = max(len(p1), len(p2), 1)
    fig, axes = plt.subplots(4, n, figsize=(n*4, 16))
    
    for i in range(n):
        if i < len(p1):
            d1 = np.load(p1[i]); m1 = d1['mask']
            axes[0,i].imshow(((d1['adc_gt']-0.6e-3)/2.4e-3)*m1, cmap='gray', vmin=0, vmax=1)
            axes[1,i].imshow(d1['adc_gt']*m1, cmap='viridis', vmin=0, vmax=3e-3)
        else:
            axes[0,i].text(0.5,0.5,'N/A',ha='center',va='center')
            axes[1,i].text(0.5,0.5,'N/A',ha='center',va='center')
        
        if i < len(p2):
            d2 = np.load(p2[i]); m2 = d2['mask']
            axes[2,i].imshow(((d2['adc_gt']-0.6e-3)/2.4e-3)*m2, cmap='gray', vmin=0, vmax=1)
            axes[3,i].imshow(d2['adc_gt']*m2, cmap='viridis', vmin=0, vmax=3e-3)
        else:
            axes[2,i].text(0.5,0.5,'N/A',ha='center',va='center')
            axes[3,i].text(0.5,0.5,'N/A',ha='center',va='center')
        
        for ax in axes[:,i]: ax.axis('off')
    
    for j, label in enumerate(['P1 T2-like','P1 ADC','P2 T2-like','P2 ADC']):
        axes[j,0].set_ylabel(label, fontsize=12)
    
    plt.suptitle(f'{subject_name}: Phase 1 vs Phase 2', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, f'{subject_name}_p1vs2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Comparison: {subject_name}_p1vs2.png')


# ============================================================
# ============================================================
# Part 4c: Phase 2 — RandomForest Inference (sklearn, no PyTorch)
# ============================================================

def extract_patch_features_rf(slice_2d, mask):
    """为RF模型提取像素级局部特征"""
    from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude
    local_mean_3 = uniform_filter(slice_2d, size=3)
    local_mean_5 = uniform_filter(slice_2d, size=5)
    local_mean_7 = uniform_filter(slice_2d, size=7)
    local_std_3 = np.sqrt(np.clip(uniform_filter(slice_2d**2, size=3) - local_mean_3**2, 0, None))
    local_std_5 = np.sqrt(np.clip(uniform_filter(slice_2d**2, size=5) - local_mean_5**2, 0, None))
    local_std_7 = np.sqrt(np.clip(uniform_filter(slice_2d**2, size=7) - local_mean_7**2, 0, None))
    grad_mag = gaussian_gradient_magnitude(slice_2d, sigma=1.0)
    H, W = slice_2d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    features, positions = [], []
    for y, x in zip(*np.where(mask)):
        features.append([slice_2d[y,x], local_mean_3[y,x], local_mean_5[y,x], local_mean_7[y,x],
                         local_std_3[y,x], local_std_5[y,x], local_std_7[y,x], grad_mag[y,x],
                         xx[y,x]/W*2-1, yy[y,x]/H*2-1])
        positions.append((y,x))
    return np.array(features, dtype=np.float32), positions


def phase2_inference_subject_rf(subject_name, subject_dir, output_dir,
                                 model_path=None, target_size=(192,192),
                                 bvals=None, noise_sigma=0.03, max_slices=20,
                                 slice_strategy='uniform', sample_counter=0):
    """Phase 2 RF: Apply RandomForest model to one subject"""
    if model_path is None:
        p = os.path.join(MODELS_DIR, 't1_to_t2_rf_model.pkl')
        if not os.path.exists(p):
            print('  ERROR: RF model not found. Run phase2_sklearn_train.py first.')
            return 0
        model_path = p
    print(f'\n[{subject_name}] Phase 2 RF inference (slice={slice_strategy})...')
    try:
        t1_volume, _ = load_dicom_subject(subject_dir)
    except Exception as e:
        print(f'  ERROR: {e}.'); return 0
    mask_3d = auto_brain_mask(t1_volume)
    if mask_3d.sum() < 1000:
        print('  WARNING: brain too small'); return 0
    t1_norm = normalize_volume(t1_volume, mask_3d, smooth_sigma=1.0)
    slices_t1, masks, indices = extract_slices(
        t1_norm, mask_3d, axis=2, target_size=target_size,
        max_slices=max_slices, strategy=slice_strategy)
    print(f'  Slices: {len(slices_t1)}')
    import joblib
    print(f'  Loading model: {os.path.basename(model_path)}')
    model = joblib.load(model_path)
    bvals = bvals or np.array([0,500,1000,1500,2000], dtype=np.int32)
    n_gen = 0
    for t1_sl, bm, sidx in zip(slices_t1, masks, indices):
        feats, pos = extract_patch_features_rf(t1_sl, bm)
        y_pred = model.predict(feats)
        t2_syn = np.zeros_like(t1_sl)
        for (y,x), val in zip(pos, y_pred):
            t2_syn[y,x] = np.clip(val,0,1)
        t2_syn[~bm] = 0.0
        seed = abs(hash(f'{subject_name}_{sidx}_rf')) % (2**31)
        sample = generate_full_sample(t2_syn, bm, sidx, subject_name, bvals=bvals,
            noise_sigma=noise_sigma, add_perturbation=True, add_bias=True,
            source='phase2_rf_t1t2', seed=seed)
        fname = os.path.join(output_dir, f'sample_{sample_counter + n_gen:04d}.npz')
        np.savez(fname, **sample)
        n_gen += 1
    print(f'  Generated {n_gen} samples (sample_{sample_counter:04d} ~ sample_{sample_counter + n_gen - 1:04d})')

    # Save preview visualization
    try:
        save_subject_preview(subject_name, output_dir, phase_label='Phase 2 RF',
                              t1_slices=slices_t1)
    except Exception as e:
        print(f'  [Preview warning] {e}')

    return n_gen


def phase2_inference_rf_all(n_subjects=None, output_dir=None, slice_strategy='uniform'):
    """Phase 2 RF: Inference on all subjects"""
    output_dir = output_dir or OUTPUT_PHASE2 + '_rf'
    os.makedirs(output_dir, exist_ok=True)
    subjects = sorted([d for d in os.listdir(T1_DATA_DIR) if os.path.isdir(os.path.join(T1_DATA_DIR, d))])
    print(f'\n{"="*60}')
    print(f'  Phase 2 RF: RandomForest Inference')
    print(f'  Slice strategy: {slice_strategy}')
    print(f'  File format: sample_0000.npz ... sample_NNNN.npz')
    print(f'  Subjects: {len(subjects)}')
    print(f'{"="*60}')
    if n_subjects: subjects = subjects[:n_subjects]
    total_gen = 0
    subjects_ok = 0
    for s in subjects:
        n = phase2_inference_subject_rf(s, os.path.join(T1_DATA_DIR, s), output_dir,
                                         slice_strategy=slice_strategy,
                                         sample_counter=total_gen)
        total_gen += n
        if n > 0:
            subjects_ok += 1
    print(f'\nPhase 2 RF done: {subjects_ok}/{len(subjects)} subjects, {total_gen} samples')

    # Save cross-subject summary
    try:
        save_all_subjects_summary(output_dir, phase_label='Phase 2 RF')
    except Exception as e:
        print(f'  [Summary warning] {e}')
    return subjects_ok


def main():
    parser = argparse.ArgumentParser(
        description='T1->ADC Pipeline (Phase1: inversion | Phase2: UNet/RF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python t1_comprehensive_pipeline.py --phase 1 --n_subjects 3
  python t1_comprehensive_pipeline.py --phase 1 --all
  python t1_comprehensive_pipeline.py --phase 2 --train         (UNet, needs PyTorch)
  python t1_comprehensive_pipeline.py --phase 2 --inference     (UNet inference)
  python t1_comprehensive_pipeline.py --phase 2 --inference --rf (RF inference, no PyTorch)
  python t1_comprehensive_pipeline.py --compare subject_name''')
    
    parser.add_argument('--phase', type=int, choices=[1,2])
    parser.add_argument('--n_subjects', type=int)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--rf', action='store_true', help='Use RandomForest (sklearn) instead of UNet')
    parser.add_argument('--visualize', type=str, default=None, nargs='?', const='auto')
    parser.add_argument('--compare', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target_size', type=int, nargs=2, default=[192,192])
    parser.add_argument('--noise_sigma', type=float, default=0.03)
    parser.add_argument('--slice_strategy', type=str, default='uniform',
                        choices=['sequential', 'uniform', 'central'],
                        help='Slice selection: sequential (first N), uniform (evenly spaced), central (largest)')

    args = parser.parse_args()
    
    if args.visualize:
        subj = args.visualize if args.visualize != 'auto' else None
        if subj:
            visualize_subject(subj, args.output_dir or OUTPUT_PHASE1, 'Phase 1')
        else:
            files = glob.glob(os.path.join(args.output_dir or OUTPUT_PHASE1, '*.npz'))
            subs = sorted(set(f.split('_slice')[0] for f in files))
            if subs:
                print(f'Subjects: {subs[:5]}...')
                visualize_subject(subs[0], args.output_dir or OUTPUT_PHASE1)
        return
    
    if args.compare:
        compare_phases(args.compare)
        return
    
    if args.phase == 1:
        n = None if args.all else args.n_subjects
        phase1_process_all(n_subjects=n, output_dir=args.output_dir, slice_strategy=args.slice_strategy)
        
        subjects = sorted([d for d in os.listdir(T1_DATA_DIR)
                         if os.path.isdir(os.path.join(T1_DATA_DIR, d))])
        if args.n_subjects:
            subjects = subjects[:args.n_subjects]
        for s in subjects[:3]:
            visualize_subject(s, args.output_dir or OUTPUT_PHASE1, 'Phase 1')
    
    elif args.phase == 2:
        if args.train and not args.rf:
            phase2_train(n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        if args.inference and args.rf:
            # RF inference (sklearn, no PyTorch needed)
            n = None if args.all else args.n_subjects
            phase2_inference_rf_all(n_subjects=n, output_dir=args.output_dir, slice_strategy=args.slice_strategy)
        elif args.inference and not args.rf:
            # UNet inference (needs PyTorch)
            n = None if args.all else args.n_subjects
            phase2_inference_all(n_subjects=n, output_dir=args.output_dir, slice_strategy=args.slice_strategy)


if __name__ == '__main__':
    main()
