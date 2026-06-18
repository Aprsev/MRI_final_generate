#!/usr/bin/env python3
"""
phase2_sklearn_train.py
=========================
Phase 2 备选方案：用RandomForest从MNI152学习T1→T2映射。
无需PyTorch/CUDA，使用sklearn（已安装），CPU训练约1-2分钟。

用法:
  python phase2_sklearn_train.py
  
输出: models/t1_to_t2_rf_model.pkl (joblib格式)
"""

import os, sys, time, numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_fill_holes, label as ndimage_label
from sklearn.ensemble import RandomForestRegressor
import joblib

# === Config ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data',
                            'data', 'templates', 'mni_icbm152_nlin_sym_09a_minc1')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def extract_patch_features(slice_2d, mask, patch_radius=3):
    """
    为每个像素提取局部特征:
    - 中心像素强度
    - 局部均值 (3x3, 5x5, 7x7)
    - 局部标准差 (3x3, 5x5, 7x7)
    - 局部梯度幅值
    - 归一化坐标 (x, y)
    """
    from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude
    
    features = []
    positions = []
    
    # 预计算特征图
    local_mean_3 = uniform_filter(slice_2d, size=3)
    local_mean_5 = uniform_filter(slice_2d, size=5)
    local_mean_7 = uniform_filter(slice_2d, size=7)
    
    local_std_3 = np.sqrt(uniform_filter(slice_2d**2, size=3) - local_mean_3**2)
    local_std_5 = np.sqrt(uniform_filter(slice_2d**2, size=5) - local_mean_5**2)
    local_std_7 = np.sqrt(uniform_filter(slice_2d**2, size=7) - local_mean_7**2)
    
    grad_mag = gaussian_gradient_magnitude(slice_2d, sigma=1.0)
    
    H, W = slice_2d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    yy_norm = yy / H * 2 - 1  # [-1, 1]
    xx_norm = xx / W * 2 - 1
    
    mask_indices = np.where(mask)
    for y, x in zip(mask_indices[0], mask_indices[1]):
        feat = [
            slice_2d[y, x],
            local_mean_3[y, x], local_mean_5[y, x], local_mean_7[y, x],
            local_std_3[y, x], local_std_5[y, x], local_std_7[y, x],
            grad_mag[y, x],
            xx_norm[y, x], yy_norm[y, x],
        ]
        features.append(feat)
        positions.append((y, x))
    
    return np.array(features, dtype=np.float32), positions


def prepare_training_data(t1_norm, t2_norm, mask, max_pixels_per_slice=3000):
    """从MNI152 T1-T2中提取训练像素对"""
    X_list, y_list = [], []
    n_slices = t1_norm.shape[2]
    
    for i in range(n_slices):
        t1_sl, t2_sl, ml = t1_norm[:,:,i], t2_norm[:,:,i], mask[:,:,i]
        if np.sum(ml) < 500:
            continue
        
        # 提取特征
        feats, positions = extract_patch_features(t1_sl, ml)
        targets = np.array([t2_sl[y, x] for y, x in positions])
        
        # 随机采样以控制数据量（每层最多N个像素）
        if len(feats) > max_pixels_per_slice:
            perm = np.random.RandomState(42 + i).permutation(len(feats))
            perm = perm[:max_pixels_per_slice]
            feats = feats[perm]
            targets = targets[perm]
        
        X_list.append(feats)
        y_list.append(targets)
        
        if (i+1) % 20 == 0:
            print(f'    Slice {i+1}/{n_slices}: {feats.shape[0]} pixels')
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def train():
    """Main training function"""
    import nibabel as nib
    
    print('='*60)
    print('  Phase 2 (sklearn): RandomForest T1->T2 Mapping')
    print('='*60)
    
    # 1. Load templates
    print('\n[1/4] Loading MNI152 templates...')
    t1 = nib.load(os.path.join(TEMPLATE_DIR, 'mni_icbm152_t1_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
    t2 = nib.load(os.path.join(TEMPLATE_DIR, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
    print(f'  T1: {t1.shape}, T2: {t2.shape}')
    
    # 2. Brain mask
    print('\n[2/4] Generating brain mask...')
    mask = (t1 > t1.max()*0.1) & (t2 > t2.max()*0.1)
    for i in range(mask.shape[-1]):
        mask[:,:,i] = binary_fill_holes(mask[:,:,i])
    labeled, nf = ndimage_label(mask)
    if nf > 0:
        sizes = np.bincount(labeled.ravel())
        mask = labeled == np.argmax(sizes[1:]) + 1
    print(f'  Brain voxels: {mask.sum()}')
    
    # 3. Normalize
    print('\n[3/4] Normalizing and extracting features...')
    def _norm(v):
        vn = (v - v[mask].min()) / (v[mask].max() - v[mask].min() + 1e-10)
        vn[~mask] = 0; return np.clip(vn, 0, 1)
    t1_n, t2_n = _norm(t1), _norm(t2)
    print('  Preparing training data...')
    
    t0 = time.time()
    X, y = prepare_training_data(t1_n, t2_n, mask, max_pixels_per_slice=3000)
    print(f'  Training samples: {X.shape[0]}, Features: {X.shape[1]}')
    print(f'  Data preparation: {time.time()-t0:.1f}s')
    
    # 4. Train RandomForest
    print('\n[4/4] Training RandomForest (this may take ~1-2 min)...')
    t0 = time.time()
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=25,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    model.fit(X, y)
    
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s')
    print(f'  R² score: {model.score(X, y):.4f}')
    
    # 5. Save
    model_path = os.path.join(MODELS_DIR, 't1_to_t2_rf_model.pkl')
    joblib.dump(model, model_path)
    print(f'\n  Model saved: {model_path}')
    print(f'  Model size: {os.path.getsize(model_path)/1e6:.2f} MB')
    
    # 6. Quick test on one slice
    print('\n  Quick test...')
    test_idx = t1_n.shape[2] // 2
    t1_test = t1_n[:,:,test_idx]
    mask_test = mask[:,:,test_idx]
    
    feats_test, pos_test = extract_patch_features(t1_test, mask_test)
    y_pred = model.predict(feats_test)
    
    # Reconstruct image
    t2_pred = np.zeros_like(t1_test)
    for (y, x), val in zip(pos_test, y_pred):
        t2_pred[y, x] = np.clip(val, 0, 1)
    
    t2_gt = t2_n[:,:,test_idx]
    
    # Compute metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mask_flat = mask_test.ravel()
    y_true_flat = t2_gt.ravel()[mask_flat]
    y_pred_flat = t2_pred.ravel()[mask_flat]
    print(f'  Test slice MAE: {mean_absolute_error(y_true_flat, y_pred_flat):.4f}')
    print(f'  Test slice R²: {r2_score(y_true_flat, y_pred_flat):.4f}')
    
    # Save test visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0,0].imshow(t1_test * mask_test, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('T1 Input'); axes[0,0].axis('off')
        axes[0,1].imshow(t2_pred * mask_test, cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title('Synth T2 (RF)'); axes[0,1].axis('off')
        axes[0,2].imshow(t2_gt * mask_test, cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title('Real T2 (GT)'); axes[0,2].axis('off')
        
        diff = np.abs(t2_pred - t2_gt) * mask_test
        axes[1,0].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[1,0].set_title(f'Abs Error (mean={diff[mask_test].mean():.3f})'); axes[1,0].axis('off')
        
        axes[1,1].scatter(y_true_flat[::10], y_pred_flat[::10], alpha=0.3, s=1)
        axes[1,1].plot([0,1],[0,1],'r--')
        axes[1,1].set_xlabel('Real T2'); axes[1,1].set_ylabel('Pred T2')
        axes[1,1].set_title(f'R²={r2_score(y_true_flat, y_pred_flat):.3f}')
        
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'rf_model_test.png'), dpi=150)
        plt.close()
        print(f'  Test preview: models/rf_model_test.png')
    except:
        pass
    
    print('\n' + '='*60)
    print('  Training complete!')
    print(f'  Model: {model_path}')
    print(f'  Use with: python t1_comprehensive_pipeline.py --phase 2 --inference --all')
    print('='*60)
    
    return model_path


def test_model_on_subject(model_path, subject_name=None):
    """Test trained RF model on a real T1 subject"""
    import pydicom as dcm
    
    print(f'\nTesting on real T1 subject...')
    model = joblib.load(model_path)
    print(f'  Model loaded: {model_path}')
    
    # Find first subject
    t1_dir = os.path.join(CURRENT_DIR, 'All_Subjects_T1_Raw', 'All_Subjects_T1_Raw')
    subjects = sorted([d for d in os.listdir(t1_dir) if os.path.isdir(os.path.join(t1_dir, d))])
    if not subjects:
        print('  No subjects found')
        return
    
    subj = subject_name or subjects[0]
    subj_dir = os.path.join(t1_dir, subj)
    
    # Find DICOM dir
    items = os.listdir(subj_dir)
    dcm_dirs = [os.path.join(subj_dir, d) for d in items if os.path.isdir(os.path.join(subj_dir, d))]
    if not dcm_dirs:
        print(f'  No DICOM folder in {subj_dir}')
        return
    
    dcm_dir = dcm_dirs[0]
    files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
    ds = dcm.dcmread(os.path.join(dcm_dir, files[0]))
    H, W = ds.pixel_array.shape
    
    # Stack slices
    slices = []
    for f in files:
        ds = dcm.dcmread(os.path.join(dcm_dir, f))
        slices.append(ds.pixel_array.astype(np.float64))
    volume = np.stack(slices, axis=-1)
    print(f'  Subject: {subj}, Shape: {volume.shape}')
    
    # Brain mask
    mask = volume > volume.max() * 0.1
    for i in range(mask.shape[-1]):
        mask[:,:,i] = binary_fill_holes(mask[:,:,i])
    
    # Normalize
    vmin, vmax = volume[mask].min(), volume[mask].max()
    t1_norm = np.clip((volume - vmin) / (vmax - vmin + 1e-10), 0, 1)
    t1_norm[~mask] = 0
    
    # Apply RF model to middle slice
    mid = t1_norm.shape[2] // 2
    t1_sl = t1_norm[:,:,mid]
    mask_sl = mask[:,:,mid]
    
    # Resize to 192x192 if needed
    if t1_sl.shape != (192, 192):
        f = (192/t1_sl.shape[0], 192/t1_sl.shape[1])
        t1_sl = zoom(t1_sl, f, order=1)
        mask_sl = zoom(mask_sl.astype(float), f, order=1) > 0.5
    
    feats, pos = extract_patch_features(t1_sl, mask_sl)
    y_pred = model.predict(feats)
    
    t2_pred = np.zeros_like(t1_sl)
    for (y, x), val in zip(pos, y_pred):
        t2_pred[y, x] = np.clip(val, 0, 1)
    
    # Visualize
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(t1_sl * mask_sl, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original T1'); axes[0].axis('off')
        
        t2_inv = 1 - t1_sl  # Phase 1 method for comparison
        axes[1].imshow(t2_inv * mask_sl, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Phase 1: T1 Inversion'); axes[1].axis('off')
        
        axes[2].imshow(t2_pred * mask_sl, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Phase 2: RF Synthesis'); axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'rf_real_subject_test.png'), dpi=150)
        plt.close()
        print(f'  Result: models/rf_real_subject_test.png')
    except:
        pass


if __name__ == '__main__':
    model_path = train()
    # Optional: test on a real subject
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_model_on_subject(model_path)
