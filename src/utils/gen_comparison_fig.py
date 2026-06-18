"""
Generate MNI152 T1 vs T2 training data comparison figures
and Phase 1 vs Phase 2 comparison for document.
"""
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_fill_holes, label as ndimage_label
import os

# ========== 1. Define UNet ==========
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
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

# ========== 2. Paths ==========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data',
                    'data', 'templates', 'mni_icbm152_nlin_sym_09a_minc1')
OUT_DIR = os.path.join(CURRENT_DIR, 'data')
MODEL_PATH = os.path.join(CURRENT_DIR, 'models', 't1_to_t2_unet_best.pth')

# ========== 3. Load MNI152 ==========
print('Loading MNI152 templates...')
t1 = nib.load(os.path.join(BASE, 'mni_icbm152_t1_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
t2 = nib.load(os.path.join(BASE, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
print(f'  T1: {t1.shape}, T2: {t2.shape}')

mask = (t1 > t1.max()*0.1) & (t2 > t2.max()*0.1)
for i in range(mask.shape[-1]):
    mask[:,:,i] = binary_fill_holes(mask[:,:,i])
labeled, nf = ndimage_label(mask)
if nf > 0:
    sizes = np.bincount(labeled.ravel())
    mask = labeled == np.argmax(sizes[1:]) + 1

def norm(v):
    vn = (v - v[mask].min()) / (v[mask].max() - v[mask].min() + 1e-10)
    vn[~mask] = 0
    return np.clip(vn, 0, 1)

t1_n, t2_n = norm(t1), norm(t2)

# ========== 4. Load model ==========
print('Loading UNet model...')
device = torch.device('cpu')
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
cfg = ckpt.get('model_config', {'n_channels':1, 'n_classes':1, 'base_filters':64})
model = UNet(**cfg).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Epoch={ckpt["epoch"]+1}, val_loss={ckpt["val_loss"]:.6f}')

# ========== 5. Select slices ==========
valid = [i for i in range(mask.shape[2]) if mask[:,:,i].sum() > 1000]
n_cols = 6
indices = np.linspace(valid[15], valid[-15], n_cols, dtype=int)
print(f'  Slices: {list(indices)}')

# ========== 6. Figure 1: MNI152 T1 vs T2 paired slices ==========
print('\n[1/3] MNI152 T1 vs T2 paired slices...')
fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.8, 6))
for j, idx in enumerate(indices):
    sl1 = t1_n[:,:,idx].copy()
    sl2 = t2_n[:,:,idx].copy()
    ml = mask[:,:,idx].copy()
    if sl1.shape != (192, 192):
        f = (192/sl1.shape[0], 192/sl1.shape[1])
        sl1 = zoom(sl1, f, order=1)
        sl2 = zoom(sl2, f, order=1)
        ml = zoom(ml.astype(float), f, order=1) > 0.5
    
    axes[0, j].imshow(sl1 * ml, cmap='gray', vmin=0, vmax=1)
    axes[0, j].set_title(f'Slice {idx}\nT1 Input', fontsize=9)
    axes[0, j].axis('off')
    
    axes[1, j].imshow(sl2 * ml, cmap='gray', vmin=0, vmax=1)
    axes[1, j].set_title(f'Slice {idx}\nT2 GT', fontsize=9)
    axes[1, j].axis('off')

axes[0, 0].set_ylabel('T1 Input', fontsize=11)
axes[1, 0].set_ylabel('T2 GT', fontsize=11)
plt.suptitle('MNI152 Training Data: T1 (Input) vs T2 (Ground Truth) Paired Slices', 
             fontsize=13, y=1.02)
plt.tight_layout()
p1 = os.path.join(OUT_DIR, 'MNI152_T1_T2_training_pairs.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p1}')

# ========== 7. Figure 2: Phase 1 vs Phase 2 comparison ==========
print('\n[2/3] Phase 1 vs Phase 2 comparison...')
fig, axes = plt.subplots(5, n_cols, figsize=(n_cols * 2.8, 14))
row_labels = ['T1 Input', 'Phase 1\n(T1 Inversion)', 'Phase 2\n(UNet Synth)', 
              'T2 GT (MNI152)', 'MAE Comparison']

for j, idx in enumerate(indices):
    sl1 = t1_n[:,:,idx].copy()
    sl2 = t2_n[:,:,idx].copy()
    ml = mask[:,:,idx].copy()
    if sl1.shape != (192, 192):
        f = (192/sl1.shape[0], 192/sl1.shape[1])
        sl1 = zoom(sl1, f, order=1)
        sl2 = zoom(sl2, f, order=1)
        ml = zoom(ml.astype(float), f, order=1) > 0.5
    
    # UNet prediction
    x = torch.from_numpy(sl1[np.newaxis, np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        yp = np.clip(model(x).cpu().numpy()[0, 0], 0, 1)
    yp[~ml] = 0.0
    
    # Phase 1 inversion
    p1 = 1.0 - sl1
    p1[~ml] = 0.0
    
    err_unet = np.abs(yp - sl2) * ml
    err_p1 = np.abs(p1 - sl2) * ml
    
    # Row 0: T1 input
    axes[0, j].imshow(sl1 * ml, cmap='gray', vmin=0, vmax=1)
    axes[0, j].set_title(f'Slice {idx}', fontsize=9)
    axes[0, j].axis('off')
    if j == 0: axes[0, j].set_ylabel(row_labels[0], fontsize=10)
    
    # Row 1: Phase 1 inversion
    axes[1, j].imshow(p1 * ml, cmap='gray', vmin=0, vmax=1)
    axes[1, j].axis('off')
    if j == 0: axes[1, j].set_ylabel(row_labels[1], fontsize=10)
    
    # Row 2: UNet synth
    axes[2, j].imshow(yp * ml, cmap='gray', vmin=0, vmax=1)
    axes[2, j].axis('off')
    if j == 0: axes[2, j].set_ylabel(row_labels[2], fontsize=10)
    
    # Row 3: T2 GT
    axes[3, j].imshow(sl2 * ml, cmap='gray', vmin=0, vmax=1)
    axes[3, j].axis('off')
    if j == 0: axes[3, j].set_ylabel(row_labels[3], fontsize=10)
    
    # Row 4: MAE bars
    mae_u = err_unet[ml].mean()
    mae_p = err_p1[ml].mean()
    impr = (mae_p - mae_u) / mae_p * 100
    axes[4, j].barh(['P1 Inv', 'UNet'], [mae_p, mae_u], 
                     color=['#ff7f7f', '#7fbfff'], height=0.5)
    axes[4, j].set_xlim(0, max(mae_p, mae_u) * 1.5)
    axes[4, j].text(mae_u + 0.002, 0, f'{mae_u:.4f}', fontsize=7, color='blue', va='center')
    axes[4, j].text(mae_p + 0.002, 1, f'{mae_p:.4f}', fontsize=7, color='red', va='center')
    axes[4, j].tick_params(labelsize=7)
    axes[4, j].set_title(f'MAE\n(imp {impr:.0f}%)', fontsize=8)
    if j == 0: axes[4, j].set_ylabel(row_labels[4], fontsize=10)

plt.suptitle('Phase 1 vs Phase 2: MNI152 T1-to-T2 Synthesis Comparison', 
             fontsize=13, y=1.01)
plt.tight_layout()
p2 = os.path.join(OUT_DIR, 'MNI152_Phase1_vs_Phase2_comparison.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p2}')

# ========== 8. Figure 3: Single mid-slice detail ==========
print('\n[3/3] Single slice detail...')
mid_idx = valid[len(valid)//2]
sl1 = t1_n[:,:,mid_idx].copy()
sl2 = t2_n[:,:,mid_idx].copy()
ml = mask[:,:,mid_idx].copy()
if sl1.shape != (192, 192):
    f = (192/sl1.shape[0], 192/sl1.shape[1])
    sl1 = zoom(sl1, f, order=1)
    sl2 = zoom(sl2, f, order=1)
    ml = zoom(ml.astype(float), f, order=1) > 0.5

x = torch.from_numpy(sl1[np.newaxis, np.newaxis].astype(np.float32)).to(device)
with torch.no_grad():
    yp = np.clip(model(x).cpu().numpy()[0, 0], 0, 1)
yp[~ml] = 0.0

p1_inv = 1.0 - sl1
p1_inv[~ml] = 0.0

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

titles = ['T1 Input', 'T2 GT', 'Phase 1\nInversion', 'Phase 2\nUNet Synth', 'T2 GT - P1 Diff']
images = [sl1*ml, sl2*ml, p1_inv*ml, yp*ml, np.abs(p1_inv-sl2)*ml]
for i in range(5):
    axes[0, i].imshow(images[i], cmap='gray' if i != 4 else 'hot', 
                      vmin=0, vmax=1 if i != 4 else 0.3)
    axes[0, i].set_title(titles[i], fontsize=10)
    axes[0, i].axis('off')

titles2 = ['T1 Overlay\nT2', 'T2 GT - UNet Diff', 'UNet Error\nMap', 'Scatter\nUNet vs GT', 'Scatter\nP1 vs GT']
axes[1, 0].imshow(np.dstack((sl1*ml, sl2*ml, np.zeros_like(sl1))))
axes[1, 0].set_title(titles2[0], fontsize=10); axes[1, 0].axis('off')

im2 = axes[1, 1].imshow(np.abs(yp-sl2)*ml, cmap='hot', vmin=0, vmax=0.3)
axes[1, 1].set_title(titles2[1], fontsize=10); axes[1, 1].axis('off')

# Error map with mask overlay
err_show = np.abs(yp - sl2)
err_show[~ml] = 0
im3 = axes[1, 2].imshow(err_show, cmap='jet', vmin=0, vmax=0.3)
axes[1, 2].set_title(titles2[2], fontsize=10); axes[1, 2].axis('off')

# Scatter plots
mask_flat = ml.ravel()
axes[1, 3].scatter(sl2.ravel()[mask_flat][::20], yp.ravel()[mask_flat][::20], 
                   alpha=0.3, s=1, c='blue')
axes[1, 3].plot([0,1], [0,1], 'r--', alpha=0.7)
axes[1, 3].set_xlabel('T2 GT'); axes[1, 3].set_ylabel('UNet Pred')
axes[1, 3].set_title(titles2[3], fontsize=10)
axes[1, 3].set_aspect('equal')

axes[1, 4].scatter(sl2.ravel()[mask_flat][::20], p1_inv.ravel()[mask_flat][::20], 
                   alpha=0.3, s=1, c='red')
axes[1, 4].plot([0,1], [0,1], 'r--', alpha=0.7)
axes[1, 4].set_xlabel('T2 GT'); axes[1, 4].set_ylabel('P1 Pred')
axes[1, 4].set_title(titles2[4], fontsize=10)
axes[1, 4].set_aspect('equal')

plt.suptitle(f'MNI152 Mid-Brain Slice {mid_idx}: Detailed Comparison', fontsize=13, y=1.02)
plt.tight_layout()
p3 = os.path.join(OUT_DIR, 'MNI152_single_slice_detail.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p3}')

# Print metrics
print('\nPer-slice metrics:')
print(f'{"Slice":>6} | {"P1 MAE":>8} | {"UNet MAE":>10} | {"Improve":>8}')
print('-' * 40)
for idx in indices:
    sl1 = t1_n[:,:,idx].copy()
    sl2 = t2_n[:,:,idx].copy()
    ml = mask[:,:,idx].copy()
    if sl1.shape != (192, 192):
        f = (192/sl1.shape[0], 192/sl1.shape[1])
        sl1 = zoom(sl1, f, order=1)
        sl2 = zoom(sl2, f, order=1)
        ml = zoom(ml.astype(float), f, order=1) > 0.5
    x = torch.from_numpy(sl1[np.newaxis, np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        yp = np.clip(model(x).cpu().numpy()[0, 0], 0, 1)
    yp[~ml] = 0.0
    p1_inv = 1.0 - sl1; p1_inv[~ml] = 0.0
    mae_u = np.abs(yp - sl2)[ml].mean()
    mae_p = np.abs(p1_inv - sl2)[ml].mean()
    impr = (mae_p - mae_u) / mae_p * 100
    print(f'{idx:6d} | {mae_p:8.4f} | {mae_u:10.4f} | {impr:7.1f}%')

print('\nAll figures generated!')
