#!/usr/bin/env python
"""
Compare pre-trained 2D U-Net (80 epoch) T1→T2 with 3D Pix2Pix GAN T1→T2.
Uses model from: final/models/t1_to_t2_unet_best.pth
GAN results from: GAN-MAT_build/input/*/output_MNI.nii.gz
"""

import os, sys, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths
BASE = r"D:\Desktop\ZJU\grade3\25-26spring\磁共振成像原理及应用\Labatory\final"
MODEL_PATH = os.path.join(BASE, "models", "t1_to_t2_unet_best.pth")
GAN_INPUT = os.path.join(BASE, "GAN-MAT_build", "input")
FIGS_OUT = os.path.join(BASE, "summary", "GAN-MAT", "figures")
os.makedirs(FIGS_OUT, exist_ok=True)

device = torch.device('cpu')

# ── U-Net Architecture ──
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))
    def forward(self, x): return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = ConvBlock(1, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.d4 = Down(512, 1024)
        self.u1 = Up(1024, 512)
        self.u2 = Up(512, 256)
        self.u3 = Up(256, 128)
        self.u4 = Up(128, 64)
        self.out = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.d1(x1); x3 = self.d2(x2)
        x4 = self.d3(x3); x5 = self.d4(x4)
        x = self.u1(x5, x4); x = self.u2(x, x3)
        x = self.u3(x, x2); x = self.u4(x, x1)
        return self.out(x)

# Load U-Net
print("Loading U-Net model...")
unet = UNet2D().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
unet.load_state_dict(ckpt['model_state_dict'])
unet.eval()
print(f"  U-Net loaded (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.6f})")

# Load GAN model info - use cached GAN results
print("Loading GAN results from:", GAN_INPUT)

# ── Subjects to compare ──
np.random.seed(42)
all_subs = sorted([d for d in os.listdir(GAN_INPUT) if os.path.isdir(os.path.join(GAN_INPUT, d))])
test_subs = ['mricourse_child1_20240803', 'mricourse_jinmeng_20240507', 'mricourse_hemingming_20230603']

def prepare_slice(t1_slice):
    """Normalize T1 slice to [0,1] for U-Net input."""
    brain = t1_slice > 0
    if brain.any():
        bv = t1_slice[brain]
        t1_slice = t1_slice.copy()
        t1_slice[brain] = (bv - bv.min()) / (bv.max() - bv.min() + 1e-8)
    return t1_slice

def run_unet_on_volume(t1_volume):
    """Run 2D U-Net slice-by-slice on a 3D volume, return T2 volume."""
    d, h, w = t1_volume.shape
    t2_vol = np.zeros((d, 192, 192), dtype=np.float32)
    for z in range(d):
        t1_slice = t1_volume[z]
        if t1_slice.max() == 0:
            continue
        # Resize to 192x192 if needed
        from skimage.transform import resize
        t1_192 = resize(t1_slice, (192, 192), order=1, preserve_range=True)
        t1_norm = prepare_slice(t1_192)
        # Inference
        tensor = torch.from_numpy(t1_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            t2_pred = unet(tensor).cpu().numpy()[0, 0]
        t2_vol[z] = t2_pred
    return t2_vol

# ── Generate comparison data ──
print("\nGenerating U-Net T2 predictions...")
results = []
for sub in test_subs:
    t1_path = os.path.join(GAN_INPUT, sub, "T1w_MNI.nii.gz")
    t2_gan_path = os.path.join(GAN_INPUT, sub, "output_MNI.nii.gz")
    if not os.path.exists(t1_path) or not os.path.exists(t2_gan_path):
        print(f"  Skipping {sub}: missing data")
        continue
    
    t1 = nib.load(t1_path).get_fdata().astype(np.float32)
    t2_gan = nib.load(t2_gan_path).get_fdata().astype(np.float32)
    
    # Run U-Net: process each axial slice
    t2_unet = run_unet_on_volume(t1.transpose(2, 0, 1))  # (Z, 227, 272) -> (Z, 192, 192)
    
    results.append((sub, t1, t2_gan, t2_unet))
    print(f"  {sub}: GAN shape={t2_gan.shape}, UNet shape={t2_unet.shape}")

# ── Figure 1: GAN vs U-Net side by side (axial slices) ──
print("\nGenerating comparison figures...")
fig, axes = plt.subplots(3, 5, figsize=(20, 10))

for row, (sub, t1, t2_gan, t2_unet) in enumerate(results):
    # Find best slice
    brain_areas = [np.sum(t1[:,:,z] > 0) for z in range(t1.shape[2])]
    z = np.argmax(brain_areas)
    
    # T1 reference (cropped to 192 region)
    t1_s = t1[:, :, z]
    from skimage.transform import resize
    t1_192 = resize(t1_s, (192, 192), order=1, preserve_range=True)
    
    # GAN T2 (cropped to center 192x192 region)
    cy, cx = t2_gan.shape[0]//2, t2_gan.shape[1]//2
    t2_gan_crop = t2_gan[cy-96:cy+96, cx-96:cx+96, z]
    
    # U-Net T2 (already 192x192)
    t2_unet_s = t2_unet[z]
    
    # Normalize for display
    def norm_01(x):
        x = x.copy()
        b = x > 0
        if b.any():
            x[b] = (x[b] - x[b].min()) / (x[b].max() - x[b].min() + 1e-8)
        return x
    
    t2_g = norm_01(t2_gan_crop)
    t2_u = norm_01(t2_unet_s)
    
    # Row: T1, GAN, U-Net, Difference, Myelin comparison
    cols = [
        (t1_192, 'T1 Input', 'gray', None),
        (t2_g, '3D Pix2Pix GAN', 'gray', (0, 1)),
        (t2_u, '2D U-Net (80ep)', 'gray', (0, 1)),
        (np.abs(t2_g - t2_u), '|GAN-UNet| Diff', 'hot', (0, 0.5)),
    ]
    
    for col, (data, title, cmap, vrange) in enumerate(cols):
        ax = axes[row, col]
        kwargs = {'vmin': vrange[0], 'vmax': vrange[1]} if vrange else {}
        ax.imshow(np.rot90(data), cmap=cmap, aspect='auto', **kwargs)
        ax.set_title(title, fontsize=9, fontweight='bold',
                    color=['black', 'green', 'blue', 'red'][col])
        ax.axis('off')
        if row == 0 and col == 3:
            plt.colorbar(ax.images[-1], ax=ax, fraction=0.046)
    
    # Myelin map comparison (5th column)
    ax = axes[row, 4]
    t1_n = norm_01(t1_192)
    myelin_g = np.clip(t1_n / (t2_g + 1e-8), 0, 5)
    myelin_u = np.clip(t1_n / (t2_u + 1e-8), 0, 5)
    myelin_diff = np.abs(myelin_g - myelin_u)
    im = ax.imshow(np.rot90(myelin_diff), cmap='hot', aspect='auto', vmin=0, vmax=2)
    ax.set_title('Myelin |GAN-UNet|', fontsize=9, fontweight='bold', color='red')
    ax.axis('off')
    if row == 0: plt.colorbar(im, ax=ax, fraction=0.046)
    
    short = sub.replace('mricourse_', '').replace('MRIcourse_', '')[:14]
    axes[row, 0].set_ylabel(f'{short}\n(z={z})', fontsize=7, fontweight='bold')

plt.suptitle('GAN (3D Pix2Pix) vs U-Net (2D, 80 epochs): T1→T2 Conversion',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_OUT, "fig_gan_vs_unet_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_gan_vs_unet_comparison.png")

# ── Figure 2: Quantitative metrics ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
all_gan_l1, all_unet_l1, all_gu_diff = [], [], []

for sub, t1, t2_gan, t2_unet in results:
    brain_areas = [np.sum(t1[:,:,z] > 0) for z in range(t1.shape[2])]
    z = np.argmax(brain_areas)
    cy, cx = t2_gan.shape[0]//2, t2_gan.shape[1]//2
    t2_g = t2_gan[cy-96:cy+96, cx-96:cx+96, z]
    t2_u = t2_unet[z]
    t1_s = resize(t1[:,:,z], (192, 192), order=1, preserve_range=True)
    
    brain = t1_s > 0
    all_gan_l1.append(np.abs(t2_g[brain] - t1_s[brain]).mean())
    all_unet_l1.append(np.abs(t2_u[brain] - t1_s[brain]).mean())
    all_gu_diff.append(np.abs(t2_g[brain] - t2_u[brain]).mean())

# Bar: L1 to input T1
ax = axes[0, 0]
x = np.arange(len(results))
w = 0.35
ax.bar(x - w/2, all_gan_l1, w, label='3D Pix2Pix GAN', color='green', alpha=0.7)
ax.bar(x + w/2, all_unet_l1, w, label='2D U-Net (80ep)', color='blue', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([s[:10] for s in test_subs[:len(results)]], fontsize=7)
ax.set_ylabel('L1 Distance from T1'); ax.set_title('Output Contrast vs T1 Input\n(lower = more T2-like)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

# Bar: GAN vs U-Net agreement
ax = axes[0, 1]
ax.bar(range(len(results)), all_gu_diff, color='red', alpha=0.6, edgecolor='black')
ax.set_xticks(range(len(results))); ax.set_xticklabels([s[:10] for s in test_subs[:len(results)]], fontsize=7)
ax.set_ylabel('Mean |GAN - UNet|'); ax.set_title('GAN vs U-Net Disagreement')
ax.grid(True, alpha=0.3, axis='y')

# Scatter: GAN vs U-Net intensities (all subjects combined)
ax = axes[0, 2]
all_g, all_u = [], []
for sub, t1, t2_gan, t2_unet in results:
    for z in range(0, t1.shape[2], 5):  # sample every 5 slices
        if np.sum(t1[:,:,z] > 0) < 100: continue
        cy, cx = t2_gan.shape[0]//2, t2_gan.shape[1]//2
        g = t2_gan[cy-96:cy+96, cx-96:cx+96, z]
        u = t2_unet[z]
        b = resize(t1[:,:,z], (192, 192), order=1, preserve_range=True) > 0
        all_g.extend(g[b][::5]); all_u.extend(u[b][::5])
ax.scatter(all_g, all_u, c='purple', alpha=0.2, s=1)
ax.plot([0, max(all_g)], [0, max(all_g)], 'k--', alpha=0.5, label='Identity')
ax.set_xlabel('3D Pix2Pix GAN'); ax.set_ylabel('2D U-Net')
ax.set_title('Intensity Scatter (all slices)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Histogram: distribution comparison
ax = axes[1, 0]
sub, t1, t2_gan, t2_unet = results[0]
brain_areas = [np.sum(t1[:,:,z] > 0) for z in range(t1.shape[2])]
z = np.argmax(brain_areas)
cy, cx = t2_gan.shape[0]//2, t2_gan.shape[1]//2
t2_g = t2_gan[cy-96:cy+96, cx-96:cx+96, z]
t2_u = t2_unet[z]
t1_s = resize(t1[:,:,z], (192, 192), order=1, preserve_range=True)
b = t1_s > 0
ax.hist(t2_g[b], bins=50, alpha=0.5, color='green', label='GAN-T2', density=True)
ax.hist(t2_u[b], bins=50, alpha=0.5, color='blue', label='UNet-T2', density=True)
ax.set_xlabel('T2 Intensity'); ax.set_ylabel('Density')
ax.set_title('T2 Intensity Distribution (brain voxels)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Myelin proxy comparison
ax = axes[1, 1]
t1_n = (t1_s[b] - t1_s[b].min()) / (t1_s[b].max() - t1_s[b].min() + 1e-8)
myelin_g = np.clip(t1_n / (t2_g[b] + 1e-8), 0, 5)
myelin_u = np.clip(t1_n / (t2_u[b] + 1e-8), 0, 5)
ax.hist(myelin_g, bins=50, alpha=0.5, color='green', label='GAN Myelin', density=True)
ax.hist(myelin_u, bins=50, alpha=0.5, color='blue', label='UNet Myelin', density=True)
ax.set_xlabel('T1w/T2w Ratio'); ax.set_ylabel('Density')
ax.set_title('Myelin Proxy Distribution')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Summary text
ax = axes[1, 2]; ax.axis('off')
txt = (
    f"Comparison Summary (3 subjects)\n\n"
    f"GAN L1 from T1: {np.mean(all_gan_l1):.4f}\n"
    f"UNet L1 from T1: {np.mean(all_unet_l1):.4f}\n"
    f"GAN-UNet |diff|: {np.mean(all_gu_diff):.4f}\n\n"
    "3D Pix2Pix GAN:\n"
    "  • 3D convolutions → spatial consistency\n"
    "  • Adversarial loss → realistic texture\n"
    "  • ~20s inference (CPU)\n\n"
    "2D U-Net (80 epochs):\n"
    "  • Slice-by-slice → fast inference\n"
    "  • L1 loss → may blur fine details\n"
    "  • <1s per volume (CPU)\n\n"
    f"U-Net trained {ckpt.get('epoch')} epochs,\n"
    f"val_loss = {ckpt.get('val_loss'):.5f}"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8, va='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Quantitative: 3D Pix2Pix GAN vs 2D U-Net (80 epochs)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_OUT, "fig_gan_vs_unet_quantitative.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_gan_vs_unet_quantitative.png")

# ── Figure 3: Multi-slice error maps ──
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for row, (sub, t1, t2_gan, t2_unet) in enumerate(results):
    brain_areas = [np.sum(t1[:,:,z] > 0) for z in range(t1.shape[2])]
    zs = np.argsort(brain_areas)[-4:][::-1]
    for col, z in enumerate(zs):
        ax = axes[row, col]
        cy, cx = t2_gan.shape[0]//2, t2_gan.shape[1]//2
        t2_g = t2_gan[cy-96:cy+96, cx-96:cx+96, z]
        t2_u = t2_unet[z]
        diff = np.abs(t2_g - t2_u)
        im = ax.imshow(np.rot90(diff), cmap='hot', aspect='auto', vmin=0, vmax=0.4)
        ax.set_title(f'z={z}', fontsize=8); ax.axis('off')
        if row == 0 and col == 3: plt.colorbar(im, ax=ax, fraction=0.046, label='|GAN-UNet|')
    short = sub.replace('mricourse_', '').replace('MRIcourse_', '')[:16]
    axes[row, 0].set_ylabel(short, fontsize=7, fontweight='bold')

plt.suptitle('Spatial Error Maps: |3D Pix2Pix GAN − 2D U-Net| per slice',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_OUT, "fig_gan_vs_unet_errormap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_gan_vs_unet_errormap.png")

# ── Figure 4: Loss curve from U-Net training ──
import matplotlib.image as mpimg
loss_curve_path = os.path.join(BASE, "models", "loss_curves.png")
if os.path.exists(loss_curve_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    img = mpimg.imread(loss_curve_path)
    ax.imshow(img); ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_OUT, "unet_training_loss.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved unet_training_loss.png (from pre-existing training curves)")

print(f"\nAll comparison figures saved to {FIGS_OUT}")
