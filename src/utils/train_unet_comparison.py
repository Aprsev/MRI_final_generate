#!/usr/bin/env python
"""
2D U-Net for T1→T2 slice-wise conversion — compare with 3D Pix2Pix GAN.
Pre-loads volumes for fast training.
"""
import os, sys, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"D:\Desktop\ZJU\grade3\25-26spring\磁共振成像原理及应用\Labatory\final\GAN-MAT_build"
INPUT_DIR = os.path.join(BASE, "input")
FIGS_OUT = r"D:\Desktop\ZJU\grade3\25-26spring\磁共振成像原理及应用\Labatory\final\summary\GAN-MAT\figures"
os.makedirs(FIGS_OUT, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups, self.pool = nn.ModuleList(), nn.ModuleList(), nn.MaxPool2d(2)
        for f in features: self.downs.append(DoubleConv(in_ch, f)); in_ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], out_ch, 1)
    def forward(self, x):
        skips = []
        for down in self.downs: x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i//2]
            if x.shape != s.shape: x = nn.functional.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((s, x), dim=1); x = self.ups[i+1](x)
        return self.final(x)

def main():
    # Pick a few subjects, pre-load their volumes into memory
    all_subs = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    np.random.seed(42)
    np.random.shuffle(all_subs)
    # Use 4 subjects: 2 train + 1 test (keep 1 spare)
    train_subs = all_subs[:2]
    test_subs = all_subs[2:3]
    print(f"Train subjects: {train_subs}")
    print(f"Test subjects: {test_subs}")

    # Pre-load volumes
    def load_vol(sub):
        t1 = nib.load(os.path.join(INPUT_DIR, sub, "T1w_MNI.nii.gz")).get_fdata().astype(np.float32)
        t2 = nib.load(os.path.join(INPUT_DIR, sub, "output_MNI.nii.gz")).get_fdata().astype(np.float32)
        brain = t1 > 0
        bv = t1[brain]
        if len(bv) > 0: t1[brain] = (bv - bv.min()) / (bv.max() - bv.min() + 1e-8)
        return t1, t2

    train_vols = [load_vol(s) for s in train_subs]
    test_vols_raw = [load_vol(s) for s in test_subs]

    # Build training slices (sample max 30 per subject)
    train_slices = []
    for t1, t2 in train_vols:
        candidates = [(z, np.sum(t1[:,:,z] > 0)) for z in range(t1.shape[2]) if np.sum(t1[:,:,z] > 0) > 50]
        candidates.sort(key=lambda x: -x[1])
        for z, _ in candidates[:30]:
            train_slices.append((t1[:,:,z], t2[:,:,z]))
    print(f"Train slices: {len(train_slices)}")

    # DataLoaders
    class MemDataset(Dataset):
        def __init__(self, slices): self.slices = slices
        def __len__(self): return len(self.slices)
        def __getitem__(self, i):
            return (torch.from_numpy(self.slices[i][0]).unsqueeze(0).float(),
                    torch.from_numpy(self.slices[i][1]).unsqueeze(0).float())

    train_ds = MemDataset(train_slices)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    # Use smaller U-Net for faster CPU training
    model = UNet(features=[16, 32, 64, 128]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    losses = []
    for epoch in range(3):
        model.train(); el = 0
        for t1, t2 in train_loader:
            t1, t2 = t1.to(device), t2.to(device)
            pred = model(t1); loss = criterion(pred, t2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            el += loss.item()
        avg = el / len(train_loader); losses.append(avg)
        print(f"  Epoch {epoch+1}/3, Loss={avg:.6f}")

    # Save loss curve
    plt.figure(figsize=(8,4)); plt.plot(losses, 'b-o', markersize=3)
    plt.xlabel('Epoch'); plt.ylabel('L1 Loss'); plt.title('U-Net Training Convergence')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGS_OUT, "unet_training_loss.png"), dpi=150); plt.close()
    print("Saved unet_training_loss.png")

    # ─────────────────────────────────────────────────────────────
    # Generate comparison figures
    # ─────────────────────────────────────────────────────────────
    print("\nGenerating comparison figures...")
    model.eval()

    for sub_idx, (sub, (t1, t2_gan)) in enumerate(zip(test_subs, test_vols_raw)):
        # Run U-Net on all slices
        t1_t = torch.from_numpy(t1.transpose(2,0,1)).unsqueeze(1).float().to(device)
        with torch.no_grad(): t2_u = model(t1_t).cpu().numpy().squeeze(1).transpose(1,2,0)
        test_vols_raw[sub_idx] = (sub, t1, t2_gan, t2_u)

    # ── Fig 1: GAN vs U-Net slice gallery ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    sub, t1, t2_gan, t2_u = test_vols_raw[0]
    brain_area = [np.sum(t1[:,:,z] > 0) for z in range(t1.shape[2])]
    z_best = np.argmax(brain_area)

    row_labels = [('GAN', t2_gan), ('U-Net', t2_u)]
    for row, (label, data) in enumerate(row_labels):
        for col, (axis, vname) in enumerate([(2, 'Axial'), (1, 'Coronal'), (0, 'Sagittal')]):
            ax = axes[row, col]
            s = np.rot90(np.take(data, data.shape[axis]//2 if axis!=2 else z_best, axis=axis))
            ax.imshow(s, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'{label} - {vname}', fontsize=10, fontweight='bold',
                        color='green' if 'GAN' in label else 'blue')
            ax.axis('off')

        # Difference
        ax = axes[row, 3]
        diff = np.abs(t2_gan - t2_u)
        s = np.rot90(np.take(diff, z_best, axis=2))
        im = ax.imshow(s, cmap='hot', aspect='auto', vmin=0, vmax=0.5)
        ax.set_title(f'Error Map (row={row})', fontsize=9, color='red')
        ax.axis('off')
        if row == 0: plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'GAN vs U-Net: Three-View Comparison ({sub[:30]})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_OUT, "fig_gan_vs_unet_slices.png"), dpi=150, bbox_inches='tight'); plt.close()
    print("Saved fig_gan_vs_unet_slices.png")

    # ── Fig 2: Quantitative + scatter ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sub, t1, t2_gan, t2_u = test_vols_raw[0]
    brain = t1 > 0

    # Bar chart
    ax = axes[0]
    gan_err = np.abs(t2_gan[brain] - t1[brain]).mean()
    unet_err = np.abs(t2_u[brain] - t1[brain]).mean()
    ax.bar(['3D Pix2Pix GAN', '2D U-Net'], [gan_err, unet_err],
           color=['green', 'blue'], alpha=0.7, edgecolor='black', width=0.5)
    ax.set_ylabel('Mean L1 from Input T1'); ax.set_title('Output vs Input (lower=more contrast)')
    ax.grid(True, alpha=0.3, axis='y')

    # Scatter
    ax = axes[1]
    ax.scatter(t2_gan[brain][::5], t2_u[brain][::5], c='purple', alpha=0.3, s=2)
    ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Identity')
    ax.set_xlabel('3D Pix2Pix GAN'); ax.set_ylabel('2D U-Net')
    ax.set_title('Intensity Scatter'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    # Analysis
    ax = axes[2]; ax.axis('off')
    diff_mean = np.abs(t2_gan[brain] - t2_u[brain]).mean()
    txt = (
        "GAN vs U-Net Analysis\n\n"
        "3D Pix2Pix GAN:\n"
        "  + 3D context across slices\n"
        "  + Adversarial training\n"
        "  + Smooth WM/GM boundaries\n"
        "  - ~20s/slice (CPU)\n\n"
        "2D U-Net:\n"
        "  + Fast (<1s/slice)\n"
        "  + Simple architecture\n"
        "  - No 3D consistency\n\n"
        f"GAN-UNet Agreement:\n"
        f"  Mean |diff| = {diff_mean:.4f}\n"
        "  (0 = identical, 1 = opposite)"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Quantitative: 3D Pix2Pix GAN vs 2D U-Net', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_OUT, "fig_gan_vs_unet_quantitative.png"), dpi=150, bbox_inches='tight'); plt.close()
    print("Saved fig_gan_vs_unet_quantitative.png")

    # ── Fig 3: Myelin map comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sub, t1, t2_gan, t2_u = test_vols_raw[0]
    brain = t1 > 0

    for col, (t2_src, label) in enumerate([(t2_gan, 'GAN-based'), (t2_u, 'U-Net based')]):
        t1_n = (t1[brain] - t1[brain].min()) / (t1[brain].max() - t1[brain].min() + 1e-8)
        myelin = np.clip(t1_n / (t2_src[brain] + 1e-8), 0, 5)
        myelin_map = np.zeros_like(t1)
        myelin_map[brain] = myelin

        ax = axes[col]
        s = np.rot90(np.take(myelin_map, z_best, axis=2))
        im = ax.imshow(s, cmap='hot', aspect='auto', vmin=0, vmax=3)
        ax.set_title(f'Myelin Map ({label})', fontsize=10, fontweight='bold')
        ax.axis('off')
        if col == 1: plt.colorbar(im, ax=ax, fraction=0.046)

    # Difference
    ax = axes[2]
    myelin_gan = np.clip(t1_n / (t2_gan[brain] + 1e-8), 0, 5)
    myelin_unet = np.clip(t1_n / (t2_u[brain] + 1e-8), 0, 5)
    diff_m = np.abs(myelin_gan - myelin_unet)
    diff_map = np.zeros_like(t1)
    diff_map[brain] = diff_m
    s = np.rot90(np.take(diff_map, z_best, axis=2))
    im = ax.imshow(s, cmap='hot', aspect='auto', vmin=0, vmax=2)
    ax.set_title('Myelin Map |Diff|', fontsize=10, fontweight='bold', color='red')
    ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'Impact on Myelin Proxy: GAN vs U-Net', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_OUT, "fig_gan_vs_unet_myelin.png"), dpi=150, bbox_inches='tight'); plt.close()
    print("Saved fig_gan_vs_unet_myelin.png")

    print(f"\nAll figures saved to {FIGS_OUT}")

if __name__ == '__main__':
    print("="*50); print("UNet vs GAN for T1->T2"); print("="*50)
    main()
