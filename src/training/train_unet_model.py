#!/usr/bin/env python3
"""
train_unet_model.py
===================
独立训练脚本：在MNI152 T1-T2模板对上训练UNet模型。
安装PyTorch后直接运行：
  python train_unet_model.py

输出: models/t1_to_t2_unet_best.pth
"""

import os, sys
# OpenMP冲突处理：matplotlib/pytorch/numpy可能使用不同OpenMP运行时
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_fill_holes, label as ndimage_label

# === Config ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, 'MRI_analog_data', 'MRI_analog_data',
                            'data', 'templates', 'mni_icbm152_nlin_sym_09a_minc1')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_prepare_data():
    """Load MNI152 T1 and T2, normalize, extract slice pairs"""
    import nibabel as nib
    
    print('Loading MNI152 templates...')
    t1 = nib.load(os.path.join(TEMPLATE_DIR, 'mni_icbm152_t1_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
    t2 = nib.load(os.path.join(TEMPLATE_DIR, 'mni_icbm152_t2_tal_nlin_sym_09a.mnc')).get_fdata().astype(np.float64)
    
    # Joint brain mask
    mask = (t1 > t1.max()*0.1) & (t2 > t2.max()*0.1)
    for i in range(mask.shape[-1]):
        mask[:,:,i] = binary_fill_holes(mask[:,:,i])
    labeled, nf = ndimage_label(mask)
    if nf > 0:
        sizes = np.bincount(labeled.ravel())
        mask = labeled == np.argmax(sizes[1:]) + 1
    
    # Normalize
    def _norm(v):
        vn = (v - v[mask].min()) / (v[mask].max() - v[mask].min() + 1e-10)
        vn[~mask] = 0; return np.clip(vn, 0, 1)
    
    t1_n, t2_n = _norm(t1), _norm(t2)
    
    # Extract 2D slices (axial)
    slices_t1, slices_t2 = [], []
    target_size = (192, 192)
    
    for i in range(t1_n.shape[2]):
        sl1, sl2, ml = t1_n[:,:,i], t2_n[:,:,i], mask[:,:,i]
        if np.sum(ml) < 500:
            continue
        if sl1.shape != target_size:
            f = (target_size[0]/sl1.shape[0], target_size[1]/sl1.shape[1])
            sl1 = zoom(sl1, f, order=1)
            sl2 = zoom(sl2, f, order=1)
        slices_t1.append(sl1); slices_t2.append(sl2)
    
    # Train/val split
    n = len(slices_t1)
    n_val = max(1, int(n * 0.1))
    perm = np.random.RandomState(42).permutation(n)
    
    train = ([slices_t1[i] for i in perm[:-n_val]], [slices_t2[i] for i in perm[:-n_val]])
    val = ([slices_t1[i] for i in perm[-n_val:]], [slices_t2[i] for i in perm[-n_val:]])
    
    print(f'Total slices: {n}, Train: {len(train[0])}, Val: {len(val[0])}')
    return train, val


def define_unet():
    """Define 2D UNet"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
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
    
    return UNet(n_channels=1, n_classes=1, base_filters=64)


def train():
    """Main training function"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Data
    print('='*60)
    print('  UNet T1->T2 Training on MNI152')
    print('='*60)
    train_data, val_data = load_and_prepare_data()
    
    train_t1, train_t2 = train_data
    val_t1, val_t2 = val_data
    
    # To tensors
    def _to_tensor(slices):
        return torch.from_numpy(np.array(slices)[:, np.newaxis].astype(np.float32))
    
    X_train, y_train = _to_tensor(train_t1), _to_tensor(train_t2)
    X_val, y_val = _to_tensor(val_t1), _to_tensor(val_t2)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = define_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    l1_loss = nn.L1Loss()
    
    # SSIM loss
    def ssim_loss(y_true, y_pred, w=11):
        pad = w//2
        mu1 = nn.functional.avg_pool2d(y_true, w, 1, pad)
        mu2 = nn.functional.avg_pool2d(y_pred, w, 1, pad)
        s1 = nn.functional.avg_pool2d(y_true*y_true, w, 1, pad) - mu1**2
        s2 = nn.functional.avg_pool2d(y_pred*y_pred, w, 1, pad) - mu2**2
        s12 = nn.functional.avg_pool2d(y_true*y_pred, w, 1, pad) - mu1*mu2
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1*mu2+c1)*(2*s12+c2))/((mu1**2+mu2**2+c1)*(s1+s2+c2))
        return (1 - ssim_map).mean().mean()
    
    best_val = float('inf')
    n_epochs = 80
    history = {'train_loss': [], 'val_loss': []}

    # Prepare visualization directory
    VIS_DIR = os.path.join(CURRENT_DIR, 'training_vis')
    os.makedirs(VIS_DIR, exist_ok=True)

    # Validation samples for consistent visualization
    vis_indices = list(range(min(6, len(val_t1))))
    vis_t1 = [val_t1[i] for i in vis_indices]
    vis_t2 = [val_t2[i] for i in vis_indices]

    print(f'Training for {n_epochs} epochs...')
    print('-'*50)

    for epoch in range(n_epochs):
        # Train
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
        
        # Val
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

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {tl:.6f} | Val Loss: {vl:.6f}')

        # --- Per-epoch visualization snapshot ---
        if (epoch+1) % 10 == 0 or epoch == 0:
            model.eval()
            n_vis = len(vis_t1)
            fig, axes = plt.subplots(4, n_vis, figsize=(n_vis*3.5, 12))
            with torch.no_grad():
                for i in range(n_vis):
                    x = torch.from_numpy(vis_t1[i][np.newaxis, np.newaxis].astype(np.float32)).to(device)
                    yp = np.clip(model(x).cpu().numpy()[0, 0], 0, 1)
                    err = np.abs(yp - vis_t2[i])
                    # T1 input
                    axes[0, i].imshow(vis_t1[i], cmap='gray', vmin=0, vmax=1)
                    axes[0, i].set_title(f'T1 Input'); axes[0, i].axis('off')
                    if i == 0: axes[0, i].set_ylabel(f'Epoch {epoch+1}', fontsize=11)
                    # Predicted T2
                    axes[1, i].imshow(yp, cmap='gray', vmin=0, vmax=1)
                    axes[1, i].set_title('Synth T2'); axes[1, i].axis('off')
                    # Ground truth T2
                    axes[2, i].imshow(vis_t2[i], cmap='gray', vmin=0, vmax=1)
                    axes[2, i].set_title('Real T2'); axes[2, i].axis('off')
                    # Error map
                    im = axes[3, i].imshow(err, cmap='hot', vmin=0, vmax=0.3)
                    axes[3, i].set_title(f'|Error|'); axes[3, i].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, f'epoch_{epoch+1:03d}.png'), dpi=150)
            plt.close()

        if vl < best_val:
            best_val = vl
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl,
                'model_config': {'n_channels': 1, 'n_classes': 1, 'base_filters': 64},
            }, os.path.join(MODELS_DIR, 't1_to_t2_unet_best.pth'))
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'n_channels': 1, 'n_classes': 1, 'base_filters': 64},
    }, os.path.join(MODELS_DIR, 't1_to_t2_unet_final.pth'))
    
    print('-'*50)
    print(f'Best val loss: {best_val:.6f}')
    print(f'Models saved to {MODELS_DIR}/')
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pth'):
            print(f'  {f} ({os.path.getsize(os.path.join(MODELS_DIR,f))/1e6:.2f} MB)')
    print('Done!')

    # ============ Comprehensive Final Visualization ============
    try:
        from matplotlib.gridspec import GridSpec

        # Reload best model for final evaluation
        best_ckpt = torch.load(os.path.join(MODELS_DIR, 't1_to_t2_unet_best.pth'),
                               map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        model.eval()

        # ---------- 1. Loss curves ----------
        fig0, ax0 = plt.subplots(1, 1, figsize=(8, 5))
        epochs_arr = np.arange(1, n_epochs + 1)
        ax0.plot(epochs_arr, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        ax0.plot(epochs_arr, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
        ax0.axvline(x=best_ckpt['epoch'] + 1, color='g', linestyle='--', alpha=0.7,
                    label=f"Best @ epoch {best_ckpt['epoch'] + 1}")
        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('Loss (L1 + 0.5×SSIM)')
        ax0.set_title('Training History')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'loss_curves.png'), dpi=150)
        plt.close()
        print(f'Loss curves: models/loss_curves.png')

        # ---------- 2. Full validation evaluation with metrics ----------
        n_show = min(10, len(val_t1))
        fig1 = plt.figure(figsize=(n_show * 2.8, 10))
        gs = GridSpec(5, n_show, figure=fig1, hspace=0.15, wspace=0.05)

        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for i in range(n_show):
                x = torch.from_numpy(val_t1[i][np.newaxis, np.newaxis].astype(np.float32)).to(device)
                yt = torch.from_numpy(val_t2[i][np.newaxis, np.newaxis].astype(np.float32)).to(device)
                yp_t = model(x)
                yp_np = np.clip(yp_t.cpu().numpy()[0, 0], 0, 1)
                yt_np = val_t2[i]

                # Metrics
                mse = np.mean((yp_np - yt_np) ** 2)
                psnr_val = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))
                # SSIM (simplified 2D)
                mu_x = gaussian_filter(yp_np, sigma=1.5)
                mu_y = gaussian_filter(yt_np, sigma=1.5)
                sigma_x = gaussian_filter(yp_np ** 2, sigma=1.5) - mu_x ** 2
                sigma_y = gaussian_filter(yt_np ** 2, sigma=1.5) - mu_y ** 2
                sigma_xy = gaussian_filter(yp_np * yt_np, sigma=1.5) - mu_x * mu_y
                c1, c2 = (0.01) ** 2, (0.03) ** 2
                ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
                ssim_val = np.mean(ssim_map)

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                err = np.abs(yp_np - yt_np)

                # Row 0: T1 input
                ax = fig1.add_subplot(gs[0, i])
                ax.imshow(val_t1[i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'T1 #{i+1}', fontsize=9)
                ax.axis('off')

                # Row 1: Predicted T2
                ax = fig1.add_subplot(gs[1, i])
                ax.imshow(yp_np, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Synth T2', fontsize=9)
                ax.axis('off')

                # Row 2: Ground truth T2
                ax = fig1.add_subplot(gs[2, i])
                ax.imshow(yt_np, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Real T2', fontsize=9)
                ax.axis('off')

                # Row 3: Error map
                ax = fig1.add_subplot(gs[3, i])
                im = ax.imshow(err, cmap='hot', vmin=0, vmax=0.3)
                ax.set_title(f'|Error|', fontsize=9)
                ax.axis('off')

                # Row 4: Metrics text
                ax = fig1.add_subplot(gs[4, i])
                ax.text(0.5, 0.6, f'PSNR\n{psnr_val:.2f} dB', ha='center', va='center',
                        fontsize=8, transform=ax.transAxes)
                ax.text(0.5, 0.25, f'SSIM\n{ssim_val:.4f}', ha='center', va='center',
                        fontsize=8, transform=ax.transAxes)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.axis('off')

        plt.suptitle(f'UNet T1→T2 — Validation Results (Best Model @ Epoch {best_ckpt["epoch"]+1})',
                     fontsize=13, y=1.02)
        plt.savefig(os.path.join(MODELS_DIR, 'validation_results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Validation results: models/validation_results.png')

        # ---------- 3. Metrics distribution ----------
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(psnr_list, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.axvline(np.mean(psnr_list), color='darkred', linestyle='--',
                    label=f'Mean: {np.mean(psnr_list):.2f} dB')
        ax1.set_xlabel('PSNR (dB)')
        ax1.set_ylabel('Number of Slices')
        ax1.set_title(f'PSNR Distribution (n={len(psnr_list)})')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.hist(ssim_list, bins=15, color='coral', edgecolor='white', alpha=0.8)
        ax2.axvline(np.mean(ssim_list), color='darkred', linestyle='--',
                    label=f'Mean: {np.mean(ssim_list):.4f}')
        ax2.set_xlabel('SSIM')
        ax2.set_ylabel('Number of Slices')
        ax2.set_title(f'SSIM Distribution (n={len(ssim_list)})')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'metrics_distribution.png'), dpi=150)
        plt.close()
        print(f'Metrics distribution: models/metrics_distribution.png')

        # ---------- 4. Full volume montage (all validation slices) ----------
        n_all = len(val_t1)
        n_cols = 8
        n_rows = (n_all + n_cols - 1) // n_cols
        fig3, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2.5, 10))
        with torch.no_grad():
            for i in range(min(n_all, n_cols)):
                x = torch.from_numpy(val_t1[i][np.newaxis, np.newaxis].astype(np.float32)).to(device)
                yp = np.clip(model(x).cpu().numpy()[0, 0], 0, 1)
                axes[0, i].imshow(val_t1[i], cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title(f'T1 #{i+1}', fontsize=8); axes[0, i].axis('off')
                axes[1, i].imshow(yp, cmap='gray', vmin=0, vmax=1)
                axes[1, i].set_title('Synth T2', fontsize=8); axes[1, i].axis('off')
                axes[2, i].imshow(val_t2[i], cmap='gray', vmin=0, vmax=1)
                axes[2, i].set_title('Real T2', fontsize=8); axes[2, i].axis('off')
                axes[3, i].imshow(np.abs(yp - val_t2[i]), cmap='hot', vmin=0, vmax=0.3)
                axes[3, i].set_title('|Error|', fontsize=8); axes[3, i].axis('off')
            for j in range(min(n_all, n_cols), n_cols):
                for k in range(4):
                    axes[k, j].axis('off')
        plt.suptitle(f'T1→T2 Synthesis — All Validation Slices  (Mean PSNR: {np.mean(psnr_list):.2f} dB, '
                     f'Mean SSIM: {np.mean(ssim_list):.4f})', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'validation_montage.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Validation montage: models/validation_montage.png')

        # ---------- 5. Per-epoch GIF (if imageio available) ----------
        try:
            import imageio
            import glob
            images = []
            for fpath in sorted(glob.glob(os.path.join(VIS_DIR, 'epoch_*.png'))):
                images.append(imageio.imread(fpath))
            if images:
                gif_path = os.path.join(MODELS_DIR, 'training_progress.gif')
                imageio.mimsave(gif_path, images, fps=2, loop=0)
                print(f'Training progress GIF: models/training_progress.gif')
        except ImportError:
            print('  (imageio not installed, skip GIF creation)')
        except Exception as e:
            print(f'  (GIF creation skipped: {e})')

        print('\nAll visualizations saved to models/')
    except Exception as e:
        print(f'\nVisualization warning: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    train()
