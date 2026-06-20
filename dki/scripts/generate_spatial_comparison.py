#!/usr/bin/env python3
"""
Generate spatial D/K map comparison figures for the final report.
"""

import sys, os
from pathlib import Path

# Add the DKI pipeline code to path
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dki_utils import (
    DkiData, load_phase2_sample, add_rician_noise,
    polynomial_dki_fit, mono_adc_fit_as_d,
    predict_model, predict_cnn_model,
    phase2_voxel_arrays, list_phase2_files,
)

# ---------- Configuration ----------
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "03_Phase2_UNet_Synthesis_DKI"
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "real_dki" / "checkpoints"
FIGURE_OUT = Path(__file__).resolve().parent.parent / "outputs" / "real_dki" / "figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_INDICES = [0, 10, 20]
TRAIN_SIGMA = 0.01
SEED = 42
D_MAX = 0.0035
K_MAX = 2.0
NOISE_SIGMA = 0.03

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Method definitions ----------
METHODS = [
    ("GT", None, False),
    ("mono_adc_as_d", None, False),
    ("poly_dki_fit", None, False),
    ("supervised_mlp", "supervised", False),
    ("pinn_log_no_gt", "pinn_log", False),
    ("semi_supervised_mlp", "semi_supervised", False),
    ("cnn_supervised", "cnn_supervised", True),
    ("cnn_pinn_log", "cnn_pinn_log", True),
]

DISPLAY_NAMES = {
    "GT": "Ground Truth",
    "mono_adc_as_d": "Mono ADC (as D)",
    "poly_dki_fit": "Poly DKI Fit",
    "supervised_mlp": "Supervised MLP",
    "pinn_log_no_gt": "PINN Log (no GT)",
    "semi_supervised_mlp": "Semi-Supervised",
    "cnn_supervised": "CNN Supervised",
    "cnn_pinn_log": "CNN PINN Log",
}

# ---------- Load data ----------
print("Loading Phase2 data ...")
all_files = sorted(list_phase2_files(DATA_ROOT, max_files=0))
print(f"  Total files available: {len(all_files)}")

selected_files = [all_files[i] for i in SAMPLE_INDICES if i < len(all_files)]
n_samples = len(selected_files)
print(f"  Selected {n_samples} samples: indices {SAMPLE_INDICES[:n_samples]}")

rng = np.random.default_rng(42)
dwi_list, s0_list, d_list, k_list, mask_list = [], [], [], [], []
bvals = None
for fpath in selected_files:
    s = load_phase2_sample(fpath)
    noisy = add_rician_noise(rng, s["dwi_clean"], NOISE_SIGMA)
    dwi_list.append(noisy)
    s0_list.append(s["s0"])
    d_list.append(s["d"])
    k_list.append(s["k"])
    mask_list.append(s["mask"])
    if bvals is None:
        bvals = s["bvals"]

data = DkiData(
    dwi_noisy=np.stack(dwi_list).astype(np.float32),
    dwi_clean=np.stack([load_phase2_sample(f)["dwi_clean"] for f in selected_files]).astype(np.float32),
    s0=np.stack(s0_list).astype(np.float32),
    d=np.stack(d_list).astype(np.float32),
    k=np.stack(k_list).astype(np.float32),
    mask=np.stack(mask_list).astype(np.float32),
    bvals=bvals,
    sigma=np.full(n_samples, NOISE_SIGMA, dtype=np.float32),
)
print(f"  Data shape: {data.dwi_noisy.shape}")

# ---------- Run predictions ----------
predictions = {}

print("\nRunning traditional fits ...")
d_mono = mono_adc_fit_as_d(data, D_MAX)
k_mono = np.zeros_like(d_mono)
predictions["mono_adc_as_d"] = (d_mono, k_mono)
print("  mono_adc_as_d done")

d_poly, k_poly = polynomial_dki_fit(data, D_MAX, K_MAX)
predictions["poly_dki_fit"] = (d_poly, k_poly)
print("  poly_dki_fit done")

for method_name, ckpt_name, is_cnn in METHODS:
    if ckpt_name is None or method_name == "GT":
        continue

    ckpt_path = CHECKPOINT_ROOT / f"train_sigma_{TRAIN_SIGMA:.2g}" / f"seed{SEED}_{ckpt_name}.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found: {ckpt_path}")
        continue

    print(f"  Loading {method_name} from {ckpt_path.name} ...")
    try:
        if is_cnn:
            d_pred, k_pred = predict_cnn_model(ckpt_path, data, device)
        else:
            d_pred, k_pred = predict_model(ckpt_path, data, device)
        predictions[method_name] = (d_pred, k_pred)
        print(f"    -> D range: [{d_pred.min():.6f}, {d_pred.max():.6f}], K range: [{k_pred.min():.4f}, {k_pred.max():.4f}]")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

# ---------- Plotting helpers ----------
def _imshow_with_colorbar(ax, data_arr, mask, cmap="viridis", vmin=None, vmax=None,
                           title="", cbar_label="", fontsize=8):
    masked = np.where(mask, data_arr, np.nan)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=fontsize - 2)
    cbar.ax.tick_params(labelsize=fontsize - 3)
    return im

# ---------- Compact overview: key methods ----------
print("\nGenerating compact overview figures ...")
KEY_METHODS = ["GT", "poly_dki_fit", "supervised_mlp", "pinn_log_no_gt",
               "semi_supervised_mlp", "cnn_supervised", "cnn_pinn_log"]
KEY_DISPLAY = {k: DISPLAY_NAMES[k] for k in KEY_METHODS}

for idx in range(min(2, n_samples)):
    fig, axes = plt.subplots(2, len(KEY_METHODS), figsize=(3 * len(KEY_METHODS) + 1, 6))
    m = data.mask[idx].astype(bool)
    gt_d = data.d[idx]
    gt_k = data.k[idx]

    d_min, d_max = gt_d[m].min(), gt_d[m].max()
    k_min, k_max = gt_k[m].min(), gt_k[m].max()
    d_err_max, k_err_max = 0.0, 0.0
    for method_name in KEY_METHODS[1:]:
        if method_name not in predictions:
            continue
        p_d, p_k = predictions[method_name]
        v_d = p_d[idx] if p_d.ndim == 3 else p_d
        v_k = p_k[idx] if p_k.ndim == 3 else p_k
        d_min = min(d_min, v_d[m].min())
        d_max = max(d_max, v_d[m].max())
        k_min = min(k_min, v_k[m].min())
        k_max = max(k_max, v_k[m].max())
        d_err_max = max(d_err_max, abs(v_d[m] - gt_d[m]).max())
        k_err_max = max(k_err_max, abs(v_k[m] - gt_k[m]).max())

    d_margin = (d_max - d_min) * 0.05
    k_margin = (k_max - k_min) * 0.05
    if d_margin < 1e-8:
        d_margin = 0.01
    if k_margin < 1e-8:
        k_margin = 0.01
    d_err_max *= 1.1
    k_err_max *= 1.1

    for col, method_name in enumerate(KEY_METHODS):
        if method_name == "GT":
            _imshow_with_colorbar(axes[0, col], gt_d, m, "viridis",
                                 vmin=d_min - d_margin, vmax=d_max + d_margin,
                                 title="GT D", cbar_label="D (mm²/s)", fontsize=9)
            _imshow_with_colorbar(axes[1, col], gt_k, m, "plasma",
                                 vmin=k_min - k_margin, vmax=k_max + k_margin,
                                 title="GT K", cbar_label="K", fontsize=9)
        elif method_name in predictions:
            p_d, p_k = predictions[method_name]
            v_d = p_d[idx] if p_d.ndim == 3 else p_d
            v_k = p_k[idx] if p_k.ndim == 3 else p_k
            _imshow_with_colorbar(axes[0, col], v_d, m, "viridis",
                                 vmin=d_min - d_margin, vmax=d_max + d_margin,
                                 title=KEY_DISPLAY[method_name],
                                 cbar_label="D (mm²/s)", fontsize=9)
            _imshow_with_colorbar(axes[1, col], v_k, m, "plasma",
                                 vmin=k_min - k_margin, vmax=k_max + k_margin,
                                 title=KEY_DISPLAY[method_name],
                                 cbar_label="K", fontsize=9)
        else:
            axes[0, col].axis("off")
            axes[1, col].axis("off")

    fig.suptitle(f"D/K Map Comparison — Sample #{SAMPLE_INDICES[idx]} (σ_noise={NOISE_SIGMA})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = FIGURE_OUT / f"dki_spatial_overview_sample{SAMPLE_INDICES[idx]}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {fname}")

# ---------- Error overview ----------
print("\nGenerating error overview figures ...")
for idx in range(min(2, n_samples)):
    fig, axes = plt.subplots(2, len(KEY_METHODS) - 1, figsize=(3 * (len(KEY_METHODS) - 1) + 1, 6))
    m = data.mask[idx].astype(bool)
    gt_d = data.d[idx]
    gt_k = data.k[idx]

    d_errs, k_errs = [], []
    for method_name in KEY_METHODS[1:]:
        if method_name not in predictions:
            continue
        p_d, p_k = predictions[method_name]
        v_d = p_d[idx] if p_d.ndim == 3 else p_d
        v_k = p_k[idx] if p_k.ndim == 3 else p_k
        d_errs.append((v_d - gt_d)[m])
        k_errs.append((v_k - gt_k)[m])
    d_err_max = max(abs(e).max() for e in d_errs) * 1.1 if d_errs else 0.001
    k_err_max = max(abs(e).max() for e in k_errs) * 1.1 if k_errs else 0.1

    for i, method_name in enumerate(KEY_METHODS[1:]):
        if method_name not in predictions:
            axes[0, i].axis("off")
            axes[1, i].axis("off")
            continue
        p_d, p_k = predictions[method_name]
        v_d = p_d[idx] if p_d.ndim == 3 else p_d
        v_k = p_k[idx] if p_k.ndim == 3 else p_k
        d_err = v_d - gt_d
        k_err = v_k - gt_k

        _imshow_with_colorbar(axes[0, i], d_err, m, "RdBu_r",
                             vmin=-d_err_max, vmax=d_err_max,
                             title=f"D err: {KEY_DISPLAY[method_name]}",
                             cbar_label="ΔD (mm²/s)", fontsize=8)
        _imshow_with_colorbar(axes[1, i], k_err, m, "RdBu_r",
                             vmin=-k_err_max, vmax=k_err_max,
                             title=f"K err: {KEY_DISPLAY[method_name]}",
                             cbar_label="ΔK", fontsize=8)

    fig.suptitle(f"Error Maps — Sample #{SAMPLE_INDICES[idx]} (σ_noise={NOISE_SIGMA})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = FIGURE_OUT / f"dki_error_overview_sample{SAMPLE_INDICES[idx]}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {fname}")

print(f"\n{'='*60}")
print("GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\nFigures saved to: {FIGURE_OUT}")
print(f"\nFiles created:")
for f in sorted(FIGURE_OUT.glob("dki_spatial_overview*.png")):
    print(f"  {f.name}")
for f in sorted(FIGURE_OUT.glob("dki_error_overview*.png")):
    print(f"  {f.name}")
