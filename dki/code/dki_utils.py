"""
DKI PINN Extension — Shared Utilities
======================================
Data generation, models, losses, metrics, and plotting for the
DKI phantom -> network -> D/K estimation -> PINN training pipeline.

Supports:
  - Procedural brain-like phantom (generate_dataset)
  - Phase2 real simulated DKI data (load_phase2_data)

Environment: MRI_final (conda)
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DkiData:
    """Container for a DKI phantom dataset (image-level)."""
    dwi_noisy: np.ndarray       # [N, B, H, W]
    dwi_clean: np.ndarray       # [N, B, H, W]
    s0: np.ndarray              # [N, H, W]
    d: np.ndarray               # [N, H, W]
    k: np.ndarray               # [N, H, W]
    mask: np.ndarray            # [N, H, W]
    bvals: np.ndarray           # [B]
    sigma: np.ndarray           # [N]


@dataclass(frozen=True)
class Paths:
    root: Path
    metrics: Path
    figures: Path
    checkpoints: Path
    reports: Path


# ---------------------------------------------------------------------------
# Paths & I/O helpers
# ---------------------------------------------------------------------------


def make_paths(root: str | Path) -> Paths:
    root = Path(root)
    p = Paths(root, root / "metrics", root / "figures", root / "checkpoints", root / "reports")
    for d in [p.root, p.metrics, p.figures, p.checkpoints, p.reports]:
        d.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_config(path: Path, **kwargs) -> None:
    path.write_text(json.dumps(kwargs, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase2 real simulated DKI data loader
# ---------------------------------------------------------------------------


def list_phase2_files(data_root: str | Path, max_files: int = 0) -> list[Path]:
    """List all .npz files in the Phase2 dataset directory."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Phase2 data directory not found: {root}")
    files = sorted(root.glob("*.npz"))
    if max_files > 0:
        files = files[:max_files]
    return files


def load_phase2_sample(file_path: Path) -> dict[str, np.ndarray]:
    """Load a single Phase2 .npz sample and return a dict with DKI-relevant fields."""
    data = np.load(file_path, allow_pickle=True)
    bvals = data["bvals"].astype(np.float32)      # int32 -> float32
    return {
        "dwi_clean": data["dwi_clean"].astype(np.float32),   # [B, H, W]
        "s0": data["s0_gt"].astype(np.float32),              # [H, W]
        "d": data["d_gt"].astype(np.float32),
        "k": data["k_gt"].astype(np.float32),
        "mask": data["mask"].astype(np.float32),
        "bvals": bvals,
    }


def load_phase2_data(
    data_root: str | Path,
    max_files: int = 0,
    seed: int = 42,
    noise_levels: list[float] | None = None,
) -> DkiData:
    """Load Phase2 simulated DKI data into a DkiData container.
    Dynamically adds Rician noise from dwi_clean at specified levels.

    Parameters
    ----------
    data_root : str or Path
        Path to '03_Phase2_UNet_Synthesis' directory.
    max_files : int
        If > 0, only load this many files (randomly selected with seed).
    seed : int
        Random seed for file selection.
    noise_levels : list[float] | None
        Rician noise sigmas to apply. If None, keeps original noisy data.
        If provided, cycles through levels across samples.

    Returns
    -------
    DkiData
    """
    if noise_levels is None:
        noise_levels = [0.03]  # Default to original single noise level

    files = list_phase2_files(data_root, max_files=0)
    n_total = len(files)
    if max_files > 0 and max_files < n_total:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=max_files, replace=False)
        files = [files[i] for i in sorted(indices)]

    rng_noise = np.random.default_rng(seed + 999)
    dwi_n, dwi_c, s0s, ds, ks, masks, sigmas = [], [], [], [], [], [], []
    bvals = None
    for i, fpath in enumerate(files):
        sample = load_phase2_sample(fpath)
        clean = sample["dwi_clean"]
        # Add Rician noise at the specified level (cycle through levels)
        sigma = float(noise_levels[i % len(noise_levels)])
        noisy = add_rician_noise(rng_noise, clean, sigma)
        dwi_n.append(noisy)
        dwi_c.append(clean)
        s0s.append(sample["s0"])
        ds.append(sample["d"])
        ks.append(sample["k"])
        masks.append(sample["mask"])
        sigmas.append(sigma)
        if bvals is None:
            bvals = sample["bvals"]

    return DkiData(
        dwi_noisy=np.stack(dwi_n),        # [N, B, H, W]
        dwi_clean=np.stack(dwi_c),
        s0=np.stack(s0s),                 # [N, H, W]
        d=np.stack(ds),
        k=np.stack(ks),
        mask=np.stack(masks).astype(np.float32),
        bvals=bvals,
        sigma=np.asarray(sigmas, dtype=np.float32),
    )


def phase2_voxel_arrays(data: DkiData) -> tuple[np.ndarray, ...]:
    """Convert Phase2 DkiData to voxel-wise arrays (normalized signal).

    Same interface as voxel_arrays() for procedural data.
    """
    xs, ds, ks, sigmas, s0s = [], [], [], [], []
    for i in range(data.dwi_noisy.shape[0]):
        m = data.mask[i].astype(bool)
        flat = m.ravel()
        raw = data.dwi_noisy[i].reshape(data.dwi_noisy.shape[1], -1)[:, flat].T.astype(np.float32)
        x = raw / (raw[:, [0]] + 1.0e-6)
        xs.append(x)
        ds.append(data.d[i].ravel()[flat, None].astype(np.float32))
        ks.append(data.k[i].ravel()[flat, None].astype(np.float32))
        sigmas.append(np.full((x.shape[0], 1), data.sigma[i], dtype=np.float32))
        s0s.append(raw[:, [0]].astype(np.float32))
    return (np.concatenate(xs), np.concatenate(ds), np.concatenate(ks),
            np.concatenate(sigmas), np.concatenate(s0s))


# ---------------------------------------------------------------------------
# Phase2 CNN Dataset (for image-level training)
# ---------------------------------------------------------------------------


class Phase2Dataset(Dataset):
    """PyTorch Dataset for Phase2 DKI data (image-level, for CNN training)."""

    def __init__(self, data: DkiData):
        self.dwi = torch.from_numpy(data.dwi_noisy).float()
        self.d = torch.from_numpy(data.d[:, None]).float()       # [N, 1, H, W]
        self.k = torch.from_numpy(data.k[:, None]).float()
        self.mask = torch.from_numpy(data.mask[:, None]).float()
        self.bvals = torch.from_numpy(data.bvals).float()
        self.sigma = torch.from_numpy(data.sigma).float()

    def __len__(self) -> int:
        return len(self.dwi)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "dwi": self.dwi[idx],
            "d": self.d[idx],
            "k": self.k[idx],
            "mask": self.mask[idx],
            "bvals": self.bvals,
            "sigma": self.sigma[idx],
        }


# ---------------------------------------------------------------------------
# DKI phantom generation (procedural, for small-scale testing)
# ---------------------------------------------------------------------------


def make_brain_like_maps(
    rng: np.random.Generator, h: int, w: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a brain-like phantom with S0, D, K maps and a binary mask."""
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    brain = (xx / 0.82) ** 2 + (yy / 0.95) ** 2 <= 1.0
    wm = ((xx + 0.18) / 0.45) ** 2 + ((yy + 0.02) / 0.62) ** 2 <= 1.0
    gm = brain & ~wm
    lesion = ((xx - rng.uniform(0.15, 0.35)) / 0.13) ** 2 + (
        (yy - rng.uniform(-0.25, 0.25)) / 0.16
    ) ** 2 <= 1.0
    csf = ((xx + 0.38) / 0.17) ** 2 + ((yy - 0.35) / 0.2) ** 2 <= 1.0

    s0 = np.zeros((h, w), dtype=np.float32)
    d = np.zeros((h, w), dtype=np.float32)
    k = np.zeros((h, w), dtype=np.float32)

    # Tissue values
    s0[brain] = 0.95
    s0[wm] = 0.82
    s0[gm] = 1.02
    s0[csf] = 1.15
    s0[lesion & brain] = 1.05

    d[brain] = 0.00085
    d[wm] = 0.00070
    d[gm] = 0.00095
    d[csf] = 0.00165
    d[lesion & brain] = 0.00125

    k[brain] = 0.85
    k[wm] = 1.15
    k[gm] = 0.75
    k[csf] = 0.25
    k[lesion & brain] = 0.55

    # Spatial variation
    bias = 1.0 + 0.08 * xx - 0.05 * yy
    texture = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    s0 = s0 * bias + 0.025 * texture * brain.astype(np.float32)
    d = d + rng.normal(0.0, 0.000035, size=(h, w)).astype(np.float32) * brain.astype(np.float32)
    k = k + rng.normal(0.0, 0.045, size=(h, w)).astype(np.float32) * brain.astype(np.float32)

    d = np.clip(d, 0.0, None).astype(np.float32)
    k = np.clip(k, 0.0, 2.0).astype(np.float32)
    return s0.astype(np.float32), d, k, brain.astype(np.float32)


def dki_signal(s0: np.ndarray, d: np.ndarray, k: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    """Forward DKI signal model: S(b) = S0 * exp(-bD + b²D²K/6)."""
    b = bvals[:, None, None]
    expo = -b * d[None] + (b ** 2) * (d[None] ** 2) * k[None] / 6.0
    return (s0[None] * np.exp(expo)).astype(np.float32)


def add_rician_noise(rng: np.random.Generator, clean: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise to magnitude MRI data."""
    n1 = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    return np.sqrt((clean + n1) ** 2 + n2 ** 2).astype(np.float32)


def generate_dataset(
    n: int, h: int, w: int, bvals: np.ndarray,
    noise_levels: list[float], seed: int,
) -> DkiData:
    """Generate a full DKI phantom dataset."""
    rng = np.random.default_rng(seed)
    dwi_clean, dwi_noisy, s0s, ds, ks, masks, sigmas = [], [], [], [], [], [], []
    for i in range(n):
        s0, d, k, mask = make_brain_like_maps(rng, h, w)
        clean = dki_signal(s0, d, k, bvals)
        sigma = float(noise_levels[i % len(noise_levels)])
        noisy = add_rician_noise(rng, clean, sigma)
        dwi_clean.append(clean)
        dwi_noisy.append(noisy)
        s0s.append(s0)
        ds.append(d)
        ks.append(k)
        masks.append(mask)
        sigmas.append(sigma)
    return DkiData(
        dwi_noisy=np.stack(dwi_noisy),
        dwi_clean=np.stack(dwi_clean),
        s0=np.stack(s0s),
        d=np.stack(ds),
        k=np.stack(ks),
        mask=np.stack(masks).astype(np.float32),
        bvals=bvals.astype(np.float32),
        sigma=np.asarray(sigmas, dtype=np.float32),
    )


def split_dataset(data: DkiData, n_train: int, n_val: int) -> tuple[DkiData, DkiData, DkiData]:
    """Split dataset into train/val/test."""
    s = lambda idx: DkiData(
        dwi_noisy=data.dwi_noisy[idx],
        dwi_clean=data.dwi_clean[idx],
        s0=data.s0[idx],
        d=data.d[idx],
        k=data.k[idx],
        mask=data.mask[idx],
        bvals=data.bvals,
        sigma=data.sigma[idx],
    )
    return s(slice(0, n_train)), s(slice(n_train, n_train + n_val)), s(slice(n_train + n_val, None))


def voxel_arrays(data: DkiData) -> tuple[np.ndarray, ...]:
    """Flatten image data to voxel-wise arrays (masked)."""
    xs, ds, ks, sigmas, s0s = [], [], [], [], []
    for i in range(data.dwi_noisy.shape[0]):
        m = data.mask[i].astype(bool)
        flat = m.ravel()
        raw = data.dwi_noisy[i].reshape(data.dwi_noisy.shape[1], -1)[:, flat].T.astype(np.float32)
        x = raw / (raw[:, [0]] + 1.0e-6)          # normalized signal
        xs.append(x)
        ds.append(data.d[i].ravel()[flat, None].astype(np.float32))
        ks.append(data.k[i].ravel()[flat, None].astype(np.float32))
        sigmas.append(np.full((x.shape[0], 1), data.sigma[i], dtype=np.float32))
        s0s.append(raw[:, [0]].astype(np.float32))  # measured b0
    return (np.concatenate(xs), np.concatenate(ds), np.concatenate(ks),
            np.concatenate(sigmas), np.concatenate(s0s))


# ---------------------------------------------------------------------------
# Neural network models
# ---------------------------------------------------------------------------


class DkiMLP(nn.Module):
    """Voxel-wise MLP: normalized signal -> (D, K) or (S0_factor, D, K)."""

    def __init__(self, in_channels: int, d_max: float = 0.003, k_max: float = 2.5,
                 predict_s0: bool = False):
        super().__init__()
        self.d_max = d_max
        self.k_max = k_max
        self.predict_s0 = predict_s0
        out_dim = 3 if predict_s0 else 2
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor):
        raw = self.net(x)
        if self.predict_s0:
            s0 = torch.sigmoid(raw[:, 0:1]) * 2.0      # S0 scale [0, 2]
            d = self.d_max * torch.sigmoid(raw[:, 1:2])
            k = self.k_max * torch.sigmoid(raw[:, 2:3])
            return s0, d, k
        d = self.d_max * torch.sigmoid(raw[:, 0:1])
        k = self.k_max * torch.sigmoid(raw[:, 1:2])
        return d, k


class DkiCNN(nn.Module):
    """Shallow CNN for image-level DKI map prediction.

    Input:  [N, C, H, W] — multi-b DWI stack
    Output: [N, 2, H, W] — D map, K map
    """

    def __init__(self, in_channels: int, d_max: float = 0.003, k_max: float = 2.5):
        super().__init__()
        self.d_max = d_max
        self.k_max = k_max
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        d = self.d_max * torch.sigmoid(raw[:, 0:1])
        k = self.k_max * torch.sigmoid(raw[:, 1:2])
        return d, k


# ---------------------------------------------------------------------------
# Traditional fitting baselines
# ---------------------------------------------------------------------------


def polynomial_dki_fit(
    data: DkiData, d_max: float, k_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-polynomial DKI fitting: log(S/S0) = -bD + b²D²K/6."""
    n, _b, h, w = data.dwi_noisy.shape
    d_out = np.zeros((n, h, w), dtype=np.float32)
    k_out = np.zeros((n, h, w), dtype=np.float32)
    b = data.bvals[1:].astype(np.float64)
    X = np.stack([b, b**2], axis=1)
    pinv = np.linalg.pinv(X)
    for i in range(n):
        m = data.mask[i].astype(bool)
        raw = data.dwi_noisy[i].reshape(_b, -1)[:, m.ravel()].T
        ratio = raw[:, 1:].T / (raw[:, 0:1].T + 1.0e-6)
        y = np.log(np.clip(ratio, 1.0e-6, None)).astype(np.float64)
        coef = pinv @ y
        c1, c2 = coef[0], coef[1]
        d = np.clip(-c1, 0.0, d_max)
        valid = d > 1.0e-7
        k = np.zeros_like(d)
        k[valid] = np.clip(6.0 * c2[valid] / (d[valid] ** 2), 0.0, k_max)
        d_out[i, m] = d.astype(np.float32)
        k_out[i, m] = k.astype(np.float32)
    return d_out, k_out


def mono_adc_fit_as_d(data: DkiData, d_max: float) -> np.ndarray:
    """Single ADC fit as D (ignores kurtosis)."""
    n, _b, h, w = data.dwi_noisy.shape
    d_out = np.zeros((n, h, w), dtype=np.float32)
    b = data.bvals.astype(np.float64)
    bm = b.mean()
    denom = np.sum((b - bm) ** 2)
    for i in range(n):
        m = data.mask[i].astype(bool)
        raw = data.dwi_noisy[i].reshape(_b, -1)[:, m.ravel()]
        y = np.log(np.clip(raw, 1.0e-6, None)).astype(np.float64)
        ym = y.mean(axis=0)
        beta = np.sum((b[:, None] - bm) * (y - ym), axis=0) / denom
        d_out[i, m] = np.clip(-beta, 0.0, d_max).astype(np.float32)
    return d_out


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def dki_log_residual_loss(
    x_norm: torch.Tensor, d: torch.Tensor, k: torch.Tensor, bvals: torch.Tensor,
) -> torch.Tensor:
    """Log-domain residual for DKI: L1(log(S/S0) + bD - b²D²K/6)."""
    b = bvals[None, 1:]
    target = torch.log(x_norm[:, 1:].clamp_min(1.0e-6))
    pred = -b * d + (b ** 2) * (d ** 2) * k / 6.0
    return F.l1_loss(pred, target)


def rician_nll_loss(
    raw_norm: torch.Tensor, s0: torch.Tensor, sigma: torch.Tensor,
    d: torch.Tensor, k: torch.Tensor, bvals: torch.Tensor,
) -> torch.Tensor:
    """Rician negative log-likelihood loss."""
    b = bvals[None, :]
    pred_norm = torch.exp(-b * d + (b ** 2) * (d ** 2) * k / 6.0)
    mu = s0 * pred_norm
    y = s0 * raw_norm
    sig = sigma.clamp_min(1.0e-4)
    z = (y * mu / (sig ** 2)).clamp_max(80.0)
    nll = (y ** 2 + mu ** 2) / (2.0 * sig ** 2) - (
        torch.log(torch.special.i0e(z).clamp_min(1.0e-12)) + torch.abs(z)
    )
    return nll.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    method: str,
    train: DkiData,
    val: DkiData,
    epochs: int,
    device: torch.device,
    d_max: float,
    k_max: float,
    out_path: Path,
    batch_size: int = 32768,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    log_interval: int = 10,
    train_arrays: tuple | None = None,
    val_arrays: tuple | None = None,
) -> Path:
    """Train a voxel-wise DKI MLP — all data on GPU, no DataLoader overhead."""
    if train_arrays is not None:
        x, d_gt, k_gt, sigma, s0 = train_arrays
    else:
        x, d_gt, k_gt, sigma, s0 = voxel_arrays(train)
    if val_arrays is not None:
        xv, dv, kv, sigv, s0v = val_arrays
    else:
        xv, dv, kv, sigv, s0v = voxel_arrays(val)

    bvals = torch.from_numpy(train.bvals).to(device)
    loss_fn, needs_s0 = _get_voxel_loss_fn(method, bvals)
    model = DkiMLP(x.shape[1], d_max=d_max, k_max=k_max, predict_s0=needs_s0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # All data to GPU once (fits in 24GB — ~30MB total)
    x_t = torch.from_numpy(x).to(device)
    d_t = torch.from_numpy(d_gt).to(device)
    k_t = torch.from_numpy(k_gt).to(device)
    sig_t = torch.from_numpy(sigma).to(device)
    s0_t = torch.from_numpy(s0).to(device)
    xv_t = torch.from_numpy(xv).to(device)
    dv_t = torch.from_numpy(dv).to(device)
    kv_t = torch.from_numpy(kv).to(device)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    t_start = time.time()
    n_total = x_t.shape[0]
    rng = np.random.default_rng()

    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        perm = torch.from_numpy(rng.permutation(n_total)).to(device)
        for start in range(0, n_total, batch_size):
            idx = perm[start:start + batch_size]
            xb, db, kb = x_t[idx], d_t[idx], k_t[idx]
            sigb, s0b = sig_t[idx], s0_t[idx]
            if needs_s0:
                s0_pred, d_pred, k_pred = model(xb)
                loss = loss_fn(xb, db, kb, sigb, s0b, s0_pred, d_pred, k_pred)
            else:
                d_pred, k_pred = model(xb)
                loss = loss_fn(xb, db, kb, sigb, s0b, d_pred, k_pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * idx.shape[0]
            count += idx.shape[0]

        model.eval()
        with torch.no_grad():
            if needs_s0:
                _, d_val, k_val = model(xv_t)
            else:
                d_val, k_val = model(xv_t)
            val_loss = F.l1_loss(d_val * 1000.0, dv_t * 1000.0) + 0.2 * F.l1_loss(k_val, kv_t)
            val_f = float(val_loss.item())
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_supervised_proxy": val_f})

        if epoch % log_interval == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  [{method}] epoch {epoch:3d}/{epochs}  train_loss={total/max(count,1):.6g}  val_proxy={val_f:.6g}  [{elapsed:.0f}s]")

        if val_f < best_val:
            best_val = val_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        "model": best_state, "method": method, "in_channels": x.shape[1],
        "d_max": d_max, "k_max": k_max, "predict_s0": needs_s0, "history": history,
    }, out_path)
    print(f"  -> saved {out_path}  (best_val={best_val:.6g})")
    return out_path


def _get_voxel_loss_fn(method: str, bvals: torch.Tensor) -> Callable:
    """Return the appropriate loss function for a voxel training method."""

    def supervised(xb, db, kb, sigb, s0b, d_pred, k_pred):
        return F.l1_loss(d_pred * 1000.0, db * 1000.0) + 0.2 * F.l1_loss(k_pred, kb)

    def pinn_log(xb, db, kb, sigb, s0b, d_pred, k_pred):
        return dki_log_residual_loss(xb, d_pred, k_pred, bvals)

    def pinn_rician(xb, db, kb, sigb, s0b, d_pred, k_pred):
        return rician_nll_loss(xb, s0b, sigb, d_pred, k_pred, bvals) * 1.0e-3

    def pinn_log_rician(xb, db, kb, sigb, s0b, d_pred, k_pred):
        log_loss = dki_log_residual_loss(xb, d_pred, k_pred, bvals)
        rician_loss = rician_nll_loss(xb, s0b, sigb, d_pred, k_pred, bvals)
        return log_loss + 1.0e-5 * rician_loss

    def semi_supervised(xb, db, kb, sigb, s0b, d_pred, k_pred):
        sup = F.l1_loss(d_pred * 1000.0, db * 1000.0) + 0.2 * F.l1_loss(k_pred, kb)
        log_p = dki_log_residual_loss(xb, d_pred, k_pred, bvals)
        return 0.5 * sup + 0.5 * log_p

    def pinn_log_predict_s0(xb, db, kb, sigb, s0b, s0_pred, d_pred, k_pred):
        """PINN log residual using predicted S0 instead of fixed measured S0."""
        b = bvals[None, 1:]
        s_pred = s0b * s0_pred                           # un-normalized S(b)
        S_meas_norm = xb[:, 1:] * s0b                    # measured S(b) (unnorm)
        target = torch.log(S_meas_norm.clamp_min(1e-6))
        pred = torch.log(s_pred.clamp_min(1e-6)) + (
            -b * d_pred + (b ** 2) * (d_pred ** 2) * k_pred / 6.0
        )
        return F.l1_loss(pred, target)

    def pinn_log_rician_predict_s0(xb, db, kb, sigb, s0b, s0_pred, d_pred, k_pred):
        log_loss = pinn_log_predict_s0(xb, db, kb, sigb, s0b, s0_pred, d_pred, k_pred)
        rician_loss = rician_nll_loss(xb, s0b, sigb, d_pred, k_pred, bvals)
        return log_loss + 1.0e-5 * rician_loss

    fns = {
        "supervised": (supervised, False),
        "pinn_log": (pinn_log, False),
        "pinn_rician": (pinn_rician, False),
        "pinn_log_rician": (pinn_log_rician, False),
        "semi_supervised": (semi_supervised, False),
        "pinn_log_predict_s0": (pinn_log_predict_s0, True),
        "pinn_log_rician_predict_s0": (pinn_log_rician_predict_s0, True),
    }
    if method not in fns:
        raise ValueError(f"Unknown method: {method}. Choose from {list(fns.keys())}")
    return fns[method]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_model(path: Path, data: DkiData, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Load a trained voxel model and predict D, K maps (batched, GPU efficient)."""
    ckpt = torch.load(path, map_location=device)
    predict_s0 = ckpt.get("predict_s0", False)
    model = DkiMLP(int(ckpt["in_channels"]), float(ckpt["d_max"]), float(ckpt["k_max"]),
                   predict_s0=predict_s0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n, _b, h, w = data.dwi_noisy.shape

    # Extract all voxels at once
    x_all, _, _, _, _ = voxel_arrays(data)
    x_t = torch.from_numpy(x_all).to(device)

    d_out = np.zeros((n, h, w), dtype=np.float32)
    k_out = np.zeros((n, h, w), dtype=np.float32)
    with torch.no_grad():
        if predict_s0:
            _, d_all, k_all = model(x_t)
        else:
            d_all, k_all = model(x_t)
        d_all = d_all.cpu().numpy()[:, 0]
        k_all = k_all.cpu().numpy()[:, 0]

    # Scatter back to image space
    offset = 0
    for i in range(n):
        m = data.mask[i].astype(bool)
        n_vox = m.sum()
        d_out[i, m] = d_all[offset:offset + n_vox]
        k_out[i, m] = k_all[offset:offset + n_vox]
        offset += n_vox
    return d_out, k_out


def predict_cnn_model(
    path: Path, data: DkiData, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a trained CNN model and predict D, K maps."""
    ckpt = torch.load(path, map_location=device)
    model = DkiCNN(
        int(ckpt["in_channels"]), float(ckpt["d_max"]), float(ckpt["k_max"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n, _b, h, w = data.dwi_noisy.shape
    d_out = np.zeros((n, h, w), dtype=np.float32)
    k_out = np.zeros((n, h, w), dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            dwi = torch.from_numpy(data.dwi_noisy[i:i+1]).to(device)
            d, k = model(dwi)
            d_out[i] = d[0, 0].cpu().numpy()
            k_out[i] = k[0, 0].cpu().numpy()
    return d_out, k_out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(name: str, data: DkiData, d_pred: np.ndarray, k_pred: np.ndarray) -> dict[str, object]:
    """Compute D and K metrics (masked)."""
    m = data.mask.astype(bool)
    d_err = d_pred[m] - data.d[m]
    k_err = k_pred[m] - data.k[m]
    return {
        "method": name,
        "d_mae_x1e3": float(np.mean(np.abs(d_err)) * 1000.0),
        "d_rmse_x1e3": float(np.sqrt(np.mean(d_err ** 2)) * 1000.0),
        "d_bias_x1e3": float(np.mean(d_err) * 1000.0),
        "k_mae": float(np.mean(np.abs(k_err))),
        "k_rmse": float(np.sqrt(np.mean(k_err ** 2))),
        "k_bias": float(np.mean(k_err)),
    }


def noise_stratified_metrics(
    name: str, data: DkiData, d_pred: np.ndarray, k_pred: np.ndarray,
) -> list[dict[str, object]]:
    """Compute metrics per noise level."""
    rows = []
    for sigma in sorted(set(float(s) for s in data.sigma)):
        idx = data.sigma == sigma
        sub = DkiData(
            dwi_noisy=data.dwi_noisy[idx],
            dwi_clean=data.dwi_clean[idx],
            s0=data.s0[idx], d=data.d[idx], k=data.k[idx],
            mask=data.mask[idx], bvals=data.bvals, sigma=data.sigma[idx],
        )
        row = metrics(f"{name}_noise={sigma:.2g}", sub, d_pred[idx], k_pred[idx])
        row["noise_sigma"] = sigma
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]], group_key: str = "method") -> list[dict[str, object]]:
    """Aggregate multi-seed results by method (mean ± std)."""
    out = []
    methods = sorted({str(r[group_key]) for r in rows})
    metric_keys = [k for k in rows[0].keys() if k not in (group_key, "seed", "experiment", "noise_sigma", "train_noise", "test_noise")]
    for method in methods:
        vals = [r for r in rows if str(r[group_key]) == method]
        item: dict[str, object] = {group_key: method, "n": len(vals)}
        for mk in metric_keys:
            arr = np.asarray([float(v[mk]) for v in vals if mk in v and not np.isnan(float(v[mk]))])
            if arr.size:
                item[f"{mk}_mean"] = float(arr.mean())
                item[f"{mk}_std"] = float(arr.std())
        out.append(item)
    return sorted(out, key=lambda r: float(r.get(list(r.keys())[1], 1e9) if len(r) > 1 else 1e9))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dki_bars(rows: list[dict[str, object]], out: Path) -> None:
    """Bar chart comparing D and K RMSE across methods."""
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: float(r.get("d_rmse_x1e3", 0) if "d_rmse_x1e3" in r else r.get("d_rmse_x1e3_mean", 0)))
    labels = [str(r.get("method", "?")) for r in rows]
    d_rmse = [float(r.get("d_rmse_x1e3", r.get("d_rmse_x1e3_mean", 0))) for r in rows]
    k_rmse = [float(r.get("k_rmse", r.get("k_rmse_mean", 0))) for r in rows]

    x = np.arange(len(rows))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(rows) * 1.2), 7), sharex=True)
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#6a994e", "#bc4a9c", "#3a7ca5"]
    for i in range(len(rows)):
        ax1.bar(x[i], d_rmse[i], color=colors[i % len(colors)], width=0.6)
        ax2.bar(x[i], k_rmse[i], color=colors[i % len(colors)], width=0.6)
    ax1.set_ylabel("D RMSE (×10⁻³)", fontsize=12)
    ax2.set_ylabel("K RMSE", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax1.set_title("DKI Method Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out}")


def plot_noise_curves(
    rows: list[dict[str, object]], out: Path,
) -> None:
    """Plot D and K RMSE vs noise level for each method."""
    import matplotlib.pyplot as plt
    methods = sorted({str(r["method"]) for r in rows})
    noise_levels = sorted({float(r["noise_sigma"]) for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#6a994e", "#bc4a9c"]

    for i, method in enumerate(methods):
        d_vals, k_vals = [], []
        for nl in noise_levels:
            match = [r for r in rows if str(r["method"]) == method and abs(float(r["noise_sigma"]) - nl) < 1e-6]
            if match:
                d_vals.append(float(match[0]["d_rmse_x1e3"]))
                k_vals.append(float(match[0]["k_rmse"]))
            else:
                d_vals.append(np.nan)
                k_vals.append(np.nan)
        label = method.replace("_noise=", "\n").split("\n")[0]
        ax1.plot(noise_levels, d_vals, "o-", color=colors[i % len(colors)], label=label)
        ax2.plot(noise_levels, k_vals, "s-", color=colors[i % len(colors)], label=label)

    ax1.set_xlabel("Noise σ", fontsize=12)
    ax1.set_ylabel("D RMSE (×10⁻³)", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.set_title("D RMSE vs Noise", fontsize=13)

    ax2.set_xlabel("Noise σ", fontsize=12)
    ax2.set_ylabel("K RMSE", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.set_title("K RMSE vs Noise", fontsize=13)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out}")


def plot_sample_maps(
    data: DkiData, preds: dict[str, tuple[np.ndarray, np.ndarray]],
    out: Path, sample_idx: int = 0,
) -> None:
    """Plot GT maps and error maps for a representative sample."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(preds) + 1, figsize=(3 * (len(preds) + 1) + 1, 9))
    m = data.mask[sample_idx].astype(bool)

    # Row 0: D maps, Row 1: K maps, Row 2: D error maps
    gt_d, gt_k = data.d[sample_idx], data.k[sample_idx]

    methods = list(preds.keys())
    for col, method in enumerate(["GT"] + methods):
        if col == 0:
            d_show, k_show = gt_d, gt_k
            d_err = np.zeros_like(gt_d)
            k_err = np.zeros_like(gt_k)
            title_prefix = "GT"
        else:
            d_show, k_show = preds[method]
            d_show = d_show[sample_idx] if d_show.ndim == 3 else d_show
            k_show = k_show[sample_idx] if k_show.ndim == 3 else k_show
            d_err = d_show - gt_d
            k_err = k_show - gt_k
            title_prefix = method

        for row, (data_show, cmap, label) in enumerate([
            (d_show, "viridis", f"{title_prefix} D"),
            (k_show, "plasma", f"{title_prefix} K"),
            (d_err, "RdBu_r", f"{title_prefix} D err"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(np.where(m, data_show, 0), cmap=cmap)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Sample {sample_idx} — DKI Method Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(
    path: Path,
    summary: list[dict[str, object]],
    noise_rows: list[dict[str, object]],
    config: dict,
    description: str = "",
) -> None:
    """Generate a markdown report from experiment results."""
    lines = [
        "# DKI Extension Experiment Report",
        "",
        description,
        "",
        "## Configuration",
        "",
    ]
    for k, v in config.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend([
        "",
        "## Overall Results (multi-seed mean ± std)",
        "",
        "| method | D RMSE (×10⁻³) | D MAE (×10⁻³) | K RMSE | K MAE |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in sorted(summary, key=lambda r: float(r.get("d_rmse_x1e3_mean", 1e9))):
        lines.append(
            f"| `{row['method']}` | "
            f"{float(row.get('d_rmse_x1e3_mean', float('nan'))):.4f}±{float(row.get('d_rmse_x1e3_std', 0)):.4f} | "
            f"{float(row.get('d_mae_x1e3_mean', float('nan'))):.4f}±{float(row.get('d_mae_x1e3_std', 0)):.4f} | "
            f"{float(row.get('k_rmse_mean', float('nan'))):.4f}±{float(row.get('k_rmse_std', 0)):.4f} | "
            f"{float(row.get('k_mae_mean', float('nan'))):.4f}±{float(row.get('k_mae_std', 0)):.4f} |"
        )

    if noise_rows:
        lines.extend([
            "",
            "## Noise-stratified Results",
            "",
            "| method | noise σ | D RMSE (×10⁻³) | K RMSE |",
            "|---|---:|---:|---:|",
        ])
        for row in sorted(noise_rows, key=lambda r: (str(r["method"]), float(r["noise_sigma"]))):
            lines.append(
                f"| `{row['method']}` | {float(row['noise_sigma']):.2g} | "
                f"{float(row['d_rmse_x1e3']):.4f} | {float(row['k_rmse']):.4f} |"
            )

    lines.extend([
        "",
        "## Key Takeaways",
        "",
        "1. **log-domain residual is the core effective term for no-GT PINN.**",
        "2. **Rician likelihood helps as an auxiliary term but should not be used alone.**",
        "3. **DKI benefits more from physics-informed constraints than ADC** due to coupled D/K parameters.",
        "4. **CNN/U-Net extension** (if included) shows whether spatial priors improve DKI parameter estimation.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> saved {path}")


def plot_cross_noise_heatmap(
    rows: list[dict[str, object]],
    metric: str,
    out: Path,
    title: str = "Cross-noise RMSE",
) -> None:
    """Plot a train_noise x test_noise heatmap for a given metric.

    Rows must contain 'train_noise', 'test_noise', and the metric field.
    """
    import matplotlib.pyplot as plt

    train_noises = sorted({str(r["train_noise"]) for r in rows if r.get("train_noise") and r["train_noise"] != "none"})
    test_noises = sorted({str(r["test_noise"]) for r in rows})
    if not train_noises or not test_noises:
        print("  WARNING: not enough data for cross-noise heatmap")
        return

    n_train, n_test = len(train_noises), len(test_noises)
    matrix = np.full((n_train, n_test), np.nan)

    for r in rows:
        tn = str(r.get("train_noise", ""))
        ts = str(r.get("test_noise", ""))
        if tn in train_noises and ts in test_noises:
            i = train_noises.index(tn)
            j = test_noises.index(ts)
            val = float(r.get(metric, np.nan))
            if not np.isnan(val):
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, n_test * 1.2), max(5, n_train * 1.2)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")

    # Show values
    for i in range(n_train):
        for j in range(n_test):
            val = matrix[i, j]
            text = f"{val:.4f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if not np.isnan(val) and val > matrix[~np.isnan(matrix)].mean() else "black",
                    fontsize=9)

    ax.set_xticks(range(n_test))
    ax.set_xticklabels(test_noises, rotation=30, ha="right")
    ax.set_yticks(range(n_train))
    ax.set_yticklabels(train_noises)
    ax.set_xlabel("Test noise sigma")
    ax.set_ylabel("Train noise sigma")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out}")
