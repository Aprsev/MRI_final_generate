"""
DKI PINN Extension — Main Experiment Runner
============================================
Comprehensive DKI experiment covering:
  - Data sources: procedural phantom (built-in) OR Phase2 real simulated DKI data
  - Multiple random seeds (>=3)
  - Baseline fits: mono_adc_as_d, poly_dki_fit
  - Voxel MLP: supervised, pinn_log, pinn_rician, pinn_log_rician, semi_supervised
  - Optional CNN image-level training (supervised, pinn_log, supervised_log)
  - Noise-stratified evaluation
  - Representative sample error maps

Usage:
  # Procedural phantom (small-scale test)
  python run_dki_extension.py --data-source procedural --seeds 42 --samples 120 --epochs 30

  # Phase2 real simulated DKI data
  python run_dki_extension.py --data-source phase2 --phase2-max-files 200 --seeds 42 123 --epochs 50

Environment: MRI_final (conda)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the task_A directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from dki_utils import (
    DkiData,
    Paths,
    make_paths,
    write_csv,
    save_config,
    generate_dataset,
    split_dataset,
    load_phase2_data,
    phase2_voxel_arrays,
    polynomial_dki_fit,
    mono_adc_fit_as_d,
    train_model,
    predict_model,
    predict_cnn_model,
    metrics,
    noise_stratified_metrics,
    summarize,
    plot_dki_bars,
    plot_noise_curves,
    plot_sample_maps,
    plot_cross_noise_heatmap,
    write_report,
    DkiCNN,
    Phase2Dataset,
    load_phase2_sample,
    add_rician_noise,
)


# ---------------------------------------------------------------------------
# CNN training (image-level DKI)
# ---------------------------------------------------------------------------


def train_cnn_model(
    method: str,
    train: DkiData,
    val: DkiData,
    epochs: int,
    device: torch.device,
    d_max: float,
    k_max: float,
    out_path: Path,
    batch_size: int = 4,
    lr: float = 5.0e-4,
    log_interval: int = 10,
) -> Path:
    """Train a CNN for image-to-image D/K map prediction.

    Supports both small procedural (e.g. 48x48) and larger Phase2 (192x192) images.
    For Phase2 data, use small batch_size (2-4) due to memory.
    """
    from torch.utils.data import DataLoader

    is_phase2 = train.dwi_noisy.shape[-1] > 100  # heuristic: large image = Phase2

    if is_phase2:
        train_dataset = Phase2Dataset(train)
        val_dataset = Phase2Dataset(val)
    else:
        n, _b, h, w = train.dwi_noisy.shape
        x_train = torch.from_numpy(train.dwi_noisy).float()
        d_train = torch.from_numpy(train.d[:, None]).float()
        k_train = torch.from_numpy(train.k[:, None]).float()
        m_train = torch.from_numpy(train.mask[:, None]).float()
        train_dataset = torch.utils.data.TensorDataset(x_train, d_train, k_train, m_train)
        x_val = torch.from_numpy(val.dwi_noisy).float().to(device)
        d_val = torch.from_numpy(val.d[:, None]).float().to(device)
        k_val = torch.from_numpy(val.k[:, None]).float().to(device)
        m_val = torch.from_numpy(val.mask[:, None]).float().to(device)

    model = DkiCNN(train.dwi_noisy.shape[1], d_max=d_max, k_max=k_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            if is_phase2:
                xb = batch["dwi"].to(device)
                db = batch["d"].to(device)
                kb = batch["k"].to(device)
                mb = batch["mask"].to(device)
                # bvals = batch["bvals"].to(device)
            else:
                xb, db, kb, mb = [b.to(device) for b in batch]
            d_pred, k_pred = model(xb)
            loss = _cnn_loss(method, xb, db, kb, mb, d_pred, k_pred, train.bvals, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]

        model.eval()
        with torch.no_grad():
            if is_phase2:
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                d_vals, k_vals, d_gts, k_gts, m_gts = [], [], [], [], []
                for b in val_loader:
                    xv = b["dwi"].to(device)
                    d_v, k_v = model(xv)
                    d_vals.append(d_v.cpu())
                    k_vals.append(k_v.cpu())
                    d_gts.append(b["d"])
                    k_gts.append(b["k"])
                    m_gts.append(b["mask"])
                d_v_all = torch.cat(d_vals, dim=0).to(device)
                d_gt_all = torch.cat(d_gts, dim=0).to(device)
                k_v_all = torch.cat(k_vals, dim=0).to(device)
                k_gt_all = torch.cat(k_gts, dim=0).to(device)
                m_all = torch.cat(m_gts, dim=0).to(device)
                val_loss = _masked_l1(d_v_all, d_gt_all, m_all) + 0.2 * _masked_l1(k_v_all, k_gt_all, m_all)
            else:
                d_v, k_v = model(x_val)
                val_loss = _masked_l1(d_v, d_val, m_val) + 0.2 * _masked_l1(k_v, k_val, m_val)
            val_f = float(val_loss.item())

        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": val_f})
        if epoch % log_interval == 0 or epoch == 1:
            print(f"  [cnn_{method}] epoch {epoch:3d}/{epochs}  train_loss={total/max(count,1):.6g}  val_loss={val_f:.6g}")
        if val_f < best_val:
            best_val = val_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        "model": best_state, "method": f"cnn_{method}", "in_channels": train.dwi_noisy.shape[1],
        "d_max": d_max, "k_max": k_max, "history": history,
    }, out_path)
    print(f"  -> saved {out_path}  (best_val={best_val:.6g})")
    return out_path


def _cnn_loss(
    method: str, xb: torch.Tensor, db: torch.Tensor, kb: torch.Tensor,
    mb: torch.Tensor, d_pred: torch.Tensor, k_pred: torch.Tensor,
    bvals: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Loss function for CNN training (supports supervised and PINN modes)."""
    b_vals_t = torch.from_numpy(np.asarray(bvals, dtype=np.float32)).to(device)

    def masked_l1(p, t, m):
        return (p - t).abs().mul(m).sum() / m.sum().clamp_min(1.0)

    if method == "supervised":
        return masked_l1(d_pred, db, mb) + 0.2 * masked_l1(k_pred, kb, mb)

    elif method == "pinn_log":
        s0 = xb[:, 0:1]
        b_img = b_vals_t[1:, None, None]
        target = torch.log(xb[:, 1:] / (s0 + 1.0e-6)).clamp_min(-10)
        pred = -b_img * d_pred + (b_img ** 2) * (d_pred ** 2) * k_pred / 6.0
        return F.mse_loss(pred * mb, target * mb.expand_as(target))

    elif method == "supervised_log":
        sup = masked_l1(d_pred, db, mb) + 0.2 * masked_l1(k_pred, kb, mb)
        s0 = xb[:, 0:1]
        b_img = b_vals_t[1:, None, None]
        target = torch.log(xb[:, 1:] / (s0 + 1.0e-6)).clamp_min(-10)
        pred = -b_img * d_pred + (b_img ** 2) * (d_pred ** 2) * k_pred / 6.0
        log_loss = F.mse_loss(pred * mb, target * mb.expand_as(target))
        return sup + 0.1 * log_loss

    else:
        raise ValueError(f"Unknown CNN method: {method}")


def _masked_l1(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (p - t).abs().mul(m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Experiment runners: procedural vs Phase2
# ---------------------------------------------------------------------------


def run_procedural_experiment(
    paths: Paths,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run DKI experiment on procedural brain-like phantom data."""
    print(f"\n{'='*60}")
    print(f"  Seed {seed} — Procedural phantom")
    print(f"{'='*60}")
    bvals = np.asarray(args.bvals, dtype=np.float32)
    d_max, k_max = args.d_max, args.k_max
    n_train, n_val = args.train, args.val
    n_test = args.samples - n_train - n_val
    if n_test <= 0:
        raise ValueError("samples must be > train + val")

    data = generate_dataset(args.samples, args.size, args.size, bvals, args.noise_levels, seed)
    train, val, test = split_dataset(data, n_train, n_val)
    print(f"  {args.samples} samples ({n_train}/{n_val}/{n_test})  size={args.size}  noise={args.noise_levels}")

    return _run_common_pipeline(paths, seed, args, device, train, val, test)


def run_phase2_cross_experiment(
    paths: Paths,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run DKI experiment on Phase2 data with Stage 1 style cross-noise evaluation.

    For each noise level:
      - Load separate training data with that noise
      - Train models
    Then evaluate each model across ALL noise levels.
    """
    print(f"\n{'='*60}")
    print(f"  Seed {seed} — Phase2 cross-noise evaluation (Stage 1 style)")
    print(f"{'='*60}")

    d_max, k_max = args.d_max, args.k_max
    noise_levels = args.phase2_noise_levels
    n_sigma = len(noise_levels)
    total_files = args.phase2_max_files

    # Number of files per sigma (evenly split)
    n_per_sigma = max(1, total_files // n_sigma)
    n_val_per_sigma = max(1, n_per_sigma // 5)
    n_test_per_sigma = max(1, n_per_sigma // 5)

    from dki_utils import list_phase2_files, load_phase2_sample, add_rician_noise, DkiData
    all_files = list_phase2_files(args.phase2_root, max_files=0)

    def _load_sigma_data(sigma_files: list[Path], sigma: float) -> DkiData:
        rng_n = np.random.default_rng(seed + 999)
        dwi_n, dwi_c, s0s, ds, ks, masks, sigmas = [], [], [], [], [], [], []
        bvals = None
        for fpath in sigma_files:
            s = load_phase2_sample(fpath)
            c = s["dwi_clean"]
            n = add_rician_noise(rng_n, c, sigma)
            dwi_n.append(n); dwi_c.append(c); s0s.append(s["s0"])
            ds.append(s["d"]); ks.append(s["k"]); masks.append(s["mask"])
            sigmas.append(sigma)
            if bvals is None: bvals = s["bvals"]
        return DkiData(dwi_noisy=np.stack(dwi_n), dwi_clean=np.stack(dwi_c),
            s0=np.stack(s0s), d=np.stack(ds), k=np.stack(ks),
            mask=np.stack(masks).astype(np.float32),
            bvals=bvals, sigma=np.asarray(sigmas, dtype=np.float32))

    # Shuffle & assign files to noise levels
    rng_f = np.random.default_rng(seed)
    rng_f.shuffle(all_files)
    files_pool = all_files[:total_files]

    sigma_data = {}
    for i, s in enumerate(noise_levels):
        start = i * n_per_sigma
        end = start + n_per_sigma
        pool = files_pool[start:end] if end <= len(files_pool) else files_pool[start:]
        sigma_data[s] = {
            "train": pool[:-n_val_per_sigma - n_test_per_sigma] if len(pool) > n_val_per_sigma + n_test_per_sigma else pool[:max(1,len(pool)//2)],
            "val": pool[-n_val_per_sigma - n_test_per_sigma:-n_test_per_sigma] if len(pool) > n_val_per_sigma + n_test_per_sigma else [],
            "test": pool[-n_test_per_sigma:] if len(pool) > n_test_per_sigma else [],
        }

    display_names = {
        "supervised": "supervised_mlp", "pinn_log": "pinn_log_no_gt",
        "pinn_rician": "pinn_rician_no_gt", "pinn_log_rician": "pinn_log_rician_no_gt",
        "semi_supervised": "semi_supervised_mlp",
        "pinn_log_predict_s0": "pinn_log_predict_s0",
        "pinn_log_rician_predict_s0": "pinn_log_rician_predict_s0",
    }
    voxel_methods = [m for m in args.voxel_methods if m in display_names]
    cnn_methods = [m for m in (args.cnn_methods or []) if m in ("supervised", "pinn_log")]

    all_rows = []

    # ---- Train separate models per noise ----
    for train_sigma in noise_levels:
        s_str = f"{train_sigma:.2g}"
        print(f"\n  --- Training noise={s_str} ---")
        tr_files = sigma_data[train_sigma]["train"]
        va_files = sigma_data[train_sigma]["val"]
        if len(tr_files) < 3:
            print(f"  SKIP: only {len(tr_files)} train files")
            continue

        train_data = _load_sigma_data(tr_files, train_sigma)
        val_data = _load_sigma_data(va_files, train_sigma) if va_files else train_data

        ckpt_dir = paths.checkpoints / f"train_sigma_{s_str}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Pre-extract voxel arrays ONCE for all methods at this noise level
        train_vox = phase2_voxel_arrays(train_data)
        val_vox = phase2_voxel_arrays(val_data) if va_files else train_vox

        for method in voxel_methods:
            print(f"    voxel {method} ...")
            ckpt = ckpt_dir / f"seed{seed}_{method}.pt"
            train_model(method, train_data, val_data, args.epochs, device,
                        d_max, k_max, ckpt, batch_size=args.batch_size, lr=args.lr,
                        train_arrays=train_vox, val_arrays=val_vox)

        # ---- CNN training per noise (image-level DKI map prediction) ----
        for cnn_method in cnn_methods:
            print(f"    CNN {cnn_method} ...")
            ckpt = ckpt_dir / f"seed{seed}_cnn_{cnn_method}.pt"
            train_cnn_model(cnn_method, train_data, val_data, args.epochs, device,
                            d_max, k_max, ckpt, batch_size=4, lr=5.0e-4)

    # ---- Cross-evaluate on all noise levels ----
    for test_sigma in noise_levels:
        te_files = sigma_data[test_sigma]["test"]
        if not te_files:
            continue
        test_data = _load_sigma_data(te_files, test_sigma)
        ts_str = f"{test_sigma:.2g}"

        # Baselines (noise-agnostic)
        d_adc = mono_adc_fit_as_d(test_data, d_max)
        d_poly, k_poly = polynomial_dki_fit(test_data, d_max, k_max)
        for method, d_pred, k_pred in [
            ("mono_adc_as_d", d_adc, np.zeros_like(d_adc)),
            ("poly_dki_fit", d_poly, k_poly),
        ]:
            r = metrics(method, test_data, d_pred, k_pred)
            r["train_noise"] = "none"
            r["test_noise"] = ts_str
            r["seed"] = seed
            all_rows.append(r)

        # Each trained model
        for train_sigma in noise_levels:
            s_str = f"{train_sigma:.2g}"
            ckpt_dir = paths.checkpoints / f"train_sigma_{s_str}"

            # Voxel MLP models
            for method in voxel_methods:
                ckpt = ckpt_dir / f"seed{seed}_{method}.pt"
                if not ckpt.exists():
                    continue
                d_pred, k_pred = predict_model(ckpt, test_data, device)
                r = metrics(display_names[method], test_data, d_pred, k_pred)
                r["train_noise"] = s_str
                r["test_noise"] = ts_str
                r["method"] = display_names[method]
                r["seed"] = seed
                all_rows.append(r)

            # CNN models
            for cnn_method in cnn_methods:
                ckpt = ckpt_dir / f"seed{seed}_cnn_{cnn_method}.pt"
                if not ckpt.exists():
                    continue
                d_pred, k_pred = predict_cnn_model(ckpt, test_data, device)
                cnn_name = f"cnn_{cnn_method}"
                r = metrics(cnn_name, test_data, d_pred, k_pred)
                r["train_noise"] = s_str
                r["test_noise"] = ts_str
                r["method"] = cnn_name
                r["seed"] = seed
                all_rows.append(r)

    return all_rows, []


# ---------------------------------------------------------------------------
# Legacy Phase2 (mixed-noise) — keep as fallback
# ---------------------------------------------------------------------------


def run_phase2_experiment(
    paths: Paths,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Legacy mode: mixed-noise training (no cross-evaluation). Use --cross-eval for Stage-1 style."""
    print(f"\n{'='*60}")
    print(f"  Seed {seed} — Phase2 mixed-noise (legacy mode)")
    print(f"{'='*60}")
    data = load_phase2_data(args.phase2_root, max_files=args.phase2_max_files,
                            seed=seed, noise_levels=args.phase2_noise_levels)
    n_total = data.dwi_noisy.shape[0]
    n_train = max(1, int(n_total * 0.7))
    n_val = max(1, int(n_total * 0.15))
    if n_total - n_train - n_val <= 0:
        n_val = n_total - n_train - 1
    train, val, test = split_dataset(data, n_train, n_val)
    print(f"  {n_total}s ({n_train}/{n_val}/{n_total-n_train-n_val}) size={data.dwi_noisy.shape[-1]}")
    return _run_common_pipeline(paths, seed, args, device, train, val, test)


def _run_common_pipeline(
    paths: Paths,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    train: DkiData,
    val: DkiData,
    test: DkiData,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Shared pipeline: baselines -> voxel MLP training -> (optional) CNN -> metrics."""
    d_max, k_max = args.d_max, args.k_max

    # ----- Baseline fits -----
    rows = []
    d_adc = mono_adc_fit_as_d(test, d_max)
    rows.append(metrics("mono_adc_as_d", test, d_adc, np.zeros_like(d_adc)))
    d_poly, k_poly = polynomial_dki_fit(test, d_max, k_max)
    rows.append(metrics("poly_dki_fit", test, d_poly, k_poly))
    print(f"  mono_adc_as_d  D RMSE={rows[-2]['d_rmse_x1e3']:.4f}  K RMSE={rows[-2]['k_rmse']:.4f}")
    print(f"  poly_dki_fit   D RMSE={rows[-1]['d_rmse_x1e3']:.4f}  K RMSE={rows[-1]['k_rmse']:.4f}")

    # ----- Noise-stratified baselines -----
    noise_rows = []
    baseline_preds = {
        "mono_adc_as_d": (d_adc, np.zeros_like(d_adc)),
        "poly_dki_fit": (d_poly, k_poly),
    }
    for r in rows:
        if r["method"] in baseline_preds:
            d_p, k_p = baseline_preds[r["method"]]
            noise_rows.extend(noise_stratified_metrics(r["method"], test, d_p, k_p))

    # ----- Voxel MLP training -----
    display_names = {
        "supervised": "supervised_mlp",
        "pinn_log": "pinn_log_no_gt",
        "pinn_rician": "pinn_rician_no_gt",
        "pinn_log_rician": "pinn_log_rician_no_gt",
        "semi_supervised": "semi_supervised_mlp",
    }
    voxel_methods = [m for m in args.voxel_methods if m in display_names]
    for method in voxel_methods:
        print(f"  Training voxel {method} ...")
        ckpt = paths.checkpoints / f"seed{seed}_{method}.pt"
        train_model(method, train, val, args.epochs, device, d_max, k_max, ckpt,
                    batch_size=args.batch_size, lr=args.lr)
        d_pred, k_pred = predict_model(ckpt, test, device)
        rows.append(metrics(display_names[method], test, d_pred, k_pred))
        noise_rows.extend(noise_stratified_metrics(display_names[method], test, d_pred, k_pred))

    # ----- CNN training (optional) -----
    cnn_methods = [m for m in args.cnn_methods if m in ("supervised", "pinn_log", "supervised_log")]
    for method in cnn_methods:
        print(f"  Training CNN {method} ...")
        ckpt = paths.checkpoints / f"seed{seed}_cnn_{method}.pt"
        train_cnn_model(method, train, val, args.epochs, device, d_max, k_max, ckpt,
                        batch_size=args.cnn_batch_size, lr=args.cnn_lr)
        d_pred, k_pred = predict_cnn_model(ckpt, test, device)
        name = f"cnn_{method}"
        rows.append(metrics(name, test, d_pred, k_pred))
        noise_rows.extend(noise_stratified_metrics(name, test, d_pred, k_pred))

    # ----- Seed tracking -----
    for r in rows:
        r["seed"] = seed
    for r in noise_rows:
        r["seed"] = seed

    # ----- Sample maps -----
    if args.save_maps:
        preds = {}
        for r in rows:
            m = r["method"]
            if m in ("mono_adc_as_d",):
                preds[m] = (d_adc, np.zeros_like(d_adc))
            elif m == "poly_dki_fit":
                preds[m] = (d_poly, k_poly)
            elif "supervised" in m or "pinn" in m:
                if m.startswith("cnn_"):
                    ckpt = paths.checkpoints / f"seed{seed}_cnn_{m[4:]}.pt"
                    if ckpt.exists():
                        d_p, k_p = predict_cnn_model(ckpt, test, device)
                        preds[m] = (d_p, k_p)
                else:
                    rev_map = {v: k for k, v in display_names.items()}
                    if m in rev_map:
                        ckpt = paths.checkpoints / f"seed{seed}_{rev_map[m]}.pt"
                        if ckpt.exists():
                            d_p, k_p = predict_model(ckpt, test, device)
                            preds[m] = (d_p, k_p)
        if preds:
            plot_sample_maps(test, preds, paths.figures / f"sample_maps_seed{seed}.png", sample_idx=0)

    return rows, noise_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DKI PINN Extension — Comprehensive Experiment")

    # Data source
    parser.add_argument("--data-source", default="phase2",
                        choices=["procedural", "phase2"],
                        help="Data source: procedural phantom or Phase2 real simulated DKI data")
    parser.add_argument("--phase2-root", default="../data/simulated/03_Phase2_UNet_Synthesis",
                        help="Path to Phase2 data directory")
    parser.add_argument("--phase2-max-files", type=int, default=200,
                        help="Max Phase2 samples to use (0 = all 860)")
    parser.add_argument("--phase2-noise-levels", nargs="*", type=float,
                        default=[0.01, 0.03, 0.05, 0.08],
                        help="Rician noise levels for Phase2 data (added dynamically from dwi_clean)")

    # Procedural data params (ignored when --data-source=phase2)
    parser.add_argument("--samples", type=int, default=240,
                        help="Total samples per seed (procedural only)")
    parser.add_argument("--train", type=int, default=160,
                        help="Training samples per seed (procedural only)")
    parser.add_argument("--val", type=int, default=40,
                        help="Validation samples per seed (procedural only)")
    parser.add_argument("--size", type=int, default=48,
                        help="Image size H=W (procedural only)")
    parser.add_argument("--bvals", nargs="*", type=float,
                        default=[0, 500, 1000, 1500, 2000],
                        help="B-values (procedural only)")
    parser.add_argument("--noise-levels", nargs="*", type=float,
                        default=[0.01, 0.03, 0.05, 0.08],
                        help="Rician noise levels for procedural data (ignored for phase2; use --phase2-noise-levels)")

    # Multi-seed evaluation
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 514],
                        help="Random seeds for multi-seed evaluation")

    # Training params
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for all methods")
    parser.add_argument("--d-max", type=float, default=0.0035,
                        help="Max diffusivity for sigmoid clamping")
    parser.add_argument("--k-max", type=float, default=2.0,
                        help="Max kurtosis for sigmoid clamping")

    # Method selection
    parser.add_argument("--voxel-methods", nargs="*",
                        default=["supervised", "pinn_log", "pinn_log_rician"],
                        help="Voxel MLP methods to train")
    parser.add_argument("--cnn-methods", nargs="*", default=[],
                        help="CNN methods to train (supervised, pinn_log, supervised_log)")

    # Hyperparams
    parser.add_argument("--batch-size", type=int, default=32768,
                        help="Batch size for voxel MLP training")
    parser.add_argument("--lr", type=float, default=1.0e-3,
                        help="Learning rate for voxel MLP")
    parser.add_argument("--cnn-batch-size", type=int, default=2,
                        help="Batch size for CNN training (use 2 for 192x192 Phase2)")
    parser.add_argument("--cnn-lr", type=float, default=5.0e-4,
                        help="Learning rate for CNN")

    # Misc
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--save-maps", action="store_true", default=True,
                        help="Save representative sample maps")
    parser.add_argument("--cross-eval", action="store_true", default=True,
                        help="Stage-1 style cross-noise evaluation (separate train per sigma)")
    parser.add_argument("--output-root", default="outputs/dki_extension",
                        help="Output directory")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  |  Data source: {args.data_source}")

    # Paths
    paths = make_paths(Path(args.output_root))
    save_config(paths.root / "config.json", **vars(args))

    # Run experiments
    all_rows: list[dict[str, object]] = []
    all_noise_rows: list[dict[str, object]] = []
    do_cross = args.data_source == "phase2" and args.cross_eval
    for seed in args.seeds:
        if args.data_source == "procedural":
            rows, noise_rows = run_procedural_experiment(paths, seed, args, device)
        elif do_cross:
            rows, noise_rows = run_phase2_cross_experiment(paths, seed, args, device)
        elif args.data_source == "phase2":
            rows, noise_rows = run_phase2_experiment(paths, seed, args, device)
        else:
            raise ValueError(f"Unknown data source: {args.data_source}")
        all_rows.extend(rows)
        all_noise_rows.extend(noise_rows)

    # Save metric CSVs
    write_csv(paths.metrics / "per_seed_metrics.csv", all_rows)
    write_csv(paths.metrics / "noise_stratified.csv", all_noise_rows)

    # Summarize
    summary = summarize(all_rows, "method")
    write_csv(paths.metrics / "summary.csv", summary)
    if all_noise_rows:
        noise_summary = summarize(all_noise_rows, "method")
        write_csv(paths.metrics / "noise_summary.csv", noise_summary)

    # Figures
    plot_dki_bars(summary, paths.figures / "dki_method_comparison.png")
    if all_noise_rows:
        plot_noise_curves(all_noise_rows, paths.figures / "dki_noise_curves.png")

    # Cross-noise heatmaps (only for cross-eval mode)
    if do_cross:
        for method_name in sorted(set(str(r["method"]) for r in all_rows)):
            method_rows = [r for r in all_rows if str(r["method"]) == method_name]
            for metric_name in ["d_rmse_x1e3", "k_rmse"]:
                fname = f"cross_noise_{metric_name}_{method_name}.png"
                plot_cross_noise_heatmap(
                    method_rows, metric_name, paths.figures / fname,
                    title=f"{method_name} — {metric_name}",
                )

    # Report
    if args.data_source == "phase2":
        desc = (
            f"Phase2 real simulated DKI data: up to {args.phase2_max_files} samples per seed, "
            f"192x192 px, {args.epochs} epochs, seeds={args.seeds}."
        )
    else:
        desc = (
            f"Procedural DKI phantom: {args.samples} samples/seed, {args.size}x{args.size}, "
            f"{args.epochs} epochs, seeds={args.seeds}."
        )
    write_report(paths.reports / "dki_extension_report.md", summary, all_noise_rows, vars(args), desc)

    # Print final summary table
    print(f"\n{'='*60}")
    print(f"  Final Summary ({args.data_source.upper()}) — sorted by D RMSE")
    print(f"{'='*60}")
    h1, h2 = "D RMSE (x1e-3)", "K RMSE"
    print(f"  {'Method':<30s}  {h1:<20s}  {h2:<15s}")
    print(f"  {'-'*30}  {'-'*20}  {'-'*15}")
    for row in sorted(summary, key=lambda r: float(r.get("d_rmse_x1e3_mean", 1e9))):
        d_str = f"{float(row.get('d_rmse_x1e3_mean', 0)):.4f} {chr(0xb1)}{float(row.get('d_rmse_x1e3_std', 0)):.4f}"
        k_str = f"{float(row.get('k_rmse_mean', 0)):.4f} {chr(0xb1)}{float(row.get('k_rmse_std', 0)):.4f}"
        print(f"  {row['method']:<30s}  {d_str:<20s}  {k_str:<15s}")

    print(f"\nOutputs: {paths.root}")


if __name__ == "__main__":
    main()

