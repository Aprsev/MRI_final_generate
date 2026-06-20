"""Extract training history from all checkpoints and save as CSV + plot curves."""
import torch, sys, json
from pathlib import Path
import numpy as np

root = Path(r"D:\Desktop\ZJU\grade3\25-26spring\磁共振成像原理及应用\Labatory\ADC_PINN_mapping\dki_pipeline\outputs_excellent\checkpoints")

rows = []
for ckpt_path in sorted(root.rglob("*.pt")):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    method = ckpt.get("method", ckpt_path.stem.split("_",1)[1] if "_" in ckpt_path.stem else "?")
    train_sigma = ckpt_path.parent.name  # e.g. train_sigma_0.01
    seed = ckpt_path.stem.split("_")[0]   # e.g. seed42
    history = ckpt.get("history", [])
    best_val = min((h["val_loss"] if "val_loss" in h else h.get("val_supervised_proxy", 1e9)) for h in history) if history else None
    final_train = history[-1]["train_loss"] if history else None
    final_val = history[-1]["val_loss"] if history and "val_loss" in history[-1] else (history[-1].get("val_supervised_proxy") if history else None)
    
    for h in history:
        rows.append({
            "train_sigma": train_sigma, "seed": seed, "method": method,
            "epoch": h["epoch"],
            "train_loss": h["train_loss"],
            "val_loss": h.get("val_loss", h.get("val_supervised_proxy", None)),
        })

# Save
import csv
out_csv = root.parent / "metrics" / "training_history.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["train_sigma","seed","method","epoch","train_loss","val_loss"])
    w.writeheader()
    w.writerows(rows)
print(f"Saved {len(rows)} rows to {out_csv}")

# Summary: best val per checkpoint
sum_rows = []
for ckpt_path in sorted(root.rglob("*.pt")):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    method = ckpt.get("method", "?")
    train_sigma = ckpt_path.parent.name
    seed = ckpt_path.stem.split("_")[0]
    history = ckpt.get("history", [])
    if not history: continue
    losses = [(h.get("val_loss", h.get("val_supervised_proxy")), h["train_loss"]) for h in history]
    best_val, best_train = min(losses, key=lambda x: x[0] if x[0] else 1e9)
    final = history[-1]
    sum_rows.append({
        "train_sigma": train_sigma, "seed": seed, "method": method,
        "best_val_loss": best_val, "final_train_loss": final["train_loss"],
        "best_epoch": min(range(len(losses)), key=lambda i: losses[i][0] if losses[i][0] else 1e9) + 1,
    })

sum_csv = root.parent / "metrics" / "training_summary.csv"
with open(sum_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["train_sigma","seed","method","best_val_loss","final_train_loss","best_epoch"])
    w.writeheader()
    w.writerows(sum_rows)
print(f"Saved summary to {sum_csv}")

# Plot training curves (all methods per sigma, averaged over 3 seeds)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for sigma_name in sorted(set(r["train_sigma"] for r in sum_rows)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    methods_plotted = set()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0
    
    for method in sorted(set(r["method"] for r in sum_rows if r["train_sigma"] == sigma_name)):
        method_rows = [r for r in rows if r["train_sigma"] == sigma_name and r["method"] == method]
        if not method_rows: continue
        epochs = sorted(set(r["epoch"] for r in method_rows))
        # Average over 3 seeds
        train_avg = [np.mean([r["train_loss"] for r in method_rows if r["epoch"] == e]) for e in epochs]
        val_avg = [np.mean([r["val_loss"] for r in method_rows if r["epoch"] == e and r["val_loss"]]) for e in epochs]
        
        c = colors[color_idx % 10]
        ax1.plot(epochs, train_avg, color=c, label=method, linewidth=1.2)
        val_epochs = [e for e, v in zip(epochs, val_avg) if not np.isnan(v)]
        val_vals = [v for v in val_avg if not np.isnan(v)]
        if val_vals:
            ax2.plot(val_epochs, val_vals, color=c, label=method, linewidth=1.2)
        color_idx += 1
    
    ax1.set_title(f"Train Loss — {sigma_name}", fontsize=12)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=7, loc="upper right")
    ax2.set_title(f"Val Loss (proxy) — {sigma_name}", fontsize=12)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out_png = root.parent / "figures" / f"training_curves_{sigma_name}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")

print("Done.")