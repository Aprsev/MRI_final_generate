"""修复 Phase2 数据：用真正的 DKI 方程重新生成 dwi_clean，保存到新目录。
   S(b) = S0 * exp(-b*D + b^2*D^2*K/6)
"""
import numpy as np, os, sys, shutil
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/03_Phase2_UNet_Synthesis")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/03_Phase2_UNet_Synthesis_DKI")
dst.mkdir(parents=True, exist_ok=True)

files = sorted(src.glob("*.npz"))
print(f"Source: {src} ({len(files)} files)")
print(f"Target: {dst}")

dki_equation = "S(b) = S0 * exp(-b*D + b^2 * D^2 * K / 6)"

for i, fpath in enumerate(files):
    d = np.load(fpath, allow_pickle=True)
    s0 = d["s0_gt"].astype(np.float32)
    d_gt = d["d_gt"].astype(np.float32)
    k_gt = d["k_gt"].astype(np.float32)
    bvals = d["bvals"].astype(np.float32)
    mask = d["mask"].astype(np.float32)

    # True DKI forward model
    b = bvals[:, None, None]
    expo = -b * d_gt[None] + (b ** 2) * (d_gt[None] ** 2) * k_gt[None] / 6.0
    dwi_clean_new = (s0[None] * np.exp(expo)).astype(np.float32)
    dwi_clean_new[:, ~mask.astype(bool)] = 0.0

    old = d["dwi_clean"]
    diff = np.abs(dwi_clean_new - old).max()

    # Save to new directory
    out = {k: d[k] for k in d.keys()}
    out["dwi_clean"] = dwi_clean_new
    out["dwi_noisy"] = dwi_clean_new  # clean, pipeline adds noise
    np.savez_compressed(dst / fpath.name, **out)

    if i < 3 or i % 200 == 0:
        print(f"  [{i+1}/{len(files)}] {fpath.name}  signal_diff={diff:.6f}")

print(f"Done. {len(files)} files -> {dst}")
print(f"DWI signal: {dki_equation}")