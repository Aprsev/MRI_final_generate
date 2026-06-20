"""诊断：定位 3090 上每 epoch 13s 的瓶颈"""
import torch, time, numpy as np, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dki_utils import (list_phase2_files, load_phase2_sample, add_rician_noise,
                       DkiData, phase2_voxel_arrays, DkiMLP)
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. 加载数据
t0 = time.time()
files = list_phase2_files(str(Path(__file__).resolve().parent.parent / "data" / "03_Phase2_UNet_Synthesis"), max_files=0)
rng = np.random.default_rng(42)
files = files[:50]
dwi_n, dwi_c, s0s, ds, ks, masks, sigmas, bvals = [], [], [], [], [], [], [], None
for f in files:
    s = load_phase2_sample(f)
    c = s["dwi_clean"]
    dwi_n.append(add_rician_noise(rng, c, 0.01))
    dwi_c.append(c); s0s.append(s["s0"]); ds.append(s["d"]); ks.append(s["k"])
    masks.append(s["mask"]); sigmas.append(0.01)
    if bvals is None: bvals = s["bvals"]
data = DkiData(dwi_noisy=np.stack(dwi_n), dwi_clean=np.stack(dwi_c),
    s0=np.stack(s0s), d=np.stack(ds), k=np.stack(ks),
    mask=np.stack(masks).astype(np.float32), bvals=bvals, sigma=np.asarray(sigmas))
print(f"1. 加载50文件: {time.time()-t0:.1f}s")

# 2. voxel_arrays
t0 = time.time()
xv, dv, kv, sigv, s0v = phase2_voxel_arrays(data)
print(f"2. voxel_arrays: {time.time()-t0:.1f}s")

# 3. 一个 epoch
t0 = time.time()
model = DkiMLP(5, 0.0035, 2.0).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(
    torch.from_numpy(xv), torch.from_numpy(dv), torch.from_numpy(kv),
    torch.from_numpy(sigv), torch.from_numpy(s0v)), batch_size=32768, shuffle=True)

model.train()
for xb, db, kb, sigb, s0b in loader:
    xb=xb.to(device); db=db.to(device); kb=kb.to(device)
    sigb=sigb.to(device); s0b=s0b.to(device)
    d_pred, k_pred = model(xb)
    loss = torch.nn.functional.l1_loss(d_pred*1000,db*1000)+0.2*torch.nn.functional.l1_loss(k_pred,kb)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"3. 一个epoch: {time.time()-t0:.1f}s")

# 4. 空循环 DataLoader
t0 = time.time()
for xb, db, kb, sigb, s0b in loader:
    pass
print(f"4. 空循环DataLoader: {time.time()-t0:.3f}s")

# 5. 信息
print(f"5. xv.shape: {xv.shape}")
print(f"6. 模型参数在GPU: {next(model.parameters()).device}")

# 6. 数据已在GPU时的10次训练
t0 = time.time()
xb_t = torch.from_numpy(xv).to(device)
db_t = torch.from_numpy(dv).to(device)
kb_t = torch.from_numpy(kv).to(device)
torch.cuda.synchronize()
t1 = time.time()
for _ in range(10):
    d_pred, k_pred = model(xb_t)
    loss = torch.nn.functional.l1_loss(d_pred*1000,db_t*1000)+0.2*torch.nn.functional.l1_loss(k_pred,kb_t)
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
print(f"7. 10次GPU训练(数据已在GPU): {time.time()-t1:.2f}s (上传: {t1-t0:.2f}s)")