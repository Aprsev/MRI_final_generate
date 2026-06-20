import numpy as np, os
root = r'D:\Desktop\ZJU\grade3\25-26spring\磁共振成像原理及应用\Labatory\ADC_PINN_mapping\data\simulated'
for dname in ['01_MNI152_Grayscale', '02_Phase1_T1_Inversion', '03_Phase2_UNet_Synthesis']:
    dpath = os.path.join(root, dname)
    files = sorted(os.listdir(dpath))
    f = os.path.join(dpath, files[0])
    d = np.load(f, allow_pickle=True)
    print(f'=== {dname} ===')
    print(f'  files: {len(files)}, bvals: {d["bvals"]}')
    has_k = 'k_gt' in d.keys()
    print(f'  has k_gt: {has_k}')
    if has_k:
        dwi = d['dwi_clean']
        s0v = d['s0_gt']
        b = d['bvals']
        mask = d['mask'].astype(bool)
        ys, xs = np.where(mask)
        mono_mse_list = []
        dki_mse_list = []
        for idx in range(0, len(ys), 500):
            i, j = ys[idx], xs[idx]
            sig = dwi[:, i, j]
            sv = s0v[i, j]
            dv = d['d_gt'][i, j]
            kv = d['k_gt'][i, j]
            mono_pred = sv * np.exp(-b * dv)
            dki_pred = sv * np.exp(-b * dv + b**2 * dv**2 * kv / 6)
            mono_mse_list.append(((sig-mono_pred)**2).mean())
            dki_mse_list.append(((sig-dki_pred)**2).mean())
        print(f'  monoexp MSE: mean={np.mean(mono_mse_list):.2e}, max={np.max(mono_mse_list):.2e}')
        print(f'  DKI MSE:     mean={np.mean(dki_mse_list):.2e}, max={np.max(dki_mse_list):.2e}')
        is_dki = np.mean(dki_mse_list) < np.mean(mono_mse_list) * 0.01
        print(f'  Signal uses K: {is_dki}')
    print()
