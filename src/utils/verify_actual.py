import numpy as np

datasets = [
    '01_MNI152_Grayscale',
    '02_Phase1_T1_Inversion', 
    '03_Phase2_UNet_Synthesis'
]

print('=' * 70)
print('验证1: DWI是否由 ADC 单指数模型 S0*exp(-b*D) 生成')
print('验证2: 如果使用 DKI 模型 S0*exp(-b*D + 1/6*b^2*D^2*K)，会有多大差异')
print('=' * 70)

for ds in datasets:
    f = np.load(f'data/{ds}/sample_0000.npz')
    s0 = f['s0_gt']
    d  = f['d_gt']
    k  = f['k_gt']
    mask = f['mask']
    bvals = f['bvals']
    
    print(f'\n--- {ds} (sample_0000) ---')
    print(f'source: {str(f["source"])}')
    
    for b_idx, b in enumerate(bvals):
        if b == 0:
            continue
        
        dwi_actual = f['dwi_clean'][b_idx]  # 实际文件中存储的 DWI
        
        # ADC模型重建 (无K项)
        adc_recon = s0 * np.exp(-b * d)
        
        # DKI模型重建 (有K项)
        dki_recon = s0 * np.exp(-b * d + (1/6) * b**2 * d**2 * k)
        
        # 与实际dwi_clean比较
        err_adc = np.abs(dwi_actual - adc_recon)[mask].max()
        err_dki = np.abs(dwi_actual - dki_recon)[mask].max()
        
        print(f'  b={b:5d}: |实际 - ADC模型| = {err_adc:.3e}  '
              f'|实际 - DKI模型| = {err_dki:.3e}')
    
    f.close()

print('\n结论:')
print('  - 若 |实际 - ADC模型| ≈ 0 (浮点误差) → DWI由 ADC单指数模型 生成')
print('  - 若 |实际 - DKI模型| ≈ 0 (浮点误差) → DWI由 DKI模型 生成')
print('  - 两者皆远大于0 → 使用了其他模型')
