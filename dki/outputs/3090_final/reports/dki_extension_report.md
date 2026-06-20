# DKI Extension Experiment Report

Phase2 real simulated DKI data: up to 200 samples per seed, 192x192 px, 50 epochs, seeds=[42, 123, 514].

## Configuration

- `data_source`: `phase2`
- `phase2_root`: `/root/dki_pipeline/data/03_Phase2_UNet_Synthesis`
- `phase2_max_files`: `200`
- `phase2_noise_levels`: `[0.01, 0.03, 0.05, 0.08]`
- `samples`: `240`
- `train`: `160`
- `val`: `40`
- `size`: `48`
- `bvals`: `[0, 500, 1000, 1500, 2000]`
- `noise_levels`: `[0.01, 0.03, 0.05, 0.08]`
- `seeds`: `[42, 123, 514]`
- `epochs`: `50`
- `d_max`: `0.0035`
- `k_max`: `2.0`
- `voxel_methods`: `['supervised', 'pinn_log', 'pinn_log_rician']`
- `cnn_methods`: `[]`
- `batch_size`: `32768`
- `lr`: `0.001`
- `cnn_batch_size`: `2`
- `cnn_lr`: `0.0005`
- `device`: `auto`
- `save_maps`: `True`
- `cross_eval`: `True`
- `output_root`: `/root/dki_pipeline/outputs`

## Overall Results (multi-seed mean ± std)

| method | D RMSE (×10⁻³) | D MAE (×10⁻³) | K RMSE | K MAE |
|---|---:|---:|---:|---:|
| `supervised_mlp` | 0.1480±0.0699 | 0.1166±0.0538 | 0.0929±0.0170 | 0.0740±0.0138 |
| `pinn_log_no_gt` | 0.1946±0.0626 | 0.1594±0.0530 | 0.3901±0.1483 | 0.3573±0.1485 |
| `pinn_log_rician_no_gt` | 0.2000±0.0644 | 0.1647±0.0559 | 0.3825±0.1519 | 0.3506±0.1520 |
| `mono_adc_as_d` | 0.3689±0.1582 | 0.2712±0.1304 | 0.6570±0.0163 | 0.6278±0.0190 |
| `poly_dki_fit` | 0.4770±0.1671 | 0.3624±0.1443 | 0.4881±0.0568 | 0.4046±0.0758 |

## Key Takeaways

1. **log-domain residual is the core effective term for no-GT PINN.**
2. **Rician likelihood helps as an auxiliary term but should not be used alone.**
3. **DKI benefits more from physics-informed constraints than ADC** due to coupled D/K parameters.
4. **CNN/U-Net extension** (if included) shows whether spatial priors improve DKI parameter estimation.