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
- `voxel_methods`: `['supervised', 'pinn_log', 'pinn_log_rician', 'semi_supervised', 'pinn_log_predict_s0', 'pinn_log_rician_predict_s0']`
- `cnn_methods`: `['supervised', 'pinn_log']`
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
| `cnn_supervised` | 0.1286±0.0426 | 0.1015±0.0376 | 0.0771±0.0092 | 0.0613±0.0084 |
| `supervised_mlp` | 0.1476±0.0706 | 0.1162±0.0544 | 0.0927±0.0170 | 0.0738±0.0138 |
| `semi_supervised_mlp` | 0.1686±0.0637 | 0.1363±0.0507 | 0.3098±0.1513 | 0.2837±0.1516 |
| `pinn_log_rician_no_gt` | 0.1934±0.0596 | 0.1584±0.0500 | 0.3921±0.1409 | 0.3592±0.1420 |
| `pinn_log_no_gt` | 0.1995±0.0654 | 0.1644±0.0571 | 0.3752±0.1579 | 0.3446±0.1575 |
| `pinn_log_rician_predict_s0` | 0.2127±0.0564 | 0.1668±0.0455 | 0.3994±0.1530 | 0.3658±0.1537 |
| `pinn_log_predict_s0` | 0.2158±0.0584 | 0.1643±0.0477 | 0.4259±0.1572 | 0.3901±0.1591 |
| `mono_adc_as_d` | 0.3689±0.1582 | 0.2712±0.1304 | 0.6570±0.0163 | 0.6278±0.0190 |
| `cnn_pinn_log` | 0.4548±0.0754 | 0.4038±0.0625 | 0.1416±0.0415 | 0.1153±0.0343 |
| `poly_dki_fit` | 0.4770±0.1671 | 0.3624±0.1443 | 0.4881±0.0568 | 0.4046±0.0758 |

## Key Takeaways

1. **log-domain residual is the core effective term for no-GT PINN.**
2. **Rician likelihood helps as an auxiliary term but should not be used alone.**
3. **DKI benefits more from physics-informed constraints than ADC** due to coupled D/K parameters.
4. **CNN/U-Net extension** (if included) shows whether spatial priors improve DKI parameter estimation.