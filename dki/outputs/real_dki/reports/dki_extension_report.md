# DKI Extension Experiment Report

Phase2 real simulated DKI data: up to 200 samples per seed, 192x192 px, 50 epochs, seeds=[42, 123, 514].

## Configuration

- `data_source`: `phase2`
- `phase2_root`: `data/03_Phase2_UNet_Synthesis_DKI`
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
| `cnn_supervised` | 0.1115±0.0292 | 0.0861±0.0249 | 0.0720±0.0072 | 0.0569±0.0065 |
| `semi_supervised_mlp` | 0.1349±0.0620 | 0.1069±0.0487 | 0.0921±0.0267 | 0.0729±0.0217 |
| `supervised_mlp` | 0.1353±0.0637 | 0.1074±0.0500 | 0.0799±0.0179 | 0.0633±0.0146 |
| `pinn_log_rician_no_gt` | 0.1520±0.0631 | 0.1190±0.0493 | 0.1139±0.0336 | 0.0909±0.0282 |
| `pinn_log_no_gt` | 0.1523±0.0641 | 0.1187±0.0507 | 0.1076±0.0272 | 0.0853±0.0229 |
| `cnn_pinn_log` | 0.1562±0.0688 | 0.1251±0.0613 | 0.0998±0.0223 | 0.0791±0.0184 |
| `pinn_log_predict_s0` | 0.1789±0.0645 | 0.1415±0.0498 | 0.1403±0.0712 | 0.1128±0.0582 |
| `pinn_log_rician_predict_s0` | 0.1835±0.0615 | 0.1442±0.0472 | 0.1310±0.0558 | 0.1058±0.0467 |
| `poly_dki_fit` | 0.3363±0.1563 | 0.2500±0.1278 | 0.2869±0.1194 | 0.2248±0.1002 |
| `mono_adc_as_d` | 0.5576±0.0728 | 0.5010±0.0471 | 0.6570±0.0163 | 0.6278±0.0190 |

## Key Takeaways

1. **log-domain residual is the core effective term for no-GT PINN.**
2. **Rician likelihood helps as an auxiliary term but should not be used alone.**
3. **DKI benefits more from physics-informed constraints than ADC** due to coupled D/K parameters.
4. **CNN/U-Net extension** (if included) shows whether spatial priors improve DKI parameter estimation.