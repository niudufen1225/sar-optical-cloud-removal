# GLF-CR x DADIGAN Screen-10 Evaluation (seed 2026)

## Scope

- S1 run: `outputs/allclear/2026-07-10T23-05-39_stage1_glfcr_s1_screen10_seed2026`
- S2 run: `outputs/allclear/2026-07-10T23-43-14_stage1_glfcr_s2_screen10_seed2026`
- Split: ALLClear `val`, 346 samples, evaluator batch size 1, `shuffle=False`.
- Checkpoints: S1 `best_epoch_0010_pixel_total_1.689622.pt`; S2 `best_epoch_0010_pixel_total_1.687005.pt`.

S1 is unique. Two complete S2 directories have identical resolved configuration, paired-initialization provenance, seed, and 10-epoch completion. This report deterministically selects the earliest S2 (`23-43-14`) and records the later `23-43-22` run as a duplicate rather than silently mixing it into the paired comparison.

## Integrity

Both selected runs logged train and validation epochs 1--10, finished normally, and contain no non-finite loss terms. The paired initialization is strict and uses the same `pair_id`: `34dc108df71aacea8ca26726`; S1 uses `s1_model_init.pt` and S2 uses `s2_model_init.pt`, both with seed 2026. For each selected run, the best and last checkpoint are both epoch 10 and their model state dictionaries are exactly equal.

The resolved configurations have no unexpected difference. Their only differences are:

| Field | S1 | S2 |
|---|---|---|
| `run_name` | `...s1_screen10` | `...s2_screen10` |
| `model.cloud_post_ddin_sar_filter` | `none` | `glfcr_dynamic` |
| `model.cloud_post_ddin_sar_filter_kernel_size` | `null` | `5` |

Data manifests, batch size 4, gradient accumulation 2, effective batch 8, Adam optimizer, cosine-warmup runtime scheduler, learning rate, loss, augmentation, and seed are otherwise identical. S1 took 28m39s (0.245 s/batch); S2 took 50m29s (0.437 s/batch). This is not a valid module-speed comparison because the two S2 duplicate jobs ran concurrently. Peak GPU memory was not logged and cannot be reconstructed after completion.

## Core Validation Metrics

All values are in the model supervision domain after the configured RGB reflectance stretch `[0, 0.35] -> [0, 1]`. SSIM uses RGB mean aggregation, `data_range=1`, uniform window size 7, and an in-region tight bounding box; empty or smaller-than-3x3 regions are excluded.

| Final output metric | S1 | S2 |
|---|---:|---:|
| Cloud MAE | 0.070797 | 0.070656 |
| Cloud RMSE | 0.130244 | 0.129865 |
| Cloud PSNR | 17.7049 | 17.7302 |
| Cloud SSIM | 0.781019 | 0.780962 |
| Full MAE | 0.066366 | 0.066297 |
| Known/clear MAE | 0.062110 | 0.062110 |
| Full RGB mean absolute channel bias | 0.005294 | 0.005152 |
| Cloud Haar LL MAE | 0.129651 | 0.129397 |
| Cloud Haar LH MAE | 0.010220 | 0.010213 |
| Cloud Haar HL MAE | 0.009820 | 0.009784 |
| Cloud Haar HH MAE | 0.005451 | 0.005438 |
| Cloud Haar HF energy-ratio absolute difference | 0.001479 | 0.001474 |

The one-level undecimated Haar SWT uses fixed Haar filters, one-pixel right/bottom reflect padding, and no decimation. `LH` is low-y/high-x and `HL` is high-y/low-x.

## Paired Comparison

All paired statistics use the exact `sample_id` intersection, `delta = S2 - S1`, 2,000 paired bootstrap resamples, seed 2026, and percentile 95% CIs. Lower is better for MAE/RMSE/RGB bias/wavelet errors; higher is better for PSNR/SSIM.

| Metric | Mean delta | 95% CI | S2 better | S1 better |
|---|---:|---:|---:|---:|
| Cloud MAE | -0.000116 | [-0.000508, 0.000189] | 44.6% | 55.4% |
| Cloud RMSE | -0.000168 | [-0.000609, 0.000179] | 47.1% | 52.9% |
| Cloud PSNR | +0.002006 | [-0.027306, 0.032951] | 47.1% | 52.9% |
| Cloud SSIM | -0.000057 | [-0.000314, 0.000232] | 34.4% | 65.6% |
| Full MAE | -0.000069 | [-0.000364, 0.000185] | 40.8% | 49.4% |
| Full RGB bias | -0.000054 | [-0.000409, 0.000243] | 45.4% | 44.8% |
| Cloud HF energy-ratio absolute difference | -0.00000494 | [-0.00000932, -0.00000129] | 58.3% | 41.7% |

S2 has a small, directionally consistent HF energy-ratio change, but the cloud MAE/RMSE/PSNR/SSIM and color CIs all include zero. This screen therefore does not establish an overall S2 advantage.

## Visuals and SAR Counterfactual

`visuals/` contains 12 compact S1/S2 comparisons: three fixed-order samples for each evaluator cloud-coverage bucket. Each panel includes the same cloudy input, target, stage output, SAR, restore mask, and hard-shadow view for S1 and S2. The first-three selection is fixed by validation order; it is not a cherry-picked best-case subset.

Within this displayed subset, the clearest cloud-MAE S2 gain is `roi696354_2022-01-24_2022-01-31_roi696354_s2_toa_2022_1_24_median` (`high_02`, delta -0.037161). The displayed S2 degradation is `roi13712_2022-03-11_2022-03-26_roi13712_s2_toa_2022_3_11_median` (`heavy_02`, delta +0.001981). Full sample identities and deltas are in `selected_samples.json`.

The optional 32-sample SAR real/zero/shuffle counterfactual was not executed: S2 was not clearly superior overall, and the slight HF improvement was not paired with a statistically supported color regression or a confirmed visual SAR-leakage signal.

## Interpretation Boundary

This directory reports a controlled 10-epoch screen only. It does not make a final architecture decision or claim that the post-DDIN dynamic filter is beneficial. The later duplicate S2 run also differs numerically despite identical recorded provenance, so any next comparison should avoid concurrent duplicate launches and retain the same deterministic evaluation protocol.
