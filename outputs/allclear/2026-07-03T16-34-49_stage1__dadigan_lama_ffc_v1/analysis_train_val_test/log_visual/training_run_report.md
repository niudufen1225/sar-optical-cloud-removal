# Training Run Report: `2026-07-03T16-34-49_stage1__dadigan_lama_ffc_v1`

## 1. Run 状态
- Run dir: `/home/students/sushaoqi/CR/main/outputs/allclear/2026-07-03T16-34-49_stage1__dadigan_lama_ffc_v1`
- 已记录 epoch: `200` / 配置 epoch: `200` (100.00%)
- Train batches: `243` | Val batches: `32`
- 报告主指标: `recon_total`
- Best val recon_total: `14.5051` at epoch `109`
- Train recon_total improvement: `26.55%` | Val recon_total improvement: `11.27%`
- Final generalization gap (val - train recon_total): `2.0549`

## 2. 核心异常与风险
| severity | topic         | message                                                             |
| -------- | ------------- | ------------------------------------------------------------------- |
| info     | discriminator | 最近若干 epoch 的 real/fake logit 非常接近；判别器区分度弱或处于均衡状态，需要结合视觉和 loss 振荡判断。 |
| info     | softshadow    | 最近 10 个 epoch 的主要 SoftShadow loss 几乎无波动；可能已经平台期，也可能验证/样本构成固定导致。     |

## 3. Loss 调度状态
- 已参与训练的主要项: `cloud_l1, cloud_kl, cloud_adv, feature_matching, perceptual, disc_total`

| metric           | target_weight | current_weight | start_epoch | ramp_epochs | active_epoch_count | status   |
| ---------------- | ------------- | -------------- | ----------- | ----------- | ------------------ | -------- |
| final_l1         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| grad             | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| shadow_removal   | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| shadow_mask      | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| shadow_penumbra  | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_l1         | 100.000000    | 100.000000     | 1           | 0           | 200                | active   |
| cloud_kl         | 30.000000     | 30.000000      | 1           | 0           | 200                | active   |
| cloud_adv        | 1.000000      | 1.000000       | 20          | 20          | 181                | active   |
| feature_matching | 10.000000     | 10.000000      | 20          | 20          | 181                | active   |
| perceptual       | 5.000000      | 5.000000       | 20          | 20          | 181                | active   |
| disc_total       | 1.000000      | 1.000000       | 20          | 20          | 181                | active   |

## 4. GAN/FM 启动冲击
- 这里比较 adversarial 或 feature matching 权重首次大于 0 之前的 train 均值，与启动后前 3 个 epoch 的 train 均值。
| transition_epoch | metric           | pre_mean  | early_mean | delta     | delta_pct  |
| ---------------- | ---------------- | --------- | ---------- | --------- | ---------- |
| 20               | cloud_l1         | 0.043600  | 0.037272   | -0.006328 | -14.514256 |
| 20               | recon_total      | 15.244450 | 14.607659  | -0.636791 | -4.177200  |
| 20               | total            | 15.244450 | 14.821716  | -0.422734 | -2.773034  |
| 20               | gan_total        | 0.000000  | 0.214057   | 0.214057  | NA         |
| 20               | cloud_adv        | 0.000000  | 0.433914   | 0.433914  | NA         |
| 20               | feature_matching | 0.000000  | 0.098849   | 0.098849  | NA         |
| 20               | perceptual       | 0.000000  | 0.146649   | 0.146649  | NA         |
| 20               | shadow_removal   | 0.000000  | 0.000000   | 0.000000  | NA         |
| 20               | shadow_mask      | 0.000000  | 0.000000   | 0.000000  | NA         |
| 20               | shadow_penumbra  | 0.000000  | 0.000000   | 0.000000  | NA         |

## 5. SoftShadow 诊断
- 这部分只基于 epoch 聚合日志，不能直接证明每个 batch 都有梯度；但可以判断 loss 项是否长期为 0、是否在权重为正时失效。
| split | term            | weight_positive_epoch_count | nonzero_epoch_count | zero_while_weight_positive_epoch_count | first_nonzero_epoch | latest_value | latest_weight | tail10_std |
| ----- | --------------- | --------------------------- | ------------------- | -------------------------------------- | ------------------- | ------------ | ------------- | ---------- |
| train | shadow_removal  | 0                           | 0                   | 0                                      | NA                  | 0.000000     | 0.000000      | 0.000000   |
| train | shadow_mask     | 0                           | 0                   | 0                                      | NA                  | 0.000000     | 0.000000      | 0.000000   |
| train | shadow_penumbra | 0                           | 0                   | 0                                      | NA                  | 0.000000     | 0.000000      | 0.000000   |
| val   | shadow_removal  | 0                           | 0                   | 0                                      | NA                  | 0.000000     | 0.000000      | 0.000000   |
| val   | shadow_mask     | 0                           | 0                   | 0                                      | NA                  | 0.000000     | 0.000000      | 0.000000   |
| val   | shadow_penumbra | 0                           | 0                   | 0                                      | NA                  | 0.000000     | 0.000000      | 0.000000   |

## 6. 指标趋势摘要
| split | metric         | first     | last      | best_epoch | best      | improvement_pct_if_lower_better | num_points |
| ----- | -------------- | --------- | --------- | ---------- | --------- | ------------------------------- | ---------- |
| train | total          | 17.496627 | 14.326547 | 192.000000 | 14.229851 | 18.118234                       | 200        |
| train | recon_total    | 17.496627 | 12.850519 | 192.000000 | 12.820503 | 26.554304                       | 200        |
| train | gan_total      | 0.000000  | 1.476028  | 1.000000   | 0.000000  | NA                              | 200        |
| train | shadow_removal | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 200        |
| train | shadow_mask    | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 200        |
| train | cloud_l1       | 0.066579  | 0.020608  | 197.000000 | 0.020216  | 69.047761                       | 200        |
| train | cloud_kl       | 0.361290  | 0.359658  | 187.000000 | 0.358479  | 0.451644                        | 200        |
| train | cloud_adv      | 0.000000  | 0.466012  | 1.000000   | 0.000000  | NA                              | 200        |
| train | disc_total     | 1.310474  | 1.106125  | 199.000000 | 1.104629  | 15.593485                       | 181        |
| val   | total          | 16.798783 | 15.535447 | 20.000000  | 14.877758 | 7.520402                        | 200        |
| val   | recon_total    | 16.798783 | 14.905447 | 109.000000 | 14.505072 | 11.270674                       | 200        |
| val   | gan_total      | 0.000000  | 0.630000  | 1.000000   | 0.000000  | NA                              | 200        |
| val   | shadow_removal | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 200        |
| val   | shadow_mask    | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 200        |
| val   | cloud_l1       | 0.057083  | 0.036408  | 158.000000 | 0.032162  | 36.218576                       | 200        |
| val   | cloud_kl       | 0.369684  | 0.375488  | 7.000000   | 0.368290  | -1.569949                       | 200        |
| val   | cloud_adv      | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 200        |

诊断：
- Val recon_total 从 epoch 1 到当前下降 `11.27%`，早期优化方向是正常的。
- 当前 val recon_total 高于 train recon_total，需要继续观察 gap 是否扩大；若持续扩大才考虑过拟合。

## 7. Loss 贡献比例
| split | term             | weighted_contribution | share_of_recon_total | share_of_total |
| ----- | ---------------- | --------------------- | -------------------- | -------------- |
| train | cloud_kl         | 10.789739             | 0.839634             | 0.753129       |
| train | cloud_l1         | 2.060780              | 0.160366             | 0.143843       |
| train | perceptual       | 0.508793              | 0.039593             | 0.035514       |
| train | feature_matching | 0.501222              | 0.039004             | 0.034986       |
| train | cloud_adv        | 0.466012              | NA                   | 0.032528       |
| val   | cloud_kl         | 11.264635             | 0.755740             | 0.725092       |
| val   | cloud_l1         | 3.640812              | 0.244260             | 0.234355       |
| val   | perceptual       | 0.630000              | 0.042266             | 0.040552       |

## 8. 训练速度
| split | epochs | seconds_mean | seconds_median | seconds_p95 | sec_per_batch_mean | sec_per_batch_median | sec_per_batch_p95 | latest_sec_per_batch |
| ----- | ------ | ------------ | -------------- | ----------- | ------------------ | -------------------- | ----------------- | -------------------- |
| train | 200    | 169.533      | 170.503        | 171.293     | 0.698              | 0.702                | 0.705             | 0.704                |
| val   | 200    | 8.055        | 8.000          | 9.000       | 0.252              | 0.250                | 0.281             | 0.250                |
- 平均训练耗时: `2.83` min/epoch, `0.698` sec/batch
- 完整逐 epoch 时间表见 `timing_summary.csv`。

## 9. 可视化结果评价
- 评价对象是保存的 RGB PNG，可判断展示质量和视觉趋势；这不是完整 13 波段 checkpoint 定量评测。
- 可视化 epoch 覆盖: `200` 个；low/medium/high 三档完整覆盖: `200` 个。
| epoch | file_count | buckets_present | missing_buckets | complete | total_size_mb |
| ----- | ---------- | --------------- | --------------- | -------- | ------------- |
| 189   | 3          | high,low,medium |                 | 1        | 4.889859      |
| 190   | 3          | high,low,medium |                 | 1        | 4.887316      |
| 191   | 3          | high,low,medium |                 | 1        | 4.895805      |
| 192   | 3          | high,low,medium |                 | 1        | 4.865683      |
| 193   | 3          | high,low,medium |                 | 1        | 4.863634      |
| 194   | 3          | high,low,medium |                 | 1        | 4.892408      |
| 195   | 3          | high,low,medium |                 | 1        | 4.867346      |
| 196   | 3          | high,low,medium |                 | 1        | 4.897144      |
| 197   | 3          | high,low,medium |                 | 1        | 4.893250      |
| 198   | 3          | high,low,medium |                 | 1        | 4.888900      |
| 199   | 3          | high,low,medium |                 | 1        | 4.899522      |
| 200   | 3          | high,low,medium |                 | 1        | 4.887810      |

| bucket | mae      | psnr      | ssim     | mae_improvement_vs_cloudy_pct |
| ------ | -------- | --------- | -------- | ----------------------------- |
| high   | 0.175639 | 13.016064 | 0.310354 | 23.152990                     |
| low    | 0.059742 | 21.052779 | 0.777060 | 7.380559                      |
| medium | 0.120723 | 16.055389 | 0.378944 | 44.611343                     |

Latest visualization panel quality:
| bucket | panel            | brightness_mean | contrast_p95_p05 | sharpness_laplacian_var | entropy  |
| ------ | ---------------- | --------------- | ---------------- | ----------------------- | -------- |
| high   | cloudy_s2        | 0.387806        | 0.677643         | 88.061123               | 7.662962 |
| high   | target           | 0.460183        | 0.597282         | 627.439976              | 7.526941 |
| high   | stage1_output    | 0.515023        | 0.475420         | 535.271419              | 7.209456 |
| high   | shadow_candidate | 0.596871        | 0.000000         | 303.568817              | 0.079811 |
| high   | cloud_candidate  | 0.023857        | 0.000000         | 58.158875               | 0.006243 |
| low    | cloudy_s2        | 0.514656        | 0.386036         | 420.029475              | 6.886140 |
| low    | target           | 0.539050        | 0.429514         | 478.278264              | 7.104777 |
| low    | stage1_output    | 0.546653        | 0.431621         | 551.770145              | 7.050198 |
| low    | shadow_candidate | 0.054675        | 0.579047         | 991.929231              | 0.302272 |
| low    | cloud_candidate  | 0.032824        | 0.000000         | 442.007443              | 0.108002 |
| medium | cloudy_s2        | 0.267848        | 0.627420         | 110.488016              | 7.272139 |
| medium | target           | 0.389803        | 0.508180         | 784.407707              | 7.250542 |
| medium | stage1_output    | 0.421334        | 0.465110         | 774.340636              | 7.132021 |
| medium | shadow_candidate | 0.387672        | 0.579047         | 2966.445356             | 0.951539 |
| medium | cloud_candidate  | 0.245201        | 0.650804         | 3081.727981             | 0.925403 |

可视化诊断：
- 建议同时查看 `latest_visual_contact_sheet.png`，用肉眼确认云区、阴影区和 hard mask 边界是否存在接缝或过暗问题。

## 10. Checkpoint 完整性
| file                                     | kind | epoch | metric      | value     | size_mb    | mtime               |
| ---------------------------------------- | ---- | ----- | ----------- | --------- | ---------- | ------------------- |
| best_epoch_0109_recon_total_14.505072.pt | best | 109   | recon_total | 14.505072 | 173.130192 | 2026-07-03T21:56:35 |
| last.pt                                  | last | 200   | last        | NA        | 172.935202 | 2026-07-04T02:28:47 |

## 11. 生成文件
- `anomalies.csv`
- `checkpoint_inventory.csv`
- `checkpoint_timeline.png`
- `gan_feature_matching_transition.csv`
- `gan_transition_impact.png`
- `generalization_gap.png`
- `generalization_gaps.csv`
- `latest_visual_contact_sheet.png`
- `loss_contributions.csv`
- `loss_curves.png`
- `loss_schedule_status.csv`
- `metric_summary.csv`
- `scheduled_gan_curves.png`
- `softshadow_diagnostics.csv`
- `source_config.resolved.json`
- `source_latest.json`
- `source_train.log`
- `timing.png`
- `timing_summary.csv`
- `training_log_numeric.csv`
- `visual_brightness.png`
- `visual_candidate_comparison.csv`
- `visual_panel_quality.csv`
- `visual_proxy_curves.png`
- `visualization_inventory.csv`
- `weight_schedule.png`

## 12. 后续建议
- `cloud_adv` 和判别器已经启动，应重点检查 GAN 开启前后 cloud 区视觉质量与重建指标是否劣化。
- `cloud_kl` 与 `shadow_penumbra` 已经启动，应结合曲线判断它们是否只增加 loss 尺度而没有改善可视化质量。
- 若要严谨报告模型性能，应再运行 checkpoint 级别评测脚本，计算 global/clear/shadow/cloud 的 MAE、RMSE、PSNR，并补充 RGB SSIM/SAM。