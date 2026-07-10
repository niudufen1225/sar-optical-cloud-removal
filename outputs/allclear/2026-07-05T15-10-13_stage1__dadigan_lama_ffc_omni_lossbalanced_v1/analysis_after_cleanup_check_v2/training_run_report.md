# Training Run Report: `2026-07-05T15-10-13_stage1__dadigan_lama_ffc_omni_lossbalanced_v1`

## 1. Run 状态
- Run dir: `/home/students/sushaoqi/CR/main/outputs/allclear/2026-07-05T15-10-13_stage1__dadigan_lama_ffc_omni_lossbalanced_v1`
- 已记录 epoch: `50` / 配置 epoch: `200` (25.00%)
- Train batches: `327` | Val batches: `44`
- 报告主指标: `recon_total`
- Best val recon_total: `4.4514` at epoch `14`
- Train recon_total improvement: `20.08%` | Val recon_total improvement: `11.62%`
- Final generalization gap (val - train recon_total): `-0.4734`
- 结论：当前日志是早期训练片段，只能评价 warm-up 初期趋势，不能评价完整收敛、GAN 稳定性或最终泛化。

## 2. 核心异常与风险
| severity | topic          | message                                                                                              |
| -------- | -------------- | ---------------------------------------------------------------------------------------------------- |
| warn     | loss_semantics | `gan_total` 在 adversarial/FM 权重为 0 时仍非零；这通常表示 perceptual/HRF 等项被归入 gan_total 命名桶，报告时不要把它解释成纯 GAN 损失。 |
| info     | progress       | 当前日志到 epoch 50/200，结论只覆盖未完成训练。                                                                       |
| info     | visual_proxy   | 最新可视化中这些 bucket 的 Stage1 RGB proxy MAE 未优于 cloudy 输入：high                                            |

## 3. Loss 调度状态
- 已参与训练的主要项: `cloud_l1, cloud_kl, cloud_adv, feature_matching, perceptual, disc_total`

| metric           | target_weight | current_weight | start_epoch | ramp_epochs | active_epoch_count | status   |
| ---------------- | ------------- | -------------- | ----------- | ----------- | ------------------ | -------- |
| final_l1         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| grad             | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| shadow_removal   | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| shadow_mask      | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| shadow_penumbra  | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_l1         | 100.000000    | 100.000000     | 1           | 0           | 50                 | active   |
| cloud_kl         | 4.000000      | 4.000000       | 5           | 25          | 46                 | active   |
| cloud_adv        | 2.000000      | 2.000000       | 10          | 40          | 41                 | active   |
| feature_matching | 25.000000     | 25.000000      | 10          | 40          | 41                 | active   |
| perceptual       | 15.000000     | 15.000000      | 5           | 25          | 46                 | active   |
| disc_total       | 2.000000      | 2.000000       | 10          | 40          | 41                 | active   |

## 4. GAN/FM 启动冲击
- 这里比较 adversarial 或 feature matching 权重首次大于 0 之前的 train 均值，与启动后前 3 个 epoch 的 train 均值。
| transition_epoch | metric           | pre_mean | early_mean | delta     | delta_pct  |
| ---------------- | ---------------- | -------- | ---------- | --------- | ---------- |
| 10               | gan_total        | 0.156672 | 0.859601   | 0.702929  | 448.661714 |
| 10               | perceptual       | 0.089766 | 0.146010   | 0.056244  | 62.655747  |
| 10               | cloud_l1         | 0.052475 | 0.046547   | -0.005929 | -11.298294 |
| 10               | total            | 5.506562 | 5.943535   | 0.436973  | 7.935488   |
| 10               | recon_total      | 5.349890 | 5.083934   | -0.265956 | -4.971245  |
| 10               | cloud_adv        | 0.000000 | 0.493772   | 0.493772  | NA         |
| 10               | feature_matching | 0.000000 | 0.156452   | 0.156452  | NA         |
| 10               | shadow_removal   | 0.000000 | 0.000000   | 0.000000  | NA         |
| 10               | shadow_mask      | 0.000000 | 0.000000   | 0.000000  | NA         |
| 10               | shadow_penumbra  | 0.000000 | 0.000000   | 0.000000  | NA         |

## 6. 指标趋势摘要
| split | metric         | first    | last      | best_epoch | best     | improvement_pct_if_lower_better | num_points |
| ----- | -------------- | -------- | --------- | ---------- | -------- | ------------------------------- | ---------- |
| train | total          | 6.737670 | 10.429269 | 4.000000   | 5.232428 | -54.790443                      | 50         |
| train | recon_total    | 6.737670 | 5.384970  | 9.000000   | 4.980936 | 20.076673                       | 50         |
| train | gan_total      | 0.000000 | 5.044299  | 1.000000   | 0.000000 | NA                              | 50         |
| train | shadow_removal | 0.000000 | 0.000000  | 1.000000   | 0.000000 | NA                              | 50         |
| train | shadow_mask    | 0.000000 | 0.000000  | 1.000000   | 0.000000 | NA                              | 50         |
| train | cloud_l1       | 0.067377 | 0.038648  | 49.000000  | 0.038215 | 42.638373                       | 50         |
| train | cloud_kl       | 0.000000 | 0.380033  | 1.000000   | 0.000000 | NA                              | 50         |
| train | cloud_adv      | 0.000000 | 0.506089  | 1.000000   | 0.000000 | NA                              | 50         |
| train | disc_total     | 1.326376 | 1.192861  | 20.000000  | 1.174382 | 10.066150                       | 41         |
| val   | total          | 5.557401 | 6.992629  | 5.000000   | 4.595729 | -25.825511                      | 50         |
| val   | recon_total    | 5.557401 | 4.911535  | 14.000000  | 4.451390 | 11.621739                       | 50         |
| val   | gan_total      | 0.000000 | 2.081094  | 1.000000   | 0.000000 | NA                              | 50         |
| val   | shadow_removal | 0.000000 | 0.000000  | 1.000000   | 0.000000 | NA                              | 50         |
| val   | shadow_mask    | 0.000000 | 0.000000  | 1.000000   | 0.000000 | NA                              | 50         |
| val   | cloud_l1       | 0.055574 | 0.034937  | 39.000000  | 0.032836 | 37.134631                       | 50         |
| val   | cloud_kl       | 0.000000 | 0.354463  | 1.000000   | 0.000000 | NA                              | 50         |
| val   | cloud_adv      | 0.000000 | 0.000000  | 1.000000   | 0.000000 | NA                              | 50         |

诊断：
- Val recon_total 从 epoch 1 到当前下降 `11.62%`，早期优化方向是正常的。
- 当前 val recon_total 低于 train recon_total，这通常不是过拟合，可能来自训练/验证样本云量分布差异、训练 batch 的难度更高或训练态正则影响。

## 7. Loss 贡献比例
| split | term             | weighted_contribution | share_of_recon_total | share_of_total |
| ----- | ---------------- | --------------------- | -------------------- | -------------- |
| train | cloud_l1         | 3.864837              | 0.717708             | 0.370576       |
| train | feature_matching | 2.175588              | 0.404011             | 0.208604       |
| train | perceptual       | 1.856532              | 0.344762             | 0.178012       |
| train | cloud_kl         | 1.520133              | 0.282292             | 0.145756       |
| train | cloud_adv        | 1.012179              | NA                   | 0.097052       |
| val   | cloud_l1         | 3.493681              | 0.711322             | 0.499623       |
| val   | perceptual       | 2.081094              | 0.423716             | 0.297613       |
| val   | cloud_kl         | 1.417854              | 0.288678             | 0.202764       |

## 8. 训练速度
| split | epochs | seconds_mean | seconds_median | seconds_p95 | sec_per_batch_mean | sec_per_batch_median | sec_per_batch_p95 | latest_sec_per_batch |
| ----- | ------ | ------------ | -------------- | ----------- | ------------------ | -------------------- | ----------------- | -------------------- |
| train | 50     | 230.074      | 231.367        | 232.712     | 0.704              | 0.708                | 0.712             | 0.719                |
| val   | 50     | 13.540       | 14.000         | 14.000      | 0.308              | 0.318                | 0.318             | 0.318                |
- 平均训练耗时: `3.83` min/epoch, `0.704` sec/batch
- 完整逐 epoch 时间表见 `timing_summary.csv`。

## 9. 可视化结果评价
- 评价对象是保存的 RGB PNG，可判断展示质量和视觉趋势；这不是完整 13 波段 checkpoint 定量评测。
- 可视化 epoch 覆盖: `50` 个；配置要求的 `medium/high/heavy` 完整覆盖: `50` 个。
| epoch | file_count | buckets_present   | missing_buckets | complete | total_size_mb |
| ----- | ---------- | ----------------- | --------------- | -------- | ------------- |
| 39    | 3          | heavy,high,medium |                 | 1        | 8.152805      |
| 40    | 3          | heavy,high,medium |                 | 1        | 8.220374      |
| 41    | 3          | heavy,high,medium |                 | 1        | 8.137961      |
| 42    | 3          | heavy,high,medium |                 | 1        | 8.151885      |
| 43    | 3          | heavy,high,medium |                 | 1        | 8.166419      |
| 44    | 3          | heavy,high,medium |                 | 1        | 8.147036      |
| 45    | 3          | heavy,high,medium |                 | 1        | 8.129724      |
| 46    | 3          | heavy,high,medium |                 | 1        | 8.100514      |
| 47    | 3          | heavy,high,medium |                 | 1        | 8.214267      |
| 48    | 3          | heavy,high,medium |                 | 1        | 8.135282      |
| 49    | 3          | heavy,high,medium |                 | 1        | 8.137158      |
| 50    | 3          | heavy,high,medium |                 | 1        | 8.134422      |

| bucket | mae      | psnr      | ssim     | mae_improvement_vs_cloudy_pct |
| ------ | -------- | --------- | -------- | ----------------------------- |
| heavy  | 0.187888 | 12.506552 | 0.145314 | 36.329852                     |
| high   | 0.247398 | 10.334574 | 0.235048 | -5.367811                     |
| medium | 0.095140 | 17.406237 | 0.591594 | 25.771410                     |

Latest visualization panel quality:
| bucket | panel         | brightness_mean | contrast_p95_p05 | sharpness_laplacian_var | entropy  |
| ------ | ------------- | --------------- | ---------------- | ----------------------- | -------- |
| heavy  | cloudy_s2     | 0.460028        | 0.689894         | 163.413348              | 7.619457 |
| heavy  | target        | 0.337155        | 0.545245         | 2687.249327             | 7.368997 |
| heavy  | stage1_output | 0.424014        | 0.471133         | 1268.179593             | 7.243544 |
| high   | cloudy_s2     | 0.376349        | 0.749079         | 357.006872              | 7.745106 |
| high   | target        | 0.401982        | 0.752946         | 1531.945642             | 7.768469 |
| high   | stage1_output | 0.348968        | 0.568967         | 1026.591743             | 7.392151 |
| medium | cloudy_s2     | 0.376848        | 0.856267         | 1019.736469             | 7.727921 |
| medium | target        | 0.412458        | 0.848220         | 2216.972317             | 7.797006 |
| medium | stage1_output | 0.423760        | 0.789930         | 1913.793082             | 7.812174 |

可视化诊断：
- 至少一个云量 bucket 中 Stage1 Output 的 RGB proxy MAE 没有优于 Cloudy 输入；需要用后续 epoch 和真实 val/test 指标确认。
- 建议同时查看 `latest_visual_contact_sheet.png`，用肉眼确认云区、阴影区和 hard mask 边界是否存在接缝或过暗问题。

## 10. Checkpoint 完整性
| file                                    | kind | epoch | metric      | value    | size_mb    | mtime               |
| --------------------------------------- | ---- | ----- | ----------- | -------- | ---------- | ------------------- |
| best_epoch_0014_recon_total_4.451390.pt | best | 14    | recon_total | 4.451390 | 173.124899 | 2026-07-05T16:06:19 |
| last.pt                                 | last | 50    | last        | NA       | 172.935202 | 2026-07-05T18:33:40 |

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
- `cloud_kl` 已按当前配置启动后，应结合曲线判断它是否只增加 loss 尺度而没有改善光谱质量。
- 若要严谨报告模型性能，应再运行 checkpoint 级别评测脚本，计算 global/clear/shadow/cloud 的 MAE、RMSE、PSNR，并补充 RGB SSIM/SAM。