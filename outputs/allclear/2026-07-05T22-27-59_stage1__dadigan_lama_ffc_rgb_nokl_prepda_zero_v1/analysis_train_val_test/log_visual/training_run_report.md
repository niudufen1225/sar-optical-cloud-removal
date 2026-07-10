# Training Run Report: `2026-07-05T22-27-59_stage1__dadigan_lama_ffc_rgb_nokl_prepda_zero_v1`

## 1. Run 状态
- Run dir: `/home/students/sushaoqi/CR/main/outputs/allclear/2026-07-05T22-27-59_stage1__dadigan_lama_ffc_rgb_nokl_prepda_zero_v1`
- 已记录 epoch: `160` / 配置 epoch: `200` (80.00%)
- Train batches: `327` | Val batches: `44`
- 报告主指标: `recon_total`
- Best val recon_total: `3.8988` at epoch `33`
- Train recon_total improvement: `56.66%` | Val recon_total improvement: `-4.63%`
- Final generalization gap (val - train recon_total): `1.8344`
- 结论：当前日志已接近完整训练，但还不是最终完成状态；可评价主要训练趋势，最终 checkpoint 和末尾收敛仍需等训练结束后复核。

## 2. 核心异常与风险
| severity | topic         | message                                                             |
| -------- | ------------- | ------------------------------------------------------------------- |
| info     | discriminator | 最近若干 epoch 的 real/fake logit 非常接近；判别器区分度弱或处于均衡状态，需要结合视觉和 loss 振荡判断。 |
| info     | progress      | 当前日志到 epoch 160/200，结论只覆盖未完成训练。                                     |

## 3. Loss 调度状态
- 已参与训练的主要项: `cloud_l1, cloud_adv, feature_matching, perceptual, disc_total`

| metric           | target_weight | current_weight | start_epoch | ramp_epochs | active_epoch_count | status   |
| ---------------- | ------------- | -------------- | ----------- | ----------- | ------------------ | -------- |
| cloud_l1         | 120.000000    | 120.000000     | 1           | 0           | 160                | active   |
| cloud_kl         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_adv        | 0.250000      | 0.250000       | 55          | 90          | 106                | active   |
| feature_matching | 8.000000      | 8.000000       | 30          | 70          | 131                | active   |
| perceptual       | 6.000000      | 6.000000       | 8           | 40          | 153                | active   |
| disc_total       | 0.250000      | 0.250000       | 55          | 90          | 106                | active   |

## 4. GAN/FM 启动冲击
- 这里比较 adversarial 或 feature matching 权重首次大于 0 之前的 train 均值，与启动后前 3 个 epoch 的 train 均值。
| transition_epoch | metric           | pre_mean | early_mean | delta     | delta_pct  |
| ---------------- | ---------------- | -------- | ---------- | --------- | ---------- |
| 30               | perceptual_total | 0.189742 | 0.492314   | 0.302572  | 159.464585 |
| 30               | perceptual       | 0.113888 | 0.136758   | 0.022869  | 20.080449  |
| 30               | cloud_l1         | 0.040157 | 0.037014   | -0.003144 | -7.828198  |
| 30               | pixel_total      | 4.818900 | 4.441667   | -0.377233 | -7.828198  |
| 30               | recon_total      | 5.008642 | 4.933980   | -0.074661 | -1.490654  |
| 30               | total            | 5.008642 | 4.933980   | -0.074661 | -1.490654  |
| 30               | gan_total        | 0.000000 | 0.000000   | 0.000000  | NA         |
| 30               | cloud_adv        | 0.000000 | 0.000000   | 0.000000  | NA         |
| 30               | feature_matching | 0.000000 | 0.000000   | 0.000000  | NA         |

## 6. 指标趋势摘要
| split | metric      | first    | last     | best_epoch | best     | improvement_pct_if_lower_better | num_points |
| ----- | ----------- | -------- | -------- | ---------- | -------- | ------------------------------- | ---------- |
| train | total       | 6.379978 | 3.570672 | 160.000000 | 3.570672 | 44.033156                       | 160        |
| train | recon_total | 6.379978 | 2.765344 | 160.000000 | 2.765344 | 56.655902                       | 160        |
| train | gan_total   | 0.000000 | 0.805328 | 1.000000   | 0.000000 | NA                              | 160        |
| train | cloud_l1    | 0.053166 | 0.017307 | 160.000000 | 0.017307 | 67.446809                       | 160        |
| train | cloud_kl    | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 160        |
| train | cloud_adv   | 0.000000 | 0.489836 | 1.000000   | 0.000000 | NA                              | 160        |
| train | disc_total  | 1.393302 | 1.200929 | 157.000000 | 1.197587 | 13.806977                       | 106        |
| val   | total       | 4.396114 | 4.599757 | 33.000000  | 3.898783 | -4.632341                       | 160        |
| val   | recon_total | 4.396114 | 4.599757 | 33.000000  | 3.898783 | -4.632341                       | 160        |
| val   | gan_total   | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 160        |
| val   | cloud_l1    | 0.036634 | 0.031443 | 151.000000 | 0.027641 | 14.169815                       | 160        |
| val   | cloud_kl    | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 160        |
| val   | cloud_adv   | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 160        |

诊断：
- 当前 val recon_total 高于 train recon_total，需要继续观察 gap 是否扩大；若持续扩大才考虑过拟合。

## 7. Loss 贡献比例
| split | term             | weighted_contribution | share_of_recon_total | share_of_total |
| ----- | ---------------- | --------------------- | -------------------- | -------------- |
| train | cloud_l1         | 2.076887              | 0.751041             | 0.581651       |
| train | perceptual       | 0.688457              | 0.248959             | 0.192809       |
| train | feature_matching | 0.682869              | 0.246938             | 0.191244       |
| train | cloud_adv        | 0.122459              | NA                   | 0.034296       |
| val   | cloud_l1         | 3.773193              | 0.820303             | 0.820303       |
| val   | perceptual       | 0.826564              | 0.179697             | 0.179697       |

## 8. 训练速度
| split | epochs | seconds_mean | seconds_median | seconds_p95 | sec_per_batch_mean | sec_per_batch_median | sec_per_batch_p95 | latest_sec_per_batch |
| ----- | ------ | ------------ | -------------- | ----------- | ------------------ | -------------------- | ----------------- | -------------------- |
| train | 160    | 288.180      | 289.041        | 290.535     | 0.881              | 0.884                | 0.888             | 0.878                |
| val   | 160    | 14.988       | 15.000         | 15.000      | 0.341              | 0.341                | 0.341             | 0.341                |
- 平均训练耗时: `4.80` min/epoch, `0.881` sec/batch
- 完整逐 epoch 时间表见 `timing_summary.csv`。

## 9. 可视化结果评价
- 评价对象是保存的 RGB PNG，可判断展示质量和视觉趋势；这不是完整 13 波段 checkpoint 定量评测。
- 可视化 epoch 覆盖: `160` 个；配置要求的 `medium/high/heavy` 完整覆盖: `160` 个。
| epoch | file_count | buckets_present   | missing_buckets | complete | total_size_mb |
| ----- | ---------- | ----------------- | --------------- | -------- | ------------- |
| 149   | 3          | heavy,high,medium |                 | 1        | 8.177419      |
| 150   | 3          | heavy,high,medium |                 | 1        | 8.162697      |
| 151   | 3          | heavy,high,medium |                 | 1        | 8.180367      |
| 152   | 3          | heavy,high,medium |                 | 1        | 8.177558      |
| 153   | 3          | heavy,high,medium |                 | 1        | 8.187490      |
| 154   | 3          | heavy,high,medium |                 | 1        | 8.161819      |
| 155   | 3          | heavy,high,medium |                 | 1        | 8.176039      |
| 156   | 3          | heavy,high,medium |                 | 1        | 8.175033      |
| 157   | 3          | heavy,high,medium |                 | 1        | 8.158462      |
| 158   | 3          | heavy,high,medium |                 | 1        | 8.179729      |
| 159   | 3          | heavy,high,medium |                 | 1        | 8.174408      |
| 160   | 3          | heavy,high,medium |                 | 1        | 8.168137      |


可视化诊断：
- 建议同时查看 `latest_visual_contact_sheet.png`，用肉眼确认云区、阴影区和 hard mask 边界是否存在接缝或过暗问题。

## 10. Checkpoint 完整性
| file                                    | kind | epoch | metric      | value    | size_mb    | mtime               |
| --------------------------------------- | ---- | ----- | ----------- | -------- | ---------- | ------------------- |
| best_epoch_0151_pixel_total_3.316924.pt | best | 151   | pixel_total | 3.316924 | 207.980337 | 2026-07-06T11:12:17 |
| last.pt                                 | last | 160   | last        | NA       | 207.557425 | 2026-07-06T11:57:40 |

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
- `visual_candidate_comparison.csv`
- `visual_panel_quality.csv`
- `visualization_inventory.csv`
- `weight_schedule.png`

## 12. 后续建议
- `cloud_adv` 和判别器已经启动，应重点检查 GAN 开启前后 cloud 区视觉质量与重建指标是否劣化。
- `cloud_kl` 已按当前配置启动后，应结合曲线判断它是否只增加 loss 尺度而没有改善光谱质量。
- 若要严谨报告模型性能，应再运行 checkpoint 级别评测脚本，计算 global/clear/shadow/cloud 的 MAE、RMSE、PSNR，并补充 RGB SSIM/SAM。