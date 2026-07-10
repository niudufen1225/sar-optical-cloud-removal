# Training Run Report: `2026-07-06T15-01-55_stage1__dadigan_lama_ffc_lamanative_sigmoid_v1`

## 1. Run 状态
- Run dir: `/home/students/sushaoqi/CR/main/outputs/allclear/2026-07-06T15-01-55_stage1__dadigan_lama_ffc_lamanative_sigmoid_v1`
- 已记录 epoch: `169` / 配置 epoch: `200` (84.50%)
- Train batches: `327` | Val batches: `44`
- 报告主指标: `recon_total`
- Best val recon_total: `6.2662` at epoch `149`
- Train recon_total improvement: `43.82%` | Val recon_total improvement: `44.32%`
- Final generalization gap (val - train recon_total): `1.0895`
- 结论：当前日志已接近完整训练，但还不是最终完成状态；可评价主要训练趋势，最终 checkpoint 和末尾收敛仍需等训练结束后复核。

## 2. 核心异常与风险
| severity | topic         | message                                                             |
| -------- | ------------- | ------------------------------------------------------------------- |
| info     | discriminator | 最近若干 epoch 的 real/fake logit 非常接近；判别器区分度弱或处于均衡状态，需要结合视觉和 loss 振荡判断。 |
| info     | progress      | 当前日志到 epoch 169/200，结论只覆盖未完成训练。                                     |

## 3. Loss 调度状态
- 已参与训练的主要项: `cloud_adv, feature_matching, perceptual, disc_total`

| metric           | target_weight | current_weight | start_epoch | ramp_epochs | active_epoch_count | status   |
| ---------------- | ------------- | -------------- | ----------- | ----------- | ------------------ | -------- |
| cloud_l1         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_kl         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_adv        | 10.000000     | 10.000000      | 1           | 0           | 169                | active   |
| feature_matching | 100.000000    | 100.000000     | 1           | 0           | 169                | active   |
| perceptual       | 30.000000     | 30.000000      | 1           | 0           | 169                | active   |
| disc_total       | 10.000000     | 10.000000      | 1           | 0           | 169                | active   |

## 6. 指标趋势摘要
| split | metric      | first     | last      | best_epoch | best      | improvement_pct_if_lower_better | num_points |
| ----- | ----------- | --------- | --------- | ---------- | --------- | ------------------------------- | ---------- |
| train | total       | 40.829301 | 17.131479 | 96.000000  | 16.558157 | 58.041214                       | 169        |
| train | recon_total | 9.382856  | 5.271246  | 169.000000 | 5.271246  | 43.820454                       | 169        |
| train | gan_total   | 31.446445 | 11.860233 | 86.000000  | 10.851463 | 62.284344                       | 169        |
| train | cloud_l1    | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 169        |
| train | cloud_kl    | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 169        |
| train | cloud_adv   | 0.448110  | 0.412243  | 69.000000  | 0.403194  | 8.004097                        | 169        |
| train | disc_total  | 1.241300  | 1.169128  | 165.000000 | 1.167087  | 5.814225                        | 169        |
| val   | total       | 11.423475 | 6.360757  | 149.000000 | 6.266220  | 44.318551                       | 169        |
| val   | recon_total | 11.423475 | 6.360757  | 149.000000 | 6.266220  | 44.318551                       | 169        |
| val   | gan_total   | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 169        |
| val   | cloud_l1    | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 169        |
| val   | cloud_kl    | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 169        |
| val   | cloud_adv   | 0.000000  | 0.000000  | 1.000000   | 0.000000  | NA                              | 169        |

诊断：
- Val recon_total 从 epoch 1 到当前下降 `44.32%`，早期优化方向是正常的。
- 当前 val recon_total 高于 train recon_total，需要继续观察 gap 是否扩大；若持续扩大才考虑过拟合。

## 7. Loss 贡献比例
| split | term             | weighted_contribution | share_of_recon_total | share_of_total |
| ----- | ---------------- | --------------------- | -------------------- | -------------- |
| train | feature_matching | 7.737802              | 1.467927             | 0.451672       |
| train | perceptual       | 5.181076              | 0.982894             | 0.302430       |
| train | cloud_adv        | 4.122431              | NA                   | 0.240635       |
| val   | perceptual       | 6.186796              | 0.972651             | 0.972651       |

## 8. 训练速度
| split | epochs | seconds_mean | seconds_median | seconds_p95 | sec_per_batch_mean | sec_per_batch_median | sec_per_batch_p95 | latest_sec_per_batch |
| ----- | ------ | ------------ | -------------- | ----------- | ------------------ | -------------------- | ----------------- | -------------------- |
| train | 169    | 404.845      | 412.980        | 465.647     | 1.238              | 1.263                | 1.424             | 1.410                |
| val   | 169    | 18.438       | 19.000         | 21.000      | 0.419              | 0.432                | 0.477             | 0.409                |
- 平均训练耗时: `6.75` min/epoch, `1.238` sec/batch
- 完整逐 epoch 时间表见 `timing_summary.csv`。

## 9. 可视化结果评价
- 评价对象是保存的 RGB PNG，可判断展示质量和视觉趋势；这不是完整 13 波段 checkpoint 定量评测。
- 可视化 epoch 覆盖: `169` 个；配置要求的 `medium/high/heavy` 完整覆盖: `169` 个。
| epoch | file_count | buckets_present   | missing_buckets | complete | total_size_mb |
| ----- | ---------- | ----------------- | --------------- | -------- | ------------- |
| 158   | 3          | heavy,high,medium |                 | 1        | 7.266875      |
| 159   | 3          | heavy,high,medium |                 | 1        | 7.286729      |
| 160   | 3          | heavy,high,medium |                 | 1        | 7.272752      |
| 161   | 3          | heavy,high,medium |                 | 1        | 7.236424      |
| 162   | 3          | heavy,high,medium |                 | 1        | 7.259645      |
| 163   | 3          | heavy,high,medium |                 | 1        | 7.285583      |
| 164   | 3          | heavy,high,medium |                 | 1        | 7.255689      |
| 165   | 3          | heavy,high,medium |                 | 1        | 7.292336      |
| 166   | 3          | heavy,high,medium |                 | 1        | 7.282742      |
| 167   | 3          | heavy,high,medium |                 | 1        | 7.250613      |
| 168   | 3          | heavy,high,medium |                 | 1        | 7.244674      |
| 169   | 3          | heavy,high,medium |                 | 1        | 7.290865      |


可视化诊断：
- 建议同时查看 `latest_visual_contact_sheet.png`，用肉眼确认云区、阴影区和 hard mask 边界是否存在接缝或过暗问题。

## 10. Checkpoint 完整性
| file                                    | kind | epoch | metric      | value    | size_mb    | mtime               |
| --------------------------------------- | ---- | ----- | ----------- | -------- | ---------- | ------------------- |
| best_epoch_0068_recon_total_6.400761.pt | best | 68    | recon_total | 6.400761 | 222.023230 | 2026-07-06T21:54:58 |
| best_epoch_0149_recon_total_6.266220.pt | best | 149   | recon_total | 6.266220 | 222.023230 | 2026-07-07T08:17:52 |
| last.pt                                 | last | 169   | last        | NA       | 221.549963 | 2026-07-07T10:58:00 |

## 11. 生成文件
- `anomalies.csv`
- `checkpoint_inventory.csv`
- `checkpoint_timeline.png`
- `gan_feature_matching_transition.csv`
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