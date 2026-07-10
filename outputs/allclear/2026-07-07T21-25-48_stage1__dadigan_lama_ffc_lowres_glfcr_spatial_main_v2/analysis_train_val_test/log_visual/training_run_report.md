# Training Run Report: `2026-07-07T21-25-48_stage1__dadigan_lama_ffc_lowres_glfcr_spatial_main_v2`

## 1. Run 状态
- Run dir: `/home/students/sushaoqi/CR/main/outputs/allclear/2026-07-07T21-25-48_stage1__dadigan_lama_ffc_lowres_glfcr_spatial_main_v2`
- 已记录 epoch: `115` / 配置 epoch: `120` (95.83%)
- Train batches: `654` | Val batches: `58`
- 报告主指标: `recon_total`
- Best val recon_total: `2.1792` at epoch `2`
- Train recon_total improvement: `-3.66%` | Val recon_total improvement: `-82.72%`
- Final generalization gap (val - train recon_total): `1.5981`
- 结论：当前日志已接近完整训练，但还不是最终完成状态；可评价主要训练趋势，最终 checkpoint 和末尾收敛仍需等训练结束后复核。

## 2. 核心异常与风险
| severity | topic          | message                                                             |
| -------- | -------------- | ------------------------------------------------------------------- |
| warn     | gan_transition | GAN/FM 启动后早期 train total 均值上升 58.3%，需要检查 loss 权重尺度和视觉质量。            |
| warn     | gan_transition | GAN/FM 启动后 recon_total 均值上升 38.8%，可能出现对抗项干扰重建项。                     |
| info     | discriminator  | 最近若干 epoch 的 real/fake logit 非常接近；判别器区分度弱或处于均衡状态，需要结合视觉和 loss 振荡判断。 |
| info     | progress       | 当前日志到 epoch 115/120，结论只覆盖未完成训练。                                     |

## 3. Loss 调度状态
- 已参与训练的主要项: `cloud_adv, feature_matching, perceptual, disc_total`

| metric           | target_weight | current_weight | start_epoch | ramp_epochs | active_epoch_count | status   |
| ---------------- | ------------- | -------------- | ----------- | ----------- | ------------------ | -------- |
| cloud_l1         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_kl         | 0.000000      | 0.000000       | 1           | 0           | 0                  | disabled |
| cloud_adv        | 1.500000      | 1.500000       | 8           | 12          | 108                | active   |
| feature_matching | 15.000000     | 15.000000      | 8           | 12          | 108                | active   |
| perceptual       | 12.000000     | 12.000000      | 3           | 8           | 113                | active   |
| disc_total       | 1.500000      | 1.500000       | 8           | 12          | 108                | active   |

## 4. GAN/FM 启动冲击
- 这里比较 adversarial 或 feature matching 权重首次大于 0 之前的 train 均值，与启动后前 3 个 epoch 的 train 均值。
| transition_epoch | metric           | pre_mean | early_mean | delta     | delta_pct  |
| ---------------- | ---------------- | -------- | ---------- | --------- | ---------- |
| 8                | perceptual_total | 0.701198 | 2.099524   | 1.398325  | 199.419340 |
| 8                | total            | 3.132417 | 4.959649   | 1.827232  | 58.332974  |
| 8                | recon_total      | 3.132417 | 4.347071   | 1.214654  | 38.776886  |
| 8                | perceptual       | 0.161407 | 0.200232   | 0.038825  | 24.054369  |
| 8                | pixel_total      | 2.431218 | 2.247547   | -0.183672 | -7.554715  |
| 8                | gan_total        | 0.000000 | 0.612578   | 0.612578  | NA         |
| 8                | cloud_l1         | 0.000000 | 0.000000   | 0.000000  | NA         |
| 8                | cloud_adv        | 0.000000 | 0.432285   | 0.432285  | NA         |
| 8                | feature_matching | 0.000000 | 0.200009   | 0.200009  | NA         |

## 6. 指标趋势摘要
| split | metric      | first    | last     | best_epoch | best     | improvement_pct_if_lower_better | num_points |
| ----- | ----------- | -------- | -------- | ---------- | -------- | ------------------------------- | ---------- |
| train | total       | 2.797293 | 5.192877 | 2.000000   | 2.518682 | -85.639351                      | 115        |
| train | recon_total | 2.797293 | 2.899546 | 2.000000   | 2.518682 | -3.655436                       | 115        |
| train | gan_total   | 0.000000 | 2.293330 | 1.000000   | 0.000000 | NA                              | 115        |
| train | cloud_l1    | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 115        |
| train | cloud_kl    | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 115        |
| train | cloud_adv   | 0.000000 | 0.428194 | 1.000000   | 0.000000 | NA                              | 115        |
| train | disc_total  | 1.221828 | 1.127935 | 114.000000 | 1.125393 | 7.684619                        | 108        |
| val   | total       | 2.461544 | 4.497652 | 2.000000   | 2.179236 | -82.716668                      | 115        |
| val   | recon_total | 2.461544 | 4.497652 | 2.000000   | 2.179236 | -82.716668                      | 115        |
| val   | gan_total   | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 115        |
| val   | cloud_l1    | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 115        |
| val   | cloud_kl    | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 115        |
| val   | cloud_adv   | 0.000000 | 0.000000 | 1.000000   | 0.000000 | NA                              | 115        |

诊断：
- 当前 val recon_total 高于 train recon_total，需要继续观察 gap 是否扩大；若持续扩大才考虑过拟合。

## 7. Loss 贡献比例
| split | term             | weighted_contribution | share_of_recon_total | share_of_total |
| ----- | ---------------- | --------------------- | -------------------- | -------------- |
| train | perceptual       | 1.940567              | 0.669266             | 0.373698       |
| train | feature_matching | 1.651040              | 0.569413             | 0.317943       |
| train | cloud_adv        | 0.642291              | NA                   | 0.123687       |
| val   | perceptual       | 2.435927              | 0.541600             | 0.541600       |

## 8. 训练速度
| split | epochs | seconds_mean | seconds_median | seconds_p95 | sec_per_batch_mean | sec_per_batch_median | sec_per_batch_p95 | latest_sec_per_batch |
| ----- | ------ | ------------ | -------------- | ----------- | ------------------ | -------------------- | ----------------- | -------------------- |
| train | 115    | 461.449      | 465.431        | 532.830     | 0.706              | 0.712                | 0.815             | 0.813                |
| val   | 115    | 21.504       | 22.000         | 24.000      | 0.371              | 0.379                | 0.414             | 0.414                |
- 平均训练耗时: `7.69` min/epoch, `0.706` sec/batch
- 完整逐 epoch 时间表见 `timing_summary.csv`。

## 9. 可视化结果评价
- 评价对象是保存的 RGB PNG，可判断展示质量和视觉趋势；这不是完整 13 波段 checkpoint 定量评测。
- 可视化 epoch 覆盖: `115` 个；配置要求的 `medium/high/heavy` 完整覆盖: `115` 个。
| epoch | file_count | buckets_present   | missing_buckets | complete | total_size_mb |
| ----- | ---------- | ----------------- | --------------- | -------- | ------------- |
| 104   | 3          | heavy,high,medium |                 | 1        | 7.517466      |
| 105   | 3          | heavy,high,medium |                 | 1        | 7.539203      |
| 106   | 3          | heavy,high,medium |                 | 1        | 7.536313      |
| 107   | 3          | heavy,high,medium |                 | 1        | 7.525987      |
| 108   | 3          | heavy,high,medium |                 | 1        | 7.520615      |
| 109   | 3          | heavy,high,medium |                 | 1        | 7.552332      |
| 110   | 3          | heavy,high,medium |                 | 1        | 7.516225      |
| 111   | 3          | heavy,high,medium |                 | 1        | 7.494001      |
| 112   | 3          | heavy,high,medium |                 | 1        | 7.528855      |
| 113   | 3          | heavy,high,medium |                 | 1        | 7.532322      |
| 114   | 3          | heavy,high,medium |                 | 1        | 7.497153      |
| 115   | 3          | heavy,high,medium |                 | 1        | 7.449355      |


可视化诊断：
- 建议同时查看 `latest_visual_contact_sheet.png`，用肉眼确认云区、阴影区和 hard mask 边界是否存在接缝或过暗问题。

## 10. Checkpoint 完整性
| file                                    | kind | epoch | metric      | value    | size_mb    | mtime               |
| --------------------------------------- | ---- | ----- | ----------- | -------- | ---------- | ------------------- |
| best_epoch_0093_pixel_total_1.682863.pt | best | 93    | pixel_total | 1.682863 | 498.433950 | 2026-07-08T09:33:17 |
| last.pt                                 | last | 115   | last        | NA       | 498.085257 | 2026-07-08T12:52:28 |

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