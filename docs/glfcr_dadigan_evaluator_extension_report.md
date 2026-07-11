# GLF-CR x DADIGAN Phase 2 Evaluator Extension Report

审查和实现日期：2026-07-10（Asia/Shanghai）  
仓库：`niudufen1225/sar-optical-cloud-removal`  
范围：只扩展 checkpoint evaluator 和数值测试；没有启动训练。

## 1. 范围与保护项

已确认 Phase 1 commit 已存在：

```text
branch: main
HEAD: 22bde2ce2ad958ff752a2c5404d52c6156631960
commit: 22bde2c refactor: decouple lowres DDIN and GLF-CR fusion
```

`python -m unittest -q tests.test_glfcr_dadigan_phase1` 通过。当前工作树中的训练日志、v3/v4 配置、v4 输出、计划文档和 Gate 0 报告属于已有用户内容，均被保留。本阶段没有修改：

- `src/allclear/dataset.py`、manifest 或数据预处理；
- `src/allclear/model.py`、任何模型模块；
- `src/allclear/losses.py`、训练损失和训练逻辑；
- 任何 YAML 配置；
- `outputs/`、checkpoint、活动日志。

新增评估文件只读取现有 dataset/model 接口。既有 `evaluate_training_run.py` 是日志和 PNG 诊断器，不具备 checkpoint tensor 评估入口，因此保持不变；checkpoint 指标扩展接入现有 `evaluate_stage1_branches.py`，批量入口接入现有 `evaluate_allclear_runs.py`。

## 2. 修改文件

### 新增

- `src/allclear/eval_metrics.py`
  - 固定数据域和 SSIM 定义；
  - 一级 Haar SWT；
  - SAR 反事实构造、输出差异和高频相关性；
  - paired bootstrap。
- `tests/test_glfcr_dadigan_evaluator_phase2.py`
  - SWT、mask/SSIM/RGB bias、SAR、bootstrap、paired CSV 和 tiny model smoke 测试。
- `docs/glfcr_dadigan_evaluator_extension_report.md`

### 修改

- `scripts/evaluate_stage1_branches.py`
  - 扩展 MAE/RMSE/PSNR/SSIM、RGB bias、known 区域、bucket 汇总；
  - 扩展 Haar 指标和三个新 JSON；
  - 增加可选 SAR 反事实第二遍评估；
  - 保留旧字段和旧命令；
  - 修正原脚本把累计 accumulator 值写入逐样本 CSV 的问题。现在旧字段名不变，但每行是真正的单样本值，汇总仍使用全体样本累计统计。
- `scripts/evaluate_allclear_runs.py`
  - 增加可选 SAR 反事实参数；
  - 增加两个 run 的按 `sample_id` 配对 bootstrap，默认 2000 次。

## 3. 反归一化、监督域和 RGB 定义

评估不使用可视化亮度增强后的像素。实际数据流是：

```text
TIFF optical
  -> dataset clip [0, 10000] / optical_scale(10000)
  -> optional model_band_indices
  -> optional model_reflectance_range [lo, hi] -> [0, 1]
  -> model and numerical metrics
```

当前 v2/v3/v4 配置为 `model_band_indices=[3,2,1]` 和 `model_reflectance_min=0.0, model_reflectance_max=0.35`。因此新指标明确使用：

```text
metric_domain = model_supervision
data_range = 1.0
model_reflectance_stretch = [0.0, 0.35]
```

输入、target、prediction 都经过同一个 model-domain 变换；没有把物理反射率值和 `[0,1]` 视觉域混算。`save_visuals` 使用的 gamma/gain/panel stretch 只用于 PNG 展示，不进入数值指标。

RGB bias 也在 `model_supervision` 域计算。若配置先选取原始波段，评估器会把配置中的原始 RGB 索引映射到当前 model tensor；当前 RGB v2/v3/v4 的 `[3,2,1]` 已正确映射到选择后的 `[0,1,2]`。记录：

```text
bias_r = mean(pred_R - target_R)
bias_g = mean(pred_G - target_G)
bias_b = mean(pred_B - target_B)
mean_abs_channel_bias = mean(|bias_r|, |bias_g|, |bias_b|)
```

区域 bias 使用同一区域 mask；CSV 对每个 candidate/region 都写入这些字段。

## 4. 基础指标

对每个 candidate 分别计算 `full`、`clear`、`known`、`shadow`、`cloud`。`known` 是 `clear` 的明确别名，保留 `clear` 旧字段并新增 `known` 字段。`masks_from_cld_shdw` 的现有语义保持不变：cloud 优先于 shadow，所以重叠像素不被重复计入 shadow/known。

对区域权重 `M` 和所有 model channels：

```text
MAE  = sum(M * |P - Y|) / sum(M)
RMSE = sqrt(sum(M * (P - Y)^2) / sum(M))
PSNR = 20 * log10(data_range / RMSE)
```

这里 `data_range=1`。空区域的 MAE/RMSE/PSNR/bias/SAM/SSIM 记为 NaN，在汇总时排除，不以 0 伪装成好结果。cloud coverage bucket 继续沿用现有 `cloud_bucket_name`，bucket 汇总同时包含新指标。

SSIM 是 RGB 评价：

- 局部窗口为 `7 x 7` uniform window；
- `C1=(0.01*1)^2`，`C2=(0.03*1)^2`；
- `count_include_pad=False`；
- 先形成每通道 SSIM map，再对 RGB channel 和 spatial map 求均值；
- full 区域使用整幅图；
- cloud/known/clear/shadow 区域使用 `mask > 0.5` 的 tight bounding box，不把外部背景填入 crop；
- bbox 小于 `3 x 3` 或为空时为 NaN；3--7 的 crop 使用不超过 crop 的最大奇数窗口。

新的 `metrics_summary.json` 将上述定义、data domain 和 stretch 范围一并写入；标准 JSON 中未定义值写为 `null`。原有 `<split>_branch_metrics_summary.json` 保持旧文件名和 Python JSON 的历史 NaN 兼容行为。

## 5. 一级固定 Haar SWT

### 5.1 定义

使用固定、无可训练参数的一级 undecimated Haar 变换：

```text
lo = [1, 1] / sqrt(2)
hi = [-1, 1] / sqrt(2)
LL = outer(lo, lo)
LH = outer(lo, hi)
HL = outer(hi, lo)
HH = outer(hi, hi)
```

第一维是 y，第二维是 x，因此：

- `LH`：low-y/high-x，主要响应垂直边缘；
- `HL`：high-y/low-x，主要响应水平边缘；
- `HH`：对角/棋盘高频。

变换不做下采样，每个 band 保持 `[B,C,H,W]`。边界规则是卷积前在右侧和底侧各做一像素 `reflect` padding，然后取与输入相同的 H/W；没有 circular padding、zero padding 或隐式缩放。

### 5.2 记录值

在 full 和 cloud 区域对每个 candidate 记录：

```text
LL MAE, LH MAE, HL MAE, HH MAE
HF energy ratio (prediction)
HF energy ratio (target)
absolute difference of the two ratios
```

其中：

```text
HF energy ratio = (E_LH + E_HL + E_HH) /
                  (E_LL + E_LH + E_HL + E_HH + eps)
```

`E` 是对应 band 的平方能量，mask 在相同空间位置应用。该评价不使用 WGSR 的 8-bit YCbCr 偏置公式，也不进入训练。

## 6. SAR 反事实评价

SAR 反事实通过 `--sar-counterfactual` 显式开启，避免旧评估命令自动增加五倍前向成本。反事实在第二个 DataLoader pass 中执行，默认 batch size 为 4，从而可以构造真正的 batch shuffle。

给定已经由 dataset 归一化到 DADIGAN 输入域的 SAR：

```text
SAR_real = current S1 tensor
SAR_zero = zeros_like(SAR_real)
SAR_shuffle = roll(SAR_real, shift=1, batch dimension)
SAR_LF = reflect-padded 5x5 uniform average(SAR_real)
SAR_HF = SAR_real - SAR_LF
```

没有对任何反事实再次 min-max、z-score 或 per-sample normalize。`SAR_shuffle` 只有 batch size 大于 1 才有效；singleton batch 的 shuffle 与 real 完全相同，相关 CSV 字段写 NaN，summary 增加 `shuffle_valid_samples`，不会将它当成有效反事实。

### 6.1 SAR 输入量

记录：

```text
D_LF_SAR = RMS(SAR_LF) / max(RMS(SAR_real), eps)
D_HF_SAR = RMS(SAR_HF) / max(RMS(SAR_real), eps)
```

并同时记录 `sar_lf_energy_ratio`、`sar_hf_energy_ratio`。

### 6.2 输出反事实变化

同一 optical/mask/target 下，对 real、zero、shuffle、LF、HF 各运行模型一次。以 real 输出为基准，对每个替代输出记录：

```text
MAE(real_output - alternative_output)
RMSE(real_output - alternative_output)
LL MAE of the output difference
HF MAE of the output difference
```

这些字段名形如 `sar_real_vs_lf_mae`、`sar_real_vs_shuffle_ll_mae`。

### 6.3 SAR 高频相关性

先计算：

```text
output_error = I_hat(real SAR) - target
output_error_HF = sqrt(mean over RGB and LH/HL/HH Haar magnitudes^2)
SAR_HF_magnitude = sqrt(mean over SAR channels and LH/HL/HH Haar magnitudes^2)
```

再对整幅图和 cloud region 的像素向量计算 Pearson 相关系数，记录为 `sar_error_hf_corr_full/cloud`。方差近似为零或有效像素少于 2 时记 NaN。

## 7. Paired bootstrap

批量 evaluator 增加：

```text
--paired-run RUN_S1 --paired-run RUN_S2
--paired-split test
--bootstrap-resamples 2000
--bootstrap-seed 20260710
```

配对方式不是按 CSV 行号，而是两个 split CSV 的 `sample_id` 精确交集。对每个指标：

```text
delta_i = metric_i(S2) - metric_i(S1)
```

只保留两侧都为 finite 的 pair；再从这些 pair 有放回采样至少 2000 次。报告：

- `mean_delta`；
- `median_delta`；
- mean 的 percentile 95% CI；
- median 的 percentile 95% CI；
- `n_total`、`n_valid`；
- higher-is-better 和方向判断；
- seed 和重采样次数。

PSNR/SSIM 是高优指标，MAE/RMSE/绝对 bias/wavelet MAE 是低优指标。不会对两个汇总均值做非配对 bootstrap。

输出：`<batch-output-dir>/paired_comparison.json`。默认只比较 `final` 的基础质量和 Haar 误差字段；SAR 的反事实变化属于诊断量，不被误判为 restoration quality 的高优/低优指标。

## 8. 输出文件

单个 `evaluate_stage1_branches.py` split 输出目录中：

```text
<split>_branch_metrics_per_sample.csv       # 旧字段保留，新增字段追加
<split>_branch_metrics_summary.json         # 旧文件名保留
metrics_summary.json                        # 基础指标、bucket、域和 SSIM 定义
wavelet_summary.json                        # Haar SWT 汇总
sar_counterfactual_summary.json             # SAR 反事实汇总；未启用时 status=disabled
visualizations/                             # 旧可视化路径
```

当 `--sar-counterfactual` 启用时，SAR 字段合并回同一个逐样本 CSV。缺失/无效场景用 NaN，标准 JSON 汇总转换为 `null`，并保留有效样本计数。

旧命令无需新增参数即可运行基础 evaluator；新增 SAR pass 是显式 opt-in。批量命令会把新基础指标带入 `branch_metrics_all.csv`，并在显式提供两个 `--paired-run` 时生成 paired JSON。

## 9. 测试和 smoke 结果

### 9.1 单元测试

执行：

```bash
python -m unittest -q tests.test_glfcr_dadigan_phase1 tests.test_glfcr_dadigan_evaluator_phase2
```

结果：

```text
Ran 17 tests in 9.712s
OK
```

覆盖内容：

1. 常量输入的 LH/HL/HH 近零；
2. 垂直/水平渐变方向响应和棋盘 HH 响应；
3. `pred == target` 的四个子带误差和 HF ratio 差异为零；
4. CPU/GPU FP32 finite（当前环境 CUDA 可用时执行 CUDA 分支）；
5. RGB channel bias 和空/小 mask 的 SSIM NaN 规则；
6. SAR zero、roll shuffle、reflect LP、`real = LF + HF` 确定性；
7. paired bootstrap 固定 seed 确定性和方向；
8. paired CSV 按 sample_id 而不是行顺序连接；
9. tiny model 前向和指标 smoke。

另行通过：

```bash
python -m py_compile \
  src/allclear/eval_metrics.py \
  scripts/evaluate_stage1_branches.py \
  scripts/evaluate_allclear_runs.py \
  tests/test_glfcr_dadigan_evaluator_phase2.py
```

### 9.2 真实 checkpoint smoke

使用 v3 的现有 `last.pt`，只评估 val 的一个样本，并把输出写到项目外临时目录：

```bash
python scripts/evaluate_stage1_branches.py \
  --config outputs/allclear/2026-07-10T16-23-41_stage1__v3_loss_calibration/config.resolved.json \
  --checkpoint outputs/allclear/2026-07-10T16-23-41_stage1__v3_loss_calibration/checkpoints/last.pt \
  --split val \
  --output-dir /tmp/glfcr_eval_phase2_smoke \
  --gpu 1 \
  --limit 1 \
  --num-workers 0 \
  --save-visuals 0 \
  --visual-samples-per-bucket 0 \
  --sar-counterfactual \
  --sar-batch-size 2 \
  --sar-low-pass-kernel 5
```

结果：模型成功加载并完成 regular + SAR pass；生成了：

```text
/tmp/glfcr_eval_phase2_smoke/val_branch_metrics_per_sample.csv
/tmp/glfcr_eval_phase2_smoke/metrics_summary.json
/tmp/glfcr_eval_phase2_smoke/wavelet_summary.json
/tmp/glfcr_eval_phase2_smoke/sar_counterfactual_summary.json
```

该单样本 smoke 的示例数值为：`final/full MAE=0.03278`、`RMSE=0.05289`、`PSNR=25.53`、`SSIM=0.8737`；`D_LF_SAR=0.9949`、`D_HF_SAR=0.0880`。因为 smoke 使用 `limit=1`，`shuffle_valid_samples=0` 是定义正确的结果，不是实现失败。

## 10. 结论和使用建议

Phase 2 evaluator extension 已完成，未触碰训练路径。基础指标和 Haar 指标可直接通过旧的 branch evaluator 获得；SAR 反事实应在需要分析 SAR 依赖性时显式开启。两个模型/实验的性能比较应使用 batch evaluator 的 `--paired-run`，并以 `paired_comparison.json` 的 `sample_id` 配对 CI 为依据，而不是比较两份汇总均值。

建议正式评估时：

```bash
python scripts/evaluate_stage1_branches.py \
  --config RUN/config.resolved.json \
  --checkpoint RUN/checkpoints/best_epoch_XXXX.pt \
  --split test \
  --output-dir RUN/analysis/branch_test \
  --gpu 1 \
  --visual-samples-per-bucket 5 \
  --sar-counterfactual \
  --sar-batch-size 4
```

旧的日志/PNG 诊断仍使用：

```bash
python scripts/evaluate_training_run.py --run-dir RUN
```

两者职责不同：前者计算 checkpoint tensor 指标，后者分析训练日志和已有 PNG；`evaluate_allclear_runs.py` 可以批量串联两者。
