# GLF-CR x DADIGAN Gate 0 预审报告

审查时间：2026-07-10 19:48 CST  
审查范围：只读代码、配置、既有实验产物和公开源码；只执行 CPU 数值核验与小尺寸前向，不修改模型、损失、配置、数据集或训练逻辑，不启动训练。  
目标仓库：`niudufen1225/sar-optical-cloud-removal`

## 1. 结论

**Gate 0 结论：通过，但 S1 实现前必须按本报告修订接口与实验定义。建议进入 S1。**

没有触发计划中的硬停止条件：

- 本地 `main` 与远端 `origin/main` 均为 `335dbf6ae22daf0843dedd32912e3bd883ec3606`。
- 当前训练入口确实读取命令行指定 YAML；两个活动 run 的 `config.resolved.json` 与对应 YAML 完全相等。
- 当前 run 目录中的 checkpoint、日志和 resolved config 可以对应。
- `_glfcr_kernel2d_conv` 与公开 PyTorch 实现在 `k=3/5` 下均满足 `atol=1e-5, rtol=1e-5`。
- S1/S2 不需要修改 dataset 或 manifest。
- low-resolution 与 inside-DDIN GLF-CR 可以通过新增默认兼容键解耦，不必破坏旧 YAML。

但不能直接把当前 `v4_structure` 当成 S1：它仍在每个 DDIN stage 内执行 GLF-CR，仍有 6 个 pre-DDIN FFC block 和 2 个空间旋转 wrapper，`prox_blocks=5`，并且启用了感知损失调度。它只是一个多因素结构实验，不是计划中的 clean-core 严格对照。

## 2. 仓库与工作树

| 项目 | 审查结果 |
|---|---|
| 分支 | `main` |
| 本地 commit | `335dbf6ae22daf0843dedd32912e3bd883ec3606` |
| 远端 `origin/main` | `335dbf6ae22daf0843dedd32912e3bd883ec3606` |
| remote | `https://github.com/niudufen1225/sar-optical-cloud-removal.git` |
| 工作树 | dirty；有活动训练日志、未跟踪的计划文档、v3/v4 YAML 和 v4 输出 |

审查时已有的用户改动未被修改。需要特别注意：v3/v4 YAML 当前未纳入 commit，因此仅靠 commit 不能复现实验；run 目录里的 `config.resolved.json` 暂时保存了精确配置。

计划文档的请求路径 `docs/GLFCR_DADIGAN_最小消融与Codex执行计划.md` 不存在。实际读取的是仓库根目录：

`GLFCR_DADIGAN_最小消融与Codex执行计划.md`

## 3. 当前实际训练配置

审查时有两个正式训练进程，均未被本次预审干预。

### 3.1 GPU1：v3 loss calibration

- YAML：`configs/allclear_dadigan_lama_ffc_stage1_rgb_lowres_glfcr_coupled_spatial_v3_loss_calibration.yaml`
- run：`outputs/allclear/2026-07-10T16-23-41_stage1__v3_loss_calibration`
- 审查快照：epoch 16，best 为 epoch 15，`pixel_total=1.753145`
- `config.resolved.json == yaml.safe_load(YAML)`：`True`
- 数据：RGB `[3,2,1]`、S1 2 通道、256 x 256、OmniCloudMask v4 manifest；train/val/test 路径均存在。

### 3.2 GPU0：v4 structure

- YAML：`configs/allclear_dadigan_lama_ffc_stage1_rgb_lowres_glfcr_coupled_spatial_v4_structure.yaml`
- run：`outputs/allclear/2026-07-10T18-58-25_stage1__v4_structure`
- 审查快照：epoch 7，best 为 epoch 7，`pixel_total=1.819844`
- `config.resolved.json == yaml.safe_load(YAML)`：`True`
- 数据路径和预处理与 v3 相同。

GPU1 在审查期间被 v3 训练占用，因此 KernelConv2D 等价性测试在 CPU/FP32 上完成；这不会改变被核验算子的数学定义，也避免影响活动训练。

## 4. GLF-CR 公开源码基准

对照了两个公开仓库：

1. `xufangchn/GLF-CR`，commit `d55611ab5be4a3530eee9f8c962966a244ba6627`。
2. 计划引用的镜像 `ESWARALLU/glfCRPLUS`，commit `48dc3f92260033fcd9296cfe049c82fa19470db6`。

第一项用于核对原始网络和 `submodules.py::kernel2d_conv`；第二项用于核对带 PyTorch autograd fallback 的 `KernelConv2D.py`。镜像不是唯一官方实现。

### 4.1 DFG 对照

公开 GLF-CR：

```text
Concat(OPT, SAR): 2C
  -> Conv3x3 + LeakyReLU: C
  -> DFResBlock x2: C
  -> Conv1x1 + LeakyReLU: C*k*k
```

当前 `GLFCRDynamicFilterGenerator`：

```text
Concat(opt_f, sar_f): 2C
  -> _glfcr_df_conv(2C, C, 3)
  -> GLFCRDFResBlock(C) x2
  -> _glfcr_df_conv(C, C*k*k, 1)
```

判断：**算子、通道、残差块数量、LeakyReLU(0.1)、bias 和最终激活均一致。** 公开实现的 `DFG(channels*2, k)` 在类内部取 `half_channels=channels`；当前代码直接把这一关系展开为 `2C -> C`，不是结构差异。两者都没有对动态核做 softmax、和为一或单位核约束。

### 4.2 KernelConv2D 对照

当前 `_glfcr_kernel2d_conv` 与公开 `submodules.py::kernel2d_conv` 使用相同处理：

1. replication padding；
2. `unfold` 提取逐像素 `k x k` patch；
3. 将 `C*k*k` 动态核重排到每位置、每通道；
4. 对 `k*k` 元素乘积求和。

其输出保持 `[B,C,H,W]`，动态核必须为 `[B,C*k*k,H,W]`。

### 4.3 双向 gate 对照

公开 GLF-CR 与当前代码均为：

```text
K       = DFG(OPT, SAR)
SAR_f   = KernelConv2D(SAR, K)
g_sar   = sigmoid(Conv1x1(SAR_f - OPT))
OPT_new = OPT + (SAR_f - OPT) * g_sar
g_opt   = sigmoid(Conv1x1(OPT_new - SAR_f))
SAR_new = SAR + (OPT_new - SAR_f) * g_opt
```

判断：**DFG、KernelConv2D 和双向更新公式与公开 SLFC 核心一致。**

不一致之处在网络语义和插入拓扑，而不是上述局部算子。公开 GLF-CR 在双流 RDB/SGCI stage 后执行 SLFC，共 `D=6`，再拼接各阶段 optical feature 做 GFF；当前项目没有 GLF-CR 的 SGCI 和多阶段 optical 聚合。

## 5. 固定种子数值测试

测试条件：

- seed：`20260710`
- device：CPU
- dtype：FP32
- feature：`[2,4,11,13]`
- kernel：`[2,4*k*k,11,13]`
- deterministic algorithms：开启
- 同时比较 forward、feature gradient 和 kernel gradient

| 对照实现 | k | forward 最大绝对误差 | feature grad 最大绝对误差 | kernel grad 最大绝对误差 | 结论 |
|---|---:|---:|---:|---:|---|
| 原仓库 `submodules.kernel2d_conv` | 3 | `0` | `0` | `0` | bitwise equal |
| 原仓库 `submodules.kernel2d_conv` | 5 | `0` | `0` | `0` | bitwise equal |
| `glfCRPLUS` PyTorch fallback | 3 | `1.90734863281e-06` | `1.90734863281e-06` | `0` | 通过 `1e-5` |
| `glfCRPLUS` PyTorch fallback | 5 | `2.86102294922e-06` | `2.86102294922e-06` | `0` | 通过 `1e-5` |

镜像 fallback 与原仓库的求和顺序不同，所以出现约 `2e-6` 的正常 FP32 舍入误差；不是卷积定义差异。

## 6. 当前 GLF-CR 在 DDIN 中的真实数据流

当 `cloud_lowres_glfcr_coupled=true` 时，实际路径为：

```text
masked RGB + mask --PixelUnshuffle(f=2)--> optical stem --> optical FFC
S1 VV/VH          --PixelUnshuffle(f=2)--> SAR stem
                                             |
                                             v
initialize P0, V0, S0
for t in 1..T:
    P_t = ProxP(P_{t-1} - eta_p * PGDA_grad_P)
    V_t = ProxV(V_{t-1} - eta_v * PGDA_grad_V)
    P_t, V_t = GLFCRFusionStep(P_t, V_t)
    S_t = ProxS(S_{t-1} - eta_s * PGDA_grad_S(P_t, V_t))
                                             |
                                             v
CAB(P_T,S_T) -> CAB(V_T,FM) -> MSAB -> optional post-PDAFM context
                                             |
                                             v
RB reconstruct -> PixelShuffle(f=2) -> RGB fill -> hard composite
```

关键事实：

- GLF-CR 操作的是 DADIGAN 的 `opt_private=P` 与 `sar_private=V`，不是公开 GLF-CR 的普通 optical/SAR backbone feature。
- GLF 更新发生在 P/V proximal update 之后、shared S update 之前。
- 更新后的 P/V 立即参与同一 stage 的 shared gradient，并传给下一 stage。
- v3 实测每次 forward 调用 `GLFCRCoupledDDINStep=4`、`GLFCRFusionStep=4`。
- v4 实测每次 forward 调用两者各 `3` 次。
- `GLFCRCoupledDDIN.forward` 每轮覆盖 `aux`，最终只返回最后一个 stage 的 gate 均值；当前日志无法观察前面各 stage 的 gate 或 kernel 演化。

因此当前实现应称为 **GLF-CR SLFC-inspired inside-DDIN coupling**，不能称为完整 GLF-CR。

## 7. `cloud_lowres_glfcr_coupled` 的绑定问题

该单一布尔键目前同时控制：

1. 是否创建并执行 PixelUnshuffle；
2. optical/SAR stem 的输入通道是否乘 `factor^2`；
3. 是否启用 low-resolution optical FFC context；
4. 使用普通 `DDIN` 还是 `GLFCRCoupledDDIN`；
5. reconstruction 是否使用 `Conv -> PixelShuffle -> Conv` 回到原分辨率。

结论：**PixelUnshuffle 低分辨率路径和 GLF-CR coupling 被硬绑定。** 当前不能只靠 YAML 得到“保留 128 x 128 路径但使用普通 DDIN”的 S1。

这也意味着已有 `no_lowres_glfcr` 配置同时改变了分辨率、stem、DDIN 类型、reconstruction 和低分辨率 FFC，不是 GLF-CR 的单变量消融。

## 8. CAB2 与 complement attention

### 8.1 当前 CAB1/CAB2

当前 PDAFM 顺序与 DADIGAN Eq. (21)-(26) 的 Q/K/V 来源一致：

| 模块 | Q | K/V | `SRACAB` 残差基底 |
|---|---|---|---|
| CAB1 | optical-private `P` | shared `S` | `P` |
| CAB2 | SAR-private `V` | CAB1 输出 `FM` | `V` |

`SRACAB.forward` 明确执行：

```text
x = query_tokens + attention(query, reference)
x = x + MLP(LN(x))
```

因此 CAB2 对 SAR-private 存在直接 identity residual。计划中的“CAB2 以 FM 为残差基底，SAR 只提供小增量”不是当前行为，也不是配置可切换项，需要改造接口；它属于任务适配，不是 DADIGAN 原文复现。

### 8.2 当前 complement 公式

先由 SDPA 计算标准注意力：

```text
A = softmax(QK^T / sqrt(d))
Y = A V
```

complement 模式再计算：

```text
Y_comp = (sum_j(V_j) - Y) / (N_ref - 1)
```

即每个 query 的有效权重是 `(1-A_ij)/(N_ref-1)`，总和仍为 1。

这不是 DADIGAN Eq. (22)/(25) 的字面实现，存在两项工程变化：

1. K/V 先经过 PVT spatial reduction；
2. 对 `1-A` 按 `N_ref-1` 归一化，以避免幅值随 token 数线性增长。

当前 256 输入经 PixelUnshuffle 后为 128 x 128，`sr_ratio=8`：

- query token：`128*128 = 16384`
- reference token：`16*16 = 256`

当 `N_ref=256` 时，单个相似 token 被抑制造成的权重变化约为 `1/255` 量级；输出容易趋向 reference V 的空间均值，具有明显低通倾向。它保留了“排除相似信息”的方向，但不等价于原文 full-token、未归一化的 `1-A`。

若关闭 low-resolution 而仍在 256 输入上使用 `sr_ratio=8`，reference token 将变为 `32*32=1024`，归一化 complement 的均值倾向会更强。

## 9. 实际生效的深度、FFC 和空间变换

以下参数量只计算 generator/model，不含 discriminator 和 HRF perceptual network。

| 配置 | DDIN stage | 每 stage ProxNet | Prox RB | reconstruct RB | GLF 调用 | lowres FFC | post-PDAFM FFC | spatial wrapper | CAB | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| v3 | 4 | P/V/S 各 8 RB | 96 | 3 | 4 | 9 | 6 | 6 | complement | 36,427,829 |
| v4 | 3 | P/V/S 各 5 RB | 45 | 3 | 3 | 6 | 0 | 2 | standard | 20,964,290 |

模块级参数核验：

- v3：inside-DDIN GLF fusion 共 `2,998,272` 参数；ProxNet 共 `24,912,000`；15 个 FFC block 共 `1,411,920`。
- v4：inside-DDIN GLF fusion 共 `2,248,704` 参数；ProxNet 共 `11,957,760`；6 个 FFC block 共 `564,768`。

空间变换实际都在 128 x 128 feature 上：

- v3 lowres FFC 层 `[1,4,7]`，post-PDAFM FFC 层 `[1,3,5]`，共 6 个 wrapper。
- v4 lowres FFC 层 `[1,4]`，共 2 个 wrapper。
- `train_angle=false` 不等于关闭：角度由固定 seed 随机初始化后注册为 buffer，训练时仍执行 reflection pad、FP32 rotate、模块、逆 rotate 和 crop；只是角度不学习。

DADIGAN 原文设置 feature channel 64、iteration stage `T=3`，stage 消融也显示 3 最优。当前 dim 96、v3 的 `T=4/prox=8` 明显更重；v4 的 stage 数对齐为 3，但 prox 深度仍不是计划 S1 的 2。

## 10. 已有消融与可复用性

### 10.1 可复用 E0

`outputs/allclear/2026-07-07T21-25-48_stage1__dadigan_lama_ffc_lowres_glfcr_spatial_main_v2` 可作为计划中的已有 E0：

- 完成 120 epochs；
- best 为 `best_epoch_0093_pixel_total_1.682863.pt`；
- 结构为 stage4、prox8、inside-DDIN GLF x4、complement CAB、lowres FFC9、post FFC6、spatial wrapper6；
- 已有 train/val/test 分支评估和逐样本 CSV。

它只能作为现有复杂基线，不能替代 S1/S2。

### 10.2 已有 no_* YAML

| YAML | 实际变化 | 是否单变量 | 是否等价 S1/S2/F1/F2 |
|---|---|---|---|
| `...ablate_no_ffc.yaml` | lowres/post FFC 和 spatial 全关；仍 stage4/prox8/inside GLF/complement | 否，FFC 多处同时关闭 | 否 |
| `...ablate_no_lowres_glfcr.yaml` | 同时关闭 PixelUnshuffle、lowres FFC、inside GLF，并改变 stem/reconstruct；post FFC6 仍开 | 否；batch 还从 2 改为 1、有效 batch 不同 | 否 |
| `...ablate_no_spatial.yaml` | 只关闭 6 个 spatial wrapper，FFC 仍开 | 接近单变量 | 否 |

工作区内没有找到这些 YAML 对应的 run 目录、`config.resolved.json`、checkpoint、日志或分析结果；全仓库搜索配置名也无实验引用。因此当前**没有可复用的 no_ffc/no_lowres/no_spatial 结果**，只有配置定义。

### 10.3 v3/v4 的复用边界

- v3 可复用为“复杂结构下的 loss calibration”，不是 S1。
- v4 可复用为“同时缩减 DDIN/Prox/FFC、切换 CAB 的联合结构实验”，不是严格单变量，也不是 S1。
- 当前没有 S2 one-shot filter 等价实验。
- 当前没有 F1 SWT/色度 loss 实现或等价实验。
- 当前 DDIN 只返回最终 P/S/V，没有 F2 stage states、共享辅助 head 或 stage loss，因此没有 F2 等价实验。

## 11. 训练入口、配置透传与 checkpoint

### 11.1 已验证能力

- CLI `--config` 进入 `train.py::build_model`，现有 DDIN/Prox/lowres/FFC/CAB 等键均透传到 `DADIGANBaseline -> DADIGANCloudBranch`。
- 新 run 创建时保存完整 `config.resolved.json`。
- 当前两个活动 run 的 YAML 与 resolved JSON 完全相等。
- `last.pt` 和最多一个 best checkpoint 按 `pixel_total` 保存。
- checkpoint 含 model、optimizer、epoch、best_metric，并在启用 GAN 时含 discriminator 和其 optimizer。
- 训练具备 finite loss/gradient 检查、AMP、grad clipping 和逐 epoch val。
- `evaluate_allclear_runs.py` 可调用 `evaluate_stage1_branches.py` 对 train/val/test 做逐区域 MAE/RMSE/PSNR/bias/SAM，并保存逐样本 CSV和 bucket 可视化。

### 11.2 风险和缺口

1. checkpoint payload 不包含 config、config hash、commit 或 run id；当前一一对应依赖目录结构，单独移动 `.pt` 后无法自证配置。
2. `load_checkpoint(..., strict=False)` 只警告 missing/unexpected keys；错误 YAML 仍可能继续评估，后续实验入口应主动检查结构签名。
3. 当前 val 的 GAN/FM 项不参与可比 checkpoint 选择；`pixel_total` 适合当前 paired 恢复选择，但不能覆盖计划的频率/SAR 泄漏标准。
4. 当前 branch evaluator 没有 SSIM、SWT-LL/LH/HL/HH、高频能量比、SAR zero/shuffle、SAR 高频-输出误差相关性或配对 bootstrap CI。
5. 当前训练只记录最终 gate 均值，不记录逐 stage kernel/gate；也不记录各模块梯度范数。
6. loss 日志中的 `loss_value * weight` 是 objective 数值贡献，不是该项对特定模块的实际梯度贡献。

因此现有验证足以做基本 paired 恢复评估，但不足以直接执行计划第六节的 S1/S2 选择规则。

## 12. 对计划修改项的可行性分类

| 计划项 | 分类 | 预审判断 |
|---|---|---|
| stage=3、prox=2、standard CAB、关闭 FFC/spatial/GAN/FM/perceptual/KL | 可直接配置 | 现有键已支持；但需先解耦 lowres/GLF，CAB2 residual 另算 |
| lowres PixelUnshuffle 与 inside-DDIN GLF 解耦 | 需要小范围重构 | 当前单键控制五处行为；必须保持旧键 fallback |
| S1 使用普通 DDIN | 需要上述解耦 | `DDIN` 已存在，不需重写 PGDA |
| CAB2 改为 FM residual + 小 SAR 增量 | 需要接口重构 | 当前 residual 固定为 query V；应可切换并默认旧行为 |
| S2 one-shot dynamic SAR filter | 需要新增组合路径 | DFG/KernelConv2D 可复用；不能直接复用双向 `GLFCRFusionStep` |
| 动态核不归一化 | 可直接实现 | 保持当前公开形式即可 |
| 逐 stage kernel/gate telemetry | 需要重构 | 当前只保留最后 stage aux；S2 若无 gate，应改记录 filtered delta |
| S1/S2 评估指标 | 需要扩展 evaluator | 基础逐样本框架可复用；频率与 counterfactual 指标缺失 |
| F1 SWT/色度 loss | 证据支持方向，但需新实现和测试；应在 S1/S2 后 | 当前没有相关代码；不是 WGSR 原样迁移 |
| F2 stage deep supervision | 需要较大重构；证据不足以称 ProxUnroll；延期 | DDIN 不返回 trajectory；ProxUnroll 的 PT target 依赖明确物理测量算子和 GT proximal target |
| O1 post-PDAFM residual FFC | 条件触发时可复用 | `LaMaFFCResidualContext` 已存在；不应在 S1/S2 提前启用 |
| kernel softmax/有界残差核 | 证据不足，延期 | 公开 GLF-CR 未采用，会改变滤波器性质 |
| tied-adjoint PGDA | 与最小定位目标冲突，延期 | 会同时改变 DDIN 算子自由度 |
| 高频 GAN | 与 mandatory deterministic 对照冲突，延期 | 先完成确定性结构/颜色定位 |
| dataset/manifest 修改 | 停止项 | 本轮不需要，也不得修改 |

### 12.1 ProxUnroll 对照修正

公开 ProxUnroll commit `9aa4f93b727756c07079af0483bae4f8ecf0d1df` 的 PT loss 不是普通“每 stage 对 GT”深监督。它先用已知测量矩阵构造 `prox_f`，再用 GT 构造显式 `prox_g` target，最后用固定 stage weights 对网络 trajectory 和 proximal trajectory 做 MSE/RMSE。当前 DADIGAN 没有可直接复用的同定义 target trajectory。

所以 F2 可以实现为 **DADIGAN stage deep supervision**，但不得称为 ProxUnroll PT loss 的复现；在 S1/S2 获胜前应延期。

### 12.2 WGSR 对照修正

公开 WGSR commit `0596d929e4c3db35cf94cb8e9022ef377a8faff6` 在 8-bit Y 公式上做一层或两层 SWT，用 LL/LH/HL/HH pixel losses，并让 discriminator 处理高频子带。计划提出的 Haar、RGB LL、亮度方向子带和色度约束是面向遥感反射率的新适配，不是 WGSR 原样实现。

方向有文献依据，但边界模式、filter normalization、mask reduction、暗区色度稳定性和梯度比例必须通过计划中的数值测试后才能进入 F1。

## 13. 必需的新配置键

S1/S2 最小建议键如下。为兼容旧 YAML，新键缺失时必须回退到当前 `cloud_lowres_glfcr_coupled` 行为。

| 新键 | 建议默认 | 用途/兼容策略 |
|---|---|---|
| `cloud_lowres_enabled` | `null` | `null` 时沿用旧 `cloud_lowres_glfcr_coupled`；显式控制 PixelUnshuffle/stem/reconstruct |
| `cloud_ddin_glfcr_coupled` | `null` | `null` 时沿用旧键；显式控制普通 DDIN 或每 stage GLF coupling |
| `cloud_post_ddin_sar_filter` | `none` | `none` 或 `glfcr_dynamic`；S2 只过滤最终 V |
| `cloud_post_ddin_sar_filter_kernel_size` | 回退 `cloud_lowres_glfcr_kernel_size` | S2 DFG/KernelConv2D 的 k |
| `cloud_cab2_residual_source` | `query` | `query` 保留旧 V residual；S1 用 `reference`/`fm` |
| `cloud_cab2_update_scale` | `1.0` | 保留旧行为；S1 设小值，必须明确是固定还是可学习 |
| `cloud_return_ddin_stages` | `false` | F2 才开启，避免 S1/S2 常态显存开销 |

F1/F2 延期键可在对应阶段再加，不应提前污染 S1/S2 配置：wavelet 各子带权重、色度阈值、stage loss weights、共享 head 开关等。

## 14. 对原计划的必要修订

1. **修订 S2 的动态核定义。** 公开 DFG 必须同时读取 optical 和 SAR feature，因此应写为：

   ```text
   K_T = DFG(P_T, V_T)
   V_filtered = KernelConv2D(V_T, K_T)
   ```

   而不是把 `V_T -> dynamic filtered V_T` 描述成单输入算子。

2. **删除 S2 的 gate telemetry 要求。** 若 S2 不更新 P、不反向更新 V，也不调用双向 gate，则没有 gate 可记录。应记录 `K_T` 的 mean/std/absmax、`V_filtered-V_T` 的 L1/RMS/频谱，以及最终输出对 SAR zero/shuffle 的变化。若保留 gate，就不再是“只做 DFG + KernelConv2D”的唯一差异。

3. **S1/S2 选择前先补 evaluator。** 当前缺少 SSIM、SWT、SAR counterfactual 和 bootstrap；否则计划中的接受规则无法执行。

4. **把 v4 标为联合结构实验，不计入 S1。** v4 同时改变 stage、prox、FFC 数、spatial 数、CAB 和 post context，且仍 inside-DDIN GLF，不能定位单一来源。

5. **F2 改名。** 使用“stage deep supervision inspired by unrolling trajectory training”，不得称为 ProxUnroll PT loss；除非为 DADIGAN 推导并实现与其观测算子一致的显式 proximal target。

6. **F1 明确为 WGSR-inspired 遥感适配。** 不照搬 8-bit YCbCr 偏置和 HF discriminator；固定 Haar SWT 是新设计，必须单独验证。

7. **补 checkpoint 可追溯性。** 后续新 run 应在 checkpoint 或相邻 metadata 中记录 commit、resolved config hash 和结构签名；不修改旧 checkpoint。

8. **S1/S2 必须从同一随机初始化分别训练。** 不能从当前 v3/v4 checkpoint 续训并宣称严格消融。

## 15. 风险与停止项

### 15.1 当前主要风险

- inside-DDIN 双向 GLF 重复改写 private P/V，且更新后的 P/V 继续影响 S，SAR 污染可跨 stage 累积。
- CAB2 的 V identity residual 允许 SAR-private 绕过差异注意力进入 fused feature。
- normalized complement 在 256 个 reference token 下接近 reference 均值，可能造成低频化。
- v3 的 96 个 Prox RB、15 个 FFC 和 6 次固定随机旋转令归因困难；v4 虽较小，仍不是 clean core。
- 现有 gate 只显示最后 stage，无法确认污染从哪一 stage 开始。
- no_* 结果缺失，不能用不存在的实验结论替代 S1/S2。

### 15.2 进入 S1 后仍应执行的停止检查

- 新键未提供旧行为 fallback，立即停止。
- 旧 main_v2/v3/v4 config 构建出的 state-dict key 或 forward 结果发生非预期变化，立即停止。
- S1 仍实例化 `GLFCRCoupledDDINStep` 或任何 FFC/spatial wrapper，立即停止。
- CAB2 选择 FM residual 后，SAR query 完全无梯度或输出 shape/scale 异常，立即停止。
- 需要修改 manifest/dataset 才能完成 S1，立即停止。
- checkpoint 只能依靠 `strict=False` 才能加载同结构实验，立即停止并核查配置。

## 16. 是否进入 S1

**建议进入 S1。** 理由：GLF-CR 局部算子已经通过源码和数值核验，当前主要不确定性确实集中在跨框架插入位置、重复 P/V 双向更新、CAB2 SAR residual、normalized complement 和过深 context，而不是 KernelConv2D 抄错。

进入条件：

1. 先实现 lowres 与 inside-DDIN GLF 的向后兼容解耦；
2. CAB2 residual source/scale 做成可切换并默认旧行为；
3. 增加最小单元测试和结构断言；
4. S1 配置关闭 GLF、FFC、spatial、GAN、FM、perceptual、KL，使用 stage3/prox2/standard CAB；
5. 不复用 v3/v4 权重作为严格 S1 起点；
6. 在 S1/S2 比较前补齐计划要求的关键 evaluator 指标。

## 17. 本次执行命令

以下为关键命令摘要；均为只读或一次性 `/tmp`/CPU 核验：

```bash
git rev-parse HEAD
git branch --show-current
git remote -v
git status --short
git ls-remote origin refs/heads/main

sed -n '1,620p' GLFCR_DADIGAN_最小消融与Codex执行计划.md
nl -ba src/allclear/modules/dadigan.py
nl -ba src/allclear/modules/pvt_sra_cab.py
rg -n '...' src/allclear/model.py src/allclear/losses.py src/allclear/train.py

git clone --depth 1 https://github.com/xufangchn/GLF-CR.git /tmp/codex_glfcr_xufang
git clone --depth 1 https://github.com/ESWARALLU/glfCRPLUS.git /tmp/codex_glfcr_plus
git clone --depth 1 https://github.com/pwangcs/ProxUnroll.git /tmp/codex_proxunroll
git clone --depth 1 https://github.com/mandalinadagi/WGSR.git /tmp/codex_wgsr

python - <<'PY'
# seed=20260710；比较本地 kernel、原仓库 Python kernel 和 mirror fallback；
# k=3/5；检查 forward 与两路 backward 最大绝对误差。
PY

OMP_NUM_THREADS=1 python - <<'PY'
# 32x32 CPU inference + forward hooks；验证 v3/v4 每次 forward 的
# GLFCRCoupledDDINStep/GLFCRFusionStep 实际调用次数。
PY

python - <<'PY'
# 比较 YAML 与 config.resolved.json；核对 manifest、checkpoint、参数量和模块数量。
PY

rg -l 'ablate_no_(ffc|lowres_glfcr|spatial)|gtstrong_quick' outputs
pdftotext -layout 2026IF-Dadigan.pdf /tmp/codex_dadigan.txt
```

未执行训练命令，未终止或更改现有训练进程，未使用 GPU，未修改任何模型/损失/配置/数据文件。

## 18. 参考资料

- GLF-CR 公开源码：<https://github.com/xufangchn/GLF-CR>
- 计划引用的 GLF-CR 镜像：<https://github.com/ESWARALLU/glfCRPLUS>
- DADIGAN 本地论文：`2026IF-Dadigan.pdf`
- ProxUnroll CVPR 2025：<https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Proximal_Algorithm_Unrolling_Flexible_and_Efficient_Reconstruction_Networks_for_Single-Pixel_CVPR_2025_paper.html>
- ProxUnroll 公开源码：<https://github.com/pwangcs/ProxUnroll>
- WGSR 公开源码：<https://github.com/mandalinadagi/WGSR>
- WGSR CVPR 2024：<https://cvpr.thecvf.com/virtual/2024/poster/29831>
