# GLF-CR x DADIGAN Phase 3 配置实现报告

本阶段：创建 S1/S2 严格消融配置和只读结构检查工具。  
本阶段状态：代码和文档已完成；按任务边界，未执行验证。

## 1. 执行边界和当前仓库状态

开始前只读确认：

```text
branch: main
HEAD: 22bde2ce2ad958ff752a2c5404d52c6156631960
commit: 22bde2c refactor: decouple lowres DDIN and GLF-CR fusion
```

Phase 1 测试文件、Phase 1 报告、Gate 0 报告、Phase 2 evaluator 报告、v4 structure YAML 和当前 RGB 配置均已按真实路径读取。当前工作树中已有的训练日志、v3/v4 配置、v4 输出、Phase 2 evaluator 修改和用户未提交文件均未覆盖、删除或移动。

严格遵守本阶段限制：

- 没有运行单元测试；
- 没有运行结构检查脚本；
- 没有构建模型、前向、反向或 smoke；
- 没有加载 `.pt`、`.pth` 或 checkpoint；
- 没有调用 CUDA、`nvidia-smi` 或设置 `CUDA_VISIBLE_DEVICES`；
- 没有启动训练或真实数据评估；
- 没有修改 dataset、manifest、预处理、outputs、checkpoint、日志或既有训练配置。

## 2. 修改和新增文件

新增：

```text
configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml
configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml
scripts/inspect_glfcr_s1_s2_configs.py
tests/test_glfcr_dadigan_phase3_configs.py
docs/glfcr_dadigan_phase3_config_implementation_report.md
```

Phase 1 测试 `tests/test_glfcr_dadigan_phase1.py` 没有修改。模型、损失和 dataset 代码没有修改。

## 3. 配置来源

两份配置的 data/train/eval 设置来自当前实际使用的 RGB AllClear 结构实验：

- 数据路径、RGB 波段 `[3,2,1]`、模型通道顺序 `[0,1,2]`、`optical_scale=10000`、`[0,0.35] -> [0,1]` 来自当前 v3/v4 RGB 配置；
- 训练周期 100、batch size 4、gradient accumulation 2、`lr=3e-5`、cosine warmup、验证和 checkpoint 规则沿用 v4 structure；
- 确定性 RGB 云区重建项沿用 v3/v4 的 `cloud_l1_missing=30`、`cloud_l1_known=0`、`cloud_l1_reduction=hybrid`；
- KL、GAN、feature matching、perceptual 权重全部设为 0，符合本阶段 clean-core 约束；
- `perceptual_type` 使用 `rgb_l1`，同时不保留外部 HRF 权重路径依赖，避免权重为 0 时仍初始化无关的 perceptual backbone。这不改变 active loss，因为 `perceptual=0`。

两份 YAML 的 `run_name` 故意相同。训练时必须通过命令行 `--run-name` 区分 S1 和 S2 输出目录；这样机器可读配置比较不会把实验命名差异误判成模型差异。

## 4. S1 配置

S1 文件：

`configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml`

有效结构设置：

```yaml
model:
  cloud_lowres_glfcr_coupled: false
  cloud_lowres_enabled: true
  cloud_ddin_glfcr_coupled: false
  cloud_ddin_steps: 3
  cloud_prox_blocks: 2
  cloud_reconstruct_blocks: 2
  cloud_cab_attention_mode: standard
  cloud_msab_mode: efficient
  cloud_cab2_residual_source: reference
  cloud_cab2_update_scale: 0.1
  cloud_post_ddin_sar_filter: none
  cloud_post_ddin_sar_filter_kernel_size: null
  cloud_lowres_opt_ffc_blocks: 0
  cloud_pre_pda_context: none
  cloud_prefusion_context: none
  cloud_bottleneck_context: none
  cloud_ffc_blocks: 0
  cloud_ffc_spatial_transform_layers: []

loss:
  cloud_l1_missing: 30.0
  cloud_l1_known: 0.0
  cloud_kl: 0.0
  cloud_adv: 0.0
  feature_matching: 0.0
  perceptual: 0.0
```

解释：PixelUnshuffle/PixelShuffle 保留，用于隔离“低分辨率路径”与“DDIN 内 GLF-CR coupling”；DDIN 使用普通 `DDIN`，不使用 `GLFCRCoupledDDINStep`。CAB2 的 query 仍然是 SAR-private，`reference` 只改变残差基底为 CAB1 的 FM，更新增量固定缩放为 0.1。

`cloud_msab_mode=efficient` 是本阶段的明确选择：任务禁止新增 Restormer，因此不沿用 v4 中的 `restormer_mdta`；这不是 S1/S2 之间的差异，二者完全一致。

## 5. S2 相对 S1 的唯一差异

S2 文件：

`configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml`

只改变：

```yaml
model:
  cloud_post_ddin_sar_filter: glfcr_dynamic
  cloud_post_ddin_sar_filter_kernel_size: 5
```

机器可读差异应严格等于：

```json
{
  "model.cloud_post_ddin_sar_filter": {"s1": "none", "s2": "glfcr_dynamic", "allowed": true},
  "model.cloud_post_ddin_sar_filter_kernel_size": {"s1": null, "s2": 5, "allowed": true}
}
```

实际数据流：

```text
PixelUnshuffle
  -> optical/SAR stem
  -> ordinary DDIN, 3 steps
  -> pre_pda_context=none
  -> one-shot DFG(opt_private, sar_private)
  -> KernelConv2D(sar_private, dynamic_kernel)
  -> CAB1/CAB2/MSAB
  -> reconstruct
  -> PixelShuffle
```

S2 不使用：

- inside-DDIN GLF-CR；
- `GLFCRFusionStep`；
- optical-to-SAR reverse gate；
- FFC；
- spatial transform；
- Restormer；
- SA-Gate；
- SWT/chroma/stage/monotonic loss；
- GAN、perceptual、feature matching、KL；
- 新数据增强。

post filter 只由现有 Phase 1 实现构造一个 `GLFCRDynamicFilterGenerator`，不更新 optical-private，不调用完整 `GLFCRFusionStep`。

## 6. 结构检查工具

新增：

```text
scripts/inspect_glfcr_s1_s2_configs.py
```

默认模式不读取 checkpoint，只解析 YAML、构建模型、统计模块和参数、比较配置、比较同 seed 的 state-dict 初始化。默认强制 `--device cpu`；没有 `--smoke` 时显式使用 CUDA 会被拒绝。

CPU 结构检查命令由用户执行：

```bash
python scripts/inspect_glfcr_s1_s2_configs.py \
  --s1-config configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml \
  --s2-config configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml \
  --device cpu \
  --output-json /tmp/glfcr_phase3_structure_cpu.json
```

GPU smoke 命令由用户执行：

```bash
python scripts/inspect_glfcr_s1_s2_configs.py \
  --s1-config configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml \
  --s2-config configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml \
  --device cuda:1 \
  --smoke \
  --batch-size 1 \
  --height 256 \
  --width 256 \
  --output-json /tmp/glfcr_phase3_structure_gpu.json
```

只有 CPU 结构检查的配置差异、结构断言和初始化比较全部通过后，才应执行这条 GPU smoke 命令；若 CPU 检查返回 `fail_initialization`，应先处理共同初始化机制。

smoke 输入完全由脚本随机生成，不读取 dataset、manifest 或 checkpoint：

```text
RGB:        [B, 3, H, W]
SAR:        [B, 2, H, W]
cld_shdw:   [B, 4, H, W]
```

脚本检查 forward 输出、简单平方标量 loss、backward、输出和梯度 finite。

## 7. 必须统计的结构

脚本对 S1/S2 分别输出：

- PixelUnshuffle 数量；
- PixelShuffle 数量；
- DDIN 类型；
- 普通 DDIN step 数；
- `GLFCRCoupledDDINStep` 数；
- `GLFCRFusionStep` 数；
- 总 `GLFCRDynamicFilterGenerator` 数；
- post-DDIN filter 内 DFG 数；
- FFC block 数；
- spatial wrapper 数；
- 总参数量和可训练参数量；
- CAB attention mode；
- CAB2 residual source；
- CAB2 update scale；
- lowres/DDIN coupling 有效状态。

预期结构：

| 项目 | S1 | S2 |
|---|---:|---:|
| PixelUnshuffle | 1 | 1 |
| PixelShuffle | 1 | 1 |
| DDIN 类型 | `DDIN` | `DDIN` |
| 普通 DDIN steps | 3 | 3 |
| coupled DDIN steps | 0 | 0 |
| 完整 GLFCRFusionStep | 0 | 0 |
| post-DDIN DFG | 0 | 1 |
| FFC blocks | 0 | 0 |
| spatial wrappers | 0 | 0 |
| CAB | `standard` | `standard` |
| CAB2 residual | `reference` | `reference` |
| CAB2 update scale | 0.1 | 0.1 |

参数数值不在本阶段预先填入，因为任务禁止构建模型和执行脚本；由用户运行工具产生实际参数量。

## 8. 配置差异和初始化一致性

配置检查会对 YAML 做以下标准化：

- 字典键排序；
- list/tuple 统一为 list；
- path/root/cache/manifest 等路径展开和规范化；
- Phase 1 nullable compatibility keys 的缺省值展开；
- 再比较叶子字段。

允许差异严格限制为：

```text
model.cloud_post_ddin_sar_filter
model.cloud_post_ddin_sar_filter_kernel_size
```

任何其他差异都会使工具以 `fail_config_diff` 退出。

初始化比较流程：

1. 用相同 seed 构建 S1；
2. 重置相同 seed 构建 S2；
3. 比较 state-dict key 集合；
4. 输出 S1/S2 key 数、共有 key 数、两侧独有 key；
5. 对每个共有 tensor 记录最大绝对差异；
6. 输出全局最大绝对差异；
7. 超过 `1e-7` 时返回 `fail_initialization`，并输出：

```text
需要另行实现共同初始化机制，当前不得开始严格消融训练。
```

这里没有自动复制 S1 参数到 S2，也没有使用 `strict=False` 加载模型。原因是 S2 的 post-DDIN DFG 在 `DADIGANCloudBranch` 构造顺序中位于 DDIN 后、PDAFM/context 前，新增随机层可能消耗 RNG 序列，导致其后的共有模块即使代码完全相同也获得不同初始化。这是当前最重要的待验证风险，必须由用户执行结构脚本确认。

如果初始化比较失败，下一步应单独设计“共同初始化机制”，例如显式按模块名使用独立 seed，或让新增 post-filter 参数使用不改变已有模块初始化顺序的初始化流程；不能通过复制 checkpoint 或 `strict=False` 掩盖差异。本阶段没有实现该机制。

## 9. 待执行测试命令

本阶段只写测试，不执行。用户可在人工审查后运行：

```bash
python -m unittest -q \
  tests.test_glfcr_dadigan_phase1 \
  tests.test_glfcr_dadigan_evaluator_phase2 \
  tests.test_glfcr_dadigan_phase3_configs
```

Phase 3 测试覆盖：

1. 两份 YAML 可解析；
2. 两份 YAML 只在两个 post-filter 字段不同；
3. S1/S2 clean-core 配置值；
4. S1 普通 DDIN、3 steps、1 个 PixelUnshuffle/PixelShuffle；
5. S2 只增加一个 post-DDIN DFG；
6. S1/S2 不实例化 FFC、spatial wrapper、coupled DDIN step 和完整 GLFCRFusionStep；
7. CAB2 配置；
8. 检查工具可导入；
9. smoke 随机输入形状符合模型接口；
10. Phase 1 兼容测试文件仍存在且核心测试名称未被删除。

## 10. 尚未验证的风险

1. **共有初始化风险**：新增 S2 post-filter 可能改变后续模块的 RNG 消耗顺序；结构工具必须先确认，失败时不得开始严格 S1/S2 训练。
2. **S2 filter 数值稳定性**：动态核没有额外 softmax/归一化/单位核约束，符合现有 GLF-CR 定义，但训练时可能放大 SAR private feature；本阶段不做训练验证。
3. **显存和速度**：S2 在 128x128 low-resolution feature 上增加一次 DFG 和 KernelConv2D，实际代价由用户 smoke/训练日志确认；本阶段不估算实测显存。
4. **配置字段兼容性**：`cloud_lowres_enabled=true` 与旧 combined key=false 的解析依赖 Phase 1 nullable switch 逻辑；本阶段脚本会在用户执行时验证真实构造结果。
5. **loss profile 兼容性**：active loss 只有 `cloud_l1_missing`，其余权重为 0；训练入口仍选择 `dadigan_lama_ffc` 的 CloudOnly loss profile，这是为了复用现有 deterministic RGB reconstruction 计算路径，而不是引入 LaMa perceptual/GAN 监督。

## 11. 本阶段结论

Phase 3 的两个严格配置、结构检查工具、待执行测试和实现报告已经生成。由于任务明确禁止运行测试、构建模型、加载权重、使用 GPU、训练和评估，本报告不宣称 S1/S2 已通过结构或初始化验证。正确顺序是：用户先运行 CPU 结构检查；只有配置差异、结构断言和共有初始化全部通过后，再人工审查并决定是否进入 smoke 和训练。
