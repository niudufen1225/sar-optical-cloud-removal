# GLF-CR x DADIGAN Phase 1 实现报告

## 1. 阶段结论

第一阶段“兼容性重构和单元测试”已完成，**未启动训练**。

本阶段实现了：

1. `PixelUnshuffle/stem/lowres context/reconstruction` 与 `DDIN/GLFCRCoupledDDIN` 的解耦。
2. CAB2 残差基底和 attention 增量尺度配置化。
3. DDIN 结束后的 one-shot SAR dynamic filter。
4. 旧配置兼容路径和标准库单元测试。

本阶段没有修改 dataset、manifest、预处理、loss、现有 YAML、outputs、checkpoint 或日志。

Gate 0 预审报告仍然适用，且没有触发停止条件。预审报告见：

`docs/glfcr_dadigan_preflight_audit.md`

## 2. 修改文件

实际由本阶段新增或修改的文件：

- `src/allclear/modules/dadigan.py`
- `src/allclear/modules/pvt_sra_cab.py`
- `src/allclear/model.py`
- `src/allclear/train.py`
- `tests/test_glfcr_dadigan_phase1.py`
- `docs/glfcr_dadigan_phase1_implementation_report.md`

`src/allclear/config.py` 已按要求审阅，但不需要修改。当前 main/v3/v4 YAML 保持原样，没有创建新的实验 YAML。

工作树中原有的 v3 训练日志、v4 输出、未提交配置、计划文档和 Gate 0 报告均保留，没有执行 reset、checkout、clean、stash、删除或覆盖操作。

## 3. 新配置键和兼容规则

### 3.1 低分辨率路径和 DDIN coupling

新增：

```yaml
model:
  cloud_lowres_enabled: null
  cloud_ddin_glfcr_coupled: null
```

解析规则：

```text
legacy = cloud_lowres_glfcr_coupled
lowres_enabled       = cloud_lowres_enabled       if not null else legacy
ddin_glfcr_coupled   = cloud_ddin_glfcr_coupled   if not null else legacy
```

`cloud_lowres_enabled` 独立控制：

- `PixelUnshuffle`；
- optical/SAR stem 的输入通道扩张；
- low-resolution optical context；
- reconstruction 中的 `PixelShuffle`。

`cloud_ddin_glfcr_coupled` 独立控制：

- `DDIN` 或 `GLFCRCoupledDDIN` 的选择。

因此以下组合现在有效：

```text
lowres_enabled=true,  ddin_glfcr_coupled=false
    PixelUnshuffle + 普通 DDIN + PixelShuffle

lowres_enabled=false, ddin_glfcr_coupled=false
    原分辨率 + 普通 DDIN + 普通 reconstruction
```

旧 YAML 没有新键时仍会同时得到原来的低分辨率路径和 coupled DDIN。旧属性 `lowres_glfcr_coupled` 保留为两个有效开关的 combined-state 视图，避免已有代码或诊断逻辑失效。

### 3.2 CAB2

新增：

```yaml
model:
  cloud_cab2_residual_source: query
  cloud_cab2_update_scale: 1.0
```

允许的 residual source：

- `query`：当前旧行为，使用 SAR-private `V` 作为 CAB2 residual base；
- `reference` 或 `fm`：使用 CAB1 输出 `FM` 作为 residual base。

CAB2 仍然保持：

```text
Q = V
K/V = FM
```

只有 residual base 和 attention 增量尺度改变。实现形式为：

```text
base = V                         # query 模式
base = FM                        # reference/fm 模式
F = base + update_scale * CAB_attention(V, FM)
F = F + MLP(Norm(F))
```

`cloud_cab2_update_scale` 是固定 Python float，不是 `nn.Parameter`，不会增加可训练参数。CAB1 没有改变。

### 3.3 one-shot post-DDIN SAR dynamic filter

新增：

```yaml
model:
  cloud_post_ddin_sar_filter: none
  cloud_post_ddin_sar_filter_kernel_size: null
```

模式：

- `none`：不实例化过滤模块，也不调用过滤器；
- `glfcr_dynamic`：DDIN 和 `pre_pda_context` 完成后，仅执行一次：

```text
K_T         = DFG(P_T, V_T)
V_filtered  = KernelConv2D(V_T, K_T)
```

实现类为 `GLFCRPostDDINSARFilter`，只复用：

- `GLFCRDynamicFilterGenerator`；
- `_glfcr_kernel2d_conv`。

它没有调用 `GLFCRFusionStep`，也没有：

- 更新 `P_T`；
- optical -> SAR 反向 gate；
- 每个 DDIN stage 重复调用；
- softmax、kernel normalization、单位核、tanh 或其他动态核约束。

过滤位置为：

```text
DDIN
  -> pre_pda_context
  -> one-shot SAR dynamic filter
  -> prefusion_context
  -> PDAFM
```

这与当前实际代码顺序兼容：`pre_pda_context` 先对 shared/P/V 做上下文处理，过滤后的 V 再进入已有 prefusion 和 PDAFM。`cloud_post_ddin_sar_filter_kernel_size: null` 时回退到 `cloud_lowres_glfcr_kernel_size`。

## 4. 代码级实现位置

### 4.1 `src/allclear/modules/dadigan.py`

- `PDAFMScale` 增加 `cab2_residual_source` 和 `cab2_update_scale`。
- 新增 `GLFCRPostDDINSARFilter`。
- `DADIGANCloudBranch` 增加 nullable lowres/DDIN 开关、CAB2 选项和 post filter 选项。
- lowres 相关构造和 forward 条件统一改为 `self.lowres_enabled`。
- DDIN 类型选择统一改为 `self.ddin_glfcr_coupled`。
- post filter 位于 `pre_pda_context` 后、`prefusion_context` 前。

### 4.2 `src/allclear/modules/pvt_sra_cab.py`

`SRACAB.forward` 从：

```python
forward(private_query, reference_feat)
```

扩展为：

```python
forward(
    private_query,
    reference_feat,
    residual_base=None,
    update_scale=1.0,
)
```

默认参数保持旧行为，并对 residual shape、尺度非负性和有限性进行检查。

### 4.3 `src/allclear/model.py`

新键已透传到：

- `DADIGANBaseline`；
- `AllClearTGDADSoftShadow` 内部的 `DADIGANCloudBranch`。

没有改变 SoftShadow 分支自身的模块、loss 或数据处理。

### 4.4 `src/allclear/train.py`

新增 `optional_bool`，确保 YAML 中的 `null` 不会被错误转换成 `False`。新键透传到 DADIGAN baseline 和普通 Stage1 模型路径；旧键仍保留原默认值。

## 5. 单元测试

新增文件：

`tests/test_glfcr_dadigan_phase1.py`

测试内容共 10 项：

1. main_v2、v3、v4 旧 YAML 均可构建，且仍使用 `GLFCRCoupledDDIN`。
2. legacy-only 与显式 `true/true` 的 state-dict key、参数初始化和 eval forward 完全一致。
3. `lowres_enabled=true + ddin_glfcr_coupled=false` 使用 PixelUnshuffle、普通 DDIN、PixelShuffle。
4. `lowres_enabled=false + ddin_glfcr_coupled=false` 使用原分辨率普通 DDIN，stem 不扩展 factor² 通道。
5. CAB2 默认 query 与显式 query/scale=1.0 的前向一致。
6. attention 增量置零时，reference/FM 模式以 FM 为输出基底。
7. reference/FM 模式下 SAR query 保持非零梯度。
8. post filter 的 k=3、k=5 每次 forward 各调用一次 DFG 和一次 KernelConv2D，且不实例化完整 GLF fusion。
9. post filter 不修改 `P_T`，forward/backward 输出和梯度 finite；kernel size null 正确回退。
10. 小尺寸 S1 smoke forward/backward 中输出和梯度均 finite，且没有 coupled GLF、FFC 或 spatial wrapper。

执行命令：

```bash
python -m unittest -q tests.test_glfcr_dadigan_phase1
```

结果：

```text
Ran 10 tests in 9.567s
OK
```

附加检查：

```bash
python -m compileall -q src/allclear tests/test_glfcr_dadigan_phase1.py
git diff --check -- src/allclear/modules/dadigan.py \
  src/allclear/modules/pvt_sra_cab.py \
  src/allclear/model.py src/allclear/train.py
```

两项均通过。

当前环境没有安装 `pytest`。尝试执行 `python -m pytest ...` 得到 `No module named pytest`，因此没有把测试结果伪装成 pytest 结果；本阶段使用 Python 标准库 unittest 完成了同等测试。

## 6. 参数量和模块计数

参数量只统计 generator/model，不含 discriminator、HRF perceptual network、optimizer state 和 activation memory。

### 6.1 旧配置兼容计数

| 配置 | 参数量 | 普通 DDIN step | `GLFCRCoupledDDINStep` | `GLFCRDynamicFilterGenerator` | FFC block | spatial wrapper |
|---|---:|---:|---:|---:|---:|---:|
| main_v2 legacy | 36,427,829 | 0 | 4 | 4 | 15 | 6 |
| v3 legacy | 36,427,829 | 0 | 4 | 4 | 15 | 6 |
| v4 legacy | 20,964,290 | 0 | 3 | 3 | 6 | 2 |

这些数量与 Gate 0 预审记录一致，说明旧配置没有因新增接口意外改变模块结构。

### 6.2 Phase 1 内存中 S1/S2 结构计数

基于 v4 的 RGB/SAR/dim 设置，仅在内存中覆盖 Phase 1 结构键，没有创建 YAML 或训练：

| 结构 | 参数量 | 普通 DDIN step | coupled step | DFG | FFC block | spatial wrapper |
|---|---:|---:|---:|---:|---:|---:|
| S1 clean core | 11,341,442 | 3 | 0 | 0 | 0 | 0 |
| S2 + one-shot filter | 12,072,386 | 3 | 0 | 1 | 0 | 0 |

S2 相对 S1 新增的主要参数是一次 DFG；KernelConv2D 为无参数运算。CAB2 reference/FM 选项不增加参数。

## 7. 兼容性与风险审查

### 已确认

- 旧 main_v2/v3/v4 配置不需要增加任何新键。
- 旧配置仍选择原来的 coupled DDIN、PixelUnshuffle、lowres FFC、PixelShuffle 和模块数量。
- CAB2 默认 `query + 1.0`，无新增参数和 state-dict key。
- post filter 默认 `none`，旧配置不会实例化过滤器，因此不会增加旧 checkpoint 的 state-dict 分支。
- 新键没有触碰 dataset、manifest、数据预处理和 loss。
- S1 结构检查确认没有 `GLFCRCoupledDDINStep`、FFC 或 spatial wrapper。
- reference/FM 模式没有切断 SAR query 梯度。

### 尚未做的事情

- 没有启动 S1/S2 正式训练。
- 没有修改或生成 S1/S2 YAML。
- 没有加载或覆盖现有 checkpoint。
- 没有验证训练后的视觉指标、显存和吞吐量。
- 没有实现计划中后续 F1 SWT loss 或 F2 stage deep supervision。

### 需要人工审查的设计点

1. `cloud_cab2_update_scale` 当前是固定 float；后续若需要学习尺度，必须作为独立实验，不应隐式改变本阶段结论。
2. `cloud_lowres_enabled=false + cloud_ddin_glfcr_coupled=true` 虽然接口支持，但会在原分辨率执行 coupled GLF，显存和计算成本可能很高，本阶段没有训练它。
3. S2 的 DFG 仍以 `(P_T,V_T)` 生成动态核，这是公开 GLF-CR DFG 的实际输入要求；它只过滤 V，不执行双向 gate。
4. 当前没有逐 stage kernel telemetry；本阶段只实现结构和调用次数，后续评估脚本再决定记录方式。

## 8. 停止条件复核

| 停止条件 | 结果 |
|---|---|
| 修改 dataset/manifest | 未发生 |
| 旧配置不能兼容 | 未发现；10 项测试通过 |
| 同结构必须依赖 `strict=False` | 未改变 checkpoint loader；legacy state-dict 结构检查通过 |
| S1 仍实例化 coupled step/FFC/spatial | 未发生；S1 结构测试通过 |
| CAB2 reference 使 SAR query 无梯度 | 未发生；梯度测试通过 |
| Gate 0 报告与当前代码不一致 | 未发现；本报告记录了新增实现后的状态 |

## 9. 阶段后建议

本阶段可以停止并等待人工审查。下一阶段若获准，应先新增独立的 S1/S2 YAML，并保持：

- 同一 seed、manifest、batch size、学习率和评价脚本；
- S1/S2 从同一随机初始化分别训练；
- GAN、feature matching、perceptual、KL 按计划关闭；
- 不使用当前 v3/v4 checkpoint 作为严格消融起点；
- 训练前先增加 config/state-dict 结构签名检查。

本阶段没有训练指令，因为用户明确要求“不要训练”。
