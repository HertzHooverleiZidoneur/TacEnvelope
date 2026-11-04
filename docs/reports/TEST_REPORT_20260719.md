# Franka 触觉力包络工程测试报告

- 报告日期：2026-07-19
- 数据集：`franka_usb_tactile_310_20260716`
- 工程：`~/Documents/tactile_force_envelope`
- 主要产物：`artifacts/franka_tactile310/force_envelope_transformer_v3`
- 任务：`Insert the red USB flash drive into the USB port on the black power strip`

## 1. 结论摘要

| 项目 | 状态 | 结论 |
| --- | --- | --- |
| 数据字段与时间戳审计 | 通过 | 已识别 3 路相机、8 维状态、8 维动作和左右触觉原始样本；时间窗可构建 |
| 300 ms 异步对齐 | 通过 | 直接使用窗口内真实触觉点，无插值；低质量窗口会被拒绝 |
| 左右独立标签 | 通过 | 左右分别计算 `raw_max` 与 `filtered_top2_mean`，训练时不先合并左右 |
| SmolVLA 条件特征缓存 | 通过 | 使用实际 checkpoint 和动作生成前的多模态 condition tokens，未退回 ResNet18 |
| Transformer Expert 张量接口 | 通过 | condition/action/state/query/mask 的 shape gate 和单元测试均通过 |
| 接触权重消融 | 通过 | 完成 `1.0/1.5/2.0`、每组 3 个随机种子的比较；`1.5` 最稳定 |
| 左右接触条件校准 | 部分通过 | 右侧达到目标；左侧测试集覆盖率仍低于目标 |
| 因果触觉历史输入 | 明显改善 | 左右接触 MAE 分别下降 67.3% 和 66.0%，包络显著收窄 |
| 在线推理接口 smoke test | 通过 | 模型可在 CUDA 上加载，并对真实缓存样本返回有限的左右 `mu/sigma/upper95` |
| Online RL 正式使用 | 未就绪 | 左侧接触覆盖不足，且候选动作的在线时序契约仍需验证 |

当前最合理的实验模型是“**SmolVLA condition tokens + 200 ms 严格因果触觉历史 + 300 ms 候选动作**”版本。它显著提高了点预测精度并缩小包络，但左侧接触覆盖率只有 **86.92%**，未达到目标 93%–97%，因此不能把它当作正式安全上界直接高权重接入 Online RL。

## 2. 数据审计

### 2.1 数据规模和划分

- 原始采集：310 episodes，来自 4 个采集批次。
- 质量过滤后：304 episodes，12,100 个 300 ms 窗口。
- Train：8,368 windows。
- Validation：1,271 windows，30 episodes。
- Calibration：1,234 windows，30 episodes。
- Test：1,227 windows，31 episodes。
- 划分以 episode 为单位，集合互斥，避免相邻窗口泄漏。

### 2.2 已确认字段

- 时间戳：`timestamp`
- 机器人状态：`observation.state`，8 维
- 动作：`action`，8 维
- 任务文本：`meta/episodes.jsonl`
- 相机顺序：
  1. `observation.images.camera_camera_1_color_image_raw`
  2. `observation.images.camera_camera_2_color_image_raw`
  3. `observation.images.d515_color_image_raw`
- 触觉：左右传感器分别保留原始异步样本与时间戳。

数据中没有真实的 action-chunk boundary 字段，因此当前离线窗口使用固定 stride 锚点。该做法适合离线基线实验，但在线部署时应由真实推理事件提供 `t_k` 和实际候选 chunk。

### 2.3 300 ms 窗口质量

| 指标 | 数值 |
| --- | ---: |
| 左右合计触觉样本数最小值 | 8 |
| P05 | 12 |
| 中位数 | 13 |
| P95 | 14 |
| 最大值 | 20 |
| 平均值 | 13.27 |
| 单侧样本数中位数 | 7 |
| 单侧最小样本数 | 4 |
| 最大采样间隔中位数 | 60.01 ms |
| 最大采样间隔 P95 | 69.33 ms |
| 最大采样间隔最大值 | 80.60 ms |

构造过程中拒绝的主要窗口：D515 图像过旧 1,031 个、camera 2 图像过旧 576 个、camera 1 图像过旧 409 个、未来窗口不完整 304 个、无锚点前图像 228 个、触觉样本不足 60 个、零偏基线无法确认 6 个。

### 2.4 时间戳检查

- 三路相机的已审计 timestamp rollback 均为 0。
- camera 1 图像年龄中位数约 50.7–51.6 ms。
- camera 2 图像年龄中位数约 52.5–55.1 ms。
- D515 图像年龄中位数约 22.9–26.4 ms，但原始数据中存在最高约 281–346 ms 的旧帧；构建时以 70 ms 阈值过滤。
- 触觉原始 sidecar 存在少量 timestamp rollback：四批数据左右分别为 `(2,3)`、`(0,0)`、`(1,1)`、`(5,5)`；窗口构建按时间排序并保留审计记录。
- 审计脚本曾存在“只报告最后一路相机 rollback，并可能重复报告”的变量作用域 bug；已修复并增加无相机和多相机回归测试。

## 3. 标签与监督信号测试

每个触觉点计算三轴合力：

```text
force_magnitude = sqrt(Fx^2 + Fy^2 + Fz^2)
```

左右传感器分别执行零偏处理和轻度滤波，并分别保存：

- `target_left_raw_max` / `target_right_raw_max`
- `target_left_top2_mean` / `target_right_top2_mean`

默认软包络监督使用左右独立的 `filtered_top2_mean`；`raw_max` 保留给硬安全监测。单元测试确认 top-2 在每个传感器内部计算，不会先把左右样本混在一起。

当前训练损失为：

```text
weighted Gaussian NLL + 0.5 * weighted SmoothL1(mu, target)
```

窗口中任一侧超过 train split 计算得到的接触阈值，即整个窗口使用统一接触权重。零力和非接触样本没有被删除。

## 4. SmolVLA 条件特征测试

### 4.1 Checkpoint 与特征位置

- Checkpoint：`lerobot/outputs/train/franka_tactile310_smolvla_b64_e10_20260717_152937/checkpoints/022123/pretrained_model`
- Checkpoint SHA256：`5e93bc79eaad6f03057dc6a136e7dd372011e872407742322455c07c860cad43`
- Feature tap：`policy.model.vlm_with_expert.get_vlm_model().text_model.layers[-1].input_layernorm`
- 语义：最后一层 Q/K/V 之前、动作生成之前的 multimodal prefix hidden state。
- 包含语言：是。
- 包含机器人状态：是。
- Pooling：无，保留 token 维度。
- 缓存 shape：`[12100, 241, 960]`
- Mask shape：`[12100, 241]`
- 缓存 dtype：`float16`

图像使用 SmolVLA 自身预处理：三路图像按既定顺序输入，resize-with-pad 到 `512x512`，像素从 `[0,1]` 映射到 `[-1,1]`。未使用 ImageNet normalization，也没有自动退回 ResNet18。

### 4.2 缓存一致性

缓存元数据记录 checkpoint、hash、feature tap、相机顺序、图像预处理和 feature shape。窗口缓存 hash 与触觉历史缓存匹配，配置变化时不会静默复用不匹配缓存。

## 5. Transformer Envelope Expert 结构测试

默认网络结构：

- SmolVLA condition tokens：`Linear(960,256) + LayerNorm`，作为 cross-attention K/V。
- 机器人状态：8 维编码为 1 个 state token。
- 动作：未来 300 ms 的 `9x8` 动作编码为 9 个 action tokens，带 learned positional embedding 和 mask。
- 查询：两个 learned query tokens，`LEFT_ENV` 和 `RIGHT_ENV`。
- Decoder：2 层 pre-norm Transformer decoder，`d_model=256`、8 heads、FFN 1024、dropout 0.1。
- 输出：左右各自的 `mu` 和 `log_var`；`log_var` clamp 到 `[-8,4]`。

无历史版本 shape gate：

| 张量 | Shape |
| --- | --- |
| condition raw | `[128,241,960]` |
| condition projected | `[128,241,256]` |
| action tokens | `[128,9,256]` |
| state token | `[128,1,256]` |
| query tokens | `[128,2,256]` |
| decoder input/output | `[128,12,256]` |

无历史版本可训练参数量：2,362,886。

## 6. 接触权重消融实验

每个配置使用 seeds 42、43、44；最多 80 epochs，前 20 epochs 不允许 early stopping，patience 15，以加权 validation loss 选择最佳 checkpoint。

| 权重 | 左接触 MAE | 左覆盖率 | 右接触 MAE | 右覆盖率 | 最佳 epoch 均值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 7.379 ± 0.898 | 80.78% ± 4.93% | 7.454 ± 1.071 | 97.85% ± 1.77% | 31.3 |
| 1.5 | **6.497 ± 0.516** | **80.78% ± 3.46%** | **6.501 ± 0.377** | **98.04% ± 0.24%** | 47.3 |
| 2.0 | 7.431 ± 1.242 | 77.51% ± 4.24% | 7.770 ± 1.589 | 95.46% ± 0.49% | 22.0 |

结论：`contact_weight=1.5` 的左右 MAE 和跨种子稳定性最好，因此保留 seed 44 checkpoint 做后续校准和历史输入对比。继续增大到 2.0 没有解决左侧覆盖问题，反而降低总体稳定性。

## 7. 无历史模型的接触条件校准

使用 calibration split 中每一侧真实接触样本，分别计算单侧有限样本 conformal quantile。校准集没有参与网络权重更新。

| 侧别 | contact alpha95 | Test 接触 MAE | 校准前覆盖率 | 校准后覆盖率 | 校准后 margin | Test oracle alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Left | 1.2458 | 5.770 | 84.99% | **88.56%** | 9.802 | 1.6680 |
| Right | 1.4066 | 6.938 | 97.70% | **98.28%** | 13.825 | 1.0351 |

左侧 calibration 的标准化残差 P95 为 1.2077，而 test 为 1.6497，说明 calibration 到 test 存在明显分布偏移。左侧 test 接触力也更大：calibration target 均值 39.94、P95 51.75；test 均值 44.46、P95 61.01。问题不是简单的采样点数量不足。

注意：左侧覆盖率低代表**安全上界漏覆盖、可能少惩罚**，不是“假阳性惩罚过多”。

## 8. 严格因果触觉历史实验

### 8.1 历史输入定义

- 历史区间：`[t_k - 200 ms, t_k)`，严格不包含当前及未来触觉。
- 每侧最多保留最近 6 个真实样本，不插值，不足时左侧 padding。
- 每个 token：`[Fx,Fy,Fz,magnitude,relative_time]`，共 5 维。
- 左右合计 12 个历史 tokens，并保留独立 mask。
- 因果违规检查：0。
- 每侧每窗口历史样本数：最小 0、P05 4、中位数 4、P95 5、最大 6、均值 4.425。

加入历史后的 decoder shape 为 `[128,24,256]`，可训练参数量 2,368,006。除了历史 tokens，split、seed 44、接触权重 1.5 和 Transformer 核心保持一致。

### 8.2 直接对比

| 侧别 | 配置 | Test 接触 MAE | 接触覆盖率 | Sigma | Envelope margin |
| --- | --- | ---: | ---: | ---: | ---: |
| Left | 无历史 | 5.770 | 88.56% | 7.868 | 9.802 |
| Left | 200 ms 因果历史 | **1.887** | 86.92% | **2.469** | **2.575** |
| Right | 无历史 | 6.938 | 98.28% | 9.828 | 13.825 |
| Right | 200 ms 因果历史 | **2.359** | **95.27%** | **3.166** | **4.976** |

- 左侧接触 MAE 降低 67.29%，margin 缩小 73.73%。
- 右侧接触 MAE 降低 65.99%，margin 缩小 64.01%。
- 右侧达到 93%–97% 目标区间。
- 左侧点预测显著改善，但 sigma 变得过窄，覆盖率仍只有 86.92%。

### 8.3 按因果接触状态分析

| 侧别 | 状态 | N | Coverage | MAE | Bias | Sigma | Margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Left | 历史已有接触 | 652 | 87.88% | 1.442 | -0.011 | 2.086 | 2.175 |
| Left | 新接触 onset | 21 | 57.14% | 15.719 | 10.777 | 14.378 | 14.992 |
| Left | 未来无接触 | 554 | 92.24% | 0.642 | 0.119 | 1.093 | 1.140 |
| Right | 历史已有接触 | 674 | 95.55% | 1.870 | -0.461 | 2.543 | 3.996 |
| Right | 新接触 onset | 23 | 86.96% | 16.714 | 11.672 | 21.437 | 33.688 |
| Right | 未来无接触 | 530 | 97.17% | 0.691 | -0.005 | 1.118 | 1.756 |

结论：因果历史对“接触已发生”的窗口非常有效，但新接触 onset 仍是最难部分；左侧同时存在更明显的 calibration/test 尾部分布偏移。因此下一步应优先尝试按“历史是否已有接触”做 Mondrian calibration 或按 episode/block 做 conformal，而不是继续扩大网络或只增加接触权重。

## 9. 在线推理接口 Smoke Test

已加载因果历史校准 checkpoint，在 CUDA 上使用真实缓存 test 样本调用在线 predictor。接口要求：

- condition tokens 和候选 action chunk 必须来自同一次 SmolVLA forward；
- 动作 shape 为固定窗口内的候选执行动作；
- 触觉历史 shape 为 `[2,6,5]`，mask 为 `[2,6]`；
- 缺少历史输入时主动报错，不静默填充未来信息。

一次 smoke 输出：

```text
mu_left       = -0.08360
sigma_left    =  0.71876
upper95_left  =  0.66587
mu_right      = -0.00012
sigma_right   =  0.66013
upper95_right =  1.03725
```

所有输出均为有限值，模型加载、张量路径和校准参数读取正常。该测试只证明接口可运行，不代表真机闭环安全性已经验证。

## 10. 自动化测试结果

执行命令：

```bash
cd ~/Documents/tactile_force_envelope
PYTHONPATH=. ~/Documents/lerobot/.venv/bin/python -m unittest discover -s tests -v
```

结果：**15/15 passed**。

覆盖内容：

1. 每路相机 rollback 只报告一次。
2. 无相机数据时 audit 不会触发 `NameError`。
3. 左右接触条件 alpha 只使用对应侧接触样本。
4. 有限样本 conformal quantile 实现。
5. Train/validation/calibration/test episode 划分互斥且完整。
6. 分组 coverage 统计。
7. 触觉标签只使用真实样本，不插值。
8. 左右分别计算 top-2 后再形成全局统计。
9. 接触权重作用于整个窗口。
10. KV pooling 排除 padding 并保持层顺序。
11. Masked mean 排除 padding。
12. 模型输出与 Gaussian NLL。
13. SmolVLA 特征已含语言时可关闭重复 task branch。
14. Transformer 接受带 mask 的严格因果触觉历史 tokens。
15. Transformer 左右输出和 mask shape。

Python `compileall` 也已通过。

## 11. 当前限制与风险

1. **左侧上包络未达标。** 历史模型左侧 test 接触 coverage 为 86.92%，不可宣称 95% 安全覆盖。
2. **Test split 已用于诊断和方案选择。** 完成下一轮校准方法后，需要另留新的盲测 episodes 做最终报告。
3. **离线未来动作存在信息契约风险。** 当前数据中的未来遥操动作可能包含接触后的人工反馈修正；在线 `t_k` 时刻并不知道这些修正。正式训练应使用同一次 SmolVLA forward 产生、在锚点时已知的候选 action chunk，或者记录实际推理/执行事件。
4. **缺少真实 chunk boundary。** 当前固定 stride 窗口不是实际 controller chunk 事件。
5. **新接触 onset 样本少且误差大。** 左右 onset 测试样本分别只有 21 和 23 个，需增加独立 onset 数据或使用分组校准。
6. **零偏前提依赖无接触状态。** 当前有 6 个 episode 因无法确认 baseline 而被拒绝；以后采集应明确记录零偏阶段。
7. **尚未进行真机 Online RL 闭环测试。** 当前只完成 envelope 模型、离线评估和在线函数级 smoke test。

## 12. 建议的下一步验收顺序

1. 保留当前历史模型 checkpoint，不重新训练网络。
2. 对历史已有接触、无历史接触 onset 分组做单侧 Mondrian calibration。
3. 使用 episode/block conformal，降低相邻窗口相关性造成的过度乐观。
4. 新采一批完全隔离的盲测 episodes，重点增加左侧高力和新接触 onset。
5. 在 VLA 推理端记录真实 `t_k`、同次 condition tokens、候选 chunk、实际发送 chunk 和触觉窗口。
6. 仅在左右盲测接触 coverage 都进入 93%–97%，且 margin 可接受后，再将触觉惩罚以较小权重接入 Online RL。

## 13. 关键产物索引

- 窗口审计：[windows_summary.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/windows_summary.json)
- Episode 划分：[episode_splits.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/episode_splits.json)
- SmolVLA 特征元数据：[condition_tokens.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/condition_tokens.json)
- 接触权重汇总：[contact_weight_sweep_summary.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/contact_weight_sweep_summary.json)
- 无历史接触校准：[transformer_contact_calibration_report.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/contact_recalibration_weight_1.5_seed_44/transformer_contact_calibration_report.json)
- 因果历史审计：[causal_tactile_history_report.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/causal_tactile_history_report.json)
- 历史消融对比：[history_ablation_comparison.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/tactile_history_ablation/weight_1.5_seed_44/history_ablation_comparison.json)
- 历史模型接触校准：[transformer_contact_calibration_report.json](../../artifacts/franka_tactile310/force_envelope_transformer_v3/tactile_history_ablation/weight_1.5_seed_44/contact_recalibration/transformer_contact_calibration_report.json)
- 历史模型 checkpoint：[transformer_force_envelope_contact_calibrated.pt](../../artifacts/franka_tactile310/force_envelope_transformer_v3/tactile_history_ablation/weight_1.5_seed_44/contact_recalibration/transformer_force_envelope_contact_calibrated.pt)

## 14. 最终判断

这轮实验已经验证了三件事：数据管线可用、SmolVLA token-level Expert 可训练、严格因果触觉历史对未来力预测有显著帮助。当前最主要的问题已从“模型预测不准”缩小为“左侧接触尾部的校准与分布漂移”。

因此，本工程可以继续作为触觉 envelope 的研究与 Online RL 接口开发基础；当前 checkpoint 适合离线分析和低风险联调，但尚不应作为正式的 95% 安全上包络投入真机高权重惩罚。
