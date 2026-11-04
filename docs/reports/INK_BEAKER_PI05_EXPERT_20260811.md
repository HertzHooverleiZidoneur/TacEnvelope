# 墨水任务 Pi0.5 Envelope Expert 训练结果

日期：2026-08-11

## 数据审计

| 项目 | 结果 |
|---|---:|
| 原始匹配 episode | 291 |
| 显式排除坏 episode | 4（281 无 sidecar，未进入匹配集） |
| 进入零偏检查 | 287 |
| 通过首秒无接触零偏检查 | 282 |
| 合格 300 ms 窗口 | 21,585 |
| sustained-contact 窗口 | 8,180 |
| action 频率中位数 | 30.00 Hz |
| 左触觉频率中位数 | 22.19 Hz |
| 右触觉频率中位数 | 22.17 Hz |
| 每侧 300 ms 真实触觉样本 | 通常 6–7 个 |
| 超过 150 ms 的触觉断档 | 0 |

按 episode 划分后，sustained-contact 窗口为：train 5,714、validation 774、
calibration 799、test 893。对应 episode 数为 197、28、28、29。

## 模型

- 条件特征：墨水 Pi0.5 JAX checkpoint 同次前向的 `z_rl [1,512]`
- 状态：8D
- 动作：未来 9×8 实际执行命令，对应 300 ms
- 历史：左右各 6 个、共 200 ms 的因果触觉 token
- 主体：2 层、`d_model=256`、8 heads 的 Transformer Envelope Expert
- 可训练参数：2,253,318
- 最佳 epoch：40；第 55 epoch 早停
- Loss：左右独立 masked Gaussian NLL + `0.5 × SmoothL1`

## 独立 Test 结果

下表只统计 causal stable gate 已开启且未来真实标签大于 5 N 的持续接触窗口。

| 侧 | Test 样本 | MAE | RMSE | `mu + alpha*sigma` coverage | 平均宽度 | `mu + q95` coverage | 平均宽度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 左 | 674 | 0.709 N | 1.683 N | 95.25% | 1.141 N | 95.70% | 1.142 N |
| 右 | 851 | 0.838 N | 1.872 N | 94.01% | 1.151 N | 95.18% | 1.146 N |

Calibration 参数：

```text
alpha95_left  = 1.382965
alpha95_right = 1.111248
q95_left      = 1.142094 N
q95_right     = 1.146016 N
```

对于当前 checkpoint，加性 `mu + q95` 在几乎相同平均宽度下，让左右 test coverage 都
落在 95% 附近，适合作为后续 blind shadow 的首选对照。test 没有参与参数拟合或
`alpha/q95` 校准。

## 当前判定

离线训练、校准和 test 已跑通，但暂不标为 `production_ready`：

1. 还没有使用全新独立 episode 做 blind shadow evaluation。
2. 离线 300 ms 动作来自遥操实际命令，其中可能含接触后的人工反馈修正；在线候选 chunk
   则在锚点时已经给定，两者存在因果分布差异。
3. 墨水任务 Online RL 路径尚未接入本 checkpoint 和 200 ms 因果触觉历史。

因此当前模型可以用于离线分析和实机 shadow 记录，不应直接启用强触觉惩罚更新策略。

## 产物

```text
checkpoints/ink_beaker/current_stable_contact.pt
checkpoints/ink_beaker/current_calibration.json

outputs/training/franka_ink_beaker_tactile287/pi05_jax_zrl_envelope_v1/
  audit.json
  windows_summary.json
  causal_tactile_history.pt
  condition_tokens.pt
  sustained_contact_seed_44/
    transformer_force_envelope_model.pt
    transformer_training_result.json
    stable_evaluation/
      stable_contact_calibrated_checkpoint.pt
      stable_contact_calibration.json
      stable_contact_gated_evaluation.json
```
