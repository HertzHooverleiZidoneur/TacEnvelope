# 糕点任务 Pi0.5 Envelope Expert 训练结果

日期：2026-08-24

## 数据审计

| 项目 | 结果 |
|---|---:|
| A 类 episode | 298 |
| 合格 300 ms 窗口 | 13,572 |
| 缓存的持续接触候选窗口 | 2,522 |
| action 频率中位数 | 30.00 Hz |
| 三路相机频率 | 约 30 Hz |
| 左触觉频率中位数 | 22.17 Hz |
| 右触觉频率中位数 | 22.16 Hz |
| 每侧 300 ms 真实触觉样本 | 通常 6-7 个 |
| 因果历史时间戳违规 | 0 |

按 episode 切分为 209 条 train、30 条 validation、30 条 calibration、29 条 test。
持续接触训练/验证/校准/测试样本数分别为 1,743、290、240、249；左右 gate 评估时按
侧独立筛选，因此最终左右样本数不同。

## 模型

- 冻结特征：糕点 Pi0.5 JAX checkpoint 同次前向的 `z_rl [1,512]`
- 机器人状态：8D，独立输入
- 动作：未来 9×8 实际执行命令，对应 300 ms
- 触觉历史：左右各最多 6 个真实样本，共 200 ms，不插值
- 主体：2 层、`d_model=256`、8 heads、FFN 1024 的 Transformer
- 输出：左右独立 `mu` 和 `log_var`
- 参数：2,253,318
- Loss：masked Gaussian NLL + `0.5 × SmoothL1`
- 最佳 epoch：13；第 28 epoch 早停

## 稳定接触 Test

下表只统计锚点前因果 stable-contact gate 已开启，且未来真实标签大于 `1 N` 的窗口。
`alpha95` 只在 calibration split 上估计。

| 侧 | Calibration | Test | MAE | RMSE | alpha95 | Coverage | 平均上包络宽度 | 超界窗口 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 左 | 171 | 166 | 0.697 N | 1.112 N | 1.8087 | 95.78% | 1.500 N | 7 |
| 右 | 201 | 237 | 0.663 N | 1.092 N | 1.4445 | 95.78% | 1.438 N | 10 |

加性 `mu + q95` 的 coverage 为左 `92.77%`、右 `91.56%`，低于目标；当前选择
`mu + alpha95 * sigma`。首秒无接触基线残差 P99 约为左 `0.182 N`、右 `0.141 N`，
明显低于 `tau_on=1.0 N`。

### 双侧平均力加性校准

双侧都满足因果 stable-contact gate 时，定义：

```text
y_avg  = (y_left + y_right) / 2
mu_avg = (mu_left + mu_right) / 2
r      = y_avg - mu_avg
```

仅使用 calibration split 的 147 个有效窗口计算残差分位数：

| 分位数 | 校准值 |
|---|---:|
| q05 | -1.642 N |
| q10 | -1.381 N |
| q90 | 0.250 N |
| q95 | 0.861 N |

`U90=mu_avg+q90` 和 `U95=mu_avg+q95` 是两个独立的单侧上界测试，不能分别与任意下界
混写成标准区间：

| 单侧上界 | Test coverage | Test 超上界窗口 |
|---|---:|---:|
| `U90 = mu_avg + 0.250 N` | 85.63% | 23/160 |
| `U95 = mu_avg + 0.861 N` | 90.63% | 15/160 |

对称尾概率区间的重新测试结果如下：

| 中央区间 | Test coverage | 低于下界 | 高于上界 | 总超界 | 平均宽度 |
|---|---:|---:|---:|---:|---:|
| `[L10,U90]`，名义 80% | 76.88% | 14/160 | 23/160 | 37/160 | 1.632 N |
| `[L05,U95]`，名义 90% | 86.25% | 7/160 | 15/160 | 22/160 | 2.502 N |

因此论文主结果使用标准的 `[L05,U95]`，不再使用 `[L05,U90]`。测试集没有用于重新
调整任何分位数。糕点的 U90、U95 和中央区间测试 coverage 均低于名义值，说明存在
calibration-to-test 分布偏移。

## 判定

当前 checkpoint 已完成离线训练、validation 选模、calibration 校准和 test 评估，左右
稳定接触 coverage 都位于目标 `93%-97%` 内。它可以用于实机 shadow 记录和低权重奖励
联调，但暂不标记为 production-ready：

1. 尚未用新采集的独立 episode 做 blind shadow evaluation。
2. 离线未来动作来自遥操实际命令，可能包含接触后的人工反馈修正；在线候选 chunk 在锚点
   已经确定，存在因果分布差异。
3. 在线路径必须同时提供同次 Pi0.5 `z_rl`、未来 9 步候选动作和过去 200 ms 原始触觉历史。

Test targets 没有用于调整 `alpha95`、`q90` 或 `q95`。

## 稳定入口

```text
checkpoints/pastry/current_stable_contact.pt
checkpoints/pastry/current_calibration.json
```

完整产物位于：

```text
outputs/training/franka_pastry_tactile_A298/pi05_jax_zrl_envelope_v1/
```
