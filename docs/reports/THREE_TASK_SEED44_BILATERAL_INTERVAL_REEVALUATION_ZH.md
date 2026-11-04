# 三任务 Seed-44 双侧稳定接触包络重测

日期：2026-09-07

## 1. 统计口径

本报告固定使用 USB 插入、挤墨水和夹取糕点三个任务各自已有的 seed-44 episode split
与 checkpoint，不重新训练模型，不改变 episode 划分。

对 calibration 和 test 均仅保留左右两侧因果 stable-contact gate 同时开启的窗口：

```text
valid = stable_contact_left AND stable_contact_right
```

筛选不读取未来触觉标签，也没有额外使用 `target > tau_train` 过滤窗口。双侧真实力、预测
均值和 calibration 残差定义为：

```math
y_{avg}=\frac{y_L+y_R}{2},\qquad
\mu_{avg}=\frac{\mu_L+\mu_R}{2},\qquad
r=y_{avg}-\mu_{avg}.
```

所有 `q` 只由 calibration split 的有效窗口计算。测试标签只用于最终指标，未用于重新
拟合或调整分位数。

中心 90% 和中心 95% 区间分别定义为：

```math
I_{90}(x)=[\mu_{avg}+q_{05},\ \mu_{avg}+q_{95}],
```

```math
I_{95}(x)=[\mu_{avg}+q_{2.5},\ \mu_{avg}+q_{97.5}].
```

本文中的 bias 统一定义为 `mean(y_avg-mu_avg)`：正值表示模型平均低估真实力，负值表示
模型平均高估真实力。Ground-truth std 使用总体标准差 `ddof=0`。

## 2. 有效数据量

| 任务 | Calibration split episode | Calibration 有效 episode | Calibration 窗口 | Test split episode | Test 有效 episode | Test 窗口 |
|---|---:|---:|---:|---:|---:|---:|
| USB 插入 | 30 | 30 | 580 | 31 | 31 | 598 |
| 挤墨水到烧杯 | 28 | 28 | 578 | 29 | 29 | 642 |
| 夹取糕点到蓝盒 | 30 | 25 | 147 | 29 | 25 | 160 |

“有效 episode”表示该 episode 至少包含一个左右 stable-contact gate 同时开启的窗口。
糕点任务有部分 calibration/test episode 没有满足条件的双侧稳定接触窗口，因此有效
episode 少于 split 中的总 episode 数。

## 3. Calibration 残差分位数

单位均为 N。

| 任务 | q2.5 | q05 | q95 | q97.5 |
|---|---:|---:|---:|---:|
| USB 插入 | -2.567 | -1.706 | 2.838 | 5.304 |
| 挤墨水到烧杯 | -4.133 | -2.474 | 0.893 | 2.094 |
| 夹取糕点到蓝盒 | -2.043 | -1.642 | 0.861 | 1.081 |

这些分位数直接作用于每个窗口的 `mu_avg`。例如 USB 的中心 90% 区间为
`[mu_avg-1.706 N, mu_avg+2.838 N]`；中心 95% 区间为
`[mu_avg-2.567 N, mu_avg+5.304 N]`。上下偏移不必数值对称，因为 calibration 残差分布
可以偏斜；“中心”指两侧尾概率相同，而不是上下宽度相等。

## 4. Test 结果

| 任务 | 有效 episode | 窗口 | GT mean (N) | GT std (N) | GT P95 (N) | Bias (N) | MAE (N) | RMSE (N) | MAE/mean | RMSE/mean | 90% coverage | 90% width (N) | 95% coverage | 95% width (N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USB 插入 | 31 | 598 | 51.018 | 11.559 | 67.866 | +0.118 | 1.323 | 3.210 | 2.59% | 6.29% | 88.46% | 4.544 | 94.48% | 7.871 |
| 挤墨水到烧杯 | 29 | 642 | 20.194 | 10.860 | 37.196 | -0.402 | 0.802 | 1.931 | 3.97% | 9.56% | 88.47% | 3.366 | 93.46% | 6.228 |
| 夹取糕点到蓝盒 | 25 | 160 | 3.479 | 1.821 | 7.157 | -0.169 | 0.600 | 0.928 | 17.26% | 26.67% | 86.25% | 2.502 | 90.00% | 3.124 |

区间宽度由 calibration 分位数之差确定，因此同一任务每个有效 test 窗口的加性区间宽度
相同；表中 mean width 即该固定宽度。

## 5. 结果解读

### USB 插入

- 点预测误差最低：MAE 为平均接触力的 2.59%，RMSE 为 6.29%。
- 中心 90% 区间实际覆盖 88.46%，比名义值低 1.54 个百分点。
- 中心 95% 区间实际覆盖 94.48%，比名义值低 0.52 个百分点，是三项任务中校准最好的。

### 挤墨水到烧杯

- MAE 为平均接触力的 3.97%，模型平均高估 0.402 N。
- 中心 90% 和 95% 区间分别覆盖 88.47% 和 93.46%。
- 95% 区间比名义值低 1.54 个百分点，存在轻度 calibration-to-test 偏移。

### 夹取糕点到蓝盒

- 绝对 MAE 为 0.600 N，但平均真实力只有 3.479 N，因此相对 MAE 达到 17.26%。
- 中心 90% 区间覆盖 86.25%，中心 95% 区间覆盖 90.00%。
- 糕点 calibration 只有 147 个有效窗口，且 test 只有 160 个有效窗口；当前包络在该低力
  任务上的跨 episode 泛化明显弱于另外两项任务。

## 6. 可用于论文的简洁表述

> We fixed the seed-44 episode split for all three tasks and retained only test
> windows in which both causal stable-contact gates were active. Bilateral force
> targets and predictions were formed by averaging the left and right outputs.
> Residual quantiles were estimated exclusively on the calibration split. The
> resulting central 90% intervals achieved test coverages of 88.46%, 88.47%, and
> 86.25% for USB insertion, ink squeezing, and pastry grasping, respectively. The
> corresponding central 95% coverages were 94.48%, 93.46%, and 90.00%. No test
> target was used to fit or adjust the reported intervals.

## 7. 可复现来源

配置与 checkpoint：

```text
USB
configs/experiments/pi05_jax_zrl_stable_contact.yaml
outputs/training/franka_tactile310/pi05_jax_zrl_envelope_v1/
  sustained_contact_seed_44/stable_evaluation/stable_contact_calibrated_checkpoint.pt

Ink
configs/experiments/pi05_jax_zrl_ink_beaker.yaml
outputs/training/franka_ink_beaker_tactile287/pi05_jax_zrl_envelope_v1/
  sustained_contact_seed_44/stable_evaluation/stable_contact_calibrated_checkpoint.pt

Pastry
configs/experiments/pi05_jax_zrl_pastry_A298.yaml
outputs/training/franka_pastry_tactile_A298/pi05_jax_zrl_envelope_v1/
  sustained_contact_seed_44/stable_evaluation/stable_contact_calibrated_checkpoint.pt
```

完整精度结果另存于同目录下的
`THREE_TASK_SEED44_BILATERAL_INTERVAL_REEVALUATION.json`。
