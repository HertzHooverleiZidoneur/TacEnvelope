# 输入相关触觉力包络算法更新

更新日期：2026-09-11

## 1. 更新目的

旧版双侧平均力包络使用 calibration residual 的固定牛顿偏移：

```math
L_t=\mu_t^{avg}+q_{05},\qquad U_t=\mu_t^{avg}+q_{95}.
```

该区间中心会随输入变化，但宽度恒为 `q95-q05`，不能表达不同视觉、状态、动作和
历史触觉条件下的预测难度。本次将默认算法改成 uncertainty-scale split conformal
envelope；旧 additive 方法仅保留为消融。

## 2. 标签、模型与尺度

每个 300 ms 时间窗内，左右传感器分别异步收集真实有效样本。每侧先做三点中值
滤波，再取最大的两个有效力幅值求均值，得到 `y_left_top2_mean` 和
`y_right_top2_mean`。少于 4 个有效样本的侧使该窗口成为低质量窗口。双侧监督量为：

```math
y_t^{avg}=\frac{y_{t,L}+y_{t,R}}{2}.
```

Transformer Expert 仍分别输出左右 `mu` 和 `sigma`。双侧中心和组合尺度定义为：

```math
\mu_t^{avg}=\frac{\mu_{t,L}+\mu_{t,R}}{2},\qquad
u_t^{mean}=\frac{\sigma_{t,L}+\sigma_{t,R}}{2}.
```

这里的 `u_mean` 是经验性的、随输入变化的 bilateral uncertainty scale，**不是**
`y_avg` 的标准差。左右预测误差相关性未知，因此不对它作独立高斯变量合成的解释。

训练结构、Gaussian NLL、SmoothL1、episode split、stable-contact gate 和 raw peak
保护均未改变。Gaussian NLL 的职责是学习输入相关的相对不确定性尺度；最终不直接把
`1.645*sigma` 或 `1.96*sigma` 当作边界，因为真实残差不保证服从理想高斯分布。

## 3. Calibration

仅使用 calibration split 中左右 causal stable-contact gate 同时开启的窗口：

```math
s_i=\frac{y_i^{avg}-\mu_i^{avg}}
{\max(u_i^{mean},10^{-6})}.
```

令 calibration 样本数为 `n`。下尾和上尾采用有限样本 order statistic：

```math
k_L=\lfloor(n+1)0.05\rfloor,\qquad
k_U=\lceil(n+1)0.95\rceil.
```

排序后取 `alpha05=s_(k_L)`、`alpha95=s_(k_U)`。在线动态区间为：

```math
L_t=\mu_t^{avg}+\alpha_{05}u_t^{mean},\qquad
U_t=\mu_t^{avg}+\alpha_{95}u_t^{mean}.
```

区间宽度为：

```math
U_t-L_t=(\alpha_{95}-\alpha_{05})u_t^{mean}.
```

因此不同任务阶段和输入条件可产生不同宽度。Test target 未参与 `alpha` 估计。

## 4. USB bilateral scale sanity check

固定 USB seed-44 split、checkpoint、stable-contact 窗口和校准方法，只替换组合尺度：

| Scale | 定义 | Test coverage | 距离 90% | 平均宽度 (N) | 宽度 P05/P50/P95 (N) |
|---|---|---:|---:|---:|---:|
| Mean | `0.5*(sigma_L+sigma_R)` | 89.80% | 0.20 pp | 4.513 | 2.123 / 2.948 / 10.770 |
| RSS | `0.5*sqrt(sigma_L^2+sigma_R^2)` | 90.13% | 0.13 pp | 4.529 | 2.124 / 2.952 / 10.663 |
| Max | `max(sigma_L,sigma_R)` | 88.29% | 1.71 pp | 4.460 | 2.076 / 2.946 / 9.922 |

`max` 只比 mean 窄 0.053 N，却少覆盖约 1.5 个百分点。Mean 和 RSS 均在 90% 的
0.5 个百分点内，mean 又略窄，因此保留 `u_mean`。这一步使用了已多次诊断的
development test 来选择 scale family，但没有使用 test target 重算任何 alpha；最终
结论仍需新的 blind episodes 验证。完整 JSON 为
`stable_evaluation/bilateral_uncertainty_scale_ablation.json`。

## 5. Seed-44 测试结果

以下均只统计 test split 中左右 stable-contact gate 同时开启的窗口。名义区间是中心
90%，不是“95% 区间”。

| 任务 | Calibration 窗口 | Test episode | Test 窗口 | alpha05 | alpha95 | MAE (N) | RMSE (N) | Test coverage | 平均宽度 (N) | 宽度 P05/P50/P95 (N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USB 插入 | 580 | 31 | 598 | -0.878 | 1.021 | 1.323 | 3.210 | 89.80% | 4.513 | 2.123 / 2.948 / 10.770 |
| 墨水挤入烧杯 | 578 | 29 | 642 | -1.342 | 0.884 | 0.802 | 1.931 | 88.79% | 2.267 | 0.815 / 1.274 / 10.610 |
| 糕点放入蓝盒 | 147 | 25 | 160 | -1.614 | 1.133 | 0.600 | 0.928 | 91.25% | 2.524 | 0.858 / 1.683 / 7.478 |

固定宽度 additive ablation 在相同窗口上的 coverage 分别为 88.46%、88.47% 和
86.88%。uncertainty-scale 方法的 coverage 分别变化 `+1.34`、`+0.31` 和 `+4.38`
个百分点，同时区间宽度随输入变化。USB 与墨水的平均宽度也低于对应 additive
区间；糕点平均宽度略低，且 coverage 明显改善。

## 6. Online RL 合同

Sidecar metadata schema 已升级为 v2，默认只接受：

```text
reward_envelope.type = bilateral_average_sigma_scaled_band
calibration_method   = standardized_residual_quantiles
uncertainty_scale    = u_mean
scale_is_stddev      = false
lower                = alpha05
upper                = alpha95
```

软惩罚仍由双侧稳定接触 gate 控制，基于观测 `Top2Mean` 到 `[L_t,U_t]` 的距离计算。
原始未滤波峰值阈值独立保留，不受 gate 控制。固定力阈值消融仍可通过
`fixed_upper` reward criterion 使用。

## 7. 文件

- 校准实现：`src/tactile_force_envelope/bilateral_sigma_calibration.py`
- 校准入口：`scripts/evaluation/calibrate_bilateral_sigma_envelope.py`
- Scale 对比：`scripts/evaluation/compare_bilateral_uncertainty_scales.py`
- 在线服务：`src/tactile_force_envelope/pi05_online_service.py`
- Online RL 握手：`src/tactile_force_envelope/online_rl.py`
- 每个任务的校准：`stable_evaluation/bilateral_sigma_calibration.json`

当前 test split 已用于多轮开发诊断。以上结果可用于开发对比，但最终论文结果仍应在
新的独立 episodes 上进行 blind evaluation。
