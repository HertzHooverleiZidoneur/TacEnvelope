# 三任务 Pi0.5 触觉力包络预测：论文素材

> 历史结果说明：本文档中的双侧 additive `q` 区间用于旧版结果与消融。2026-09-10
> 起，默认 Online RL 包络已更新为输入相关的 sigma-scaled `alpha05/alpha95` 区间。
> 当前算法与数值以 `SIGMA_SCALED_ENVELOPE_UPDATE_20260910_ZH.md` 为准。

生成日期：2026-09-07

## 1. 研究目标

本方法面向已经建立稳定接触后的短时力预测。给定动作窗口起点 `t_k` 时的
Pi0.5 视觉语言条件表示、机器人状态、未来 300 ms 实际执行动作，以及动作开始前
200 ms 的因果触觉历史，模型分别预测左右触觉在未来 300 ms 内的力分布：

```text
(z_rl, q_k, a_exec[k:k+9], h_left, h_right)
    -> (mu_left, sigma_left, mu_right, sigma_right)
```

该预测器不负责无接触阶段、新接触 onset 或接触释放。在线使用时由只读取历史触觉的
stable-contact gate 决定是否启用软包络惩罚；原始力硬阈值保护独立保留。

## 2. 输入、对齐与标签

### 2.1 时间对齐

- 控制频率为 30 Hz；每个预测窗口固定为 300 ms，对应 9 个实际执行动作。
- 图像使用锚点 `t_k` 之前最近的三路 RGB 帧，最大允许图像年龄为 70 ms。
- 三路图像顺序固定为 `camera_1`、`camera_2`、`d515`，并使用对应 Pi0.5 checkpoint
  的原始输入变换。
- 触觉约 22 Hz。直接收集 `[t_k, t_k+300 ms]` 中的真实异步样本，不对触觉插值；
  每侧少于 4 个有效样本的窗口被丢弃。
- 因果历史来自 `[t_k-200 ms, t_k)`，左右各保留最多 6 个真实样本。每个历史样本为
  `Fx, Fy, Fz, force magnitude, relative time` 五维；不足部分 padding，并使用 mask。

### 2.2 监督标签

对每个触觉样本先计算三轴合力：

```math
f_j = \sqrt{F_{x,j}^2 + F_{y,j}^2 + F_{z,j}^2}.
```

左右传感器分别进行首秒无接触零偏修正和三点中值滤波。每侧软包络标签取未来窗口内
最大的两个有效滤波力值之平均：

```math
y_s = \frac{f_{s,(1)} + f_{s,(2)}}{2}, \qquad s\in\{L,R\}.
```

原始未滤波最大值另行保留，只用于硬峰值保护，不作为该回归头的监督标签。

## 3. 条件特征与网络结构

Pi0.5 参数在 Envelope Expert 训练中完全冻结。对三路图像和任务文本运行任务对应的
Pi0.5 JAX checkpoint，读取策略响应中的 `z_rl`。`z_rl` 是一个 512 维视觉语言条件
token，包含图像和任务语言信息，但不包含机器人状态，因此 8 维机器人状态仍作为独立
state token 输入。缓存记录显示三个任务均使用 `joint JAX RLTPolicy response.z_rl`，
而不是 ImageNet/ResNet 特征。

Envelope Expert 使用两层 pre-norm Transformer：`d_model=256`、8 个 attention heads、
FFN 维度 1024、dropout 0.1，共 2,253,318 个可训练参数。实际 self-attention 序列依次为
两个 learned envelope query、一个 state token、左右各 6 个 history tokens 和 9 个
action tokens，共 24 个 token；`z_rl` 投影到 256 维后作为 cross-attention 的 K/V。两个 query
分别输出左右侧的 `mu` 和 `log_var`：

```math
\log\sigma_s^2 = \operatorname{clip}(\mathrm{log\_var}_s,-8,4),\qquad
\sigma_s=\exp(0.5\log\sigma_s^2).
```

## 4. 训练、验证、校准与测试划分

所有数据按完整 episode 随机划分，而不是按相邻窗口划分，从而避免同一轨迹的相邻帧
跨集合泄漏。随机种子为 44，比例为 70% train、10% validation、10% calibration、
10% test。

| 任务 | 可用 episode | Train | Validation | Calibration | Test |
|---|---:|---:|---:|---:|---:|
| 插入 USB | 304 | 213 | 30 | 30 | 31 |
| 挤墨水到烧杯 | 282 | 197 | 28 | 28 | 29 |
| 夹取糕点到蓝盒 | 298 | 209 | 30 | 30 | 29 |

四个集合的职责严格分离：

- **Train**：更新 Transformer Envelope Expert；所有归一化统计也只由 train episodes 计算。
- **Validation**：不更新参数，用于选择最低 validation loss 的 checkpoint 和 early stopping。
- **Calibration**：冻结网络后估计 `alpha95` 或加性残差分位数，不更新网络参数。
- **Test**：只计算最终误差和 coverage，不参与 checkpoint 选择或当前校准参数估计。

最佳 epoch 分别为 USB 18、墨水 40、糕点 13；分别在第 33、55、28 epoch 早停。

## 5. Stable-contact 训练目标

左右两侧使用独立因果 gate。USB 和墨水任务采用 `tau_on=5 N, tau_off=3 N`；糕点为
低力任务，采用 `tau_on=1 N, tau_off=0.5 N`。进入稳定接触要求锚点前最近 4 个有效
样本至少 3 个高于 `tau_on`。训练 loss 还要求该侧未来标签高于 `tau_train`；这一未来
条件只用于离线筛选训练样本，不用于在线 gate。

每侧损失为 masked Gaussian NLL 与 SmoothL1 的组合：

```math
\mathcal{L}_s =
\frac{\sum_i m_{i,s}
\left[\frac{1}{2}\left((y_{i,s}-\mu_{i,s})^2e^{-\log\sigma_{i,s}^2}
+\log\sigma_{i,s}^2\right)
+0.5\,\operatorname{SmoothL1}(\mu_{i,s},y_{i,s})\right]}
{\sum_i m_{i,s}+\epsilon},
```

```math
\mathcal{L}=\frac{1}{2}(\mathcal{L}_L+\mathcal{L}_R).
```

## 6. 校准方法

### 6.1 左右单侧 95% 上包络

在 calibration split 的持续接触样本上分别计算标准化单侧残差：

```math
s_{i,s}=\frac{y_{i,s}-\mu_{i,s}}{\sigma_{i,s}+\epsilon}.
```

使用有限样本 conformal rank 得到 `alpha95_s`，单侧上界为：

```math
U_{95,s}=\mu_s+\alpha_{95,s}\sigma_s.
```

### 6.2 在线奖励使用的双侧平均力区间

在线软惩罚使用双侧都已稳定接触的窗口：

```math
y_{avg}=\frac{y_L+y_R}{2},\qquad
\mu_{avg}=\frac{\mu_L+\mu_R}{2}.
```

在 calibration split 上计算加性残差 `r_i=y_avg,i-mu_avg,i`。当前运行代码保存
`q05`、`q90` 和 `q95` 三个经验分位数：

```math
L_{05}=\mu_{avg}+q_{05},\qquad
U_{90}=\mu_{avg}+q_{90},\qquad
U_{95}=\mu_{avg}+q_{95}.
```

`U90` 和 `U95` 是两个独立的单侧上界。标准中央区间必须使用对称尾概率：中央 80% 为
`[L10,U90]`，中央 90% 为 `[L05,U95]`，不使用 `[L05,U90]` 作为标准包络。这里的双侧
`q` 是 calibration split 的经验残差分位数，不是标准化残差，也没有使用 test 重新拟合。
若论文声称严格有限样本 split conformal，应将双侧经验分位数改为与单侧 `alpha95` 相同
的有限样本 rank 实现；当前更准确的表述是“held-out residual quantile calibration”。

## 7. 测试集结果

回归任务不宜使用分类“准确率”；本文报告 MAE、RMSE、bias、预测尺度和包络 coverage。

### 7.1 左右单侧持续接触预测

以下只统计该侧因果 stable-contact gate 开启且未来真实力超过任务接触阈值的 test 窗口。
Coverage 对应 `mu + alpha95*sigma`，`alpha95` 仅由 calibration split 得到。

| 任务 | 侧 | Calibration 窗口 | Test 窗口 | MAE (N) | RMSE (N) | 平均 sigma (N) | 95% coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| USB | 左 | 574 | 592 | 1.345 | 3.080 | 2.039 | 94.59% |
| USB | 右 | 589 | 610 | 1.820 | 3.896 | 2.701 | 95.57% |
| 墨水 | 左 | 583 | 674 | 0.709 | 1.683 | 0.825 | 95.25% |
| 墨水 | 右 | 784 | 851 | 0.838 | 1.872 | 1.036 | 94.01% |
| 糕点 | 左 | 171 | 166 | 0.697 | 1.112 | 0.829 | 95.78% |
| 糕点 | 右 | 201 | 237 | 0.663 | 1.092 | 0.996 | 95.78% |

三个任务的单侧上包络 coverage 均处于约 94%--96%，与目标 95% 接近。糕点的绝对
MAE 最小，但其真实力尺度也显著更小，不能仅凭绝对 MAE 判断它最准确。

### 7.2 双侧平均力点预测

以下统一统计左右 gate 同时开启的 test 窗口。

| 任务 | Test episode | Test 窗口 | 真实平均力 (N) | 真实力 P95 (N) | Bias (N) | MAE (N) | MAE/均力 | RMSE (N) | RMSE/均力 | 平均 sigma (N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USB | 31 | 598 | 51.018 | 67.866 | -0.118 | 1.323 | 2.59% | 3.210 | 6.29% | 2.377 |
| 墨水 | 29 | 642 | 20.194 | 37.196 | +0.402 | 0.802 | 3.97% | 1.931 | 9.56% | 1.018 |
| 糕点 | 25* | 160 | 3.479 | 7.157 | +0.169 | 0.600 | 17.26% | 0.928 | 26.67% | 0.919 |

`*` 糕点 test split 有 29 个 episode，其中 25 个出现双侧同时稳定接触的有效窗口。

USB 的双侧平均力预测最稳定，MAE 约为真实均力的 2.6%；墨水为 4.0%。糕点虽然绝对
MAE 仅 0.60 N，但由于任务力尺度仅约 3.48 N，相对误差达到 17.3%，是三个任务中最难
校准的一个。Bias 显示 USB 略低估 0.12 N，墨水和糕点分别略高估 0.40 N 和 0.17 N。

### 7.3 双侧加性包络

单侧上界结果：

| 任务 | Calibration 窗口 | q90 (N) | Test U90 coverage | q95 (N) | Test U95 coverage |
|---|---:|---:|---:|---:|---:|
| USB | 580 | 1.642 | 90.47% | 2.838 | 94.65% |
| 墨水 | 578 | 0.368 | 88.32% | 0.893 | 94.55% |
| 糕点 | 147 | 0.250 | 85.63% | 0.861 | 90.63% |

对称尾概率区间结果：

| 任务 | q10 (N) | `[L10,U90]` coverage | 低/高超界窗口 | q05 (N) | `[L05,U95]` coverage | 低/高超界窗口 |
|---|---:|---:|---:|---:|---:|---:|
| USB | -1.163 | 78.76% | 70/57 | -1.706 | 88.46% | 37/32 |
| 墨水 | -0.752 | 78.50% | 63/75 | -2.474 | 88.47% | 39/35 |
| 糕点 | -1.381 | 76.88% | 14/23 | -1.642 | 86.25% | 7/15 |

`U90/U95 coverage` 是单侧上界覆盖率；`[L10,U90]` 和 `[L05,U95]` 才是上下界同时
成立的中央区间 coverage。`[L05,U95]` 的名义覆盖率是 90%，不能称为“95% 区间”。

糕点任务中，`U90=mu_avg+0.250 N`、`U95=mu_avg+0.861 N`；测试 coverage 分别为
85.63% 和 90.63%，均比对应名义覆盖率低 4.37 个百分点。这两个数应作为测试结果原样
报告，不能用 calibration 的 90%/95% 名义值替代。

## 8. 可直接用于论文的结果描述

> We split complete episodes into 70% training, 10% validation, 10% calibration,
> and 10% testing subsets. The validation subset was used exclusively for model
> selection and early stopping, while the calibration subset was used only after
> training to estimate one-sided residual quantiles. No test target was used to
> fit network parameters or the reported calibration coefficients.

> On stable-contact test windows, the bilateral mean-force predictor achieved
> MAEs of 1.323 N, 0.802 N, and 0.600 N for USB insertion, ink dispensing, and
> pastry grasping, respectively. Relative to each task's mean contact force,
> these errors correspond to 2.59%, 3.97%, and 17.26%. Calibration-only 95th
> residual upper bounds attained test upper-coverages of 94.65%, 94.55%, and
> 90.63%, respectively. The reduced coverage and larger relative error on the
> pastry task indicate that low-force contact remains the most challenging
> regime.

## 9. 论文表述边界

1. 当前三任务主结果均为 seed 44；不能写成三随机种子的均值和标准差。
2. Test 没有参与当前模型训练或 calibration，但这些 test episodes 已在开发期间多次查看。
   正式论文最好增加全新 episode 的最终盲测，并将其作为最终泛化结果。
3. 离线动作是未来 300 ms 的遥操实际执行命令，可能包含接触后的人工反馈修正；在线输入
   是锚点时已经生成的候选动作。两者存在训练/部署分布差异，应在 limitation 中说明。
4. 论文不再将 `[L05,U90]` 作为标准区间。若在线实现选择更敏感的 `U90`，必须将其明确
   写成单侧高力消融；标准双侧包络与主文图统一使用 `[L05,U95]`。
5. Envelope Expert 是软奖励模型，不能替代 FCI 碰撞保护或原始触觉硬峰值阈值。

## 10. 数值来源

- `outputs/training/franka_tactile310/pi05_jax_zrl_envelope_v1/`
- `outputs/training/franka_ink_beaker_tactile287/pi05_jax_zrl_envelope_v1/`
- `outputs/training/franka_pastry_tactile_A298/pi05_jax_zrl_envelope_v1/`
- 每个任务的 `episode_splits.json`
- 每个任务的 `sustained_contact_seed_44/transformer_training_result.json`
- 每个任务的 `sustained_contact_seed_44/stable_evaluation/`
  `stable_contact_gated_evaluation.json` 与 `bilateral_average_calibration.json`
