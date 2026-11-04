# USB Envelope Expert Ablation Studies

日期：2026-09-10

## 共同实验设置

- 任务：USB insertion。
- 模型：Pi0.5 `z_rl` 条件下的两层 Transformer Envelope Expert。
- 输入基线：`z_t + q_t + A_t^(300 ms) + h_t^tac(200 ms)`。
- 标签：左右 `filtered_top2_mean`，评估使用 bilateral stable-contact test windows。
- 划分：相同 episode-level train/validation/calibration/test split。
- validation：选择最佳 checkpoint。
- calibration：只估计 residual quantiles。
- test：不参与训练、checkpoint 选择或校准。
- 随机种子：42、43、44。
- 每个种子：580 个 calibration windows，598 个 test windows。

## Ablation 1：Action Conditioning

| Method | MAE (N) | RMSE (N) | Gaussian NLL | Coverage | Width (N) |
|---|---:|---:|---:|---:|---:|
| Full action | **1.258 ± 0.049** | **3.276 ± 0.097** | 1.033 ± 0.037 | 88.80 ± 0.60% | 4.437 ± 0.235 |
| Without action | 1.311 ± 0.104 | 3.489 ± 0.176 | **1.020 ± 0.016** | 89.13 ± 1.55% | **4.253 ± 0.241** |

移除 action 后，MAE 平均增加 4.23%，RMSE 增加 6.49%。MAE 在 2/3 个种子上恶化，
RMSE 在 3/3 个种子上恶化。结果支持 action conditioning 对降低较大预测误差有一定贡献，
但 Gaussian NLL 和区间宽度没有改善，因此不应写成强显著性结论。

实现上，without-action 模型真正删除 action projection、action positional embedding 和
9 个 action tokens；decoder sequence 从 24 tokens 减为 15 tokens，不使用零向量替代动作。

### Action 时序限制

当前离线 action 是按时间戳从 `[t,t+300 ms]` 收集的实际执行命令，元信息
`action_sequence_available_at_anchor=false`。因此该消融证明的是短时动作轨迹含有额外的
力预测信息。若论文将其解释为在线 candidate action conditioning，最终还应使用同一次
Pi0.5 forward 在 anchor 时刻产生的 candidate chunk 进行因果复验。

## Ablation 2：Gaussian Probabilistic Objective

| Method | MAE (N) | RMSE (N) | Coverage | Width (N) |
|---|---:|---:|---:|---:|
| SmoothL1-only | 1.405 ± 0.087 | 3.665 ± 0.377 | 89.46 ± 1.48% | 4.263 ± 0.174 |
| Gaussian + additive calibration | **1.258 ± 0.049** | **3.276 ± 0.097** | 88.80 ± 0.60% | 4.437 ± 0.235 |
| Gaussian + sigma-scaled calibration | **1.258 ± 0.049** | **3.276 ± 0.097** | **89.86 ± 0.34%** | **4.071 ± 0.280** |

相对 SmoothL1-only，Gaussian NLL + 0.5 SmoothL1 将 MAE 和 RMSE 分别降低 10.46% 和
10.61%。MAE 在 3/3 个种子上改善，RMSE 在 2/3 个种子上改善。在 Gaussian 模型内部，
使用 calibration standardized residual 构造 `mu + alpha*sigma` 区间后，coverage 为
89.86%，并比同模型的 additive residual interval 窄 8.24%。

该结果支持 Gaussian NLL 作为 heteroscedastic probabilistic training objective，并说明
预测 sigma 含有可利用的输入相关尺度信息。它不证明真实力严格服从 Gaussian，也不证明
未经 calibration 的 raw sigma 已具有正确 coverage。

## 复现命令

```bash
cd ~/Documents/tactile_force_envelope

PYTHONPATH="$PWD/src:$HOME/Documents/rlt_openpi_franka_v4.6_dualrate/src" \
  ~/Documents/RL_Token_update/.venv/bin/python \
  scripts/training/run_action_conditioning_ablation.py --seeds 42 43 44

PYTHONPATH="$PWD/src:$HOME/Documents/rlt_openpi_franka_v4.6_dualrate/src" \
  ~/Documents/RL_Token_update/.venv/bin/python \
  scripts/training/run_gaussian_loss_ablation.py --seeds 42 43 44
```

已有 checkpoint 会被复用；只有显式加入 `--force` 才重新训练。

## 论文建议表述

> Removing action conditioning increased MAE by 4.23% and RMSE by 6.49%, with RMSE
> degradation observed across all three seeds. This provides moderate evidence that the
> short-horizon action trajectory contributes information beyond the current VLA latent,
> proprioception, and causal tactile history. Separately, probabilistic training reduced
> MAE and RMSE by 10.46% and 10.61%. Calibration of standardized residuals retained 89.86%
> test coverage while reducing interval width by 8.24%, indicating that the predicted scale
> captures useful heteroscedastic information, although empirical calibration remains necessary.
