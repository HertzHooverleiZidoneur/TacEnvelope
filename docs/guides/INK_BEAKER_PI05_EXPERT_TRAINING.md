# 墨水任务 Pi0.5 触觉 Envelope Expert 训练

## 固定输入

```text
任务            Squeeze the ink into the beaker.
Pi0.5 checkpoint /path/to/tactile_model/ink_model/24999
LeRobot v3       /path/to/VLA_RL_dataset/franka_ink_beaker_tactile_287_20260807
原始触觉 sidecar /path/to/VLA_RL_dataset/franka_remote_tactile/franka_lerobot_20260807_145251_tactile
配置             configs/experiments/pi05_jax_zrl_ink_beaker.yaml
```

训练不会把 30 Hz LeRobot 触觉列当作标签。标签来自左右传感器原始约 22 Hz sidecar，
按绝对时间戳异步选择 `[t, t+300 ms)` 内的真实样本，不做触觉插值。

原始 source episode 47、48、276、280、281 已明确排除。首秒零偏检查失败的 episode
还会在构建时自动整条剔除。

## 模型与监督

```text
三路 RGB + task + state --冻结的同一 Pi0.5 JAX forward--> z_rl [512]
z_rl [1,512]
+ 8D robot state
+ 未来实际执行动作 [9,8]（300 ms @ 30 Hz）
+ 左右过去 200 ms 因果触觉历史 [2,6,5]
    -> 2-layer Transformer Envelope Expert
    -> left/right mu, log_var
```

只缓存并训练满足以下条件的持续接触窗口：该侧过去 4 个有效样本至少 3 个大于 5 N，
且未来 300 ms 的该侧 `filtered_top2_mean > 5 N`。Loss 是左右独立的 masked Gaussian
NLL 加 `0.5 * SmoothL1`。数据按 episode 固定划分为 70% train、10% validation、
10% calibration、10% test；归一化只用 train，calibration 和 test 不更新网络。

## 首次完整训练

审计和窗口构建需要能访问 NUC 原始 30 Hz `frames.jsonl`。默认配置使用当前 GVFS
挂载路径；如果挂载点变化，只改配置中的 `data.robot_dataset`。

终端 1，启动冻结的墨水 Pi0.5 特征服务并保持打开：

```bash
cd ~/Documents/tactile_force_envelope
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh start-server
```

终端 2，执行完整流水线：

```bash
cd ~/Documents/tactile_force_envelope
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh all
```

`cache-features` 从本地 LeRobot v3 MP4 顺序读取三路图像，不会逐张通过 SFTP 读取
NUC JPEG。Pi0.5 checkpoint、任务文本、相机顺序或 `windows.jsonl` 改变时，特征缓存
签名会改变，旧缓存不会被误用。

## 分阶段执行与恢复

某阶段中断后，从该阶段重新运行即可。已完成且签名一致的特征缓存会自动复用。

```bash
cd ~/Documents/tactile_force_envelope

./scripts/training/run_ink_pi05_jax_zrl_envelope.sh audit
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh build
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh cache-tactile-history
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh cache-features
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh train
./scripts/training/run_ink_pi05_jax_zrl_envelope.sh stable-evaluate
```

`cache-tactile-history` 必须先于当前的 sustained-contact-only `cache-features`。

## 输出位置

所有墨水任务产物保存在：

```text
/path/to/tactile_force_envelope/outputs/training/
  franka_ink_beaker_tactile287/pi05_jax_zrl_envelope_v1/
```

主要文件：

```text
audit.json
windows.jsonl
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

在线推理最终加载
`stable_contact_calibrated_checkpoint.pt`。`transformer_force_envelope_model.pt` 是校准前
权重，不应直接作为 Online RL 的生产包络。

本机稳定短路径为：

```text
checkpoints/ink_beaker/current_stable_contact.pt
checkpoints/ink_beaker/current_calibration.json
```

本次训练的审计和 test 数值见
[`../reports/INK_BEAKER_PI05_EXPERT_20260811.md`](../reports/INK_BEAKER_PI05_EXPERT_20260811.md)。
