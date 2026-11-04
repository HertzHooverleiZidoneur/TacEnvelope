# 糕点任务 Pi0.5 触觉 Envelope Expert 训练

## 固定输入

```text
任务             Pick up the pastry and place it into the blue box.
Pi0.5 checkpoint /path/to/tactile_model/pastry_model/24999
LeRobot v3       /path/to/VLA_RL_dataset/franka_pastry_tactile_A298_20260820
训练源索引       /path/to/VLA_RL_dataset/franka_pastry_tactile_A298_sources
配置             configs/experiments/pi05_jax_zrl_pastry_A298.yaml
```

训练源索引保留原始机器人帧和约 22 Hz 左右触觉 sidecar 的绝对时间戳，但 episode 编号与
最终 A298 LeRobot v3 数据一一对应。标签使用 `[t, t+300 ms)` 内真实触觉样本，不插值。

## 模型与数据切分

```text
三路 RGB + task --冻结的糕点 Pi0.5 JAX forward--> z_rl [1,512]
z_rl + 8D state + 未来实际动作 [9,8] + 过去 200 ms 触觉历史 [2,6,5]
    -> 2-layer Transformer Envelope Expert
    -> left/right mu, log_var
```

模型参数为 2,253,318。数据严格按 episode 切分：209 条 train、30 条 validation、
30 条 calibration、29 条 test。归一化仅使用 train；validation 选择最佳 checkpoint；
calibration 计算左右独立的 conformal 系数；test 不参与训练和校准。

双侧平均力的在线加性包络使用 calibration split 中双侧稳定接触的 147 个窗口，按
`r=y_avg-mu_avg` 计算：

```text
q05 = -1.642 N
q10 = -1.381 N
q90 =  0.250 N  -> U90 = mu_avg + 0.250 N
q95 =  0.861 N  -> U95 = mu_avg + 0.861 N
```

`U90` 和 `U95` 分别是单侧上界，其测试 coverage 为 `85.63%` 和 `90.63%`。标准双侧
区间按相同尾概率配对：中央 80% 使用 `[L10,U90]`，测试 coverage `76.88%`；中央 90%
使用 `[L05,U95]`，测试 coverage `86.25%`。不要使用非对称的 `[L05,U90]` 作为标准
包络。测试标签不参与上述分位数计算。

糕点任务的首秒无接触残差 P99 约为 `0.18 N`（左）和 `0.14 N`（右），因此配置使用
`tau_on=1.0 N`、`tau_off=0.5 N`、`tau_train=1.0 N`。不能沿用 U 盘/墨水任务的 `5 N`
门限，否则会丢掉糕点的轻接触窗口。

## 完整训练

首次先构建本地训练源索引：

```bash
cd ~/Documents/tactile_force_envelope
./scripts/training/build_pastry_A298_sources.py
```

终端 1 启动冻结 Pi0.5 特征服务并保持运行：

```bash
cd ~/Documents/tactile_force_envelope
./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh start-server
```

终端 2 依次运行：

```bash
cd ~/Documents/tactile_force_envelope

./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh audit
./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh build
./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh cache-tactile-history
./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh cache-features
./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh train
./scripts/training/run_pastry_pi05_jax_zrl_envelope.sh stable-evaluate
```

`cache-features` 完成后可以关闭终端 1，释放 Pi0.5 占用的显存。缓存签名绑定 checkpoint、
任务、相机顺序和窗口文件，配置变化时不会误用旧特征。

## 输出

```text
outputs/training/franka_pastry_tactile_A298/pi05_jax_zrl_envelope_v1/
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

在线或 shadow 测试只加载校准后的稳定入口：

```text
checkpoints/pastry/current_stable_contact.pt
checkpoints/pastry/current_calibration.json
```

校准前的 `transformer_force_envelope_model.pt` 不作为在线入口。
