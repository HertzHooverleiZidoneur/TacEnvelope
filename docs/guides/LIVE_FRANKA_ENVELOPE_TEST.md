# Franka 实机 Envelope 记录测试

该测试是独立工具，不修改已有 ACT、pi0.5 或 SmolVLA 推理脚本。

## 1. 启动 NUC runtime

终端 1：

```bash
cd ~/Documents/lerobot/examples/franka

REMOTE_DATASET=datasets/franka_usb_tactile_310_episode0_reset \
POLICY_EXECUTE_FPS=30 \
CAMERA_FPS=30 \
./open_franka_smolvla_runtime.sh
```

等待机械臂完成复位，且终端显示 `runtime is up`。保持此终端打开。

## 2. 执行短时实机测试

终端 2：

```bash
cd ~/Documents/tactile_force_envelope

./scripts/runtime/run_live_franka_test.sh \
  --send-actions \
  --max-chunks 12 \
  --send-chunk-size 50 \
  --queue-min-execute 50
```

`--send-actions` 会让机械臂真实运动。运行前清空工作区，并保持急停可用。按
`Ctrl-C` 会清空待执行动作并切换为保持当前位置。

runtime 启动时已经复位。若保持同一个 runtime 重复测试，可额外添加：

```bash
--reset-before-run
```

## 3. 输出

每次测试保存在：

```text
outputs/live_tests/franka_live_envelope_tests/run_YYYYMMDD_HHMMSS/
```

主要文件：

- `report.html`：中文汇总报告。
- `force_and_envelope.png|pdf`：左右真实 `top2_mean`、`mu` 和 `upper95`。
- `envelope_deviation.png|pdf`：`实际 top2_mean - upper95`；正值表示超界。
- `bilateral_average_force_and_envelope.png|pdf`：左右窗口标签平均值、平均预测和
  calibration-only additive `upper95`。
- `bilateral_average_deviation.png|pdf`：`y_avg - (mu_avg + q95)`；仅双侧稳定
  contact gate 同时开启时用于软惩罚判断。
- `raw_force_timeline.png|pdf`：约 22 Hz 原始力与每个 300 ms 预测窗口。
- `prediction_windows.csv`：逐 action chunk 的数值结果。
- `inference_events.jsonl`：状态、完整实际 action chunk、Envelope 和时间戳。
- `anchor_images/`：每次推理使用的三路图像。
- `tactile_raw/`：左右千觉原始异步样本与触觉图像。

标签与训练保持一致：动作 enqueue 确认时刻为锚点，使用未来 300 ms 的真实触觉
样本；首秒三轴零偏、三点中值滤波，再计算左右各自的 `top2_mean`。不会把触觉
插值到 30 Hz。

左右平均结果使用与当前 Envelope checkpoint 匹配的 calibration split 结果：

```text
y_avg     = (y_left + y_right) / 2
mu_avg    = (mu_left + mu_right) / 2
upper_avg = mu_avg + q95
```

`q95` 不使用实机测试标签重新拟合。左右触觉为异步采样，因此不会对约 22 Hz
原始样本强行逐点求平均；只对同一 300 ms 控制窗口内各自计算完成的标签求平均。

## 4. 历史触觉来源

本机 Xense daemon 持续将左右传感器原始样本写入当前 run 的
`tactile_raw/.../{left_tip,right_tip}/samples.jsonl`。每次 SmolVLA forward 完成后、
action enqueue 前，以本机 `pc_time_ns` 为锚点，分别选择 `[t-200 ms,t)` 内最后 6
个真实有效样本，不做插值。单点特征为：

```text
[Fx-baseline_x, Fy-baseline_y, Fz-baseline_z, force_magnitude, relative_time]
```

`relative_time=(sample_time-t)/200 ms`，范围为 `[-1,0)`，与训练缓存一致。不足 6
点时左侧补零，并通过 mask 告诉模型哪些位置无效。每个推理窗口实际使用的
`history`、`history_mask` 和时间新鲜度也会复制到 `inference_events.jsonl`，便于复核。

## 5. 仅测试报表代码

不连接机器人、不读取传感器：

```bash
cd ~/Documents/tactile_force_envelope
./scripts/runtime/run_live_franka_test.sh \
  --self-test-report \
  --output-root /tmp/franka_envelope_report_test
```
