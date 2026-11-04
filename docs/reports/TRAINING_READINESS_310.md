# Tactile Force Envelope Training Readiness

> The default main model is now the token-level Transformer Envelope Expert. See
> `TRANSFORMER_ENVELOPE_REPORT_310.md`. The pooled 2048D model documented below is retained as the
> historical MLP baseline only.

Dataset collection: `franka_usb_tactile_310_20260716`

## Fixed-horizon contract

- Each sample is anchored at `t_k` and covers `[t_k, t_k + 300 ms)`.
- Images are the latest preceding frames. Every camera must have age <= 70 ms.
- Robot state is sampled at `t_k`.
- Actions are selected by timestamps over the complete 300 ms window, then represented by 9 commands.
- Online RL must use the same 300 ms horizon and pass the complete command sequence that is actually queued at `t_k`.

The recorded `action` field exactly matches `gello.q + gripper.target`; `observation.state`
matches `robot.q + gripper`. It is therefore a teleoperation command target, not robot state.
Nearly every retrospective 300 ms demonstration window contains command changes. Those changes
may include human feedback after contact and are not causal inputs available to an Online RL policy.
For causal training, future collection should record the planned action chunk at `t_k` separately
from the commands actually executed after feedback.

## Tactile labels

Labels are computed separately for `left_tip` and `right_tip` after per-episode bias removal:

- `raw_max`: maximum unfiltered force magnitude on each side, then max(left, right).
- `filtered_top2_mean`: 3-point median filter, mean of the largest two values on each side,
  then max(left, right).

`filtered_top2_mean` is the soft-envelope training target. `raw_max` is retained for direct hard
safety monitoring and must not be replaced by the learned envelope.

Each accepted window has at least 4 real samples from each sensor. No tactile interpolation is used.

## Baseline validation

The first second uses a robust per-axis median bias. A contact-free proxy is accepted only when:

- the gripper command remains >= 0.95;
- each sensor has enough valid samples;
- each sensor's baseline residual p95 is <= 1.0 force unit.

Six episodes failed this proxy and were removed. This cannot detect perfectly static preload, so a
future recorder should save an explicit operator-confirmed contact-free baseline event.

## Split and calibration

- Episode split: 70% train, 15% calibration, 15% test.
- All visual/state/action/target normalization statistics use train episodes only.
- Calibration uses the finite-sample one-sided conformal rank
  `ceil((n + 1) * 0.95)`.
- Coverage is reported overall and grouped by task, contact state, and minimum per-side sample rate.

The rebuilt workspace contains 12,100 windows from 304 accepted episodes:

- train: 8,368 windows
- calibration: 1,905 windows
- test: 1,827 windows

## Primary SmolVLA condition feature

The primary feature source is the frozen checkpoint:

`lerobot/outputs/train/franka_tactile310_smolvla_b64_e10_20260717_152937/checkpoints/022123/pretrained_model`

`checkpoints/last` currently resolves to `022123`. The cache records the resolved path and SHA-256.

- Feature tap: `past_key_values` returned by the prefix call to
  `policy.model.vlm_with_expert.forward` and passed unchanged into every `denoise_step`.
- Mask tap: valid-token mask returned by `policy.model.embed_prefix`.
- Raw shape: 16 layers, each with key and value tensors `[B,241,5,64]`; observed valid prefix
  tokens: 209 per sample.
- Pooling: for each layer and K/V tensor, masked mean over valid tokens followed by mean over
  5 KV heads; concatenate key then value across all 16 layers.
- Included modalities: three images, task language, normalized 8D robot state.
- Excluded modalities: action query, diffusion noise, denoising suffix tokens, predicted actions,
  tactile values, and tactile labels.
- Final cached feature: `[B, 2048]`.
- Extra task embedding: disabled because language is already present.
- Extra robot state: retained for the first version, despite state also being present in the prefix.

Camera order and preprocessing exactly follow the checkpoint preprocessor:

1. `observation.images.camera_camera_1_color_image_raw` -> `observation.images.camera1`
2. `observation.images.camera_camera_2_color_image_raw` -> `observation.images.camera2`
3. `observation.images.d515_color_image_raw` -> `observation.images.camera3`

Input images remain in `[0,1]`. SmolVLA itself applies `resize_with_pad(512,512)` and converts to
`[-1,1]`. The checkpoint allows zero empty cameras, so a missing camera is rejected. The checkpoint
processor appends a newline to task text, tokenizes to length 48, and applies its saved state
normalization. ImageNet preprocessing is not used.

The helper `predict_action_chunk_with_condition()` captures action and condition from one normal
SmolVLA inference. A real-window test showed exactly equal offline and same-forward features
(`max_abs_diff=0.0`) while producing an action tensor of shape `[1,50,8]`.

## SmolVLA result

The frozen SmolVLA KV-condition model completed successfully. Test MAE is 4.84, Gaussian NLL is
6.50, overall coverage is 96.61%, and contact-only coverage is 95.31%. This result is a learned soft
envelope and is not a hard-safety mechanism.

The checkpoint remains `production_ready=false` only because the historical demonstration future
actions can include post-contact human corrections unavailable at `t_k`. New Online RL collection
must persist the candidate chunk selected at `t_k` and the actually queued 300 ms prefix.

## ResNet18 ablation

ResNet18 is retained only in `configs/experiments/resnet18_ablation.yaml`. It is never selected as a
fallback when SmolVLA loading, preprocessing, feature capture, or cache validation fails.

## Commands

```bash
cd ~/Documents/gello_software
./run.sh audit
./run.sh build
./run.sh cache-features
./run.sh train
```

The existing SmolVLA TCP client supports the same-forward path without loading SmolVLA twice:

```bash
cd ~/Documents/lerobot/examples/franka
./run_franka_smolvla_tcp_client.sh \
  --send-actions \
  --envelope-checkpoint ~/Documents/tactile_force_envelope/outputs/training/franka_tactile310/force_envelope_v2/force_envelope_model.pt
```

At 30 Hz it evaluates the first 9 actually queued candidate commands as the 300 ms horizon. It
rejects policy/checkpoint mismatch and action-rate/horizon mismatch. The predictor currently reports
the envelope but does not autonomously stop the robot; direct `raw_max` remains the hard monitor.

Artifacts are under `artifacts/franka_tactile310/force_envelope_v2`.
