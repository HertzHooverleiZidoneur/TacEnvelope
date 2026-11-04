# Token-Level Transformer Envelope Expert

## Status

The token-level implementation and offline/online execution path are complete. The current
checkpoint is an experimental model, not a hard-safety controller. It is not production-ready
because historical future actions may contain human feedback unavailable at the anchor and the
left-side held-out coverage is below 95%.

## Frozen SmolVLA condition

- Checkpoint: `franka_tactile310_smolvla_b64_e10_20260717_152937/checkpoints/022123`.
- Tap: output of the final VLM layer `input_layernorm`, immediately before its Q/K/V projections.
- Shape: `[B, 241, 960]`; observed valid-token count: 209.
- Mask: the exact prefix padding mask from `SmolVLA.embed_prefix`.
- Modalities: three checkpoint-native camera streams, task language, and normalized 8D state.
- Pooling: none.
- Cache: FP16 `[12100, 241, 960]`, with checkpoint/config/code hash invalidation.
- Same-forward verification: offline and action-forward tokens had max absolute difference `0.0`.

These hidden states are the final-layer pre-QKV multimodal prefix tokens. Their projected K/V are
part of the cache consumed by SmolVLA action denoising. The action query, diffusion noise, suffix,
predicted actions, and tactile labels are excluded.

## Expert architecture

Inputs are two learned envelope queries, one normalized 8D state token, nine timestamp-selected
8D action tokens with learned positions and mask, and the SmolVLA condition tokens. The expert has
two pre-norm decoder layers with `d_model=256`, eight heads, FFN width 1024, and dropout 0.1. Each
layer performs self-attention over `[LEFT_ENV, RIGHT_ENV, STATE, ACTION_1..ACTION_9]`, then
cross-attention to condition tokens, then an FFN. Separate left/right heads predict `mu`, `log_var`,
and contact logits. Trainable parameter count is 2,362,886; SmolVLA remains frozen.

The one-batch gate recorded:

- raw condition `[128,241,960]`, projected condition `[128,241,256]`;
- action tokens `[128,9,256]`, state `[128,1,256]`, queries `[128,2,256]`;
- decoder input/output `[128,12,256]`;
- left/right distribution outputs `[128,2]` each.

## Supervision and split

Left and right labels are computed separately. Each side uses its own three-point median filtered
force magnitudes and the mean of its two largest values. The sides are never max-reduced for this
model. Gaussian NLL is averaged across sides; contact BCE uses raw per-side max >= 5 and weight 0.1.

Episodes are split 70/10/10/10:

- train: 8,368 windows;
- validation: 1,271 windows;
- calibration: 1,234 windows;
- test: 1,227 windows.

Validation selects the best checkpoint and drives early stopping. Training stopped after 12 epochs
and restored epoch 6. Calibration never updates weights and computes finite-sample one-sided
`alpha95_left` and `alpha95_right` independently.

## Held-out results

| Metric | Left | Right |
|---|---:|---:|
| MAE | 5.391 | 5.629 |
| RMSE | 8.333 | 9.292 |
| Gaussian NLL | 2.285 | 2.516 |
| alpha95 | 0.643 | 1.131 |
| Overall coverage | 81.66% | 97.39% |
| Contact coverage | 70.13% | 95.84% |
| Non-contact coverage | 95.67% | 99.43% |
| Mean envelope margin | 4.599 | 11.829 |

The left-side calibration does not transfer to held-out test episodes. One fixed global threshold,
computed as the train P95 of `max(left,right)`, is 76.03 and reaches 100% left / 98.13% right test
coverage, but is extremely conservative. Separate train P95 thresholds reach 93.48% left and
98.13% right with thresholds 56.89 and 76.01. The historical MLP baseline reports 96.61% global
max-force coverage and MAE 4.84 on a different 70/15/15 split; it is not a like-for-like per-side
comparison.

## Commands

```bash
cd ~/Documents/gello_software
./run.sh build
./run.sh cache-features
./run.sh train
```

Online same-forward self-test:

```bash
cd ~/Documents/lerobot/examples/franka
./run_franka_smolvla_tcp_client.sh --self-test --send-chunk-size 9 \
  --envelope-checkpoint ~/Documents/tactile_force_envelope/outputs/training/franka_tactile310/force_envelope_transformer_v3/transformer_force_envelope_model.pt
```

The old pooled MLP remains available only through
`configs/experiments/mlp_baseline.yaml`; ResNet18 remains a separate optional ablation.
