# Current Franka 310-episode setup

The original workstation paths at repository extraction time were:

- source code: `/path/to/tactile_force_envelope`
- combined collection: `/path/to/VLA_RL_dataset/franka_usb_tactile_310_20260716`
- SmolVLA checkpoint: `/path/to/lerobot/outputs/train/franka_tactile310_smolvla_b64_e10_20260717_152937/checkpoints/last/pretrained_model`
- generated workspace: `/path/to/tactile_force_envelope/outputs/training/franka_tactile310/force_envelope_transformer_v3`
- latest bilateral report: `stable_contact_ab/bilateral_average_report/detailed_report.html`

The generated workspace is stored under the ignored `outputs/training/` tree. In particular,
`condition_tokens.pt` is about 5.3 GB and is not tracked by Git.

Current data split for the 310-episode collection:

| split | episodes | windows | use |
|---|---:|---:|---|
| train | 213 | 8368 | gradient updates |
| validation | 30 | 1271 | best checkpoint / early stopping |
| calibration | 30 | 1234 | q90/q95/alpha only |
| test | 31 | 1227 | final metrics only |
