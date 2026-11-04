# Production checkpoint pointers

- `pi05_jax_zrl_stable_contact.pt`: seed-44 Pi0.5 `z_rl` stable-contact Transformer checkpoint.
- `pi05_jax_zrl_bilateral_sigma_calibration.json`: matching calibration-only standardized-residual
  `alpha05`/`alpha95` metadata for the dynamic bilateral uncertainty-scale band.
- `current_stable_contact.pt` and `current_bilateral_calibration.json`: legacy SmolVLA/additive
  ablation pointers; not used by the Pi0.5 Online RL sidecar.

These are local symbolic links into `outputs/training/`. They are intentionally excluded from Git
because the model and generated evaluation artifacts are machine-local.
