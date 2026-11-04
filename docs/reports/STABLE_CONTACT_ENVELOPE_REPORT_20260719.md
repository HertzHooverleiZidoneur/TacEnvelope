# Stable-Contact Force Envelope Report

## Scope

This experiment keeps the frozen SmolVLA condition tokens, robot state, 300 ms
candidate action sequence, 200 ms causal tactile history, and two-layer
Transformer Envelope Expert unchanged. It changes only when the learned
envelope is trained and evaluated.

- Entry gate per side: at least 3 of the last 4 valid pre-anchor samples are
  greater than `tau_on=5.0`.
- Online exit gate: at least 3 of the last 4 valid samples are below
  `tau_off=3.0`.
- Offline sustained-contact mask: causal entry gate is true and the side's
  future `filtered_top2_mean` is greater than `tau_train=5.0`.
- The optional `future_contact_ratio` filter is disabled in this first pass;
  the cached windows do not contain the full raw future force sequence.
- Future labels are used only for offline train/evaluation selection. They are
  not read by the online gate.
- Variant A: current all-window objective plus the online stable-contact gate.
- Variant B: per-side sustained-contact-only masked objective.

All comparisons use the same episode split and seeds 42, 43, and 44.

## Noise Threshold Audit

The verified contact-free first-second baselines from 213 train episodes give:

| Side | Median residual | MAD | Median + 6 robust sigma | Configured tau_on |
| --- | ---: | ---: | ---: | ---: |
| Left | 0.0060 | 0.0057 | 0.0565 | 5.0 |
| Right | 0.0210 | 0.0141 | 0.1461 | 5.0 |

The configured 5.0 threshold is far above measured baseline noise. This does
not prove it is the optimal task threshold, but it is not being triggered by
ordinary no-contact sensor noise in this collection.

## Three-Seed Results

Values are mean +/- standard deviation on development-test sustained-contact
windows.

| Variant | Side | Envelope | MAE | Coverage | Width | False-penalty windows |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | Left | mu + alpha sigma | 1.387 +/- 0.029 | 84.51% +/- 1.55% | 1.776 +/- 0.101 | 99.0 +/- 9.9 |
| A | Left | mu + q95 | 1.387 +/- 0.029 | 83.46% +/- 0.78% | 1.664 +/- 0.296 | 105.7 +/- 5.0 |
| A | Right | mu + alpha sigma | 1.508 +/- 0.219 | 94.40% +/- 0.57% | 3.113 +/- 0.142 | 37.0 +/- 3.7 |
| A | Right | mu + q95 | 1.508 +/- 0.219 | 94.96% +/- 0.51% | 3.729 +/- 0.462 | 33.3 +/- 3.4 |
| B | Left | mu + alpha sigma | 1.289 +/- 0.072 | 88.26% +/- 2.33% | 1.946 +/- 0.401 | 75.0 +/- 14.9 |
| B | Left | mu + q95 | 1.289 +/- 0.072 | 89.05% +/- 1.78% | 1.719 +/- 0.230 | 70.0 +/- 11.3 |
| B | Right | mu + alpha sigma | 1.478 +/- 0.058 | 95.01% +/- 1.08% | 2.879 +/- 0.635 | 33.0 +/- 7.1 |
| B | Right | mu + q95 | 1.478 +/- 0.058 | 94.55% +/- 0.65% | 3.321 +/- 0.764 | 36.0 +/- 4.3 |

Variant B improves left MAE and substantially reduces left false penalties,
but neither left envelope reaches the requested 93%-97% coverage. The right
scale envelope reaches the target interval and is narrower than Variant A.
The sustained-contact-only formulation is therefore the better development
candidate, but the left side remains the blocker.

## Causality

- Cached causal-history violations: 0.
- Evaluated history samples: 10,894.
- Latest history timestamp relative to anchor: -0.0014 ms.
- All gate samples are strictly before the action anchor.
- Online gate code has no future-label argument.
- The future target is used only by the offline sustained-contact mask.

The historical action sequence is still marked unavailable at the anchor
because it contains later human feedback corrections. Online inference passes
the candidate chunk produced at that anchor, but this offline-to-online action
distribution mismatch remains a development limitation.

The Online RL runtime must also provide the real 200 ms causal tactile history
to the adapter. The API and tests are present, but the current SmolVLA TCP
observation payload does not yet carry that history.

## Online Contract

The online soft cost is computed independently per side:

```text
gate * clip(max(0, observed_force - upper95) / force_scale, 0, cost_clip)
```

When a side's gate is false, its learned soft cost is zero. The raw-force hard
limit is evaluated independently of gate state and model availability, so it
continues to cover onset collisions, inference failures, and no-contact phases.

## Decision

`production_ready=false`.

The selected Variant B seed-44 checkpoint is a development candidate because
it has the best joint coverage proximity among the three sustained-contact
runs. It must not be treated as a production safety envelope because:

1. Left development-test coverage is still below 93%.
2. The test episodes have already been used for repeated diagnostics.
3. New independent blind shadow episodes have not been evaluated.
4. Offline future actions include human feedback unavailable at the anchor.
5. The live Online RL observation path has not yet supplied causal tactile
   history to the Expert.

The next evidence-bearing step is blind shadow evaluation on newly collected
episodes. Further calibration on this reused test set would only overfit the
diagnostic split.

## Artifacts

- `stable_contact_ab/stable_contact_ab_summary.json`
- `stable_contact_ab/stable_contact_ab_table.csv`
- `stable_contact_ab/stable_contact_per_episode_all_runs.csv`
- `stable_contact_ab/stable_contact_gate_test_report.json`
- `stable_contact_ab/<variant>_seed_<seed>/stable_evaluation/`
- `stable_contact_ab/stable_contact_sustained_candidate.pt`
- `stable_contact_gated_current/stable_contact_gated_evaluation.json`
- `stable_contact_gated_current/stable_contact_causality_audit.json`
