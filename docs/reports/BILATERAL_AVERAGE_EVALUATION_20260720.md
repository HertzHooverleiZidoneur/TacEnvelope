# Bilateral Average Stable-Contact Envelope Evaluation

## Scope

- Reused the three trained Variant B checkpoints (`seed=42,43,44`).
- Did not retrain or modify the model, loss, checkpoint, or split.
- Selected windows only when both causal pre-anchor stable-contact gates were true.
- Did not use future force targets to select windows.
- Computed `q95 = quantile(y_avg - mu_avg, 0.95)` independently for each seed using only its calibration predictions.
- Reported results once on the existing development test split. Test targets did not adjust `q95`.

## Results

All values are mean +/- population standard deviation across three seeds. Force values use the dataset's calibrated force unit.

| Metric | Bilateral average | Independent additive envelopes |
|---|---:|---:|
| MAE | 1.0412 +/- 0.0661 | n/a |
| Coverage | 92.19% +/- 0.73 pp | 85.67% +/- 1.57 pp |
| Additive width (`q95`) | 1.6865 +/- 0.0758 | side-specific |
| False-penalty windows | 50.67 +/- 4.71 | 93.00 +/- 10.20 |
| Episodes with a false penalty | 23.67 +/- 0.94 | 27.67 +/- 0.47 |

Per-seed average-envelope results:

| Seed | Calibration windows | Test windows | MAE | q95 | Coverage | False windows | False episodes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 633 | 649 | 1.1316 | 1.6915 | 91.68% | 54 | 23 |
| 43 | 633 | 649 | 1.0166 | 1.5913 | 91.68% | 54 | 25 |
| 44 | 633 | 649 | 0.9755 | 1.7768 | 93.22% | 44 | 23 |

## Decision

The bilateral average envelope reduced false-penalty windows by 45.5% and affected episodes by 14.5% relative to the current independent additive envelopes on the same windows. Its three-seed mean coverage is 92.19%, which is 0.81 percentage points below the requested 93%-97% interval. Seed 44 reaches the interval, but the aggregate criterion does not.

This result is promising for reducing spurious penalties but does not yet pass the requested coverage criterion. No parameter was adjusted after viewing development-test results.

## Artifacts

- `stable_contact_ab/bilateral_average_evaluation/bilateral_average_evaluation.json`
- `stable_contact_ab/bilateral_average_evaluation/bilateral_average_per_episode.csv`
- `stable_contact_ab/bilateral_average_evaluation/bilateral_average_episode_curves_seed_44.png`

The artifact root is configured by `data.output_dir` in `configs/production/franka_tactile310_stable_contact.yaml`.
