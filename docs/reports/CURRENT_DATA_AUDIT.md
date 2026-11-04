# Current Xense dataset audit

Audited dataset pair:

- robot/camera mirror: `franka_lerobot_20260715_170207_tactile_training_mirror`
- raw tactile sidecar: `franka_remote_tactile/franka_lerobot_20260715_170207_tactile`

The machine-readable report is
`/path/to/VLA_RL_dataset/tactile_force_envelope_20260715/audit.json`.

## Resolved fields

- timestamp: `timestamp`
- state: `observation.state`, shape 8
- executed controller target: `action`, shape 8
- images: three `observation.images.*` fields, each with a `.stamp`
- task: `meta/episodes.jsonl`
- tactile axes: `force_fx`, `force_fy`, `force_fz`

There is no action chunk boundary field in this teleoperation recording. The generated training
windows therefore use a documented 300 ms fixed-stride fallback. A future Online RL recorder
should persist `action_chunk_id` or `action_chunk_start`.

## Rates and alignment

| Stream | Median rate | Relevant timing result |
|---|---:|---:|
| recorded action command | 30.00 Hz | median interval 33.33 ms |
| camera 1 | 29.99 Hz | median age 50.71 ms |
| camera 2 | 29.98 Hz | median age 54.16 ms |
| D515 | 29.92 Hz | median age 26.43 ms; 219 rows older than 200 ms |
| Xense left | 22.19 Hz | median interval 45.06 ms |
| Xense right | 22.19 Hz | median interval 45.06 ms |

Across 300 ms windows, the combined valid tactile count has median 13 and p05/p95 12/14. The
largest per-sensor interval within a window has median 59.88 ms and p95 69.11 ms. Forty candidate
windows have fewer than four valid samples and are dropped.

No robot timestamp duplicate or rollback was found. Tactile values have no consecutive exact
duplicates. The raw sidecar contains 2 left and 3 right timestamp rollbacks, plus four gaps over
150 ms on each sensor; sorting is used only for selection, and these events remain reported as
quality issues. Recorded actions have a 21.98% consecutive duplicate ratio, expected while the
operator holds a pose.

## Built training set

- 70 matched episodes; one unmatched tactile-only episode (`70`) is ignored
- 3,725 accepted windows
- 2,600 train windows from 49 episodes
- 539 calibration windows from 10 episodes
- 586 test windows from 11 episodes
- default target: `top2_mean`
- frozen visual cache: `3725 x 3 x 512`

The target distribution is not identical across episode splits: median `top2_mean` is 23.64 N in
train, 4.71 N in calibration, and 18.82 N in test. This episode-level shift makes the current
calibrated envelope conservative and should be investigated before using it as a hard Online RL
safety threshold.

## Smoke training result

- one-sided conformal `alpha95`: 4.9751
- test coverage: 98.12%
- test center MAE: 8.40 N
- mean predicted sigma: 2.17 N
- mean upper-envelope margin: 10.79 N

The code path is operational, but these numbers are a baseline rather than a production force
model. Calibration is correctly episode-disjoint and conservative; improving episode balance,
zero-load calibration, and explicit action-chunk logging should come before Online RL integration.

