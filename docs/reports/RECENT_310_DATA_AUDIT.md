# Recent 310-episode data audit

Audited on 2026-07-16:

| Dataset | Episodes | Frames | Accepted 300 ms force windows |
|---|---:|---:|---:|
| `franka_lerobot_20260715_170207_tactile` | 70 | 36,817 | 3,725 |
| `franka_lerobot_20260716_124858_tactile` | 54 | 26,348 | 2,677 |
| `franka_lerobot_20260716_135727_tactile` | 86 | 37,195 | 3,738 |
| `franka_lerobot_20260716_162300_tactile` | 100 | 41,223 | 4,156 |
| Total | 310 | 141,583 | 14,296 |

## Validated

- Every raw robot episode has a matching workstation Xense sidecar.
- State and action are consistently 8-dimensional and contain no NaN or infinity.
- Robot/action timestamps contain no duplicate or rollback.
- Recorded action command rate is 30.00 Hz in all four datasets.
- Camera 1 and camera 2 are approximately 29.98--29.99 Hz.
- D515 is approximately 29.92--29.97 Hz.
- Both raw Xense force streams are approximately 22.18--22.20 Hz.
- Referenced raw camera files exist. Initial null references are retained as missing data rather
  than silently redirected to a future frame.
- The three newly exported v3 datasets have matching Parquet row counts and metadata lengths,
  no duplicate frame IDs, and no timestamp rollback.
- All 30 MP4 files in the three new v3 datasets decode without error. Every camera/tactile video
  stream has exactly the same number of frames as its v3 Parquet table.
- The task string is consistently
  `Insert the red USB flash drive into the USB port on the black power strip`.

The `135727` v3 snapshot initially contained only episodes 0--84 while raw episode 85 was valid.
It was re-exported and now contains all 86 episodes and 37,195 frames.

## Timing quality

For the 300 ms force-label windows, the combined two-sensor sample count has median 13. The
per-sensor maximum interval within a window is typically about 60 ms. The force-window builder
drops windows with fewer than four raw tactile samples; 14,296 windows remain.

D515 has 506 recorded rows older than 200 ms across the 141,583 raw frames (0.36%). These samples
are reported by the audit and can be excluded by an image-age quality rule.

The old nearest-sample merge logic also embedded invalid tactile values in a small number of raw
rows: 1,237 entries selected a future first sample and 520 entries reused a sample older than
200 ms. This is 0.62% of the two-sensor frame entries. The raw sidecars are intact, and the force
envelope pipeline reads those sidecars directly, so its labels are not contaminated. The merge
script now writes invalid zero values for these cases. Existing datasets were not rewritten.

## Remaining caveat

After per-episode zero-bias removal, `top2_mean` force targets reach roughly 76--97 in the current
SDK output units, and their median changes noticeably between episode splits. Before treating
these values as a hard Newton safety threshold, verify the Xense `resultant` unit and calibration
with a known load. They are usable as internally consistent supervision now, but the physical
unit should not be asserted without that check.

