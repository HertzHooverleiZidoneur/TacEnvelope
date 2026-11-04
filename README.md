# TacEnvelope

**A Plug-in Framework for Utilizing Tactile Envelope Feedback in RL Post-Training of VLA Policies**

[Project page](https://HertzHooverleiZidoneur.github.io/TacEnvelope/) · [Paper](docs/assets/TacEnvelope.pdf) · [Demo video](docs/assets/demo.mp4) · [Publishing guide](PUBLISHING_ZH.md)

Huanzhang Hu, Bo Chen, Fangdi Yu, Haoyi Zhang, Xianxin Lai, Jiahui Chen,
Haijier Chen, Xiaoquan Sun, Dongchen Zheng, Yijun Hong, Zurui Chen, Wenqi Ge,
and Jiayu Chen.

This repository combines the supplied code release, calibrated task checkpoints,
the lab-version manuscript, and the project video. The static project page lives
in `docs/` and is ready for GitHub Pages.

Vision-conditioned **tactile force envelope prediction** for contact-rich manipulation,
and its use as an **online-RL reward signal** on a real Franka arm.

An *envelope expert* is a compact Transformer that predicts, for each tactile pad
(left / right), a distribution over contact force — mean `mu`, standard deviation
`sigma`, and a calibrated `upper95` bound — rather than a single force value.
The predicted envelope drives a reward/cost signal for online RL: staying inside
the envelope reinforces compliant, stable contact; exceeding it triggers a
sigma-scaled penalty.

## Method summary

The expert is conditioned on four input streams:

| Input | Detail |
|---|---|
| Vision / language | Condition tokens from a **frozen SmolVLA** checkpoint |
| Robot state | Current joint / EE state |
| Candidate actions | Future **300 ms** of candidate execution actions (9 steps) |
| Tactile history | Causal **200 ms** window of real tactile readings |

The current online reward uses a **bilateral uncertainty-scaled conformal band**.
The per-pad means and scales are averaged, then calibrated lower and upper
standardized-residual quantiles produce `L05` and `U95` (a nominal central
**90%** interval). The averaged uncertainty scale is not interpreted as the
standard deviation of bilateral force. A `fixed_upper` criterion is provided
for ablation. Two execution modes are supported:

- `dual_rate` — high-rate tactile windows evaluated against a low-rate action chunk
- `posthoc_single_rate` — single-rate evaluation after chunk execution

## Repository layout

```text
src/tactile_force_envelope/     Core package
  model.py                      Transformer envelope expert
  transformer_train.py          Training / validation / calibration / testing
  tactile_history.py            200 ms causal history construction
  smolvla_condition.py          SmolVLA condition-token extraction
  inference.py                  Online inference interface
  online_rl.py                  Dual-rate online-RL integration
  pi05_online_service.py        Sidecar service (envelope -> penalty -> reward)
  stable_contact.py             Stable-contact gate and soft cost
tests/                          Unit tests (core, calibration, online RL, service)
configs/                        production / experiments / ablations / reference
scripts/                        training / analysis / evaluation / diagnostics
checkpoints/                    Calibrated pretrained checkpoints (see below)
integrations/xense/             Xense sensor capture example (sensor SDK not included)
examples/                       Minimal inference example
docs/                           Architecture and per-task guides
```

## Installation

```bash
python -m pip install -e .
# or
python -m pip install -r requirements.txt
```

Requires Python >= 3.10, PyTorch >= 2.2. Install the test dependency with
`python -m pip install -e '.[dev]'`.

Standalone package tests:

```bash
python -m pytest tests -q --ignore=tests/test_online_rl.py
```

The online-RL adapter test additionally imports
`rlt_openpi.rollout.reward_provider` from the separate robot-side RLT runtime.
With that runtime installed, run the complete suite with `python -m pytest tests -q`.

## Quickstart: inference

`examples/predict_from_npz.py` runs the expert on a recorded window file:

```bash
python examples/predict_from_npz.py \
  --checkpoint checkpoints/production/pi05_jax_zrl_stable_contact.pt \
  --input /path/to/recorded_window.npz \
  --device cpu
```

Online service entry point (envelope sidecar):

```bash
./run_pi05_jax_envelope_sidecar.sh
```

## Pretrained checkpoints

Calibrated production checkpoints are shipped with the repository (three tasks):

| Path | Task |
|---|---|
| `checkpoints/production/` | USB insertion (main benchmark) |
| `checkpoints/ink_beaker/` | Ink-beaker handling |
| `checkpoints/pastry/` | Pastry handling |

Each directory contains the calibrated expert (`.pt`) and its bilateral /
sigma calibration files (`.json`). Training datasets are **not** included.

## Configuration

Copy `.env.example` to `.env` and fill in your local paths (dataset roots,
SmolVLA checkpoint, output directory). Experiment configs live under
`configs/experiments/`.

## Training

```bash
./run.sh            # data audit + training pipeline (scripts/training/run_pipeline.sh)
```

Window extraction, condition-token caching, training, calibration and
evaluation are separate stages; see `docs/guides/` for per-task walkthroughs
(USB / ink-beaker / pastry).

## Hardware prerequisites

- **Xense tactile sensor** — the vendor SDK (`daimon SDK`) is proprietary and
  **not** included; obtain it from the sensor manufacturer. `integrations/xense/`
  shows the capture-side interface used on the robot PC.
- The robot-side runtime (Franka FCI impedance control, GELLO teleoperation,
  ROS 2 bridge) is a separate deployment and not part of this repository.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

If you use this work, please cite:

```bibtex
@misc{hu2026tacenvelope,
  title = {TacEnvelope: A Plug-in Framework for Utilizing Tactile
           Envelope Feedback in RL Post-Training of VLA Policies},
  author = {Hu, Huanzhang and Chen, Bo and Yu, Fangdi and
            Zhang, Haoyi and Lai, Xianxin and Chen, Jiahui and
            Chen, Haijier and Sun, Xiaoquan and Zheng, Dongchen and
            Hong, Yijun and Chen, Zurui and Ge, Wenqi and Chen, Jiayu},
  year = {2026},
  url = {https://agentic-intelligence-lab.org/files/TacEnvelope.pdf}
}
```

## Project website

The website is plain HTML, CSS, and JavaScript. It uses local images, PDF, and
MP4 assets and requires no build step, package installation, or external font service.

Preview it from the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. You can also open `docs/index.html` directly.

The checked-in GitHub Actions workflow deploys `docs/` after pushes to `main`.
Enable **Settings → Pages → Source → GitHub Actions** in the destination
repository first. See [PUBLISHING_ZH.md](PUBLISHING_ZH.md) for publication and
the lab website integration.
