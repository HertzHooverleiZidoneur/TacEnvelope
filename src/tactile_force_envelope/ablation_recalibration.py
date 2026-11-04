from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bilateral_sigma_calibration import (
    _finite_sample_tail_quantile,
    bilateral_uncertainty_scale,
)
from .common import write_json
from .stable_contact_experiment import _gates_for_rows, _load_evaluation_context


LOWER_PROBABILITY = 0.05
UPPER_PROBABILITY = 0.95
SEEDS = (42, 43, 44)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_window_fingerprint(
    rows: list[dict[str, Any]], mask: np.ndarray
) -> str:
    selected = [
        {
            "batch_id": str(row.get("batch_id", "")),
            "episode_index": int(row["episode_index"]),
            "window_index": int(row["window_index"]),
        }
        for row, keep in zip(rows, mask, strict=True)
        if bool(keep)
    ]
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _best_state_matches(checkpoint_path: Path) -> tuple[bool, str]:
    best_path = checkpoint_path.parent / "transformer_best_training_state.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing best-validation state: {best_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["model_state_dict"]
    best_state = best["model_state_dict"]
    if checkpoint_state.keys() != best_state.keys():
        return False, str(best_path)
    matches = all(
        torch.equal(checkpoint_state[key].cpu(), best_state[key].cpu())
        for key in checkpoint_state
    )
    if int(checkpoint["best_epoch"]) != int(best["epoch"]):
        matches = False
    return matches, str(best_path)


def _interval_metrics(
    target: np.ndarray,
    mu: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    residual = target - mu
    width = upper - lower
    covered = (target >= lower) & (target <= upper)
    return {
        "mae_N": float(np.mean(np.abs(residual))),
        "rmse_N": float(np.sqrt(np.mean(residual**2))),
        "coverage": float(np.mean(covered)),
        "mean_envelope_width_N": float(np.mean(width)),
        "width_p05_N": float(np.quantile(width, 0.05)),
        "width_p50_N": float(np.quantile(width, 0.50)),
        "width_p95_N": float(np.quantile(width, 0.95)),
        "out_of_band_windows": int(np.sum(~covered)),
    }


def _aggregate(runs: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = np.asarray([float(run["test"][metric]) for run in runs])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def _checkpoint_run(
    base_config: dict[str, Any],
    *,
    checkpoint_path: Path,
    seed: int,
    variant: str,
    use_action_conditioning: bool,
    envelope_type: str,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Required checkpoint is missing; refusing to retrain: {checkpoint_path}"
        )
    config = copy.deepcopy(base_config)
    config["training"]["seed"] = int(seed)
    config["model"]["use_action_conditioning"] = bool(use_action_conditioning)
    checkpoint, _, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, checkpoint_path)
    )
    matches_best, best_state_path = _best_state_matches(checkpoint_path)
    if not matches_best:
        raise RuntimeError(
            f"Deployable checkpoint does not match best-validation state: {checkpoint_path}"
        )
    actual_action_setting = bool(
        checkpoint["model_config"].get("use_action_conditioning", True)
    )
    if actual_action_setting != bool(use_action_conditioning):
        raise RuntimeError(
            f"Action-conditioning mismatch for {checkpoint_path}: "
            f"checkpoint={actual_action_setting}, expected={use_action_conditioning}"
        )

    masks: dict[str, np.ndarray] = {}
    for split in ("calibration", "test"):
        gates = _gates_for_rows(grouped[split], history, history_mask, config)
        masks[split] = gates[:, 0] & gates[:, 1]
        if not np.any(masks[split]):
            raise ValueError(f"No bilateral stable-contact windows in {split}")

    values: dict[str, dict[str, np.ndarray]] = {}
    for split in ("calibration", "test"):
        left = predictions[split]["left"]
        right = predictions[split]["right"]
        values[split] = {
            "target": 0.5 * (left["target"] + right["target"]),
            "mu": 0.5 * (left["mu"] + right["mu"]),
            "uncertainty_scale": bilateral_uncertainty_scale(
                left["sigma"], right["sigma"], "mean"
            ),
        }

    cal_mask = masks["calibration"]
    test_mask = masks["test"]
    cal_residual = (
        values["calibration"]["target"][cal_mask]
        - values["calibration"]["mu"][cal_mask]
    )
    test_target = values["test"]["target"][test_mask]
    test_mu = values["test"]["mu"][test_mask]
    calibration: dict[str, Any]
    if envelope_type == "uncertainty_scaled":
        if not bool(checkpoint["loss"].get("sigma_trained", False)):
            raise RuntimeError(
                f"Uncertainty-scaled calibration requires trained sigma: {checkpoint_path}"
            )
        cal_scale = values["calibration"]["uncertainty_scale"][cal_mask]
        test_scale = values["test"]["uncertainty_scale"][test_mask]
        scores = cal_residual / np.maximum(cal_scale, 1e-6)
        alpha05, lower_rank = _finite_sample_tail_quantile(
            scores, LOWER_PROBABILITY, upper=False
        )
        alpha95, upper_rank = _finite_sample_tail_quantile(
            scores, UPPER_PROBABILITY, upper=True
        )
        lower = test_mu + alpha05 * test_scale
        upper = test_mu + alpha95 * test_scale
        calibration = {
            "type": "uncertainty_scaled_finite_sample",
            "score": "s = (y_avg - mu_avg) / max(u_mean, 1e-6)",
            "scale_definition": "u_mean = 0.5 * (sigma_left + sigma_right)",
            "scale_is_bilateral_standard_deviation": False,
            "alpha05": alpha05,
            "alpha95": alpha95,
            "alpha05_rank_one_based": lower_rank,
            "alpha95_rank_one_based": upper_rank,
            "formula": "[mu_avg + alpha05*u_mean, mu_avg + alpha95*u_mean]",
        }
    elif envelope_type == "additive":
        if bool(checkpoint["loss"].get("sigma_trained", False)):
            raise RuntimeError(
                "Additive branch in this ablation is reserved for SmoothL1-only"
            )
        q05, lower_rank = _finite_sample_tail_quantile(
            cal_residual, LOWER_PROBABILITY, upper=False
        )
        q95, upper_rank = _finite_sample_tail_quantile(
            cal_residual, UPPER_PROBABILITY, upper=True
        )
        lower = test_mu + q05
        upper = test_mu + q95
        calibration = {
            "type": "additive_fixed_width_finite_sample",
            "score": "r = y_avg - mu_avg",
            "q05_N": q05,
            "q95_N": q95,
            "q05_rank_one_based": lower_rank,
            "q95_rank_one_based": upper_rank,
            "formula": "[mu_avg + q05, mu_avg + q95]",
            "predicted_sigma_used": False,
        }
    else:
        raise ValueError(f"Unknown envelope_type: {envelope_type}")

    test_metrics = _interval_metrics(test_target, test_mu, lower, upper)
    test_episode_keys = {
        (str(row.get("batch_id", "")), int(row["episode_index"]))
        for row, keep in zip(grouped["test"], test_mask, strict=True)
        if bool(keep)
    }
    return {
        "variant": variant,
        "seed": int(seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_validation_state": best_state_path,
        "best_validation_state_sha256": _sha256(Path(best_state_path)),
        "matches_best_validation_state": matches_best,
        "best_epoch": int(checkpoint["best_epoch"]),
        "loss": checkpoint["loss"],
        "use_action_conditioning": actual_action_setting,
        "selection": "stable_prior_left AND stable_prior_right",
        "selection_uses_future_target": False,
        "calibration_samples": int(np.sum(cal_mask)),
        "test_samples": int(np.sum(test_mask)),
        "test_episodes": len(test_episode_keys),
        "calibration_window_fingerprint": _selected_window_fingerprint(
            grouped["calibration"], cal_mask
        ),
        "test_window_fingerprint": _selected_window_fingerprint(
            grouped["test"], test_mask
        ),
        "calibration": calibration,
        "test": test_metrics,
    }


def _check_identical_windows(runs: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = {run["calibration_window_fingerprint"] for run in runs}
    test = {run["test_window_fingerprint"] for run in runs}
    if len(calibration) != 1 or len(test) != 1:
        raise RuntimeError("Compared checkpoints do not use identical selected windows")
    return {
        "identical_calibration_windows": True,
        "identical_test_windows": True,
        "calibration_window_fingerprint": next(iter(calibration)),
        "test_window_fingerprint": next(iter(test)),
    }


def _prediction_consistency(
    old_summary_path: Path, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    if not old_summary_path.is_file():
        return {"old_summary_found": False}
    old = json.loads(old_summary_path.read_text(encoding="utf-8"))
    old_by_key = {(run["variant"], int(run["seed"])): run for run in old["runs"]}
    comparisons = []
    for run in runs:
        prior = old_by_key[(run["variant"], int(run["seed"]))]["additive_test"]
        mae_delta = float(run["test"]["mae_N"] - prior["mae_N"])
        rmse_delta = float(run["test"]["rmse_N"] - prior["rmse_N"])
        comparisons.append({
            "variant": run["variant"],
            "seed": run["seed"],
            "mae_delta_N": mae_delta,
            "rmse_delta_N": rmse_delta,
            "unchanged_within_1e-6_N": (
                abs(mae_delta) <= 1e-6 and abs(rmse_delta) <= 1e-6
            ),
        })
    return {
        "old_summary_found": True,
        "old_summary": str(old_summary_path),
        "comparison_tolerance_N": 1e-6,
        "all_prediction_metrics_unchanged_within_tolerance": all(
            row["unchanged_within_1e-6_N"] for row in comparisons
        ),
        "comparisons": comparisons,
    }


def _common_metadata(config_path: Path, command: str) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "exact_command": command,
        "models_retrained": False,
        "checkpoint_selection": "best validation loss only",
        "calibration_source": "calibration split only",
        "test_used_to_fit_calibration": False,
        "nominal_coverage": 0.90,
        "evaluation_status": "development evaluation",
        "development_evaluation_reason": (
            "The existing test split has previously been used for model diagnostics "
            "and design comparisons; an independent blind episode set is still required."
        ),
    }


def _mean_std(metric: dict[str, float], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{metric['mean'] * scale:.3f} +/- {metric['std'] * scale:.3f}{suffix}"


def _write_markdown_report(
    output: Path,
    action: dict[str, Any],
    objective: dict[str, Any],
) -> None:
    action_rows = []
    for variant, label in (
        ("full_action", "Full Action"),
        ("without_action", "Without Action"),
    ):
        metrics = action["aggregate"][variant]
        action_rows.append(
            f"| {label} | {_mean_std(metrics['mae_N'])} | "
            f"{_mean_std(metrics['rmse_N'])} | "
            f"{_mean_std(metrics['coverage'], percent=True)} | "
            f"{_mean_std(metrics['mean_envelope_width_N'])} |"
        )

    action_seed_rows = []
    for run in action["runs"]:
        calibration = run["calibration"]
        metrics = run["test"]
        action_seed_rows.append(
            f"| {run['variant']} | {run['seed']} | {run['calibration_samples']} | "
            f"{run['test_samples']} | {calibration['alpha05']:.6f} | "
            f"{calibration['alpha95']:.6f} | {metrics['mae_N']:.4f} | "
            f"{metrics['rmse_N']:.4f} | {metrics['coverage'] * 100:.2f}% | "
            f"{metrics['mean_envelope_width_N']:.4f} | "
            f"{metrics['width_p05_N']:.4f}/{metrics['width_p50_N']:.4f}/"
            f"{metrics['width_p95_N']:.4f} |"
        )

    objective_rows = []
    for variant, label in (
        ("smooth_l1_only", "SmoothL1-only"),
        ("gaussian_nll", "Gaussian NLL + 0.5 SmoothL1"),
    ):
        metrics = objective["aggregate"][variant]
        envelope = (
            "additive fixed-width" if variant == "smooth_l1_only"
            else "uncertainty-scaled"
        )
        objective_rows.append(
            f"| {label} | {envelope} | {_mean_std(metrics['mae_N'])} | "
            f"{_mean_std(metrics['rmse_N'])} | "
            f"{_mean_std(metrics['coverage'], percent=True)} | "
            f"{_mean_std(metrics['mean_envelope_width_N'])} |"
        )

    objective_seed_rows = []
    for run in objective["runs"]:
        calibration = run["calibration"]
        metrics = run["test"]
        if run["variant"] == "smooth_l1_only":
            tails = f"q05={calibration['q05_N']:.6f}, q95={calibration['q95_N']:.6f} N"
        else:
            tails = (
                f"alpha05={calibration['alpha05']:.6f}, "
                f"alpha95={calibration['alpha95']:.6f}"
            )
        objective_seed_rows.append(
            f"| {run['variant']} | {run['seed']} | {tails} | "
            f"{metrics['coverage'] * 100:.2f}% | "
            f"{metrics['mean_envelope_width_N']:.4f} | "
            f"{metrics['width_p05_N']:.4f}/{metrics['width_p50_N']:.4f}/"
            f"{metrics['width_p95_N']:.4f} |"
        )

    smooth = objective["aggregate"]["smooth_l1_only"]
    gaussian = objective["aggregate"]["gaussian_nll"]
    gaussian_mae_gain = 100.0 * (
        smooth["mae_N"]["mean"] - gaussian["mae_N"]["mean"]
    ) / smooth["mae_N"]["mean"]
    gaussian_rmse_gain = 100.0 * (
        smooth["rmse_N"]["mean"] - gaussian["rmse_N"]["mean"]
    ) / smooth["rmse_N"]["mean"]
    gaussian_width_gain = 100.0 * (
        smooth["mean_envelope_width_N"]["mean"]
        - gaussian["mean_envelope_width_N"]["mean"]
    ) / smooth["mean_envelope_width_N"]["mean"]

    checkpoints = []
    seen = set()
    for run in [*action["runs"], *objective["runs"]]:
        key = run["checkpoint"]
        if key in seen:
            continue
        seen.add(key)
        checkpoints.append(
            f"- `{run['variant']}` seed {run['seed']}: `{key}`  "
            f"SHA256 `{run['checkpoint_sha256']}`"
        )

    report = f"""# USB Envelope Expert 消融重评估

## 评估协议

- 未重新训练任何模型；逐个加载已有 validation-best deployable checkpoint。
- Calibration 仅使用 calibration split，test target 未参与 alpha/q 拟合。
- Calibration/test 均只保留左右 stable-contact gate 同时开启的窗口。
- 每个模型使用相同的 580 个 calibration 窗口和 598 个 test 窗口。
- 当前 test split 此前参与过方法诊断，因此本报告属于 development evaluation；最终论文结论仍需独立 blind episodes。
- `u_mean=(sigma_L+sigma_R)/2` 是经验 uncertainty scale，不称为双侧平均力的标准差。

执行命令：

```bash
{action['exact_command']}
```

## A. Action Conditioning

| Method | MAE (N) | RMSE (N) | 90% Coverage | Mean Width (N) |
|---|---:|---:|---:|---:|
{chr(10).join(action_rows)}

移除动作后，MAE 增加 **{action['comparison']['mae_change_when_removing_action_percent']:.2f}%**，
RMSE 增加 **{action['comparison']['rmse_change_when_removing_action_percent']:.2f}%**。
这支持 action conditioning 改善点预测；不要声称它同时缩窄了区间，因为两者平均宽度接近。

| Method | Seed | Cal N | Test N | alpha05 | alpha95 | MAE | RMSE | Coverage | Mean Width | Width P05/P50/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(action_seed_rows)}

## B. Probabilistic Objective

| Method | Envelope | MAE (N) | RMSE (N) | 90% Coverage | Mean Width (N) |
|---|---|---:|---:|---:|---:|
{chr(10).join(objective_rows)}

Gaussian 完整方案相对 SmoothL1-only：MAE 降低 **{gaussian_mae_gain:.2f}%**，
RMSE 降低 **{gaussian_rmse_gain:.2f}%**。这是相同 test windows 上的纯点预测比较。

完整 envelope 设计中，Gaussian uncertainty-scaled band 的平均宽度缩小
**{gaussian_width_gain:.2f}%**。但该比较同时改变了训练目标和 calibration 方法，
不能把 coverage/width 差异单独归因于 Gaussian NLL。

| Method | Seed | Calibration parameters | Coverage | Mean Width | Width P05/P50/P95 |
|---|---:|---|---:|---:|---:|
{chr(10).join(objective_seed_rows)}

## 一致性审计

- 所有 calibration 窗口指纹一致：`{action['consistency']['calibration_window_fingerprint']}`
- 所有 test 窗口指纹一致：`{action['consistency']['test_window_fingerprint']}`
- 新旧 MAE/RMSE 在 `1e-6 N` 浮点容差内一致：
  `{action['prediction_consistency_with_previous_results']['all_prediction_metrics_unchanged_within_tolerance']}`
- 所有 deployable checkpoint 均逐 tensor 匹配对应 best-validation state。
- SmoothL1-only 的未训练 sigma head 完全未用于 calibration。

## Checkpoints

{chr(10).join(checkpoints)}

## 论文表格建议

Action ablation 表替换为本报告 A 表。Probabilistic ablation 正文优先报告 B1 的 MAE/RMSE；
B2 的 coverage/width 应标注为 complete envelope design comparison。推荐保留 Gaussian 方案，
因为它在三 seed 平均上同时降低 MAE/RMSE，并在约 90% coverage 下得到更窄的动态区间。
"""
    (output / "USB_ABLATION_SIGMA_SCALED_REEVALUATION_ZH.md").write_text(
        report, encoding="utf-8"
    )


def reevaluate_usb_ablations(
    config: dict[str, Any], *, config_path: Path, seeds: tuple[int, ...] = SEEDS
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    output = workspace / "stable_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    command = (
        f"cd {repository} && PYTHONPATH=src {sys.executable} "
        "scripts/evaluation/reevaluate_usb_ablation_sigma_scaled.py "
        f"--config {config_path} --device {config['training']['device']} "
        "--seeds " + " ".join(str(seed) for seed in seeds)
    )

    action_runs = []
    for variant, use_action in (("full_action", True), ("without_action", False)):
        for seed in seeds:
            checkpoint = (
                workspace / "gaussian_loss_ablation" / f"gaussian_nll_seed_{seed}"
                / "transformer_force_envelope_model.pt"
                if variant == "full_action"
                else workspace / "action_conditioning_ablation"
                / f"without_action_seed_{seed}" / "transformer_force_envelope_model.pt"
            )
            action_runs.append(_checkpoint_run(
                config,
                checkpoint_path=checkpoint,
                seed=seed,
                variant=variant,
                use_action_conditioning=use_action,
                envelope_type="uncertainty_scaled",
            ))

    action_metrics = (
        "mae_N",
        "rmse_N",
        "coverage",
        "mean_envelope_width_N",
        "width_p05_N",
        "width_p50_N",
        "width_p95_N",
    )
    action_aggregate = {
        variant: {
            metric: _aggregate(
                [run for run in action_runs if run["variant"] == variant], metric
            )
            for metric in action_metrics
        }
        for variant in ("full_action", "without_action")
    }
    full = action_aggregate["full_action"]
    removed = action_aggregate["without_action"]
    action_report = {
        "schema_version": 1,
        "experiment": "USB action-conditioning ablation after uncertainty-scale calibration",
        **_common_metadata(config_path, command),
        "target": "y_avg = 0.5 * (y_left + y_right)",
        "center": "mu_avg = 0.5 * (mu_left + mu_right)",
        "uncertainty_scale": "u_mean = 0.5 * (sigma_left + sigma_right)",
        "uncertainty_scale_is_standard_deviation": False,
        "consistency": _check_identical_windows(action_runs),
        "prediction_consistency_with_previous_results": _prediction_consistency(
            workspace / "action_conditioning_ablation" / "summary.json", action_runs
        ),
        "runs": action_runs,
        "aggregate": action_aggregate,
        "comparison": {
            "mae_change_when_removing_action_percent": 100.0
            * (removed["mae_N"]["mean"] - full["mae_N"]["mean"])
            / full["mae_N"]["mean"],
            "rmse_change_when_removing_action_percent": 100.0
            * (removed["rmse_N"]["mean"] - full["rmse_N"]["mean"])
            / full["rmse_N"]["mean"],
        },
    }

    objective_runs = []
    for variant, envelope_type in (
        ("smooth_l1_only", "additive"),
        ("gaussian_nll", "uncertainty_scaled"),
    ):
        for seed in seeds:
            checkpoint = (
                workspace / "gaussian_loss_ablation" / f"{variant}_seed_{seed}"
                / "transformer_force_envelope_model.pt"
            )
            objective_runs.append(_checkpoint_run(
                config,
                checkpoint_path=checkpoint,
                seed=seed,
                variant=variant,
                use_action_conditioning=True,
                envelope_type=envelope_type,
            ))

    objective_aggregate = {
        variant: {
            metric: _aggregate(
                [run for run in objective_runs if run["variant"] == variant], metric
            )
            for metric in action_metrics
        }
        for variant in ("smooth_l1_only", "gaussian_nll")
    }
    objective_report = {
        "schema_version": 1,
        "experiment": "USB probabilistic-objective ablation after calibration update",
        **_common_metadata(config_path, command),
        "prediction_only_comparison": (
            "MAE/RMSE use identical test targets and do not depend on calibration."
        ),
        "envelope_comparison_warning": (
            "SmoothL1 uses additive residual calibration, while Gaussian uses "
            "uncertainty-scaled calibration. Coverage/width differences therefore "
            "evaluate the complete envelope design and cannot be attributed solely "
            "to Gaussian NLL."
        ),
        "smooth_l1_sigma_policy": (
            "The SmoothL1-only checkpoint has sigma_trained=false; its sigma output "
            "is ignored and no replacement sigma is invented."
        ),
        "consistency": _check_identical_windows(objective_runs),
        "prediction_consistency_with_previous_results": _prediction_consistency(
            workspace / "gaussian_loss_ablation" / "summary.json", objective_runs
        ),
        "runs": objective_runs,
        "aggregate": objective_aggregate,
    }

    write_json(output / "action_conditioning_sigma_scaled_ablation.json", action_report)
    write_json(output / "probabilistic_objective_updated_ablation.json", objective_report)
    _write_markdown_report(output, action_report, objective_report)
    return action_report, objective_report
