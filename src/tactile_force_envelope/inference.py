from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ForceEnvelopeModel, TransformerEnvelopeExpert
from .stable_contact import (
    StableContactHysteresis,
    hard_force_limit_exceeded,
    normalized_soft_cost,
    stable_contact_mask,
)


class MLPForceEnvelopePredictor:
    """Inference adapter for an Online RL action-chunk penalty."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        dimensions = checkpoint["dimensions"]
        self.model = ForceEnvelopeModel(**dimensions, config=checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.norm = checkpoint["normalization"]
        self.task_ids = checkpoint["task_ids"]
        self.calibration = checkpoint["calibration"]
        self.action_steps = int(dimensions["action_steps"])
        self.action_dim = int(dimensions["action_dim"])
        self.control_window_ms = int(checkpoint.get("control_window_ms", 300))
        self.production_ready = bool(checkpoint.get("production_ready", False))
        self.feature_metadata = dict(checkpoint.get("feature_metadata", {}))

    def predict(
        self,
        visual_features: np.ndarray,
        robot_state: np.ndarray,
        control_actions: np.ndarray,
        task: str,
    ) -> dict[str, Any]:
        if task not in self.task_ids:
            raise KeyError(f"Unknown task {task!r}; known tasks: {list(self.task_ids)}")
        visual = torch.as_tensor(visual_features, dtype=torch.float32).flatten()
        state = torch.as_tensor(robot_state, dtype=torch.float32).flatten()
        actions = torch.as_tensor(control_actions, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have shape [N, {self.action_dim}], got {tuple(actions.shape)}")
        if actions.shape[0] != self.action_steps:
            raise ValueError(
                f"control_actions must cover the complete {self.control_window_ms} ms horizon "
                f"with {self.action_steps} commands; got {actions.shape[0]}"
            )
        mask = torch.ones(self.action_steps, dtype=torch.float32)
        visual = (visual - self.norm["visual_mean"]) / self.norm["visual_std"]
        state = (state - self.norm["state_mean"]) / self.norm["state_std"]
        actions = (actions - self.norm["action_mean"]) / self.norm["action_std"]
        with torch.inference_mode():
            mu_norm, log_var_norm = self.model(
                visual[None].to(self.device), state[None].to(self.device),
                actions[None].to(self.device), mask[None].to(self.device),
                torch.tensor([self.task_ids[task]], device=self.device),
            )
        target_mean = float(self.norm["target_mean"])
        target_std = float(self.norm["target_std"])
        mu = float(mu_norm.item() * target_std + target_mean)
        sigma = float(torch.exp(0.5 * log_var_norm).item() * target_std)
        alpha = float(self.calibration["alpha95"])
        return {"mu": mu, "sigma": sigma, "upper95": mu + alpha * sigma}


class TransformerForceEnvelopePredictor:
    """Online adapter for the token-level left/right envelope expert."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        dimensions = checkpoint["dimensions"]
        model_dimensions = {
            key: dimensions[key]
            for key in ("condition_dim", "state_dim", "action_dim", "action_steps")
        }
        self.model = TransformerEnvelopeExpert(
            **model_dimensions,
            config=checkpoint["model_config"],
            history_dim=int(dimensions.get("history_dim", 0)),
            history_steps=int(dimensions.get("history_steps", 0)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.norm = checkpoint["normalization"]
        self.calibration = checkpoint["calibration"]
        self.action_steps = int(dimensions["action_steps"])
        self.action_dim = int(dimensions["action_dim"])
        self.condition_steps = int(dimensions["condition_steps"])
        self.condition_dim = int(dimensions["condition_dim"])
        self.history_dim = int(dimensions.get("history_dim", 0))
        self.history_steps = int(dimensions.get("history_steps", 0))
        self.control_window_ms = int(checkpoint.get("control_window_ms", 300))
        self.production_ready = bool(checkpoint.get("production_ready", False))
        self.feature_metadata = dict(checkpoint.get("feature_metadata", {}))
        self.stable_contact_config = dict(checkpoint.get("stable_contact", {}))

    def predict(
        self,
        condition_tokens: np.ndarray,
        condition_mask: np.ndarray,
        robot_state: np.ndarray,
        control_actions: np.ndarray,
        task: str | None = None,
        tactile_history: np.ndarray | None = None,
        tactile_history_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        del task
        condition = torch.as_tensor(condition_tokens, dtype=torch.float32)
        mask = torch.as_tensor(condition_mask, dtype=torch.bool)
        if tuple(condition.shape) != (self.condition_steps, self.condition_dim):
            raise ValueError(
                f"condition_tokens must be [{self.condition_steps},{self.condition_dim}], "
                f"got {tuple(condition.shape)}"
            )
        if tuple(mask.shape) != (self.condition_steps,):
            raise ValueError(f"condition_mask must be [{self.condition_steps}], got {tuple(mask.shape)}")
        state = torch.as_tensor(robot_state, dtype=torch.float32).flatten()
        actions = torch.as_tensor(control_actions, dtype=torch.float32)
        if tuple(actions.shape) != (self.action_steps, self.action_dim):
            raise ValueError(
                f"control_actions must be [{self.action_steps},{self.action_dim}] for the complete "
                f"{self.control_window_ms} ms horizon; got {tuple(actions.shape)}"
            )
        state = (state - self.norm["state_mean"]) / self.norm["state_std"]
        actions = (actions - self.norm["action_mean"]) / self.norm["action_std"]
        action_mask = torch.ones(self.action_steps, dtype=torch.bool)
        history = None
        history_mask = None
        stable_gates = np.asarray([False, False], dtype=bool)
        if self.history_steps > 0:
            if tactile_history is None or tactile_history_mask is None:
                raise ValueError("This checkpoint requires causal tactile history and its mask")
            history = torch.as_tensor(tactile_history, dtype=torch.float32)
            history_mask = torch.as_tensor(tactile_history_mask, dtype=torch.bool)
            expected = (2, self.history_steps, self.history_dim)
            if tuple(history.shape) != expected:
                raise ValueError(f"tactile_history must be {expected}, got {tuple(history.shape)}")
            if tuple(history_mask.shape) != expected[:2]:
                raise ValueError(
                    f"tactile_history_mask must be {expected[:2]}, got {tuple(history_mask.shape)}"
                )
            raw_history = history.clone()
            stable_gates = stable_contact_mask(
                raw_history.numpy(), history_mask.numpy(),
                magnitude_index=int(self.stable_contact_config.get("magnitude_index", 3)),
                tau_on=float(self.stable_contact_config.get("tau_on", 5.0)),
                recent_samples=int(self.stable_contact_config.get("recent_samples", 4)),
                required_samples=int(self.stable_contact_config.get("required_samples", 3)),
            )
            history = (
                history - self.norm["history_mean"][:, None]
            ) / self.norm["history_std"][:, None]
            history = torch.where(history_mask[..., None], history, torch.zeros_like(history))
        with torch.inference_mode():
            output = self.model(
                condition[None].to(self.device),
                mask[None].to(self.device),
                state[None].to(self.device),
                actions[None].to(self.device),
                action_mask[None].to(self.device),
                history[None].to(self.device) if history is not None else None,
                history_mask[None].to(self.device) if history_mask is not None else None,
            )
        result: dict[str, Any] = {}
        for side in ("left", "right"):
            target_mean = float(self.norm[f"target_mean_{side}"])
            target_std = float(self.norm[f"target_std_{side}"])
            mu = float(output[f"mu_{side}"].item() * target_std + target_mean)
            sigma = float(
                torch.exp(0.5 * output[f"log_var_{side}"].float()).item() * target_std
            )
            alpha = float(self.calibration[f"alpha95_{side}"])
            additive = self.calibration.get(f"q95_{side}")
            upper_scale = mu + alpha * sigma
            result[f"mu_{side}"] = mu
            result[f"sigma_{side}"] = sigma
            result[f"upper95_scale_{side}"] = upper_scale
            if additive is not None:
                result[f"upper95_additive_{side}"] = mu + float(additive)
            result[f"upper95_{side}"] = upper_scale
            result[f"stable_contact_{side}"] = bool(stable_gates[("left", "right").index(side)])
            result[f"contact_probability_{side}"] = float(
                torch.sigmoid(output[f"contact_logit_{side}"].float()).item()
            )
        return result


class StableContactOnlineEnvelope:
    """Stateful gate/cost adapter for Online RL; hard safety never depends on the model."""

    def __init__(
        self, predictor: TransformerForceEnvelopePredictor, *,
        hard_force_limit: float,
        tau_on: float = 5.0, tau_off: float = 3.0,
        force_scale: float = 5.0, cost_clip: float = 2.0,
    ) -> None:
        self.predictor = predictor
        self.gate = StableContactHysteresis(tau_on=tau_on, tau_off=tau_off)
        self.force_scale = float(force_scale)
        self.cost_clip = float(cost_clip)
        self.hard_force_limit = float(hard_force_limit)
        if self.hard_force_limit <= 0.0:
            raise ValueError("hard_force_limit must be explicitly configured and positive")

    def update_tactile(
        self, left_force: float, right_force: float, *,
        left_valid: bool = True, right_valid: bool = True,
    ) -> dict[str, Any]:
        raw_force_max = max(float(left_force), float(right_force))
        return {
            "gate": self.gate.update_pair(
                left_force, right_force,
                left_valid=left_valid, right_valid=right_valid,
            ),
            "hard_stop": hard_force_limit_exceeded(raw_force_max, self.hard_force_limit),
            "raw_force_max": raw_force_max,
        }

    def cost(
        self, left_force: float, right_force: float, prediction: dict[str, float],
    ) -> dict[str, float]:
        active = self.gate.active
        left = normalized_soft_cost(
            left_force, prediction["upper95_left"], active["left"],
            force_scale=self.force_scale, cost_clip=self.cost_clip,
        )
        right = normalized_soft_cost(
            right_force, prediction["upper95_right"], active["right"],
            force_scale=self.force_scale, cost_clip=self.cost_clip,
        )
        return {"left": left, "right": right, "total": left + right}


def ForceEnvelopePredictor(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") == "transformer_envelope":
        return TransformerForceEnvelopePredictor(checkpoint_path, device=device)
    return MLPForceEnvelopePredictor(checkpoint_path, device=device)
