from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _branch(input_dim: int, branch_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, branch_dim),
        nn.LayerNorm(branch_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(branch_dim, branch_dim),
        nn.SiLU(),
    )


class ForceEnvelopeModel(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        state_dim: int,
        action_dim: int,
        action_steps: int,
        num_tasks: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        branch_dim = int(config["branch_dim"])
        hidden_dim = int(config["hidden_dim"])
        dropout = float(config["dropout"])
        task_dim = int(config["task_embedding_dim"])
        self.use_extra_robot_state = bool(config.get("use_extra_robot_state", True))
        self.use_extra_task_embedding = bool(config.get("use_extra_task_embedding", True))
        self.log_var_min = float(config["log_var_min"])
        self.log_var_max = float(config["log_var_max"])
        self.visual = _branch(visual_dim, branch_dim, dropout)
        self.state = _branch(state_dim, branch_dim, dropout) if self.use_extra_robot_state else None
        self.action = _branch(action_steps * action_dim + action_steps, branch_dim, dropout)
        self.task_embedding = (
            nn.Embedding(max(1, num_tasks), task_dim) if self.use_extra_task_embedding else None
        )
        self.task = _branch(task_dim, branch_dim, dropout) if self.use_extra_task_embedding else None
        branch_count = 2 + int(self.use_extra_robot_state) + int(self.use_extra_task_embedding)
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * branch_count, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_input = torch.cat((actions.flatten(1), action_mask), dim=1)
        branches = [self.visual(visual), self.action(action_input)]
        if self.state is not None:
            branches.append(self.state(state))
        if self.task is not None and self.task_embedding is not None:
            branches.append(self.task(self.task_embedding(task_id)))
        encoded = torch.cat(branches, dim=1)
        output = self.fusion(encoded)
        mu = output[:, 0]
        log_var = output[:, 1].clamp(self.log_var_min, self.log_var_max)
        return mu, log_var


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((target - mu).square() * torch.exp(-log_var)) + log_var).mean()


class PreNormEnvelopeDecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, ffn_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        sequence_padding_mask: torch.Tensor,
        condition_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(tokens)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=sequence_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        normalized = self.cross_norm(tokens)
        attended, _ = self.cross_attention(
            normalized,
            condition,
            condition,
            key_padding_mask=condition_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class TransformerEnvelopeExpert(nn.Module):
    """Token-level left/right force envelope predictor conditioned on frozen SmolVLA."""

    def __init__(
        self,
        condition_dim: int,
        state_dim: int,
        action_dim: int,
        action_steps: int,
        config: dict[str, Any],
        history_dim: int = 0,
        history_steps: int = 0,
    ) -> None:
        super().__init__()
        d_model = int(config.get("d_model", 256))
        num_heads = int(config.get("num_heads", 8))
        ffn_dim = int(config.get("ffn_dim", 1024))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        self.log_var_min = float(config.get("log_var_min", -8.0))
        self.log_var_max = float(config.get("log_var_max", 4.0))
        self.action_steps = action_steps
        self.use_action_conditioning = bool(config.get("use_action_conditioning", True))
        self.history_dim = int(history_dim)
        self.history_steps = int(history_steps)
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.LayerNorm(d_model),
        )
        if self.use_action_conditioning:
            self.action_projection = nn.Linear(action_dim, d_model)
            self.action_positions = nn.Parameter(torch.empty(action_steps, d_model))
            nn.init.normal_(self.action_positions, std=0.02)
        else:
            self.action_projection = None
            self.register_parameter("action_positions", None)
        self.state_projection = nn.Linear(state_dim, d_model)
        if self.history_dim > 0 and self.history_steps > 0:
            self.history_projection = nn.Linear(self.history_dim, d_model)
            self.history_positions = nn.Parameter(torch.empty(2, self.history_steps, d_model))
            self.history_sides = nn.Parameter(torch.empty(2, d_model))
            nn.init.normal_(self.history_positions, std=0.02)
            nn.init.normal_(self.history_sides, std=0.02)
        else:
            self.history_projection = None
            self.register_parameter("history_positions", None)
            self.register_parameter("history_sides", None)
        self.envelope_queries = nn.Parameter(torch.empty(2, d_model))
        self.layers = nn.ModuleList([
            PreNormEnvelopeDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.left_distribution = nn.Linear(d_model, 2)
        self.right_distribution = nn.Linear(d_model, 2)
        self.left_contact = nn.Linear(d_model, 1)
        self.right_contact = nn.Linear(d_model, 1)
        nn.init.normal_(self.envelope_queries, std=0.02)

    def _sequence_tokens(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        tactile_history: torch.Tensor | None,
        tactile_history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        state_token = self.state_projection(state)[:, None]
        queries = self.envelope_queries[None].expand(state.shape[0], -1, -1)
        parts = [queries, state_token]
        padding_parts = [
            torch.zeros((state.shape[0], 3), dtype=torch.bool, device=state.device)
        ]
        history_tokens = None
        if self.history_projection is not None:
            if tactile_history is None or tactile_history_mask is None:
                raise ValueError("This model requires causal tactile history and its mask")
            expected = (state.shape[0], 2, self.history_steps, self.history_dim)
            if tuple(tactile_history.shape) != expected:
                raise ValueError(
                    f"tactile_history must be {expected}, got {tuple(tactile_history.shape)}"
                )
            history_tokens = self.history_projection(tactile_history)
            history_tokens = (
                history_tokens
                + self.history_positions[None]
                + self.history_sides[None, :, None, :]
            )
            history_tokens = history_tokens.flatten(1, 2)
            parts.append(history_tokens)
            padding_parts.append(~tactile_history_mask.bool().flatten(1, 2))
        if self.action_projection is not None and self.action_positions is not None:
            action = self.action_projection(actions) + self.action_positions[None]
            parts.append(action)
            padding_parts.append(~action_mask.bool())
        return torch.cat(parts, dim=1), torch.cat(padding_parts, dim=1), history_tokens

    def token_shapes(
        self,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        tactile_history: torch.Tensor | None = None,
        tactile_history_mask: torch.Tensor | None = None,
    ) -> dict[str, list[int]]:
        with torch.no_grad():
            condition = self.condition_projection(condition_tokens)
            state_token = self.state_projection(state)[:, None]
            queries = self.envelope_queries[None].expand(state.shape[0], -1, -1)
            sequence, sequence_padding, history_tokens = self._sequence_tokens(
                state, actions, action_mask, tactile_history, tactile_history_mask
            )
            output = sequence
            for layer in self.layers:
                output = layer(output, condition, sequence_padding, ~condition_mask.bool())
        report = {
            "condition_tokens_raw": list(condition_tokens.shape),
            "condition_mask": list(condition_mask.shape),
            "condition_tokens_projected": list(condition.shape),
            "action_conditioning_enabled": self.use_action_conditioning,
            "state_token": list(state_token.shape),
            "query_tokens": list(queries.shape),
            "decoder_input": list(sequence.shape),
            "transformer_output": list(output.shape),
            "left_env_output": [state.shape[0], 2],
            "right_env_output": [state.shape[0], 2],
        }
        if self.action_projection is not None and self.action_positions is not None:
            action = self.action_projection(actions) + self.action_positions[None]
            report["action_tokens"] = list(action.shape)
        else:
            report["action_tokens"] = []
        if history_tokens is not None and tactile_history_mask is not None:
            report["tactile_history_raw"] = list(tactile_history.shape)
            report["tactile_history_mask"] = list(tactile_history_mask.shape)
            report["tactile_history_tokens"] = list(history_tokens.shape)
        return report

    def forward(
        self,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        tactile_history: torch.Tensor | None = None,
        tactile_history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        condition = self.condition_projection(condition_tokens)
        tokens, sequence_padding, _ = self._sequence_tokens(
            state, actions, action_mask, tactile_history, tactile_history_mask
        )
        for layer in self.layers:
            tokens = layer(tokens, condition, sequence_padding, ~condition_mask.bool())
        tokens = self.output_norm(tokens)
        left = self.left_distribution(tokens[:, 0])
        right = self.right_distribution(tokens[:, 1])
        return {
            "mu_left": left[:, 0],
            "log_var_left": left[:, 1].clamp(self.log_var_min, self.log_var_max),
            "mu_right": right[:, 0],
            "log_var_right": right[:, 1].clamp(self.log_var_min, self.log_var_max),
            "contact_logit_left": self.left_contact(tokens[:, 0]).squeeze(-1),
            "contact_logit_right": self.right_contact(tokens[:, 1]).squeeze(-1),
        }
