from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn


FEATURE_TAP = (
    "policy.model.vlm_with_expert.forward: past_key_values returned by prefix forward and consumed "
    "by policy.model.denoise_step"
)
MASK_TAP = "policy.model.embed_prefix: prefix_pad_masks"
TOKEN_FEATURE_TAP = (
    "policy.model.vlm_with_expert.get_vlm_model().text_model.layers[-1].input_layernorm: "
    "multimodal prefix hidden states immediately before the final-layer Q/K/V projections"
)


def checkpoint_fingerprint(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    ):
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"SmolVLA checkpoint is missing {path}")
        digest.update(name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def masked_mean(tokens: Tensor, valid_mask: Tensor) -> Tensor:
    if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"Expected tokens [B,N,D] and mask [B,N], got {tokens.shape}, {valid_mask.shape}"
        )
    weights = valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def pool_kv_cache(past_key_values: dict[int, dict[str, Tensor]], valid_mask: Tensor) -> Tensor:
    """Pool the exact per-layer K/V condition consumed by SmolVLA denoising."""
    pooled: list[Tensor] = []
    weights = valid_mask[:, :, None, None].to(dtype=torch.float32)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    for layer_index in sorted(past_key_values):
        layer = past_key_values[layer_index]
        for name in ("key_states", "value_states"):
            value = layer[name].float()
            if value.shape[:2] != valid_mask.shape:
                raise ValueError(
                    f"KV {layer_index}.{name} shape {value.shape} does not match mask {valid_mask.shape}"
                )
            token_mean = (value * weights).sum(dim=1) / denominator
            pooled.append(token_mean.mean(dim=1))
    if not pooled:
        raise RuntimeError("SmolVLA prefix forward returned an empty KV cache")
    return torch.cat(pooled, dim=-1)


class FrozenSmolVLAConditionEncoder(nn.Module):
    """Extract the post-transformer prefix used to condition SmolVLA denoising."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        text_config = self.policy.model.vlm_with_expert.config.text_config
        head_dim = int(getattr(
            text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
        ))
        return int(self.policy.config.num_vlm_layers) * 2 * head_dim

    @property
    def token_dim(self) -> int:
        return int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)

    @torch.inference_mode()
    def forward(self, processed_batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, dict[int, dict[str, Tensor]]]:
        images, image_masks = self.policy.prepare_images(processed_batch)
        state = self.policy.prepare_state(processed_batch)
        tokens = processed_batch["observation.language.tokens"]
        language_mask = processed_batch["observation.language.attention_mask"]
        prefix, pad_mask, attention_mask = self.policy.model.embed_prefix(
            images, image_masks, tokens, language_mask, state=state
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        attention_2d = make_att_2d_masks(pad_mask, attention_mask)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        (_, _), past_key_values = self.policy.model.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        feature = pool_kv_cache(past_key_values, pad_mask.bool())
        return feature, pad_mask.bool(), past_key_values

    @torch.inference_mode()
    def pooled(self, processed_batch: dict[str, Tensor]) -> Tensor:
        feature, _, _ = self(processed_batch)
        return feature

    @torch.inference_mode()
    def tokens(self, processed_batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return the final-layer pre-QKV multimodal condition tokens and valid mask."""
        captured: dict[str, Tensor] = {}
        layer_norm = (
            self.policy.model.vlm_with_expert.get_vlm_model().text_model.layers[-1].input_layernorm
        )

        def capture_tokens(_module, _inputs, output):
            if "condition_tokens" not in captured:
                captured["condition_tokens"] = output.detach()

        handle = layer_norm.register_forward_hook(capture_tokens)
        try:
            images, image_masks = self.policy.prepare_images(processed_batch)
            state = self.policy.prepare_state(processed_batch)
            tokens = processed_batch["observation.language.tokens"]
            language_mask = processed_batch["observation.language.attention_mask"]
            prefix, pad_mask, attention_mask = self.policy.model.embed_prefix(
                images, image_masks, tokens, language_mask, state=state
            )
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

            attention_2d = make_att_2d_masks(pad_mask, attention_mask)
            position_ids = torch.cumsum(pad_mask, dim=1) - 1
            self.policy.model.vlm_with_expert.forward(
                attention_mask=attention_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                use_cache=True,
                fill_kv_cache=True,
            )
        finally:
            handle.remove()
        if "condition_tokens" not in captured:
            raise RuntimeError("Failed to capture final-layer SmolVLA condition tokens")
        condition_tokens = captured["condition_tokens"]
        if condition_tokens.shape[:2] != pad_mask.shape:
            raise RuntimeError(
                f"Condition token shape {condition_tokens.shape} does not match mask {pad_mask.shape}"
            )
        return condition_tokens, pad_mask.bool()


@contextmanager
def _capture_action_condition(policy: nn.Module) -> Iterator[dict[str, Tensor]]:
    captured: dict[str, Any] = {}

    original_embed_prefix = policy.model.embed_prefix

    def capturing_embed_prefix(*args, **kwargs):
        output = original_embed_prefix(*args, **kwargs)
        captured["pad_mask"] = output[1].detach()
        return output

    original_vlm_forward = policy.model.vlm_with_expert.forward

    layer_norm = policy.model.vlm_with_expert.get_vlm_model().text_model.layers[-1].input_layernorm

    def capture_tokens(_module, _inputs, output):
        if "condition_tokens" not in captured:
            captured["condition_tokens"] = output.detach()

    token_handle = layer_norm.register_forward_hook(capture_tokens)

    def capturing_vlm_forward(*args, **kwargs):
        output = original_vlm_forward(*args, **kwargs)
        past_key_values = output[1]
        if past_key_values and "past_key_values" not in captured:
            captured["past_key_values"] = past_key_values
        return output

    policy.model.embed_prefix = capturing_embed_prefix
    policy.model.vlm_with_expert.forward = capturing_vlm_forward
    try:
        yield captured
    finally:
        token_handle.remove()
        policy.model.embed_prefix = original_embed_prefix
        policy.model.vlm_with_expert.forward = original_vlm_forward


@torch.inference_mode()
def predict_action_chunk_with_condition(
    policy: nn.Module, processed_batch: dict[str, Tensor], **kwargs: Any
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return action and its condition vector from one normal SmolVLA inference."""
    with _capture_action_condition(policy) as captured:
        actions = policy.predict_action_chunk(processed_batch, **kwargs)
    if "past_key_values" not in captured or "pad_mask" not in captured:
        raise RuntimeError("Failed to capture SmolVLA prefix condition during action inference")
    past_key_values = captured["past_key_values"]
    feature = pool_kv_cache(past_key_values, captured["pad_mask"])
    first_layer = past_key_values[min(past_key_values)]
    metadata = {
        "feature_tap": FEATURE_TAP,
        "mask_tap": MASK_TAP,
        "raw_tensor_shape": {
            "layers": len(past_key_values),
            "key_states_per_layer": list(first_layer["key_states"].shape),
            "value_states_per_layer": list(first_layer["value_states"].shape),
        },
        "valid_tokens_per_sample": captured["pad_mask"].sum(dim=1).tolist(),
        "pooling": "per_layer_key_value_masked_token_mean_then_head_mean_then_concat",
    }
    return actions, feature, metadata


@torch.inference_mode()
def predict_action_chunk_with_condition_tokens(
    policy: nn.Module, processed_batch: dict[str, Tensor], **kwargs: Any
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Return action and token-level condition from one normal SmolVLA inference."""
    with _capture_action_condition(policy) as captured:
        actions = policy.predict_action_chunk(processed_batch, **kwargs)
    required = {"past_key_values", "pad_mask", "condition_tokens"}
    if not required.issubset(captured):
        raise RuntimeError(
            f"Failed to capture SmolVLA token condition during action inference: "
            f"missing={sorted(required - set(captured))}"
        )
    condition_tokens = captured["condition_tokens"]
    condition_mask = captured["pad_mask"].bool()
    if condition_tokens.shape[:2] != condition_mask.shape:
        raise RuntimeError(
            f"Condition token shape {condition_tokens.shape} does not match mask {condition_mask.shape}"
        )
    metadata = {
        "feature_tap": TOKEN_FEATURE_TAP,
        "mask_tap": MASK_TAP,
        "condition_token_shape": list(condition_tokens.shape),
        "condition_mask_shape": list(condition_mask.shape),
        "valid_tokens_per_sample": condition_mask.sum(dim=1).tolist(),
        "pooling": "none",
        "condition_semantics": (
            "final VLM layer input-normalized prefix hidden states used to form the K/V cache "
            "consumed by action denoising"
        ),
    }
    return actions, condition_tokens, condition_mask, metadata


def read_preprocessor_camera_map(checkpoint: Path) -> dict[str, str]:
    payload = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    for step in payload.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            return dict(step.get("config", {}).get("rename_map", {}))
    return {}
