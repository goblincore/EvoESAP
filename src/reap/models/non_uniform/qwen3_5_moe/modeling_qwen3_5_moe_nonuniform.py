"""
Non-uniform Qwen3_5Moe: per-layer-variable routed expert count.

Strategy:
  - Reuse upstream attention classes verbatim (Qwen3_5MoeAttention for
    full_attention layers, Qwen3_5MoeGatedDeltaNet for linear_attention).
  - Reuse upstream Qwen3_5MoeMLP for the shared_expert (dense, always-on,
    sigmoid-gated — independent of routed-expert count).
  - Override Qwen3_5MoeExperts and Qwen3_5MoeTopKRouter to accept an explicit
    num_experts parameter (so each layer's routed-expert count can differ).
  - Override Qwen3_5MoeSparseMoeBlock with the same signature change.
  - Override Qwen3_5MoeDecoderLayer to instantiate our SparseMoeBlock with
    config.get_num_experts(layer_idx).
  - Override the text model, CausalLM, conditional-generation, and VLM
    wrapper classes to use our non-uniform decoder layer / text model.

Tensor parameter names match upstream exactly (gate_up_proj, down_proj,
weight, shared_expert.{gate_proj,up_proj,down_proj}, shared_expert_gate.weight)
so state_dicts saved from the uniform model load cleanly into our model
when num_experts_per_layer == [config.num_experts] * num_hidden_layers, and
state_dicts saved from our model (with varied shapes) re-load into our model
when num_experts_per_layer is preserved in the config.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.integrations import use_experts_implementation
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeModel,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
)

from .configuration_qwen3_5_moe_nonuniform import (
    NonUniformQwen3_5MoeConfig,
    NonUniformQwen3_5MoeTextConfig,
)


# ---------------------------------------------------------------------------
# Routed experts / router with per-layer-variable num_experts.
# Parameter layout matches upstream Qwen3_5MoeExperts / Qwen3_5MoeTopKRouter
# (verified 2026-05-12 against transformers==5.8.0.dev0):
#   gate_up_proj: [num_experts, 2*moe_intermediate, hidden]
#   down_proj:   [num_experts, hidden, moe_intermediate]
#   weight:      [num_experts, hidden]
# ---------------------------------------------------------------------------


@use_experts_implementation
class Qwen3_5MoeNonUniformExperts(nn.Module):
    """Packed routed experts with per-layer-variable num_experts.

    The @use_experts_implementation decorator is required so that the parent
    PreTrainedModel's _can_set_experts_implementation() check passes for our
    subclass (transformers/modeling_utils.py heuristically grep's source files
    for this exact string to whitelist classes that support the dispatch).
    Even if we never actually use the grouped_mm kernel (we run "eager" by
    default since variable per-layer expert counts preclude grouped MM), the
    decorator must be present to unblock model instantiation.
    """

    def __init__(self, config, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = (
            int(num_experts) if num_experts is not None else config.num_experts
        )
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors upstream Qwen3_5MoeExperts.forward exactly.
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


class Qwen3_5MoeNonUniformTopKRouter(nn.Module):
    """Top-k router with per-layer-variable num_experts."""

    def __init__(self, config, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = (
            int(num_experts) if num_experts is not None else config.num_experts
        )
        # Clamp top_k in case num_experts < config.num_experts_per_tok.
        self.top_k = min(int(config.num_experts_per_tok), int(self.num_experts))
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3_5MoeNonUniformSparseMoeBlock(nn.Module):
    """SparseMoeBlock with per-layer-variable routed-expert count.

    Forward mirrors upstream Qwen3_5MoeSparseMoeBlock.forward exactly:
      shared_expert_output = shared_expert(x)
      _, weights, indices = gate(x)
      expert_output = experts(x, indices, weights)
      shared_expert_output = sigmoid(shared_expert_gate(x)) * shared_expert_output
      return expert_output + shared_expert_output

    shared_expert + shared_expert_gate are constructed identically to upstream
    (dense MLP + 1-dim gate Linear). Only the routed-experts + router vary.
    """

    def __init__(self, config, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = (
            int(num_experts) if num_experts is not None else config.num_experts
        )
        # Routed (varies per layer).
        self.experts = Qwen3_5MoeNonUniformExperts(config, num_experts=self.num_experts)
        self.gate = Qwen3_5MoeNonUniformTopKRouter(config, num_experts=self.num_experts)
        # Shared (constant across layers, never pruned).
        self.shared_expert = Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


# ---------------------------------------------------------------------------
# Decoder layer: identical to upstream Qwen3_5MoeDecoderLayer except the mlp
# is constructed with per-layer num_experts. Attention is unchanged.
# ---------------------------------------------------------------------------


class Qwen3_5MoeNonUniformDecoderLayer(nn.Module):
    """Decoder layer with per-layer-variable mlp expert count.

    Attention branch matches upstream Qwen3_5MoeDecoderLayer.__init__:
      - linear_attention → Qwen3_5MoeGatedDeltaNet on attribute `linear_attn`
      - full_attention → Qwen3_5MoeAttention on attribute `self_attn`
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        else:
            raise ValueError(
                f"unknown layer_type at layer {layer_idx}: {self.layer_type!r}"
            )

        # Per-layer routed-expert count (falls back to config.num_experts if
        # num_experts_per_layer is absent or out-of-range).
        num_experts = (
            config.get_num_experts(layer_idx)
            if hasattr(config, "get_num_experts")
            else config.num_experts
        )
        self.mlp = Qwen3_5MoeNonUniformSparseMoeBlock(config, num_experts=num_experts)

        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states, **kwargs)
        else:
            attn_out = self.self_attn(hidden_states, **kwargs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Text model, CausalLM, VLM wrappers — subclass upstream and swap the layers
# / language_model submodule for our non-uniform versions.
# ---------------------------------------------------------------------------


class NonUniformQwen3_5MoeTextModel(Qwen3_5MoeTextModel):
    """Text model using our non-uniform decoder layers.

    We bypass Qwen3_5MoeTextModel.__init__ (which would build upstream layers)
    and reproduce its structure with our decoder layer class. This keeps the
    embed_tokens / norm / rotary_emb attributes identical so state_dict load
    works without remapping.
    """

    config_class = NonUniformQwen3_5MoeTextConfig

    def __init__(self, config: NonUniformQwen3_5MoeTextConfig):
        # Skip Qwen3_5MoeTextModel.__init__; call its grandparent (PreTrainedModel).
        super(Qwen3_5MoeTextModel, self).__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeNonUniformDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class NonUniformQwen3_5MoeForCausalLM(Qwen3_5MoeForCausalLM):
    """CausalLM (text-only) wrapper using our non-uniform text model.

    Used when EvoESAP / REAP wants to instantiate just the language model
    (e.g. during the EA search forward passes — vision tower would be wasted
    overhead since search prompts are text-only).
    """

    config_class = NonUniformQwen3_5MoeTextConfig

    def __init__(self, config: NonUniformQwen3_5MoeTextConfig):
        # Skip Qwen3_5MoeForCausalLM.__init__; call grandparent.
        super(Qwen3_5MoeForCausalLM, self).__init__(config)
        self.model = NonUniformQwen3_5MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


class NonUniformQwen3_5MoeModel(Qwen3_5MoeModel):
    """VLM model (vision tower + text model). Vision tower untouched."""

    config_class = NonUniformQwen3_5MoeConfig

    def __init__(self, config: NonUniformQwen3_5MoeConfig):
        super().__init__(config)
        # Replace the language_model submodule built by super().__init__ with
        # our non-uniform version. Vision tower (self.visual or similar) is
        # left alone.
        self.language_model = NonUniformQwen3_5MoeTextModel(config.text_config)


class NonUniformQwen3_5MoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Top-level VLM wrapper: vision + non-uniform text LM."""

    config_class = NonUniformQwen3_5MoeConfig

    def __init__(self, config: NonUniformQwen3_5MoeConfig):
        # Build via Qwen3_5MoeForConditionalGeneration.__init__ (creates
        # self.model = Qwen3_5MoeModel(config) + self.lm_head). Then replace
        # the language_model submodule.
        super().__init__(config)
        self.model.language_model = NonUniformQwen3_5MoeTextModel(config.text_config)


__all__ = [
    "Qwen3_5MoeNonUniformExperts",
    "Qwen3_5MoeNonUniformTopKRouter",
    "Qwen3_5MoeNonUniformSparseMoeBlock",
    "Qwen3_5MoeNonUniformDecoderLayer",
    "NonUniformQwen3_5MoeTextModel",
    "NonUniformQwen3_5MoeForCausalLM",
    "NonUniformQwen3_5MoeModel",
    "NonUniformQwen3_5MoeForConditionalGeneration",
]
