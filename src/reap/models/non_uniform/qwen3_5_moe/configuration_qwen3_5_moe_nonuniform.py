"""
Config classes for non-uniform Qwen3_5Moe.

Mirrors the pattern from qwen3_moe/configuration_qwen3_moe_nonuniform.py, with
the key difference that Qwen3_5Moe is a VLM whose MoE config lives on the nested
text_config sub-config (not the top-level config). So we extend BOTH:

  - NonUniformQwen3_5MoeTextConfig: extends Qwen3_5MoeTextConfig with
    num_experts_per_layer + get_num_experts(layer_idx).
  - NonUniformQwen3_5MoeConfig: extends Qwen3_5MoeConfig and ensures its
    text_config is an instance of NonUniformQwen3_5MoeTextConfig.

The vision_config sub-config is passed through unchanged (vision tower is
never pruned).
"""

from typing import List, Optional

from transformers import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)


class NonUniformQwen3_5MoeTextConfig(Qwen3_5MoeTextConfig):
    """Extends Qwen3_5MoeTextConfig with per-layer routed-expert counts.

    Field:
      num_experts_per_layer: Optional[list[int]] of length num_hidden_layers.
        If None or absent, every layer uses config.num_experts (uniform fallback).

    The shared_expert + shared_expert_gate are unaffected — they're dense
    always-on MLPs that never participate in routed-expert pruning.
    """

    model_type = "qwen3_5_moe_text"  # match parent so HF picks the right class

    def __init__(
        self,
        *args,
        num_experts_per_layer: Optional[List[int]] = None,
        **kwargs,
    ):
        # Set the field BEFORE super().__init__ so subclasses that read it during
        # init (e.g. layer_types-aware validators) see it.
        self.num_experts_per_layer = num_experts_per_layer
        super().__init__(*args, **kwargs)

    def get_num_experts(self, layer_idx: int) -> int:
        """Returns the number of routed experts for the given layer."""
        if self.num_experts_per_layer and 0 <= layer_idx < len(self.num_experts_per_layer):
            return int(self.num_experts_per_layer[layer_idx])
        return int(self.num_experts)

    def to_dict(self):
        d = super().to_dict()
        d["num_experts_per_layer"] = self.num_experts_per_layer
        return d


class NonUniformQwen3_5MoeConfig(Qwen3_5MoeConfig):
    """Top-level config for non-uniform Qwen3_5Moe (VLM).

    Inherits Qwen3_5MoeConfig but overrides sub_configs so that text_config
    deserializes into NonUniformQwen3_5MoeTextConfig. Vision sub-config is
    passed through unchanged.
    """

    model_type = "qwen3_5_moe"  # match parent

    # Override sub_configs to point text_config at our subclass.
    # Vision sub-config inherits whatever the parent declared.
    sub_configs = {
        **getattr(Qwen3_5MoeConfig, "sub_configs", {}),
        "text_config": NonUniformQwen3_5MoeTextConfig,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If text_config got constructed as the parent class (e.g. via from_dict),
        # convert it in-place to our subclass.
        if self.text_config is not None and not isinstance(
            self.text_config, NonUniformQwen3_5MoeTextConfig
        ):
            self.text_config = NonUniformQwen3_5MoeTextConfig(
                **self.text_config.to_dict()
            )

        # Register this class for trust_remote_code loading. HF AutoXxx looks at
        # auto_map first when a config has it, so saved non-uniform checkpoints
        # can be re-loaded without our code path being installed system-wide.
        if not hasattr(self, "auto_map") or self.auto_map is None:
            self.auto_map = {}
        self.auto_map.setdefault(
            "AutoConfig",
            "configuration_qwen3_5_moe_nonuniform.NonUniformQwen3_5MoeConfig",
        )
        self.auto_map.setdefault(
            "AutoModelForImageTextToText",
            "modeling_qwen3_5_moe_nonuniform.NonUniformQwen3_5MoeForConditionalGeneration",
        )
        self.auto_map.setdefault(
            "AutoModel",
            "modeling_qwen3_5_moe_nonuniform.NonUniformQwen3_5MoeModel",
        )
        if not getattr(self, "architectures", None):
            self.architectures = ["NonUniformQwen3_5MoeForConditionalGeneration"]


__all__ = [
    "NonUniformQwen3_5MoeTextConfig",
    "NonUniformQwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeConfig",
]
