import difflib

from torch import nn

from .delta_net import DeltaNetAttention
from .gated_delta_net import GatedDeltaNetAttention
from .kimi_delta_attention import KimiDeltaAttention
from .multi_head_linear import MultiHeadLinearAttention
from .multi_head_softmax import MultiHeadSoftmaxAttention

ATTENTION_LAYERS: dict[str, type[nn.Module]] = {
    "softmax": MultiHeadSoftmaxAttention,
    "linear": MultiHeadLinearAttention,
    "delta_net": DeltaNetAttention,
    "gated_delta_net": GatedDeltaNetAttention,
    "kda": KimiDeltaAttention,
}


def build_attention(name: str, d_model: int, heads: int, is_causal: bool = True) -> nn.Module:
    """Construct the attention layer registered under `name`."""
    if name not in ATTENTION_LAYERS:
        available = ", ".join(sorted(ATTENTION_LAYERS))
        suggestion = difflib.get_close_matches(name, ATTENTION_LAYERS, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(f"Unknown attention layer '{name}'.{hint} Available: {available}")
    return ATTENTION_LAYERS[name](d_model=d_model, heads=heads, is_causal=is_causal)


__all__ = [
    "ATTENTION_LAYERS",
    "build_attention",
    "DeltaNetAttention",
    "GatedDeltaNetAttention",
    "KimiDeltaAttention",
    "MultiHeadLinearAttention",
    "MultiHeadSoftmaxAttention",
]
