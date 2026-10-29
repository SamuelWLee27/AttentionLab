import torch

from attnlab.attention.multi_head_linear import MultiHeadLinearAttention


def test_linear_attention_shape():
    torch.manual_seed(0)
    layer = MultiHeadLinearAttention(d_model=8, heads=4)
    x = torch.randn(2, 5, 8)

    result = layer(x)

    assert result.shape == x.shape


def test_linear_attention_matches_quadratic_form():
    torch.manual_seed(0)
    layer = MultiHeadLinearAttention(d_model=8, heads=2, is_causal=False)
    x = torch.randn(2, 5, 8)
    B, T, D = x.shape

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, T, layer.heads, layer.head_dim).permute(0, 2, 1, 3)

    phi_Q = layer._phi(split_heads(layer.q_proj(x)))
    phi_K = layer._phi(split_heads(layer.k_proj(x)))
    V = split_heads(layer.v_proj(x))

    # Slow route: build the full (T, T) matrix, normalize rows, mix values.
    scores = phi_Q @ phi_K.transpose(-2, -1)
    weights = scores / scores.sum(dim=-1, keepdim=True)
    out = weights @ V

    expected = layer.out_proj(out.permute(0, 2, 1, 3).reshape(B, T, D))

    torch.testing.assert_close(layer(x), expected)


def test_causal_linear_attention_matches_quadratic_form():
    torch.manual_seed(0)
    layer = MultiHeadLinearAttention(d_model=8, heads=2, is_causal=True)
    x = torch.randn(2, 5, 8)
    B, T, D = x.shape

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, T, layer.heads, layer.head_dim).permute(0, 2, 1, 3)

    phi_Q = layer._phi(split_heads(layer.q_proj(x)))
    phi_K = layer._phi(split_heads(layer.k_proj(x)))
    V = split_heads(layer.v_proj(x))

    # Slow route: build the full (T, T) matrix, normalize rows, mix values.
    scores = phi_Q @ phi_K.transpose(-2, -1)

    mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
    causal_mask = torch.triu(mask, diagonal=1)
    causal_scores = scores.masked_fill(causal_mask, 0.0)

    weights = causal_scores / causal_scores.sum(dim=-1, keepdim=True)
    out = weights @ V

    expected = layer.out_proj(out.permute(0, 2, 1, 3).reshape(B, T, D))

    torch.testing.assert_close(layer(x), expected)
