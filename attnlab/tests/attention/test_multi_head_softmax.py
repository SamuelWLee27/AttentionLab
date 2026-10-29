import torch
from torch import nn

from attnlab.attention.multi_head_softmax import MultiHeadSoftmaxAttention


def test_softmax_attention_shape():
    torch.manual_seed(0)
    layer = MultiHeadSoftmaxAttention(d_model=8, heads=4)
    x = torch.randn(2, 5, 8)

    result = layer(x)

    assert result.shape == x.shape


def test_multi_head_softmax_against_pytorch():
    torch.manual_seed(0)
    layer = MultiHeadSoftmaxAttention(d_model=8, heads=2)
    x = torch.randn(2, 5, 8)
    B, T, D = x.shape

    actual = layer(x)

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, T, layer.heads, layer.head_dim).permute(0, 2, 1, 3)

    q = split_heads(layer.q_proj(x))
    k = split_heads(layer.k_proj(x))
    v = split_heads(layer.v_proj(x))

    out = nn.functional.scaled_dot_product_attention(query=q, key=k, value=v)
    expected = layer.out_proj(out.permute(0, 2, 1, 3).reshape(B, T, D))

    torch.testing.assert_close(actual, expected)
