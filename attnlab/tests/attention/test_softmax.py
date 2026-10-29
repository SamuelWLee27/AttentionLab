import torch
from torch import nn

from attnlab.attention.softmax import SoftmaxAttention


def test_softmax_attention_shape():
    torch.manual_seed(0)
    layer = SoftmaxAttention(4)
    x = torch.randn(2, 5, 4)

    result = layer(x)

    assert result.shape == x.shape


def test_softmax_against_pytorch():
    torch.manual_seed(0)
    layer = SoftmaxAttention(4)
    x = torch.randn(2, 5, 4)

    actual = layer(x)
    expected = nn.functional.scaled_dot_product_attention(
        query=layer.q_proj(x), key=layer.k_proj(x), value=layer.v_proj(x)
    )

    torch.testing.assert_close(actual, expected)
