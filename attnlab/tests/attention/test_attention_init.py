import pytest
import torch

from attnlab.attention import ATTENTION_LAYERS, build_attention


@pytest.mark.parametrize("name", sorted(ATTENTION_LAYERS))
def test_every_layer_keeps_shape(name):
    layer = build_attention(name, d_model=12, heads=3)
    x = torch.randn(2, 5, 12)
    assert layer(x).shape == x.shape


def test_unknown_layer_suggests_close_match():
    with pytest.raises(ValueError, match="Did you mean 'kda'"):
        build_attention("kdaa", d_model=12, heads=3)
