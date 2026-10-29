import torch

from attnlab.attention.delta_net import DeltaNetAttention


def test_delta_net_attention_shape():
    torch.manual_seed(0)
    layer = DeltaNetAttention(d_model=8, heads=4)
    x = torch.randn(2, 5, 8)

    result = layer(x)

    assert result.shape == x.shape


def test_delta_rule_overwrites_same_key():
    layer = DeltaNetAttention(d_model=2, heads=1)

    key = torch.tensor([1.0, 0.0])
    phi_K = torch.stack([key, key]).view(1, 1, 2, 2)  # same key twice
    phi_Q = phi_K.clone()  # read with the key
    V = torch.tensor([[3.0, 3.0], [7.0, 7.0]]).view(1, 1, 2, 2)  # Monday, then Friday
    beta = torch.ones(1, 1, 2)  # full overwrite

    out = layer._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta)

    expected = torch.tensor([[3.0, 3.0], [7.0, 7.0]]).view(1, 1, 2, 2)
    torch.testing.assert_close(out, expected)


def test_delta_rule_keeps_other_keys():
    layer = DeltaNetAttention(d_model=2, heads=1)

    key_a = torch.tensor([1.0, 0.0])
    key_b = torch.tensor([0.0, 1.0])
    phi_K = torch.stack([key_a, key_b]).view(1, 1, 2, 2)  # two different keys
    phi_Q = torch.stack([key_a, key_a]).view(1, 1, 2, 2)  # both read key_a
    V = torch.tensor([[3.0, 3.0], [7.0, 7.0]]).view(1, 1, 2, 2)
    beta = torch.ones(1, 1, 2)

    out = layer._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta)

    # Writing under key_b must not disturb what key_a holds.
    expected = torch.tensor([[3.0, 3.0], [3.0, 3.0]]).view(1, 1, 2, 2)
    torch.testing.assert_close(out, expected)
