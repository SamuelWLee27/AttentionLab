import torch

from attnlab.attention.delta_net import DeltaNetAttention
from attnlab.attention.gated_delta_net import GatedDeltaNetAttention


def test_gated_delta_net_attention_shape():
    torch.manual_seed(0)
    layer = GatedDeltaNetAttention(d_model=8, heads=4)
    x = torch.randn(2, 5, 8)

    result = layer(x)

    assert result.shape == x.shape


def test_gated_one_alpha_delta_rule_overwrites_same_key():
    layer = GatedDeltaNetAttention(d_model=2, heads=1)
    reference = DeltaNetAttention(d_model=2, heads=1)

    key = torch.tensor([1.0, 0.0])
    phi_K = torch.stack([key, key]).view(1, 1, 2, 2)  # same key twice
    phi_Q = phi_K.clone()  # read with the key
    V = torch.tensor([[3.0, 3.0], [7.0, 7.0]]).view(1, 1, 2, 2)
    beta = torch.ones(1, 1, 2)  # full overwrite
    alpha = torch.ones(1, 1, 2)

    out = layer._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta, alpha=alpha)

    expected = reference._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta)
    torch.testing.assert_close(out, expected)


def test_gated_one_alpha_delta_rule_keeps_other_keys():
    layer = GatedDeltaNetAttention(d_model=2, heads=1)
    reference = DeltaNetAttention(d_model=2, heads=1)

    key_a = torch.tensor([1.0, 0.0])
    key_b = torch.tensor([0.0, 1.0])
    phi_K = torch.stack([key_a, key_b]).view(1, 1, 2, 2)  # two different keys
    phi_Q = torch.stack([key_a, key_a]).view(1, 1, 2, 2)  # both read key_a
    V = torch.tensor([[3.0, 3.0], [7.0, 7.0]]).view(1, 1, 2, 2)
    beta = torch.ones(1, 1, 2)
    alpha = torch.ones(1, 1, 2)

    out = layer._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta, alpha=alpha)

    # Writing under key_b must not disturb what key_a holds.
    expected = reference._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta)
    torch.testing.assert_close(out, expected)


def test_gated_delta_net_reduces_to_delta_net_random_inputs():
    torch.manual_seed(0)
    B, H, T, head_dim = 2, 2, 6, 4
    layer = GatedDeltaNetAttention(d_model=H * head_dim, heads=H)
    reference = DeltaNetAttention(d_model=H * head_dim, heads=H)

    phi_Q = torch.nn.functional.normalize(torch.randn(B, H, T, head_dim), dim=-1)
    phi_K = torch.nn.functional.normalize(torch.randn(B, H, T, head_dim), dim=-1)
    V = torch.randn(B, H, T, head_dim)
    beta = torch.sigmoid(torch.randn(B, H, T))
    alpha = torch.ones(B, H, T)  # no decay: should behave exactly like DeltaNet

    out = layer._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta, alpha=alpha)
    expected = reference._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta)

    torch.testing.assert_close(out, expected)


def test_gated_delta_net_decays_memory():
    layer = GatedDeltaNetAttention(d_model=2, heads=1)

    key_a = torch.tensor([1.0, 0.0])
    key_b = torch.tensor([0.0, 1.0])
    phi_K = torch.stack([key_a, key_b]).view(1, 1, 2, 2)  # perpendicular keys
    phi_Q = torch.stack([key_a, key_a]).view(1, 1, 2, 2)  # both read key_a
    V = torch.tensor([[4.0, 4.0], [9.0, 9.0]]).view(1, 1, 2, 2)
    beta = torch.ones(1, 1, 2)
    alpha = torch.tensor([1.0, 0.5]).view(1, 1, 2)  # token 1 halves the memory

    out = layer._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta, alpha=alpha)

    # Token 0 reads [4, 4]. Token 1's write goes under key_b, so the only
    # change to key_a is the decay: 0.5 * [4, 4] = [2, 2].
    expected = torch.tensor([[4.0, 4.0], [2.0, 2.0]]).view(1, 1, 2, 2)
    torch.testing.assert_close(out, expected)
