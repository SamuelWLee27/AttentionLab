import torch
from torch import nn


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, is_causal: bool = True):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"{type(self).__name__}: D {d_model} is not divisible by H {heads}")
        self.heads = heads
        self.head_dim = d_model // heads
        self.is_causal = is_causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)

        phi_Q = self._phi(Q)
        phi_K = self._phi(K)

        if self.is_causal:
            out = self._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V)
        else:
            out = self._parallel(phi_Q=phi_Q, phi_K=phi_K, V=V)

        concat = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(concat)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(x) + 1

    def _parallel(self, phi_Q: torch.Tensor, phi_K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        memory = phi_K.transpose(-2, -1) @ V

        K_sum = torch.sum(phi_K, dim=-2)

        return (phi_Q @ memory) / (phi_Q @ K_sum.unsqueeze(-1))

    def _recurrent(self, phi_Q: torch.Tensor, phi_K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, T, head_dim = phi_K.shape

        memory = torch.zeros(B, H, head_dim, head_dim, device=phi_K.device, dtype=phi_K.dtype)
        K_sum = torch.zeros(B, H, head_dim, device=phi_K.device, dtype=phi_K.dtype)

        outputs = []

        for t in range(T):
            K_t = phi_K[:, :, t, :]
            V_t = V[:, :, t, :].unsqueeze(-2)
            Q_t = phi_Q[:, :, t, :].unsqueeze(-2)
            memory = memory + (K_t.unsqueeze(-1) @ V_t)
            K_sum = K_sum + K_t
            output_t = (Q_t @ memory) / (Q_t @ K_sum.unsqueeze(-1))
            outputs.append(output_t)

        return torch.cat(outputs, dim=2)
