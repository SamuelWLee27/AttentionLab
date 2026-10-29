import torch
from torch import nn


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"{type(self).__name__}: D {d_model} is not divisible by H {heads}")
        self.heads = heads
        self.head_dim = d_model // heads
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

        phi_Q = self.phi(Q)
        phi_K = self.phi(K)

        memory = phi_K.transpose(-2, -1) @ V

        K_sum = torch.sum(phi_K, dim=-2)

        out = (phi_Q @ memory) / (phi_Q @ K_sum.unsqueeze(-1))

        concat = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(concat)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(x) + 1
