import math

import torch
from torch import nn


class MultiHeadSoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, is_causal: bool = False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)

        score = Q @ (K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.is_causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
            causal_mask = torch.triu(mask, diagonal=1)
            score = score.masked_fill(causal_mask, float("-inf"))

        weights = torch.softmax(score, dim=-1)
        out = weights @ V
        concat = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(concat)
