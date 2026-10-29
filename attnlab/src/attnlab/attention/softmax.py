import math

import torch
from torch import nn


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        score = Q @ (K.transpose(-2, -1)) / math.sqrt(self.d_model)
        weights = torch.softmax(score, dim=-1)
        return weights @ V
