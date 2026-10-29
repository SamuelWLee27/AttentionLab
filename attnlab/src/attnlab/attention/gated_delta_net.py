import torch
from torch import nn


class GatedDeltaNetAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"{type(self).__name__}: D {d_model} is not divisible by H {heads}")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, heads)
        self.alpha_proj = nn.Linear(d_model, heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(x)).permute(0, 2, 1)
        alpha = torch.sigmoid(self.alpha_proj(x)).permute(0, 2, 1)

        phi_Q = self._phi(Q)
        phi_K = self._phi(K)

        out = self._recurrent(phi_Q=phi_Q, phi_K=phi_K, V=V, beta=beta, alpha=alpha)

        concat = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(concat)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(x)
        return torch.nn.functional.normalize(x, dim=-1)

    def _recurrent(
        self,
        phi_Q: torch.Tensor,
        phi_K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        B, H, T, head_dim = phi_K.shape

        memory = torch.zeros(B, H, head_dim, head_dim, device=phi_K.device, dtype=phi_K.dtype)

        outputs = []

        for t in range(T):
            K_t = phi_K[:, :, t, :]
            V_t = V[:, :, t, :].unsqueeze(-2)
            Q_t = phi_Q[:, :, t, :].unsqueeze(-2)
            beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)
            alpha_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)

            memory = memory * alpha_t
            stored = K_t.unsqueeze(-2) @ memory
            memory = memory + (K_t.unsqueeze(-1) @ (beta_t * (V_t - stored)))
            output_t = Q_t @ memory
            outputs.append(output_t)

        return torch.cat(outputs, dim=2)
