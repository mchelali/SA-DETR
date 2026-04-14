import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleEmbedder(nn.Module):
    """
    Combine:
      - a small learned table of K embeddings (interpolated by s in [0,1])
      - a sinusoidal encoding of s projected to d_model
      - optional per-level learnable bias
    Forward API:
      forward(s_scalar, B, L) -> (B, L, d_model)
      where s_scalar is float or 0-d tensor in [0,1],
            B batch size, L length (H*W flattened)
    """

    def __init__(self, d_model, K=8, sinus_dim=64, use_level_bias=False, device=None):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.sinus_dim = sinus_dim
        self.use_level_bias = use_level_bias

        # table of embeddings to interpolate (K x d_model)
        self.scale_table = nn.Parameter(torch.randn(K, d_model))

        # project sinusoidal encoding to d_model
        self.sinus_proj = nn.Linear(sinus_dim, d_model)

        if use_level_bias:
            # small bias vector to allow per-level shift (will be indexed externally)
            # However we still allow external per-level bias if needed.
            self.level_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        # init
        nn.init.trunc_normal_(self.scale_table, std=0.02)
        nn.init.xavier_uniform_(self.sinus_proj.weight)
        nn.init.normal_(self.sinus_proj.bias, std=1e-6)

    def _sinusoidal_encoding(self, s_scalar):
        # s_scalar: python float or tensor 0-d or shape (...,)
        # produce vector of length sinus_dim
        if not torch.is_tensor(s_scalar):
            s = torch.tensor([s_scalar], dtype=torch.float32)
        else:
            s = s_scalar.reshape(-1)
        device = s.device
        dim = self.sinus_dim
        # create freqs
        freqs = torch.exp(
            torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
        )
        # s may be multiple but we expect scalar -> produce (dim,)
        s = s.to(device).float()
        sin = torch.sin(s.unsqueeze(-1) * freqs)  # (1, dim/2)
        cos = torch.cos(s.unsqueeze(-1) * freqs)
        enc = torch.cat([sin, cos], dim=-1).view(-1)[:dim]  # (dim,)
        return enc  # 1D tensor length sinus_dim

    def forward(self, s_scalar, B, L, device=None, level_idx=None):
        """
        s_scalar : float or 0-d tensor in [0,1] (global for this level)
        B : batch size
        L : sequence length (H*W)
        level_idx: optional int for indexing per-level bias (not used by default)
        returns: tensor (B, L, d_model)
        """
        if device is None:
            device = self.scale_table.device

        # --- interpolation on table ---
        if not isinstance(s_scalar, torch.Tensor):
            s = torch.tensor(s_scalar, device=device)
        else:
            s = s_scalar.to(device)
        s = s.clamp(0.0, 1.0)
        idx = s * (self.K - 1)
        low = torch.floor(idx).long()
        high = torch.ceil(idx).long()
        alpha = idx - low

        low_vec = self.scale_table[low]  # (d_model,)
        high_vec = self.scale_table[high]
        interp = (1.0 - alpha) * low_vec + alpha * high_vec  # (d_model,)

        # --- sinusoidal part ---
        sinus = self._sinusoidal_encoding(s_scalar).to(device)  # (sinus_dim,)
        sinus_proj = self.sinus_proj(sinus)  # (d_model,)

        # combine
        combined = (interp + sinus_proj) * 0.5  # (d_model,)

        # optional level bias (broadcast)
        if self.use_level_bias:
            combined = combined + self.level_bias.view(-1)

        # expand to (B, L, d_model)
        out = (
            combined.view(1, 1, self.d_model)
            .expand(B, L, self.d_model)
            .contiguous()
            .to(device)
        )
        return out


class ScaleEmbedder_table(nn.Module):
    """
    Combine:
      - a small learned table of K embeddings (interpolated by s in [0,1])
      - a sinusoidal encoding of s projected to d_model
      - optional per-level learnable bias
    Forward API:
      forward(s_scalar, B, L) -> (B, L, d_model)
      where s_scalar is float or 0-d tensor in [0,1],
            B batch size, L length (H*W flattened)
    """

    def __init__(self, d_model, K=8, sinus_dim=64, use_level_bias=False, device=None):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.use_level_bias = use_level_bias

        # table of embeddings to interpolate (K x d_model)
        self.scale_table = nn.Parameter(torch.randn(K, d_model))

        if use_level_bias:
            # small bias vector to allow per-level shift (will be indexed externally)
            # However we still allow external per-level bias if needed.
            self.level_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        # init
        nn.init.trunc_normal_(self.scale_table, std=0.02)

    def forward(self, s_scalar, B, L, device=None, level_idx=None):
        """
        s_scalar : float or 0-d tensor in [0,1] (global for this level)
        B : batch size
        L : sequence length (H*W)
        level_idx: optional int for indexing per-level bias (not used by default)
        returns: tensor (B, L, d_model)
        """
        if device is None:
            device = self.scale_table.device

        # --- interpolation on table ---
        if not isinstance(s_scalar, torch.Tensor):
            s = torch.tensor(s_scalar, device=device)
        else:
            s = s_scalar.to(device)
        s = s.clamp(0.0, 1.0)
        idx = s * (self.K - 1)
        low = torch.floor(idx).long()
        high = torch.ceil(idx).long()
        alpha = idx - low

        low_vec = self.scale_table[low]  # (d_model,)
        high_vec = self.scale_table[high]
        interp = (1.0 - alpha) * low_vec + alpha * high_vec  # (d_model,)

        # combine
        combined = interp

        # optional level bias (broadcast)
        if self.use_level_bias:
            combined = combined + self.level_bias.view(-1)

        # expand to (B, L, d_model)
        out = (
            combined.view(1, 1, self.d_model)
            .expand(B, L, self.d_model)
            .contiguous()
            .to(device)
        )
        return out


class ScaleEmbedder_sin(nn.Module):
    """
    Combine:
      - a small learned table of K embeddings (interpolated by s in [0,1])
      - a sinusoidal encoding of s projected to d_model
      - optional per-level learnable bias
    Forward API:
      forward(s_scalar, B, L) -> (B, L, d_model)
      where s_scalar is float or 0-d tensor in [0,1],
            B batch size, L length (H*W flattened)
    """

    def __init__(self, d_model, K=8, sinus_dim=64, use_level_bias=False, device=None):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.sinus_dim = sinus_dim
        self.use_level_bias = use_level_bias

        # project sinusoidal encoding to d_model
        self.sinus_proj = nn.Linear(sinus_dim, d_model)

        if use_level_bias:
            # small bias vector to allow per-level shift (will be indexed externally)
            # However we still allow external per-level bias if needed.
            self.level_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        # init
        nn.init.xavier_uniform_(self.sinus_proj.weight)
        nn.init.normal_(self.sinus_proj.bias, std=1e-6)

    def _sinusoidal_encoding(self, s_scalar):
        # s_scalar: python float or tensor 0-d or shape (...,)
        # produce vector of length sinus_dim
        if not torch.is_tensor(s_scalar):
            s = torch.tensor([s_scalar], dtype=torch.float32)
        else:
            s = s_scalar.reshape(-1)
        device = s.device
        dim = self.sinus_dim
        # create freqs
        freqs = torch.exp(
            torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
        )
        # s may be multiple but we expect scalar -> produce (dim,)
        s = s.to(device).float()
        sin = torch.sin(s.unsqueeze(-1) * freqs)  # (1, dim/2)
        cos = torch.cos(s.unsqueeze(-1) * freqs)
        enc = torch.cat([sin, cos], dim=-1).view(-1)[:dim]  # (dim,)
        return enc  # 1D tensor length sinus_dim

    def forward(self, s_scalar, B, L, device=None, level_idx=None):
        """
        s_scalar : float or 0-d tensor in [0,1] (global for this level)
        B : batch size
        L : sequence length (H*W)
        level_idx: optional int for indexing per-level bias (not used by default)
        returns: tensor (B, L, d_model)
        """

        # --- interpolation on table ---
        if not isinstance(s_scalar, torch.Tensor):
            s = torch.tensor(s_scalar, device=device)
        else:
            s = s_scalar.to(device)
        s = s.clamp(0.0, 1.0)

        # --- sinusoidal part ---
        sinus = self._sinusoidal_encoding(s_scalar).to(device)  # (sinus_dim,)
        sinus_proj = self.sinus_proj(sinus)  # (d_model,)

        # optional level bias (broadcast)
        if self.use_level_bias:
            combined = combined + self.level_bias.view(-1)

        # expand to (B, L, d_model)
        out = (
            sinus_proj.view(1, 1, self.d_model)
            .expand(B, L, self.d_model)
            .contiguous()
            .to(device)
        )
        return out
