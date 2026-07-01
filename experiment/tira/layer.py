import math
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiraLayer:
    """Mixin class holding block-wise rank-L TIRA adapter parameters."""

    def __init__(self, in_features: int, out_features: int):
        self.tira_b = nn.ParameterDict({})
        self.tira_a = nn.ParameterDict({})
        self.tira_M = {}
        self.tira_L = {}
        self.tira_alpha = {}
        self._tira_buf_names = {}
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name: str,
        M: int,
        L: int,
        alpha: Optional[int] = None,
    ):
        assert self.out_features % M == 0, (
            f"d_out={self.out_features} must be divisible by M={M}"
        )
        assert self.in_features % M == 0, (
            f"d_in={self.in_features} must be divisible by M={M}"
        )
        assert L >= 1, f"L={L} must be >= 1"

        self.tira_M[adapter_name] = M
        self.tira_L[adapter_name] = L
        self.tira_alpha[adapter_name] = L * M if alpha is None else alpha

        n_out = self.out_features // M
        n_in = self.in_features // M
        device = self.weight.device
        dtype = self.weight.dtype

        a_param = nn.Parameter(torch.empty(M, M, L, n_in, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))
        self.tira_a.update(nn.ParameterDict({adapter_name: a_param}))
        self.tira_b.update(nn.ParameterDict({
            adapter_name: nn.Parameter(torch.zeros(M, M, L, n_out, device=device, dtype=dtype))
        }))

        offset_idx = torch.arange(M)
        row_idx = torch.arange(M)
        col_idx = (row_idx.unsqueeze(0) + offset_idx.unsqueeze(1)) % M
        a_row_idx = (row_idx.unsqueeze(0) - offset_idx.unsqueeze(1)) % M
        col_buf = f"_tira_col_idx_{adapter_name}"
        arow_buf = f"_tira_a_row_idx_{adapter_name}"
        offset_buf = f"_tira_offset_idx_{adapter_name}"
        self.register_buffer(col_buf, col_idx)
        self.register_buffer(arow_buf, a_row_idx)
        self.register_buffer(offset_buf, offset_idx)
        self._tira_buf_names[adapter_name] = (col_buf, arow_buf, offset_buf)
        self.to(self.weight.device)

    @torch.no_grad()
    def delta_weight(self, adapter_name: str = None) -> torch.Tensor:
        if adapter_name is None:
            adapter_name = self.active_adapter
        M = self.tira_M[adapter_name]
        L = self.tira_L[adapter_name]
        b = self.tira_b[adapter_name]
        a = self.tira_a[adapter_name]
        n_out = b.shape[3]
        n_in = a.shape[3]

        blocks_all = torch.einsum("kmlo,kmli->kmoi", b, a)
        delta_blocks = torch.zeros(M, M, n_out, n_in, dtype=b.dtype, device=b.device)
        rows = torch.arange(M, device=b.device).unsqueeze(0).expand(M, M)
        col_buf, _, _ = self._tira_buf_names[adapter_name]
        col_idx = getattr(self, col_buf)
        delta_blocks.index_put_((rows, col_idx), blocks_all, accumulate=False)
        delta = delta_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return delta * (self.tira_alpha[adapter_name] / (L * M))


class TiraLinear(nn.Linear, TiraLayer):
    """nn.Linear with block-wise rank-L TIRA adapter."""

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        M: int = 16,
        L: int = 1,
        alpha: int = None,
        bias: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        TiraLayer.__init__(self, in_features=in_features, out_features=out_features)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, M, L, alpha=alpha)
        self.active_adapter = adapter_name

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.active_adapter in self.tira_b:
            dw = self.delta_weight(self.active_adapter)
            self._cached_delta = dw.clone()
            self.weight.data += dw
            self.merged = True

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.active_adapter in self.tira_b:
            self.weight.data -= self._cached_delta
            self._cached_delta = None
            self.merged = False

    def _adapter_forward(self, x_flat, b, a, col_idx, a_row_idx, offset_idx, M):
        L = a.shape[2]
        n_in = a.shape[3]
        n_out = b.shape[3]
        x_blocks = x_flat.reshape(-1, M, n_in)
        batch_size = x_blocks.shape[0]

        a_aligned = a[offset_idx[:, None], a_row_idx]
        a_flat = a_aligned.permute(1, 3, 0, 2).reshape(M, n_in, M * L)
        act = torch.bmm(
            x_blocks.permute(1, 0, 2),
            a_flat,
        ).permute(1, 0, 2).reshape(batch_size, M, M, L)

        act_by_offset = act.permute(0, 2, 1, 3)
        gather_idx = col_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, M, M, L)
        act_shifted = torch.gather(act_by_offset, 2, gather_idx)

        act_flat = act_shifted.permute(2, 0, 1, 3).reshape(M, batch_size, M * L)
        b_flat = b.permute(1, 0, 2, 3).reshape(M, M * L, n_out)
        y_blocks = torch.bmm(act_flat, b_flat).permute(1, 0, 2)
        return y_blocks.reshape(-1, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return F.linear(x, self.weight, self.bias)
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        result = F.linear(x, self.weight, self.bias)
        adapter_name = self.active_adapter
        if adapter_name in self.tira_b:
            M = self.tira_M[adapter_name]
            L = self.tira_L[adapter_name]
            b = self.tira_b[adapter_name]
            a = self.tira_a[adapter_name]
            col_buf, arow_buf, offset_buf = self._tira_buf_names[adapter_name]
            col_idx = getattr(self, col_buf)
            a_row_idx = getattr(self, arow_buf)
            offset_idx = getattr(self, offset_buf)
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features).to(b.dtype)
            y_delta = self._adapter_forward(x_flat, b, a, col_idx, a_row_idx, offset_idx, M)
            scale = self.tira_alpha[adapter_name] / (L * M)
            result = result + (y_delta * scale).to(previous_dtype).reshape(*orig_shape[:-1], -1)
        return result
