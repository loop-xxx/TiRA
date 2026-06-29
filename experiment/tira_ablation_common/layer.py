import math
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTiraAblationLayer:
    placement = None

    def __init__(self, in_features: int, out_features: int):
        self.tira_b = nn.ParameterDict({})
        self.tira_a = nn.ParameterDict({})
        self.tira_M = {}
        self.tira_K = {}
        self.tira_alpha = {}
        self._tira_buf_names = {}
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def build_col_idx(cls, M: int, K: int) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build_row_col_idx(cls, M: int, K: int):
        m_idx = torch.arange(M)
        row_idx = m_idx.unsqueeze(0).expand(K, M).clone()
        return row_idx, cls.build_col_idx(M, K)

    def update_layer(
        self,
        adapter_name: str,
        M: int,
        K: int,
        alpha: Optional[int] = None,
    ):
        assert self.out_features % M == 0, (
            f"d_out={self.out_features} must be divisible by M={M}"
        )
        assert self.in_features % M == 0, (
            f"d_in={self.in_features} must be divisible by M={M}"
        )
        assert K >= M, f"K={K} must be >= M={M}"
        assert K % M == 0, f"K={K} must be a multiple of M={M}"

        self.tira_M[adapter_name] = M
        self.tira_K[adapter_name] = K
        self.tira_alpha[adapter_name] = K if alpha is None else alpha

        n_out = self.out_features // M
        n_in = self.in_features // M
        device = self.weight.device
        dtype = self.weight.dtype

        a_param = nn.Parameter(torch.empty(K, M, n_in, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))
        self.tira_a.update(nn.ParameterDict({adapter_name: a_param}))
        self.tira_b.update(nn.ParameterDict({
            adapter_name: nn.Parameter(torch.zeros(K, M, n_out, device=device, dtype=dtype))
        }))

        row_idx, col_idx = self.build_row_col_idx(M, K)
        ri_buf = f"_tira_row_idx_{adapter_name}"
        ci_buf = f"_tira_col_idx_{adapter_name}"
        self.register_buffer(ri_buf, row_idx)
        self.register_buffer(ci_buf, col_idx)
        self._tira_buf_names[adapter_name] = (ri_buf, ci_buf)
        self.to(self.weight.device)

    @torch.no_grad()
    def delta_weight(self, adapter_name: str = None) -> torch.Tensor:
        if adapter_name is None:
            adapter_name = self.active_adapter
        M = self.tira_M[adapter_name]
        K = self.tira_K[adapter_name]
        b = self.tira_b[adapter_name]
        a = self.tira_a[adapter_name]
        n_out = b.shape[2]
        n_in = a.shape[2]
        ri_buf, ci_buf = self._tira_buf_names[adapter_name]
        row_idx = getattr(self, ri_buf)
        col_idx = getattr(self, ci_buf)

        blocks_all = torch.einsum("kmo,kmi->kmoi", b, a)
        delta_blocks = torch.zeros(M, M, n_out, n_in, dtype=b.dtype, device=b.device)
        delta_blocks.index_put_((row_idx, col_idx), blocks_all, accumulate=True)
        delta = delta_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return delta * (self.tira_alpha[adapter_name] // K)

    def _adapter_forward(
        self,
        x_flat: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        n_in = a.shape[2]
        n_out = b.shape[2]
        x_blocks = x_flat.reshape(-1, M, n_in)
        batch_size = x_blocks.shape[0]
        y_blocks = None
        for k in range(col_idx.shape[0]):
            selected = x_blocks[:, col_idx[k], :]
            act = (selected * a[k].unsqueeze(0)).sum(dim=-1)
            contrib = act.unsqueeze(-1) * b[k].unsqueeze(0)
            if y_blocks is None:
                y_blocks = contrib.new_zeros(batch_size, M, n_out)
            y_blocks.index_add_(1, row_idx[k], contrib)
        return y_blocks.reshape(batch_size, self.out_features)


class BaseTiraAblationLinear(nn.Linear, BaseTiraAblationLayer):
    placement = None

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        M: int = 16,
        K: int = 16,
        alpha: int = None,
        bias: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        BaseTiraAblationLayer.__init__(self, in_features=in_features, out_features=out_features)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, M, K, alpha=alpha)
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
            K = self.tira_K[adapter_name]
            b = self.tira_b[adapter_name]
            a = self.tira_a[adapter_name]
            ri_buf, ci_buf = self._tira_buf_names[adapter_name]
            row_idx = getattr(self, ri_buf)
            col_idx = getattr(self, ci_buf)
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features).to(b.dtype)
            y_delta = self._adapter_forward(x_flat, b, a, row_idx, col_idx, M)
            scale = self.tira_alpha[adapter_name] // K
            result = result + (y_delta * scale).to(previous_dtype).reshape(*orig_shape[:-1], -1)
        return result
