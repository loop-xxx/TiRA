import math
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiraUpperTriangularLayer:
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

    def update_layer(self, adapter_name: str, M: int, L: int, alpha: Optional[int] = None):
        assert self.out_features % M == 0, f"d_out={self.out_features} must be divisible by M={M}"
        assert self.in_features % M == 0, f"d_in={self.in_features} must be divisible by M={M}"
        assert L >= 1, f"L={L} must be >= 1"

        self.tira_M[adapter_name] = M
        self.tira_L[adapter_name] = L
        self.tira_alpha[adapter_name] = L * M if alpha is None else alpha

        n_out = self.out_features // M
        n_in = self.in_features // M
        device = self.weight.device
        dtype = self.weight.dtype

        offset_idx, row_idx, col_idx, dense_col_idx, a_row_idx, a_input_mask, offset_all = self._build_indices(M)
        num_blocks = row_idx.numel()
        a_param = nn.Parameter(torch.empty(num_blocks, L, n_in, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))
        self.tira_a.update(nn.ParameterDict({adapter_name: a_param}))
        self.tira_b.update(nn.ParameterDict({
            adapter_name: nn.Parameter(torch.zeros(num_blocks, L, n_out, device=device, dtype=dtype))
        }))

        offset_buf = f"_tira_offset_idx_{adapter_name}"
        row_buf = f"_tira_row_idx_{adapter_name}"
        col_buf = f"_tira_col_idx_{adapter_name}"
        dense_col_buf = f"_tira_dense_col_idx_{adapter_name}"
        arow_buf = f"_tira_a_row_idx_{adapter_name}"
        amask_buf = f"_tira_a_input_mask_{adapter_name}"
        offset_all_buf = f"_tira_offset_all_{adapter_name}"
        self.register_buffer(offset_buf, offset_idx)
        self.register_buffer(row_buf, row_idx)
        self.register_buffer(col_buf, col_idx)
        self.register_buffer(dense_col_buf, dense_col_idx)
        self.register_buffer(arow_buf, a_row_idx)
        self.register_buffer(amask_buf, a_input_mask)
        self.register_buffer(offset_all_buf, offset_all)
        self._tira_buf_names[adapter_name] = (
            offset_buf,
            row_buf,
            col_buf,
            dense_col_buf,
            arow_buf,
            amask_buf,
            offset_all_buf,
        )
        self.to(self.weight.device)

    @staticmethod
    def _build_indices(M: int):
        offsets = []
        rows = []
        cols = []
        dense_col_idx = torch.zeros(M, M, dtype=torch.long)
        a_row_idx = torch.zeros(M, M, dtype=torch.long)
        a_input_mask = torch.zeros(M, M, dtype=torch.bool)
        for offset in range(M):
            for row in range(M - offset):
                offsets.append(offset)
                rows.append(row)
                col = row + offset
                cols.append(col)
                dense_col_idx[offset, row] = col
            for input_col in range(offset, M):
                a_row_idx[offset, input_col] = input_col - offset
                a_input_mask[offset, input_col] = True
        return (
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            dense_col_idx,
            a_row_idx,
            a_input_mask,
            torch.arange(M, dtype=torch.long),
        )

    @torch.no_grad()
    def delta_weight(self, adapter_name: str = None) -> torch.Tensor:
        if adapter_name is None:
            adapter_name = self.active_adapter
        M = self.tira_M[adapter_name]
        L = self.tira_L[adapter_name]
        b = self.tira_b[adapter_name]
        a = self.tira_a[adapter_name]
        n_out = b.shape[2]
        n_in = a.shape[2]
        _, row_buf, col_buf, _, _, _, _ = self._tira_buf_names[adapter_name]
        row_idx = getattr(self, row_buf)
        col_idx = getattr(self, col_buf)

        blocks_all = torch.einsum("plo,pli->poi", b, a)
        delta_blocks = torch.zeros(M, M, n_out, n_in, dtype=b.dtype, device=b.device)
        delta_blocks.index_put_((row_idx, col_idx), blocks_all, accumulate=False)
        delta = delta_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return delta * (self.tira_alpha[adapter_name] / (L * M))


class TiraUpperTriangularLinear(nn.Linear, TiraUpperTriangularLayer):
    """TIRA ablation that only writes rank-L blocks on or above the block diagonal."""

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
        TiraUpperTriangularLayer.__init__(self, in_features=in_features, out_features=out_features)
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

    def _adapter_forward(
        self,
        x_flat,
        b,
        a,
        offset_idx,
        row_idx,
        dense_col_idx,
        a_row_idx,
        a_input_mask,
        offset_all,
        M,
    ):
        L = a.shape[1]
        n_in = a.shape[2]
        n_out = b.shape[2]
        x_blocks = x_flat.reshape(-1, M, n_in)
        batch_size = x_blocks.shape[0]

        dense_a = a.new_zeros(M, M, L, n_in)
        dense_b = b.new_zeros(M, M, L, n_out)
        dense_a.index_put_((offset_idx, row_idx), a, accumulate=False)
        dense_b.index_put_((offset_idx, row_idx), b, accumulate=False)

        a_aligned = dense_a[offset_all[:, None], a_row_idx]
        a_aligned = a_aligned * a_input_mask.to(a.dtype).unsqueeze(-1).unsqueeze(-1)
        a_flat = a_aligned.permute(1, 3, 0, 2).reshape(M, n_in, M * L)
        act = torch.bmm(
            x_blocks.permute(1, 0, 2),
            a_flat,
        ).permute(1, 0, 2).reshape(batch_size, M, M, L)

        act_by_offset = act.permute(0, 2, 1, 3)
        gather_idx = dense_col_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, M, M, L)
        act_shifted = torch.gather(act_by_offset, 2, gather_idx)

        act_flat = act_shifted.permute(2, 0, 1, 3).reshape(M, batch_size, M * L)
        b_flat = dense_b.permute(1, 0, 2, 3).reshape(M, M * L, n_out)
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
            offset_buf, row_buf, _col_buf, dense_col_buf, arow_buf, amask_buf, offset_all_buf = self._tira_buf_names[adapter_name]
            offset_idx = getattr(self, offset_buf)
            row_idx = getattr(self, row_buf)
            dense_col_idx = getattr(self, dense_col_buf)
            a_row_idx = getattr(self, arow_buf)
            a_input_mask = getattr(self, amask_buf)
            offset_all = getattr(self, offset_all_buf)
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features).to(b.dtype)
            y_delta = self._adapter_forward(
                x_flat,
                b,
                a,
                offset_idx,
                row_idx,
                dense_col_idx,
                a_row_idx,
                a_input_mask,
                offset_all,
                M,
            )
            scale = self.tira_alpha[adapter_name] / (L * M)
            result = result + (y_delta * scale).to(previous_dtype).reshape(*orig_shape[:-1], -1)
        return result
