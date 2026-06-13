"""Core TIRA layer: TiraLayer mixin + TiraLinear.

TIRA (Tiled Rank-1 Adaptation) uses K groups, each covering a staggered
diagonal block band. Each group k (0-indexed) has M pairs of block vectors
(b_{k,m}, a_{k,m}).

Algorithm:
    ΔW = Σ_k Σ_m  [ block at (m, (m+k)%M) = b_{k,m} · a_{k,m}^T ]

    Group k places block b_{k,m} · a_{k,m}^T at row-block m, col-block
    (m+k) mod M, forming the k-th staggered diagonal band.

    a_{k,m} ∈ R^{n_in}  — input-side vector  (kaiming init, like LoRA A)
    b_{k,m} ∈ R^{n_out} — output-side vector (zero init, like LoRA B)

Forward pass avoids materializing the full ΔW (d_out × d_in) by computing
ΔW·x block-wise with rank-1 multiplications:
    For each group k:
        1. Reshape x into M blocks of size n_in
        2. Roll by -k to align input blocks with output blocks
        3. Dot-product with a[k] → scalar per block
        4. Scale by b[k] → output block contribution
    Cost: O(K × batch × (d_in + d_out))  vs  O(batch × d_in × d_out)
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiraLayer:
    """Mixin class holding TIRA adapter parameters.

    Mirrors hira's LoraLayer / CLRA's ClraLayer: stores adapter params in
    ParameterDicts keyed by adapter_name, supports multiple adapters,
    merge/unmerge.
    """

    def __init__(self, in_features: int, out_features: int):
        self.tira_b = nn.ParameterDict({})
        self.tira_a = nn.ParameterDict({})
        self.tira_M = {}
        self.tira_K = {}
        self.tira_alpha = {}
        self._tira_buf_names = {}      # adapter_name -> (a_shift, col_idx, k_idx) buffer names
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name: str,
        M: int,
        K: int,
        alpha: float = None,
    ):
        """Initialize or update adapter parameters for a given adapter name.

        Args:
            M: Number of block segments. Must divide both d_out and d_in.
            K: Number of groups (K >= M, must be a multiple of M).
        """
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
        self.tira_alpha[adapter_name] = float(K if alpha is None else alpha)

        n_out = self.out_features // M
        n_in = self.in_features // M

        device = self.weight.device
        dtype = self.weight.dtype

        # a: kaiming init (input side, like LoRA A)
        a_param = nn.Parameter(torch.empty(K, M, n_in, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))
        self.tira_a.update(nn.ParameterDict({adapter_name: a_param}))
        # b: zero init (output side, like LoRA B) → ΔW = 0 at init
        self.tira_b.update(nn.ParameterDict({
            adapter_name: nn.Parameter(
                torch.zeros(K, M, n_out, device=device, dtype=dtype)
            )
        }))

        # Precompute integer shift indices as buffers (computed once, auto-moved with .to())
        k_idx = torch.arange(K)
        m_idx = torch.arange(M)
        # a_shift[k,p] = (p-k)%M  →  a_aligned[k,p] = a[k,a_shift[k,p]]
        # col_idx[k,m] = (m+k)%M  →  act_shifted[b,k,m] = act[b,k,col_idx[k,m]]  &  delta_weight col
        a_shift = (m_idx.unsqueeze(0) - k_idx.unsqueeze(1)) % M
        col_idx = (m_idx.unsqueeze(0) + k_idx.unsqueeze(1)) % M
        as_buf = f'_tira_a_shift_{adapter_name}'
        ci_buf = f'_tira_col_idx_{adapter_name}'
        ki_buf = f'_tira_k_idx_{adapter_name}'
        self.register_buffer(as_buf, a_shift)
        self.register_buffer(ci_buf, col_idx)
        self.register_buffer(ki_buf, k_idx)
        self._tira_buf_names[adapter_name] = (as_buf, ci_buf, ki_buf)

        self.to(self.weight.device)

    @torch.no_grad()
    def delta_weight(self, adapter_name: str = None) -> torch.Tensor:
        """Compute full ΔW by placing rank-1 blocks on staggered diagonals.

        ΔW = Σ_k Σ_m  [ block at (m, (m+k)%M) = b_{k,m} a_{k,m}^T ]

        Returns:
            Tensor of shape (d_out, d_in).
        """
        if adapter_name is None:
            adapter_name = self.active_adapter
        M = self.tira_M[adapter_name]
        K = self.tira_K[adapter_name]

        b = self.tira_b[adapter_name]   # (K, M, n_out)
        a = self.tira_a[adapter_name]   # (K, M, n_in)
        n_out = b.shape[2]
        n_in = a.shape[2]

        # Compute all K*M outer products at once, then scatter.
        as_buf, ci_buf, ki_buf = self._tira_buf_names[adapter_name]
        col_idx = getattr(self, ci_buf)   # (K, M): (m+k)%M
        k_idx   = getattr(self, ki_buf)   # (K,)
        m_idx   = torch.arange(M, device=b.device)

        blocks_all = torch.einsum('kmi,kmj->kmij', b, a)            # (K, M, n_out, n_in)

        delta_blocks = torch.zeros(M, M, n_out, n_in, dtype=b.dtype, device=b.device)
        row_idx = m_idx[None, :].expand(K, M)
        delta_blocks.index_put_((row_idx, col_idx), blocks_all, accumulate=True)

        # Reshape: (M, n_out, M, n_in) → (M*n_out, M*n_in) = (d_out, d_in)
        delta = delta_blocks.permute(0, 2, 1, 3).reshape(
            self.out_features, self.in_features
        )
        alpha = self.tira_alpha[adapter_name]
        scale = alpha / K
        return delta * scale


class TiraLinear(nn.Linear, TiraLayer):
    """nn.Linear with TIRA adapter.

    Follows hira's Linear(nn.Linear, LoraLayer) pattern:
    - Inherits nn.Linear to hold the frozen base weight
    - Inherits TiraLayer for adapter parameter management
    - forward() adds block-wise ΔW·x to base linear output
    """

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        M: int = 16,
        K: int = 16,
        alpha: float = None,
        bias: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        TiraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freeze the pre-trained weight matrix
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, M, K, alpha=alpha)
        self.active_adapter = adapter_name

    def merge(self):
        """Merge adapter weights into the base weight."""
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.active_adapter in self.tira_b:
            dw = self.delta_weight(self.active_adapter)
            self._cached_delta = dw.clone()
            self.weight.data += dw
            self.merged = True

    def unmerge(self):
        """Unmerge adapter weights from the base weight."""
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

        # Base linear: W0 · x + bias
        result = F.linear(x, self.weight, self.bias)

        # TIRA delta via efficient block-wise rank-1 multiplication
        adapter_name = self.active_adapter
        if adapter_name in self.tira_b:
            M = self.tira_M[adapter_name]
            K = self.tira_K[adapter_name]
            b = self.tira_b[adapter_name]   # (K, M, n_out)
            a = self.tira_a[adapter_name]   # (K, M, n_in)
            alpha = self.tira_alpha[adapter_name]

            adapter_dtype = b.dtype
            n_out = b.shape[2]
            n_in = a.shape[2]

            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features).to(adapter_dtype)
            x_blocks = x_flat.reshape(-1, M, n_in)   # (batch, M, n_in)

            # Shift a (small, K×M×n_in) instead of x (huge, batch×K×M×n_in).
            # Buffers are precomputed in update_layer and auto-moved with .to().
            as_buf, ci_buf, ki_buf = self._tira_buf_names[adapter_name]
            a_shift = getattr(self, as_buf)   # (K, M): (p-k)%M
            col_idx = getattr(self, ci_buf)   # (K, M): (m+k)%M
            k_idx   = getattr(self, ki_buf)   # (K,)

            # Align a with input blocks: a_aligned[k,p] = a[k,(p-k)%M]
            a_aligned = a[k_idx[:, None], a_shift, :]                  # (K, M, n_in)

            # Step 1: batched GEMM — (M, batch, n_in) × (M, n_in, K) → (batch, K, M)
            act = torch.bmm(
                x_blocks.permute(1, 0, 2),      # (M, batch, n_in)
                a_aligned.permute(1, 2, 0),     # (M, n_in, K)
            ).permute(1, 2, 0)                  # (batch, K, M)

            # Step 2: re-index act from input-block to output-block coords
            act_shifted = act[:, k_idx[:, None], col_idx]                  # (batch, K, M)

            # Step 3: batched GEMM — (M, batch, K) × (M, K, n_out) → (batch, M, n_out)
            y_delta = torch.bmm(
                act_shifted.permute(2, 0, 1),     # (M, batch, K)
                b.permute(1, 0, 2),             # (M, K, n_out)
            ).permute(1, 0, 2)                  # (batch, M, n_out)

            y_delta = y_delta.reshape(-1, self.out_features)
            scale = alpha / K
            result = result + (y_delta * scale).to(previous_dtype).reshape(*orig_shape[:-1], -1)

        return result
