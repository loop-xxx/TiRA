import torch

from tira_ablation_common.layer import BaseTiraAblationLinear


class TiraRowBalancedLinear(BaseTiraAblationLinear):
    """Each group writes all columns in one row; groups uniformly cover rows."""

    placement = "row_balanced"

    @classmethod
    def build_row_col_idx(cls, M: int, K: int):
        group_rows = (torch.arange(K) % M).unsqueeze(1).expand(K, M).clone()
        all_cols = torch.arange(M).unsqueeze(0).expand(K, M).clone()
        return group_rows, all_cols

    @classmethod
    def build_col_idx(cls, M: int, K: int) -> torch.Tensor:
        return torch.arange(M).unsqueeze(0).expand(K, M).clone()

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

        act = torch.bmm(
            x_blocks.permute(1, 0, 2),
            a.permute(1, 2, 0),
        ).permute(1, 2, 0)
        group_delta = torch.einsum("bkm,kmo->bko", act, b)

        group_rows = row_idx[:, 0]
        y_blocks = group_delta.new_zeros(x_blocks.shape[0], M, n_out)
        y_blocks.index_add_(1, group_rows, group_delta)
        return y_blocks.reshape(-1, self.out_features)
