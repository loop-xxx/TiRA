import torch

from tira_ablation_common.layer import BaseTiraAblationLinear


class TiraDiagonalLinear(BaseTiraAblationLinear):
    """TIRA ablation with all subblocks placed on the main block diagonal."""

    placement = "diagonal"

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
        x_blocks = x_flat.reshape(-1, M, n_in)
        act = torch.bmm(
            x_blocks.permute(1, 0, 2),
            a.permute(1, 2, 0),
        )
        y_delta = torch.bmm(act, b.permute(1, 0, 2))
        return y_delta.permute(1, 0, 2).reshape(-1, self.out_features)
