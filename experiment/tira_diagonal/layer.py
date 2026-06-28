import torch

from tira_ablation_common.layer import BaseTiraAblationLinear


class TiraDiagonalLinear(BaseTiraAblationLinear):
    """TIRA ablation with all subblocks placed on the main block diagonal."""

    placement = "diagonal"

    @classmethod
    def build_col_idx(cls, M: int, K: int, seed: int) -> torch.Tensor:
        return torch.arange(M).unsqueeze(0).expand(K, M).clone()
