import torch

from tira_ablation_common.layer import BaseTiraAblationLinear


class TiraRandomBalancedLinear(BaseTiraAblationLinear):
    """TIRA ablation with random placement and balanced row-wise block coverage."""

    placement = "random_balanced"

    @classmethod
    def build_col_idx(cls, M: int, K: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        columns = torch.arange(M).repeat(K // M)
        rows = []
        for _ in range(M):
            rows.append(columns[torch.randperm(K, generator=generator)])
        return torch.stack(rows, dim=1)
