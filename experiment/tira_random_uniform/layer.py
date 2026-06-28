import torch

from tira_ablation_common.layer import BaseTiraAblationLinear


class TiraRandomUniformLinear(BaseTiraAblationLinear):
    """TIRA ablation with independently sampled block columns."""

    placement = "random_uniform"

    @classmethod
    def build_col_idx(cls, M: int, K: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randint(low=0, high=M, size=(K, M), generator=generator)
