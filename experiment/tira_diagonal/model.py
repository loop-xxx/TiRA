from tira_ablation_common.model import BaseTiraAblationModel
from .layer import TiraDiagonalLinear


class TiraDiagonalModel(BaseTiraAblationModel):
    linear_cls = TiraDiagonalLinear
