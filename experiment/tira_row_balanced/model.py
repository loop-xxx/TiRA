from tira_ablation_common.model import BaseTiraAblationModel
from .layer import TiraRowBalancedLinear


class TiraRowBalancedModel(BaseTiraAblationModel):
    linear_cls = TiraRowBalancedLinear
