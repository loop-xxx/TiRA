from tira_ablation_common.model import BaseTiraAblationModel
from .layer import TiraRandomBalancedLinear


class TiraRandomBalancedModel(BaseTiraAblationModel):
    linear_cls = TiraRandomBalancedLinear
