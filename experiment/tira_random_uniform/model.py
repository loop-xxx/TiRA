from tira_ablation_common.model import BaseTiraAblationModel
from .layer import TiraRandomUniformLinear


class TiraRandomUniformModel(BaseTiraAblationModel):
    linear_cls = TiraRandomUniformLinear
