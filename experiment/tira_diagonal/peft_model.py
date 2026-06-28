from tira_ablation_common.peft_model import (
    BaseTiraAblationPeftModel,
    BaseTiraAblationPeftModelForCausalLM,
)
from .config import TiraDiagonalConfig
from .model import TiraDiagonalModel


class TiraDiagonalPeftModel(BaseTiraAblationPeftModel):
    config_cls = TiraDiagonalConfig
    model_cls = TiraDiagonalModel


class TiraDiagonalPeftModelForCausalLM(BaseTiraAblationPeftModelForCausalLM):
    config_cls = TiraDiagonalConfig
    model_cls = TiraDiagonalModel
