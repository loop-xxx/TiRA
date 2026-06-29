from tira_ablation_common.peft_model import (
    BaseTiraAblationPeftModel,
    BaseTiraAblationPeftModelForCausalLM,
)
from .config import TiraRowBalancedConfig
from .model import TiraRowBalancedModel


class TiraRowBalancedPeftModel(BaseTiraAblationPeftModel):
    config_cls = TiraRowBalancedConfig
    model_cls = TiraRowBalancedModel


class TiraRowBalancedPeftModelForCausalLM(BaseTiraAblationPeftModelForCausalLM):
    config_cls = TiraRowBalancedConfig
    model_cls = TiraRowBalancedModel
