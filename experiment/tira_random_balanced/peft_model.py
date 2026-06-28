from tira_ablation_common.peft_model import (
    BaseTiraAblationPeftModel,
    BaseTiraAblationPeftModelForCausalLM,
)
from .config import TiraRandomBalancedConfig
from .model import TiraRandomBalancedModel


class TiraRandomBalancedPeftModel(BaseTiraAblationPeftModel):
    config_cls = TiraRandomBalancedConfig
    model_cls = TiraRandomBalancedModel


class TiraRandomBalancedPeftModelForCausalLM(BaseTiraAblationPeftModelForCausalLM):
    config_cls = TiraRandomBalancedConfig
    model_cls = TiraRandomBalancedModel
