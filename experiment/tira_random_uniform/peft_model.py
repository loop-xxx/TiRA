from tira_ablation_common.peft_model import (
    BaseTiraAblationPeftModel,
    BaseTiraAblationPeftModelForCausalLM,
)
from .config import TiraRandomUniformConfig
from .model import TiraRandomUniformModel


class TiraRandomUniformPeftModel(BaseTiraAblationPeftModel):
    config_cls = TiraRandomUniformConfig
    model_cls = TiraRandomUniformModel


class TiraRandomUniformPeftModelForCausalLM(BaseTiraAblationPeftModelForCausalLM):
    config_cls = TiraRandomUniformConfig
    model_cls = TiraRandomUniformModel
