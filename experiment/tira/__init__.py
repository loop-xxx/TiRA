"""TIRA: Tiled Rank-1 Adaptation for High-Rank Parameter-Efficient Fine-Tuning.

Block-structured high-rank adaptation using K groups with staggered
diagonal shift. Each group covers one diagonal band with M rank-1 blocks.
K >= M and must be a multiple of M, achieving up to rank
M^2 * min(K/M, n_out, n_in) with only K*(d_out+d_in) parameters.

Usage (standalone):
    from tira import TiraConfig, TiraPeftModel

    config = TiraConfig(
        tira_M=16,
        tira_K=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = TiraPeftModel(base_model, config)
    model.print_trainable_parameters()

Usage (for CausalLM):
    from tira import TiraConfig, TiraPeftModelForCausalLM

    config = TiraConfig(...)
    model = TiraPeftModelForCausalLM(base_model, config)
"""

from .config import TiraConfig
from .layer import TiraLayer, TiraLinear
from .model import TiraModel, mark_only_tira_as_trainable
from .peft_model import TiraPeftModel, TiraPeftModelForCausalLM
from .save_and_load import get_tira_model_state_dict, set_tira_model_state_dict

__all__ = [
    "TiraConfig",
    "TiraLayer",
    "TiraLinear",
    "TiraModel",
    "TiraPeftModel",
    "TiraPeftModelForCausalLM",
    "mark_only_tira_as_trainable",
    "get_tira_model_state_dict",
    "set_tira_model_state_dict",
]
