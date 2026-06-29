from dataclasses import dataclass

from tira.config import TiraConfig


@dataclass
class TiraRowBalancedConfig(TiraConfig):
    """TIRA ablation where each group covers one block row and all columns."""

    def __post_init__(self):
        self.peft_type = "TIRA_ROW_BALANCED"
