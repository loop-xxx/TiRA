from dataclasses import dataclass

from tira.config import TiraConfig


@dataclass
class TiraDiagonalConfig(TiraConfig):
    """TIRA ablation that places every rank-1 subblock on the main block diagonal."""

    def __post_init__(self):
        self.peft_type = "TIRA_DIAGONAL"
