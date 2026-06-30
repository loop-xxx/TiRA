from dataclasses import dataclass

from tira.config import TiraConfig


@dataclass
class TiraLowerTriangularConfig(TiraConfig):
    """TIRA ablation that only writes block updates on or below the diagonal."""

    def __post_init__(self):
        self.peft_type = "TIRA_LOWER_TRIANGULAR"
