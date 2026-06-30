from dataclasses import dataclass

from tira.config import TiraConfig


@dataclass
class TiraUpperTriangularConfig(TiraConfig):
    """TIRA ablation that only writes block updates on or above the diagonal."""

    def __post_init__(self):
        self.peft_type = "TIRA_UPPER_TRIANGULAR"
