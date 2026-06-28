from dataclasses import dataclass, field

from tira.config import TiraConfig


@dataclass
class TiraDiagonalConfig(TiraConfig):
    """TIRA ablation that places every rank-1 subblock on the main block diagonal."""

    tira_placement_seed: int = field(default=0, metadata={"help": "Unused; kept for CLI/config parity"})

    def __post_init__(self):
        self.peft_type = "TIRA_DIAGONAL"
