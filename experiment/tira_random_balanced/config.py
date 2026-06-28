from dataclasses import dataclass, field

from tira.config import TiraConfig


@dataclass
class TiraRandomBalancedConfig(TiraConfig):
    """TIRA ablation with random placement and balanced block coverage."""

    tira_placement_seed: int = field(default=0, metadata={"help": "Seed for random-balanced block placement"})

    def __post_init__(self):
        self.peft_type = "TIRA_RANDOM_BALANCED"
