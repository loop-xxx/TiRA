from dataclasses import dataclass, field

from tira.config import TiraConfig


@dataclass
class TiraRandomUniformConfig(TiraConfig):
    """TIRA ablation with independently sampled block columns."""

    tira_placement_seed: int = field(default=0, metadata={"help": "Seed for random-uniform block placement"})

    def __post_init__(self):
        self.peft_type = "TIRA_RANDOM_UNIFORM"
