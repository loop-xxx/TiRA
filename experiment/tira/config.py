"""Configuration for TIRA (Tiled Rank-1 Adaptation).

Extends hira's PeftConfig so that TIRA integrates with the PeftModel
save/load/checkpoint infrastructure.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from hira.utils import PeftConfig


@dataclass
class TiraConfig(PeftConfig):
    """Configuration for the TIRA adapter.

    TIRA gives each covered subblock a rank upper bound L.
    block band. Each group has M pairs of block vectors (b, a) that
    create rank-1 blocks along one staggered diagonal.

    a_{k,m} ∈ R^{n_in}  — input-side vector  (kaiming init, like LoRA A)
    b_{k,m} ∈ R^{n_out} — output-side vector (zero init, like LoRA B)

        ΔW = Σ_k Σ_m  [ block at (m, (m+k)%M) = b_{k,m} · a_{k,m}^T ]

    Args:
        tira_M: Number of block segments. Must divide both d_out and d_in.
        tira_L: Rank upper bound for each covered subblock.
        tira_q_M: Optional override of M for q_proj only; fallback to tira_M when unset.
        tira_q_L: Optional override of L for q_proj only; fallback to tira_L when unset.
        tira_k_M: Optional override of M for k_proj only; fallback to tira_M when unset.
        tira_k_L: Optional override of L for k_proj only; fallback to tira_L when unset.
        tira_v_M: Optional override of M for v_proj only; fallback to tira_M when unset.
        tira_v_L: Optional override of L for v_proj only; fallback to tira_L when unset.
        tira_o_M: Optional override of M for o_proj only; fallback to tira_M when unset.
        tira_o_L: Optional override of L for o_proj only; fallback to tira_L when unset.
        tira_up_M: Optional override of M for up_proj only; fallback to tira_M when unset.
        tira_up_L: Optional override of L for up_proj only; fallback to tira_L when unset.
        tira_down_M: Optional override of M for down_proj only; fallback to tira_M when unset.
        tira_down_L: Optional override of L for down_proj only; fallback to tira_L when unset.
        target_modules: Module name suffixes (or regex) to apply TIRA to.
        bias: Bias handling strategy. Can be 'none', 'all', or 'tira_only'.
        modules_to_save: Extra modules to keep trainable and save.
    """
    tira_M: int = field(default=16, metadata={"help": "Number of block segments M"})
    tira_L: int = field(default=1, metadata={"help": "Subblock rank upper bound L"})
    tira_alpha: Optional[int] = field(default=None, metadata={"help": "Scaling alpha for TIRA; effective scale is alpha / (L * M). If None, uses L * M"})
    tira_q_M: Optional[int] = field(default=None, metadata={"help": "Override M for q_proj only; fallback to tira_M"})
    tira_q_L: Optional[int] = field(default=None, metadata={"help": "Override L for q_proj only; fallback to tira_L"})
    tira_k_M: Optional[int] = field(default=None, metadata={"help": "Override M for k_proj only; fallback to tira_M"})
    tira_k_L: Optional[int] = field(default=None, metadata={"help": "Override L for k_proj only; fallback to tira_L"})
    tira_v_M: Optional[int] = field(default=None, metadata={"help": "Override M for v_proj only; fallback to tira_M"})
    tira_v_L: Optional[int] = field(default=None, metadata={"help": "Override L for v_proj only; fallback to tira_L"})
    tira_o_M: Optional[int] = field(default=None, metadata={"help": "Override M for o_proj only; fallback to tira_M"})
    tira_o_L: Optional[int] = field(default=None, metadata={"help": "Override L for o_proj only; fallback to tira_L"})
    tira_up_M: Optional[int] = field(default=None, metadata={"help": "Override M for up_proj only; fallback to tira_M"})
    tira_up_L: Optional[int] = field(default=None, metadata={"help": "Override L for up_proj only; fallback to tira_L"})
    tira_down_M: Optional[int] = field(default=None, metadata={"help": "Override M for down_proj only; fallback to tira_M"})
    tira_down_L: Optional[int] = field(default=None, metadata={"help": "Override L for down_proj only; fallback to tira_L"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module name suffixes to apply TIRA to"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type: 'none', 'all', or 'tira_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Extra modules to save besides TIRA adapter weights"},
    )

    def __post_init__(self):
        self.peft_type = "TIRA"
