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

    TIRA uses K groups, each covering a staggered diagonal
    block band. Each group has M pairs of block vectors (b, a) that
    create rank-1 blocks along one staggered diagonal.

    a_{k,m} ∈ R^{n_in}  — input-side vector  (kaiming init, like LoRA A)
    b_{k,m} ∈ R^{n_out} — output-side vector (zero init, like LoRA B)

        ΔW = Σ_k Σ_m  [ block at (m, (m+k)%M) = b_{k,m} · a_{k,m}^T ]

    Args:
        tira_M: Number of block segments. Must divide both d_out and d_in.
        tira_K: Number of groups (K >= M, must be a multiple of M).
        tira_q_M: Optional override of M for q_proj only; fallback to tira_M when unset.
        tira_q_K: Optional override of K for q_proj only; fallback to tira_K when unset.
        tira_k_M: Optional override of M for k_proj only; fallback to tira_M when unset.
        tira_k_K: Optional override of K for k_proj only; fallback to tira_K when unset.
        tira_v_M: Optional override of M for v_proj only; fallback to tira_M when unset.
        tira_v_K: Optional override of K for v_proj only; fallback to tira_K when unset.
        tira_o_M: Optional override of M for o_proj only; fallback to tira_M when unset.
        tira_o_K: Optional override of K for o_proj only; fallback to tira_K when unset.
        tira_up_M: Optional override of M for up_proj only; fallback to tira_M when unset.
        tira_up_K: Optional override of K for up_proj only; fallback to tira_K when unset.
        tira_down_M: Optional override of M for down_proj only; fallback to tira_M when unset.
        tira_down_K: Optional override of K for down_proj only; fallback to tira_K when unset.
        target_modules: Module name suffixes (or regex) to apply TIRA to.
        bias: Bias handling strategy. Can be 'none', 'all', or 'tira_only'.
        modules_to_save: Extra modules to keep trainable and save.
    """
    tira_M: int = field(default=16, metadata={"help": "Number of block segments M"})
    tira_K: int = field(default=16, metadata={"help": "Number of groups K (K >= M, multiple of M)"})
    tira_alpha: Optional[int] = field(default=None, metadata={"help": "Scaling alpha for TIRA; effective scale is alpha / K. If None, uses K"})
    tira_q_M: Optional[int] = field(default=None, metadata={"help": "Override M for q_proj only; fallback to tira_M"})
    tira_q_K: Optional[int] = field(default=None, metadata={"help": "Override K for q_proj only; fallback to tira_K"})
    tira_k_M: Optional[int] = field(default=None, metadata={"help": "Override M for k_proj only; fallback to tira_M"})
    tira_k_K: Optional[int] = field(default=None, metadata={"help": "Override K for k_proj only; fallback to tira_K"})
    tira_v_M: Optional[int] = field(default=None, metadata={"help": "Override M for v_proj only; fallback to tira_M"})
    tira_v_K: Optional[int] = field(default=None, metadata={"help": "Override K for v_proj only; fallback to tira_K"})
    tira_o_M: Optional[int] = field(default=None, metadata={"help": "Override M for o_proj only; fallback to tira_M"})
    tira_o_K: Optional[int] = field(default=None, metadata={"help": "Override K for o_proj only; fallback to tira_K"})
    tira_up_M: Optional[int] = field(default=None, metadata={"help": "Override M for up_proj only; fallback to tira_M"})
    tira_up_K: Optional[int] = field(default=None, metadata={"help": "Override K for up_proj only; fallback to tira_K"})
    tira_down_M: Optional[int] = field(default=None, metadata={"help": "Override M for down_proj only; fallback to tira_M"})
    tira_down_K: Optional[int] = field(default=None, metadata={"help": "Override K for down_proj only; fallback to tira_K"})
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
