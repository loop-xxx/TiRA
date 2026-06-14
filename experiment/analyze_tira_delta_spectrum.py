from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
ADAPTER_WEIGHTS = "adapter_model.bin"
ADAPTER_CONFIG = "adapter_config.json"
TRAINER_STATE = "trainer_state.json"


def require_torch():
    if torch is None:
        raise SystemExit(
            "PyTorch is required for SVD analysis. Please run in your project env, "
            "e.g. `conda activate <env>` then rerun."
        )


@dataclass
class RunCheckpoint:
    run_dir: str
    checkpoint_dir: str
    best_metric: Optional[float]
    source: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the singular-value spectrum of TIRA adapter delta weights. "
            "Use --root_dir to scan all training runs and write consolidated results "
            "to that root directory, or --checkpoint for a single checkpoint."
        )
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Directory containing TIRA training run folders.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Analyze one checkpoint directory directly.")
    parser.add_argument("--tau", type=float, default=0.005, help="Absolute singular-value threshold for effective rank.")
    parser.add_argument(
        "--energy_thresholds",
        type=str,
        default="0.90,0.95,0.99",
        help="Comma-separated cumulative energy thresholds for energy ranks.",
    )
    parser.add_argument(
        "--top_k_singular",
        type=int,
        default=10,
        help="Number of leading singular values to store in the detailed CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for consolidated outputs. Defaults to --root_dir, or the checkpoint directory in single mode.",
    )
    parser.add_argument("--output_prefix", type=str, default="tira_delta_spectrum")
    parser.add_argument(
        "--include_missing",
        action="store_true",
        help="Emit rows for target modules that are absent from a checkpoint.",
    )
    parser.add_argument(
        "--no_scale",
        action="store_true",
        help="Do not apply TIRA alpha/K scaling before SVD. Rank-energy metrics are unchanged by scaling.",
    )
    return parser.parse_args()


def checkpoint_step(path: str) -> Optional[int]:
    match = re.search(r"checkpoint-(\d+)$", os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else None


def is_checkpoint_dir(path: str) -> bool:
    if not (
        os.path.isdir(path)
        and re.match(r"checkpoint-\d+$", os.path.basename(os.path.normpath(path))) is not None
        and os.path.isfile(os.path.join(path, ADAPTER_WEIGHTS))
        and os.path.isfile(os.path.join(path, ADAPTER_CONFIG))
    ):
        return False

    try:
        cfg = read_json(os.path.join(path, ADAPTER_CONFIG))
    except (OSError, json.JSONDecodeError):
        return False
    return str(cfg.get("peft_type", "")).lower() == "tira"


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_checkpoint_path(path: str, run_dir: str) -> str:
    if os.path.isabs(path) and os.path.isdir(path):
        return os.path.normpath(path)

    candidates = [
        os.path.normpath(os.path.join(run_dir, path)),
        os.path.normpath(os.path.join(run_dir, os.path.basename(os.path.normpath(path)))),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return os.path.normpath(path)


def immediate_checkpoints(run_dir: str) -> List[str]:
    checkpoints = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if is_checkpoint_dir(path):
            checkpoints.append(path)
    checkpoints.sort(key=lambda p: checkpoint_step(p) or -1)
    return checkpoints


def trainer_states_for_run(run_dir: str, checkpoints: Sequence[str]) -> Iterable[Tuple[str, dict]]:
    # Prefer the final TrainerState. Earlier checkpoint states may contain an
    # outdated best_model_checkpoint when multiple checkpoints are retained.
    root_state = os.path.join(run_dir, TRAINER_STATE)
    if os.path.isfile(root_state):
        yield root_state, read_json(root_state)

    for checkpoint in reversed(checkpoints):
        state_path = os.path.join(checkpoint, TRAINER_STATE)
        if os.path.isfile(state_path):
            yield state_path, read_json(state_path)


def select_best_checkpoint(run_dir: str) -> Optional[RunCheckpoint]:
    checkpoints = immediate_checkpoints(run_dir)
    if not checkpoints:
        return None

    for state_path, state in trainer_states_for_run(run_dir, checkpoints):
        best_path = state.get("best_model_checkpoint")
        if not best_path:
            continue

        checkpoint = normalize_checkpoint_path(best_path, run_dir)
        if not is_checkpoint_dir(checkpoint):
            checkpoint = os.path.join(run_dir, os.path.basename(os.path.normpath(best_path)))

        if is_checkpoint_dir(checkpoint):
            return RunCheckpoint(
                run_dir=run_dir,
                checkpoint_dir=os.path.normpath(checkpoint),
                best_metric=state.get("best_metric"),
                source=f"trainer_state:{os.path.relpath(state_path, run_dir)}",
            )

    return RunCheckpoint(
        run_dir=run_dir,
        checkpoint_dir=os.path.normpath(checkpoints[-1]),
        best_metric=None,
        source="fallback:last_checkpoint",
    )


def discover_run_checkpoints(root_dir: str) -> List[RunCheckpoint]:
    root_dir = os.path.normpath(root_dir)
    runs: List[RunCheckpoint] = []
    seen_run_dirs = set()

    for current, dirnames, _filenames in os.walk(root_dir):
        # Checkpoint directories are leaves for this analysis; do not recurse into them.
        dirnames[:] = [d for d in dirnames if re.match(r"checkpoint-\d+$", d) is None]

        selected = select_best_checkpoint(current)
        if selected is None:
            continue
        if selected.run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(selected.run_dir)
        runs.append(selected)

    runs.sort(key=lambda r: os.path.relpath(r.run_dir, root_dir))
    return runs


def load_checkpoint(checkpoint_dir: str):
    require_torch()
    model_path = os.path.join(checkpoint_dir, ADAPTER_WEIGHTS)
    config_path = os.path.join(checkpoint_dir, ADAPTER_CONFIG)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find adapter weights: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find adapter config: {config_path}")

    cfg = read_json(config_path)
    state_dict = torch.load(model_path, map_location="cpu")
    return state_dict, cfg


def extract_layer_module(key: str):
    layer_match = re.search(r"\.layers\.(\d+)\.", key)
    module_match = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj)\.", key)
    if layer_match is None or module_match is None:
        return None
    return int(layer_match.group(1)), module_match.group(1)


def build_delta_from_tira(a: torch.Tensor, b: torch.Tensor, alpha: Optional[float], apply_scale: bool):
    K, M, n_in = a.shape
    _, _, n_out = b.shape
    delta_blocks = torch.zeros(M, M, n_out, n_in, dtype=torch.float32)

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    for k in range(K):
        for m in range(M):
            col = (m + k) % M
            delta_blocks[m, col] += torch.outer(b[k, m], a[k, m])

    delta = delta_blocks.permute(0, 2, 1, 3).reshape(M * n_out, M * n_in)
    if apply_scale:
        scale = float(K if alpha is None else alpha) / float(K)
        delta = delta * scale
    return delta


def parse_energy_thresholds(raw: str) -> List[float]:
    thresholds = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0 or value >= 1:
            raise ValueError(f"Energy threshold must be in (0, 1), got {value}")
        thresholds.append(value)
    return thresholds


def energy_rank(s: torch.Tensor, threshold: float) -> int:
    energy = s * s
    total = energy.sum()
    if total.item() <= 0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=cumulative.dtype)).item() + 1)


def get_alpha_for_module(cfg: dict, module_name: str, K: int) -> Optional[float]:
    alpha = cfg.get("tira_alpha")
    if alpha is None:
        return None
    return float(alpha)


def group_tira_weights(state_dict: Dict[str, torch.Tensor]):
    grouped: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        parsed = extract_layer_module(key)
        if parsed is None:
            continue
        layer_id, module_name = parsed
        grouped.setdefault((layer_id, module_name), {})
        if key.endswith(".tira_a"):
            grouped[(layer_id, module_name)]["a"] = value
        elif key.endswith(".tira_b"):
            grouped[(layer_id, module_name)]["b"] = value
    return grouped


def analyze_checkpoint(
    checkpoint_dir: str,
    tau: float,
    energy_thresholds: Sequence[float],
    top_k_singular: int,
    include_missing: bool,
    apply_scale: bool,
):
    state_dict, cfg = load_checkpoint(checkpoint_dir)
    grouped = group_tira_weights(state_dict)

    if not grouped:
        raise ValueError(f"No TIRA layer/module weights were found in {checkpoint_dir}")

    layer_ids = sorted({layer for layer, _module in grouped.keys()})
    rows = []

    for layer_id in layer_ids:
        modules = MODULES if include_missing else [m for m in MODULES if (layer_id, m) in grouped]
        for module_name in modules:
            item = grouped.get((layer_id, module_name))
            if item is None or "a" not in item or "b" not in item:
                rows.append(
                    {
                        "layer": layer_id,
                        "module": module_name,
                        "note": "missing_in_checkpoint",
                    }
                )
                continue

            a = item["a"]
            b = item["b"]
            K, M, n_in = a.shape
            _, _, n_out = b.shape
            alpha = get_alpha_for_module(cfg, module_name, K)
            delta = build_delta_from_tira(a, b, alpha=alpha, apply_scale=apply_scale)
            singular_values = torch.linalg.svdvals(delta)
            singular_values = torch.sort(singular_values, descending=True).values

            spectral_norm = float(singular_values[0].item()) if singular_values.numel() > 0 else 0.0
            frobenius_sq = float((singular_values * singular_values).sum().item())
            nuclear_norm = float(singular_values.sum().item())
            stable_rank = frobenius_sq / (spectral_norm * spectral_norm) if spectral_norm > 0 else 0.0

            row = {
                "layer": layer_id,
                "module": module_name,
                "K": int(K),
                "M": int(M),
                "n_out": int(n_out),
                "n_in": int(n_in),
                "delta_rows": int(delta.shape[0]),
                "delta_cols": int(delta.shape[1]),
                "rank_upper_bound": int(min(delta.shape[0], delta.shape[1], K * M)),
                "effective_rank_tau": int((singular_values > tau).sum().item()),
                "stable_rank": stable_rank,
                "spectral_norm": spectral_norm,
                "frobenius_norm": float(frobenius_sq ** 0.5),
                "frobenius_sq": frobenius_sq,
                "nuclear_norm": nuclear_norm,
                "tau": tau,
                "alpha": float(K if alpha is None else alpha),
                "scale_applied": apply_scale,
                "top_singular_values": json.dumps(
                    [float(x) for x in singular_values[:top_k_singular].tolist()]
                ),
                "note": "ok",
            }
            for threshold in energy_thresholds:
                pct = int(round(threshold * 100))
                row[f"rank_energy_{pct}"] = energy_rank(singular_values, threshold)
            rows.append(row)

    return rows, cfg


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def summarize_rows(detail_rows: Sequence[dict], energy_thresholds: Sequence[float]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in detail_rows:
        if row.get("note") != "ok":
            continue
        groups.setdefault((row["run_name"], row["module"]), []).append(row)
        groups.setdefault((row["run_name"], "ALL"), []).append(row)

    summary = []
    for (run_name, module_name), rows in sorted(groups.items()):
        eff = [float(r["effective_rank_tau"]) for r in rows]
        stable = [float(r["stable_rank"]) for r in rows]
        fro = [float(r["frobenius_norm"]) for r in rows]
        item = {
            "run_name": run_name,
            "module": module_name,
            "num_layers": len(rows),
            "checkpoint": rows[0]["checkpoint"],
            "best_metric": rows[0].get("best_metric", ""),
            "base_model": rows[0].get("base_model", ""),
            "tira_M": rows[0].get("tira_M", ""),
            "tira_K": rows[0].get("tira_K", ""),
            "mean_effective_rank_tau": mean(eff),
            "median_effective_rank_tau": median(eff),
            "max_effective_rank_tau": max(eff),
            "mean_stable_rank": mean(stable),
            "median_stable_rank": median(stable),
            "mean_frobenius_norm": mean(fro),
        }
        for threshold in energy_thresholds:
            pct = int(round(threshold * 100))
            key = f"rank_energy_{pct}"
            values = [float(r[key]) for r in rows if key in r]
            item[f"mean_{key}"] = mean(values)
            item[f"median_{key}"] = median(values)
        summary.append(item)
    return summary


def union_fieldnames(rows: Sequence[dict], preferred: Sequence[str]) -> List[str]:
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def write_csv(rows: Sequence[dict], path: str, preferred_fields: Sequence[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = union_fieldnames(rows, preferred_fields)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(output_dir: str, prefix: str, tau: float):
    tau_tag = str(tau).replace(".", "p")
    return {
        "detail_csv": os.path.join(output_dir, f"{prefix}_layers_tau_{tau_tag}.csv"),
        "summary_csv": os.path.join(output_dir, f"{prefix}_summary_tau_{tau_tag}.csv"),
        "manifest_json": os.path.join(output_dir, f"{prefix}_manifest_tau_{tau_tag}.json"),
    }


def enrich_rows(
    rows: List[dict],
    run: RunCheckpoint,
    cfg: dict,
    root_dir: Optional[str],
):
    run_name = os.path.basename(os.path.normpath(run.run_dir))
    for row in rows:
        row.update(
            {
                "run_name": run_name,
                "run_dir": run.run_dir,
                "checkpoint": run.checkpoint_dir,
                "checkpoint_step": checkpoint_step(run.checkpoint_dir),
                "best_metric": run.best_metric if run.best_metric is not None else "",
                "checkpoint_source": run.source,
                "base_model": cfg.get("base_model_name_or_path"),
                "peft_type": cfg.get("peft_type"),
                "tira_M": cfg.get("tira_M"),
                "tira_K": cfg.get("tira_K"),
                "relative_run_dir": os.path.relpath(run.run_dir, root_dir) if root_dir else run_name,
                "relative_checkpoint": os.path.relpath(run.checkpoint_dir, root_dir)
                if root_dir
                else os.path.basename(os.path.normpath(run.checkpoint_dir)),
            }
        )


def analyze_runs(
    runs: Sequence[RunCheckpoint],
    root_dir: Optional[str],
    tau: float,
    energy_thresholds: Sequence[float],
    top_k_singular: int,
    include_missing: bool,
    apply_scale: bool,
):
    all_rows = []
    manifest = {
        "root_dir": root_dir,
        "tau": tau,
        "energy_thresholds": list(energy_thresholds),
        "apply_scale": apply_scale,
        "runs": [],
        "errors": [],
    }

    for run in runs:
        print(f"analyzing: {run.checkpoint_dir}")
        try:
            rows, cfg = analyze_checkpoint(
                run.checkpoint_dir,
                tau=tau,
                energy_thresholds=energy_thresholds,
                top_k_singular=top_k_singular,
                include_missing=include_missing,
                apply_scale=apply_scale,
            )
            enrich_rows(rows, run, cfg, root_dir)
            all_rows.extend(rows)
            manifest["runs"].append(
                {
                    "run_dir": run.run_dir,
                    "checkpoint": run.checkpoint_dir,
                    "best_metric": run.best_metric,
                    "source": run.source,
                    "base_model": cfg.get("base_model_name_or_path"),
                    "tira_M": cfg.get("tira_M"),
                    "tira_K": cfg.get("tira_K"),
                    "num_rows": len(rows),
                }
            )
        except Exception as exc:
            manifest["errors"].append(
                {
                    "run_dir": run.run_dir,
                    "checkpoint": run.checkpoint_dir,
                    "error": repr(exc),
                }
            )
            print(f"warning: failed to analyze {run.checkpoint_dir}: {exc}")

    return all_rows, summarize_rows(all_rows, energy_thresholds), manifest


def main():
    args = parse_args()
    if args.root_dir is None and args.checkpoint is None:
        raise SystemExit("Please provide --root_dir for batch analysis or --checkpoint for single-checkpoint analysis.")

    energy_thresholds = parse_energy_thresholds(args.energy_thresholds)
    apply_scale = not args.no_scale

    if args.root_dir is not None:
        root_dir = os.path.normpath(args.root_dir)
        output_dir = os.path.normpath(args.output_dir or root_dir)
        runs = discover_run_checkpoints(root_dir)
        if not runs:
            raise SystemExit(f"No TIRA checkpoint runs found under: {root_dir}")
    else:
        checkpoint = os.path.normpath(args.checkpoint)
        if not is_checkpoint_dir(checkpoint):
            raise SystemExit(f"Not a valid TIRA checkpoint directory: {checkpoint}")
        root_dir = None
        output_dir = os.path.normpath(args.output_dir or checkpoint)
        runs = [
            RunCheckpoint(
                run_dir=os.path.dirname(checkpoint),
                checkpoint_dir=checkpoint,
                best_metric=None,
                source="explicit_checkpoint",
            )
        ]

    detail_rows, summary_rows, manifest = analyze_runs(
        runs,
        root_dir=root_dir,
        tau=args.tau,
        energy_thresholds=energy_thresholds,
        top_k_singular=args.top_k_singular,
        include_missing=args.include_missing,
        apply_scale=apply_scale,
    )

    paths = output_paths(output_dir, args.output_prefix, args.tau)
    detail_fields = [
        "run_name",
        "relative_run_dir",
        "relative_checkpoint",
        "checkpoint_step",
        "best_metric",
        "checkpoint_source",
        "base_model",
        "peft_type",
        "tira_M",
        "tira_K",
        "layer",
        "module",
        "K",
        "M",
        "rank_upper_bound",
        "effective_rank_tau",
        "stable_rank",
        "spectral_norm",
        "frobenius_norm",
        "frobenius_sq",
        "nuclear_norm",
        "tau",
        "note",
    ]
    summary_fields = [
        "run_name",
        "module",
        "num_layers",
        "checkpoint",
        "best_metric",
        "base_model",
        "tira_M",
        "tira_K",
        "mean_effective_rank_tau",
        "median_effective_rank_tau",
        "max_effective_rank_tau",
        "mean_stable_rank",
        "median_stable_rank",
        "mean_frobenius_norm",
    ]

    write_csv(detail_rows, paths["detail_csv"], detail_fields)
    write_csv(summary_rows, paths["summary_csv"], summary_fields)
    with open(paths["manifest_json"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"runs_found: {len(runs)}")
    print(f"rows_written: {len(detail_rows)}")
    print(f"summary_rows_written: {len(summary_rows)}")
    print(f"saved_detail_csv: {paths['detail_csv']}")
    print(f"saved_summary_csv: {paths['summary_csv']}")
    print(f"saved_manifest_json: {paths['manifest_json']}")
    if manifest["errors"]:
        print(f"warnings: {len(manifest['errors'])} run(s) failed; see manifest for details")


if __name__ == "__main__":
    main()
