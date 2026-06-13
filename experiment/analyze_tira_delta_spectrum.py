import argparse
import csv
import json
import os
import re

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is required. Please run in your project env, e.g. `conda activate <env>` then rerun."
    ) from e


MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def load_checkpoint(checkpoint_dir: str):
    model_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find adapter weights: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find adapter config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    state_dict = torch.load(model_path, map_location="cpu")
    return state_dict, cfg


def extract_layer_module(key: str):
    layer_match = re.search(r"\.layers\.(\d+)\.", key)
    module_match = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj)\.", key)
    if layer_match is None or module_match is None:
        return None
    return int(layer_match.group(1)), module_match.group(1)


def build_delta_from_tira(a: torch.Tensor, b: torch.Tensor):
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
    return delta


def analyze(checkpoint_dir: str, tau: float):
    state_dict, cfg = load_checkpoint(checkpoint_dir)

    grouped = {}
    for k, v in state_dict.items():
        parsed = extract_layer_module(k)
        if parsed is None:
            continue
        layer_id, module_name = parsed
        grouped.setdefault((layer_id, module_name), {})
        if k.endswith(".tira_a"):
            grouped[(layer_id, module_name)]["a"] = v
        elif k.endswith(".tira_b"):
            grouped[(layer_id, module_name)]["b"] = v

    if not grouped:
        raise ValueError("No TIRA layer/module weights were found in adapter_model.bin")

    layer_ids = sorted({x[0] for x in grouped.keys()})
    rows = []

    for layer_id in layer_ids:
        for module_name in MODULES:
            item = grouped.get((layer_id, module_name), None)
            if item is None or "a" not in item or "b" not in item:
                rows.append({
                    "layer": layer_id,
                    "module": module_name,
                    "effective_rank": 0,
                    "frobenius_sq": 0.0,
                    "tau": tau,
                    "note": "missing_in_checkpoint",
                })
                continue

            a = item["a"]
            b = item["b"]
            delta = build_delta_from_tira(a, b)
            s = torch.linalg.svdvals(delta)
            eff_rank = int((s > tau).sum().item())
            fro_sq = float((s * s).sum().item())

            rows.append({
                "layer": layer_id,
                "module": module_name,
                "effective_rank": eff_rank,
                "frobenius_sq": fro_sq,
                "tau": tau,
                "note": "ok",
            })

    return rows, cfg


def print_table(rows):
    print("layer,module,effective_rank,frobenius_sq,tau,note")
    for r in rows:
        print(
            f"{r['layer']},{r['module']},{r['effective_rank']},"
            f"{r['frobenius_sq']:.8f},{r['tau']},{r['note']}"
        )


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "module", "effective_rank", "frobenius_sq", "tau", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows, cfg = analyze(args.checkpoint, args.tau)

    print(f"checkpoint: {args.checkpoint}")
    print(f"base_model: {cfg.get('base_model_name_or_path')}")
    print(f"tira_M={cfg.get('tira_M')}, tira_K={cfg.get('tira_K')}")
    print_table(rows)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(args.checkpoint, f"tira_spectrum_tau_{args.tau}.csv")
    write_csv(rows, output_csv)
    print(f"saved_csv: {output_csv}")


if __name__ == "__main__":
    main()
