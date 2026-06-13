import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gsm8k HuggingFace dataset to JSON format")
    parser.add_argument("--dataset_path", default="./gsm8k", help="Path to dataset saved by save_to_disk()")
    parser.add_argument("--output_json", default="./gsm8k.json", help="Output JSON file path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (0 = compact)")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset_config = (args.dataset_config or "").strip()
    if dataset_config:
        ds = load_dataset(args.dataset_path, dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset_path, split=args.split)

    rows = [
        {
            "instruction": row["question"],
            "output": row["answer"],
        }
        for row in ds
    ]

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    indent = args.indent if args.indent > 0 else None
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=indent)

    print(f"Saved {len(rows)} records to: {args.output_json}")


if __name__ == "__main__":
    main()
