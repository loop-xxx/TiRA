import argparse
import json
import os
import random


def split_data(rows, train_ratio, seed):
    random.seed(seed)
    data = rows[:]
    random.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)

    if n > 0 and n_train >= n:
        n_train = n - 1

    train = data[:n_train]
    val = data[n_train:]
    return train, val, val


def save(path, data, indent):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"  {len(data):>5} records -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset into easy/challenge by output length, then train/val/test")
    parser.add_argument("--input_json", default="./gsm8k.json")
    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    lengths = sorted(len(r["output"]) for r in rows)
    n = len(lengths)
    t1 = lengths[n // 2]
    print(f"Total: {n}  |  threshold: easy <= {t1} < challenge")

    buckets = {"easy": [], "challenge": []}
    for r in rows:
        l = len(r["output"])
        if l <= t1:
            buckets["easy"].append(r)
        else:
            buckets["challenge"].append(r)

    indent = args.indent if args.indent > 0 else None

    for split_name in ["easy", "challenge"]:
        bucket = buckets[split_name]
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        train, val, test = split_data(bucket, args.train_ratio, args.seed)
        print(f"{split_name.capitalize()} ({len(bucket)} total):")
        save(os.path.join(split_dir, "train.json"), train, indent)
        save(os.path.join(split_dir, "validation.json"), val, indent)
        save(os.path.join(split_dir, "test.json"), test, indent)


if __name__ == "__main__":
    main()
