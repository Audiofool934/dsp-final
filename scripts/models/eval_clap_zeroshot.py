from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.datasets.esc50 import Esc50Meta
from src.models.clap import run_zero_shot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot classification with CLAP")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default="outputs/clap_zeroshot.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = Esc50Meta(args.data_root)
    query_items = meta.by_folds([5])
    rows, acc, correct, total = run_zero_shot(
        query_items,
        meta,
        model_id=args.model,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        progress=True,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "target", "predicted_target", "predicted_label", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CLAP zero-shot accuracy (fold 5): {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
