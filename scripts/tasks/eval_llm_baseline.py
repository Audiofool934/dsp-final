from __future__ import annotations

import argparse
from src.datasets.esc50 import Esc50Meta
from src.tasks.llm_baseline import evaluate_llm_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM baseline predictions")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/llm_predictions.csv",
        help="CSV with columns: filename,predicted_target",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = Esc50Meta(args.data_root)
    acc, correct, total = evaluate_llm_predictions(meta, args.predictions)
    print(f"LLM baseline accuracy (fold 5): {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
