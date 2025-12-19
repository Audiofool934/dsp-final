from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history CSV")
    parser.add_argument("--history", type=str, required=True, help="Path to history CSV")
    parser.add_argument("--output", type=str, default="outputs/plots/history.png")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def plot_history(df: pd.DataFrame, output: Path, title: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if "train_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    if "test_loss" in df.columns:
        axes[0].plot(df["epoch"], df["test_loss"], label="test_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "train_acc" in df.columns:
        axes[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    if "test_acc" in df.columns:
        axes[1].plot(df["epoch"], df["test_acc"], label="test_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.history)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))
    output = Path(args.output)
    title = args.title or Path(args.history).stem
    plot_history(df, output, title)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
