from __future__ import annotations

import argparse

import numpy as np
import torch

from src.datasets.esc50 import Esc50Meta
from src.models.panns import extract_embeddings, load_panns
from src.tasks.classification import train_linear_classifier
from src.utils.history import write_history_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning with PANNs embeddings")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--history", type=str, default="outputs/history/panns_transfer.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_panns(device)

    meta = Esc50Meta(args.data_root)
    train_items = meta.by_folds([1, 2, 3, 4])
    test_items = meta.by_folds([5])

    train_x = extract_embeddings(model, train_items, device, args.batch_size)
    test_x = extract_embeddings(model, test_items, device, args.batch_size)

    train_y = np.array([item.target for item in train_items], dtype=np.int64)
    test_y = np.array([item.target for item in test_items], dtype=np.int64)

    history, acc = train_linear_classifier(
        train_x,
        train_y,
        test_x,
        test_y,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    print(f"PANNs transfer accuracy (fold 5): {acc:.4f}")
    write_history_csv(history, args.history, ["epoch", "train_loss", "train_acc", "test_acc"])


if __name__ == "__main__":
    main()
