from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.datasets.esc50 import Esc50Meta, Esc50FeatureDataset, get_fold_splits
from src.dsp.mfcc import MfccConfig
from src.features.cache import FeatureCache
from src.models.cnn import SimpleCnn
from src.tasks.classification import build_dataloaders, train_supervised_classifier
from src.utils.history import write_history_csv
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESC-50 classifier")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="outputs/models/cnn.pt")
    parser.add_argument("--history", type=str, default="outputs/history/train_cnn.csv")
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = MfccConfig(
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    feature_cache = FeatureCache(args.feature_cache)
    def to_tensor(feat):
        return torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)

    meta = Esc50Meta(args.data_root)
    train_items, test_items = get_fold_splits(meta)

    train_ds = Esc50FeatureDataset(train_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)
    test_ds = Esc50FeatureDataset(test_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)

    train_loader, test_loader = build_dataloaders(train_ds, test_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = SimpleCnn(n_classes=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_acc, best_state = train_supervised_classifier(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=args.epochs,
        optimizer=optimizer,
    )
    for row in history:
        print(
            f"Epoch {row['epoch']:02d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
            f"test loss {row['test_loss']:.4f} acc {row['test_acc']:.4f}"
        )
    print(f"Best test accuracy: {best_acc:.4f}")
    if best_state:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "config": vars(args)}, args.output)
    write_history_csv(history, args.history, ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])


if __name__ == "__main__":
    main()
