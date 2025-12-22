from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.datasets.esc50 import Esc50Meta, Esc50FeatureDataset, get_fold_splits
from src.dsp.mfcc import MfccConfig
from src.features.cache import FeatureCache
from src.models.resnet import ResNetAudio
from src.tasks.classification import build_dataloaders, train_supervised_classifier
from src.utils.history import write_history_csv
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for classification settings")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-lengths", type=int, nargs="+", default=[1024])
    parser.add_argument("--hop-lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="outputs/classification_grid.csv")
    parser.add_argument("--history-dir", type=str, default="outputs/history/classification_grid")
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    return parser.parse_args()


def train_eval(train_ds, test_ds, device, epochs, batch_size, lr, num_workers):
    train_loader, test_loader = build_dataloaders(
        train_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )
    model = ResNetAudio(n_classes=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_acc, _ = train_supervised_classifier(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=epochs,
        optimizer=optimizer,
    )
    return best_acc, history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    meta = Esc50Meta(args.data_root)
    train_items, test_items = get_fold_splits(meta)
    feature_cache = FeatureCache(args.feature_cache)
    def to_tensor(feat):
        return torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)

    results = []
    device = torch.device(args.device)
    for frame_length in args.frame_lengths:
        for hop_length in args.hop_lengths:
            cfg = MfccConfig(
                sample_rate=args.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
                n_mels=args.n_mels,
            )
            train_ds = Esc50FeatureDataset(train_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)
            test_ds = Esc50FeatureDataset(test_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)
            best_acc, history = train_eval(
                train_ds,
                test_ds,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
            )
            results.append(
                {
                    "frame_length": frame_length,
                    "hop_length": hop_length,
                    "best_acc": best_acc,
                }
            )
            print(f"frame {frame_length} hop {hop_length} best_acc {best_acc:.4f}")
            if args.history_dir:
                history_path = Path(args.history_dir) / f"frame{frame_length}_hop{hop_length}.csv"
                write_history_csv(
                    history,
                    history_path,
                    ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_history_csv(results, args.output, ["frame_length", "hop_length", "best_acc"])


if __name__ == "__main__":
    main()
