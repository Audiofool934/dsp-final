from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.datasets.esc50 import Esc50Meta
from src.features.cache import FeatureCache
from src.models.clap import extract_audio_embeddings, load_clap_audio_model
from src.tasks.classification import train_linear_classifier
from src.utils.history import write_history_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning with CLAP embeddings")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--history", type=str, default="outputs/history/clap_transfer.csv")
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    parser.add_argument("--output", type=str, default="outputs/models/clap_transfer.pt")
    return parser.parse_args()


def get_embeddings(
    model,
    processor,
    items,
    device: torch.device,
    batch_size: int,
    cache: FeatureCache,
    cfg: dict,
) -> np.ndarray:
    embeddings = [None] * len(items)
    missing_items = []
    missing_idx = []
    for i, item in enumerate(items):
        cached = cache.load_feature(item, "embedding_clap", cfg)
        if cached is not None:
            embeddings[i] = cached
        else:
            missing_items.append(item)
            missing_idx.append(i)

    if missing_items:
        for i in range(0, len(missing_items), batch_size):
            batch = missing_items[i : i + batch_size]
            batch_emb = extract_audio_embeddings(model, processor, batch, device, batch_size=batch_size)
            for j, item in enumerate(batch):
                vec = batch_emb[j]
                path = cache.feature_path(item, "embedding_clap", cfg)
                cache.save_feature(path, vec)
                embeddings[missing_idx[i + j]] = vec

    return np.stack(embeddings, axis=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, processor = load_clap_audio_model(args.model, device)

    meta = Esc50Meta(args.data_root)
    train_items = meta.by_folds([1, 2, 3, 4])
    test_items = meta.by_folds([5])

    cache = FeatureCache(args.feature_cache)
    cfg = {
        "model_type": "clap",
        "model_id": args.model,
        "sample_rate": processor.feature_extractor.sampling_rate,
    }

    train_x = get_embeddings(model, processor, train_items, device, args.batch_size, cache, cfg)
    test_x = get_embeddings(model, processor, test_items, device, args.batch_size, cache, cfg)

    train_y = np.array([item.target for item in train_items], dtype=np.int64)
    test_y = np.array([item.target for item in test_items], dtype=np.int64)

    history, acc, best_state = train_linear_classifier(
        train_x,
        train_y,
        test_x,
        test_y,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    print(f"CLAP transfer accuracy (fold 5): {acc:.4f}")
    write_history_csv(history, args.history, ["epoch", "train_loss", "train_acc", "test_acc"])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "config": vars(args),
            },
            args.output,
        )


if __name__ == "__main__":
    main()
