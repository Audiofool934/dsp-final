from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.esc50 import Esc50Meta, Esc50FeatureDataset
from src.dsp.mfcc import MfccConfig
from src.features.cache import FeatureCache
from src.models.resnet import ResNetAudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN inference on ESC-50 fold 5")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/cnn.pt")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="outputs/predictions/cnn_fold5.csv")
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

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
    query_items = meta.by_folds([5])
    query_ds = Esc50FeatureDataset(query_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)
    loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ResNetAudio(n_classes=50).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    correct = 0
    total = 0
    rows = []
    with torch.no_grad():
        offset = 0
        for feats, targets in loader:
            feats = feats.to(device)
            targets = targets.to(device)
            logits = model(feats)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            for i in range(preds.size(0)):
                item = query_items[offset + i]
                rows.append(
                    {
                        "filename": item.filename,
                        "target": int(targets[i].item()),
                        "predicted_target": int(preds[i].item()),
                    }
                )
            offset += preds.size(0)

    acc = correct / max(1, total)
    print(f"CNN fold5 accuracy: {acc:.4f} ({correct}/{total})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "target", "predicted_target"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
