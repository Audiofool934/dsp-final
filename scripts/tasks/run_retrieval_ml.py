from __future__ import annotations

import argparse

import torch

from src.datasets.esc50 import Esc50Meta
from src.dsp.mfcc import MfccConfig
from src.features.cache import FeatureCache
from src.tasks.retrieval import run_ml_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-based retrieval on ESC-50")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--model", type=str, default="outputs/models/cnn.pt")
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn",
        choices=["cnn", "panns", "ast", "clap"],
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model id for ast/clap",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MfccConfig(
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    meta = Esc50Meta(args.data_root)
    feature_cache = FeatureCache(args.feature_cache)
    results = run_ml_retrieval(
        meta,
        cfg=cfg,
        model_path=args.model,
        device=torch.device(args.device),
        feature_cache=feature_cache,
        batch_size=args.batch_size,
        model_type=args.model_type,
        model_id=args.model_id,
    )
    for res in results:
        print(f"top{res.k} precision: {res.precision:.4f}")


if __name__ == "__main__":
    main()
