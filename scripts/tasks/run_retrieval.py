from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from src.datasets.esc50 import Esc50Meta
from src.dsp.mfcc import MfccConfig
from src.features.cache import FeatureCache
from src.tasks.retrieval import run_mfcc_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MFCC-based retrieval on ESC-50")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-lengths", type=int, nargs="+", default=[1024])
    parser.add_argument("--hop-lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--output", type=str, default="outputs/retrieval_mfcc.csv")
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    return parser.parse_args()


def run_setting(
    meta: Esc50Meta,
    frame_length: int,
    hop_length: int,
    sample_rate: int,
    n_mels: int,
    n_mfcc: int,
    feature_cache,
):
    cfg = MfccConfig(
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    return run_mfcc_retrieval(meta, cfg, k_list=[10, 20], feature_cache=feature_cache)


def main() -> None:
    args = parse_args()
    meta = Esc50Meta(args.data_root)
    feature_cache = FeatureCache(args.feature_cache)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for frame_length in args.frame_lengths:
        for hop_length in args.hop_lengths:
            results = run_setting(
                meta,
                frame_length=frame_length,
                hop_length=hop_length,
                sample_rate=args.sample_rate,
                n_mels=args.n_mels,
                n_mfcc=args.n_mfcc,
                feature_cache=feature_cache,
            )
            row = {
                "frame_length": frame_length,
                "hop_length": hop_length,
            }
            for res in results:
                row[f"top{res.k}"] = res.precision
            rows.append(row)
            print(f"frame {frame_length} hop {hop_length} -> top10 {row['top10']:.4f} top20 {row['top20']:.4f}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_length", "hop_length", "top10", "top20"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
