from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from src.datasets.esc50 import Esc50Meta
from src.dsp.mfcc import MfccConfig
from src.features.cache import FeatureCache


@dataclass
class CacheItem:
    filename: str
    fold: int
    path: str


def _compute_record(item: CacheItem, feature_type: str, cfg_params: dict, cache_root: str) -> dict:
    cfg = MfccConfig(**cfg_params)
    cache = FeatureCache(cache_root)
    obj = SimpleNamespace(filename=item.filename, fold=item.fold, path=Path(item.path))
    feat = cache.get_feature(obj, feature_type, cfg)
    path = cache.feature_path(obj, feature_type, cfg)
    return {
        "filename": item.filename,
        "fold": item.fold,
        "path": str(path),
        "shape": list(feat.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DSP features and cache them")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--feature-types", type=str, nargs="+", default=["mfcc", "log_mel"])
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--pre-emphasis", type=float, default=0.97)
    parser.add_argument("--window", type=str, default="hann")
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    parser.add_argument("--workers", type=int, default=0, help="Use process workers for parallel precompute")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = Esc50Meta(args.data_root)
    items = meta.by_folds(args.folds)
    cache_items = [CacheItem(item.filename, item.fold, str(item.path)) for item in items]

    cfg = MfccConfig(
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        f_min=args.f_min,
        f_max=args.f_max,
        pre_emphasis=args.pre_emphasis,
        window=args.window,
    )

    cache = FeatureCache(args.feature_cache)
    cfg_params = {
        "sample_rate": args.sample_rate,
        "frame_length": args.frame_length,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "pre_emphasis": args.pre_emphasis,
        "window": args.window,
    }
    for feature_type in args.feature_types:
        records = []
        if args.workers and args.workers > 0:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(_compute_record, item, feature_type, cfg_params, args.feature_cache)
                    for item in cache_items
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"precompute {feature_type}",
                ):
                    records.append(future.result())
        else:
            for item in tqdm(cache_items, desc=f"precompute {feature_type}", total=len(cache_items)):
                records.append(_compute_record(item, feature_type, cfg_params, args.feature_cache))

        manifest_path = cache.write_manifest(feature_type, cfg, records)
        print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
