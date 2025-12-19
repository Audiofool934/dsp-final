from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from src.utils.audio import load_audio, normalize_audio


@dataclass
class Esc50Item:
    filename: str
    fold: int
    target: int
    category: str
    path: Path


class Esc50Meta:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta_path = self.root / "meta" / "esc50.csv"
        self.audio_dir = self.root / "audio"
        self.items = self._load()

    def _load(self) -> List[Esc50Item]:
        items: List[Esc50Item] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                items.append(
                    Esc50Item(
                        filename=filename,
                        fold=int(row["fold"]),
                        target=int(row["target"]),
                        category=row["category"],
                        path=self.audio_dir / filename,
                    )
                )
        return items

    def by_folds(self, folds: Iterable[int]) -> List[Esc50Item]:
        folds = set(folds)
        return [item for item in self.items if item.fold in folds]


class Esc50TorchDataset:
    def __init__(
        self,
        items: List[Esc50Item],
        sample_rate: int,
        transform=None,
    ):
        self.items = items
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        audio, sr = load_audio(item.path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        if self.transform is not None:
            audio = self.transform(audio, sr)
        return audio, item.target


class Esc50FeatureDataset:
    def __init__(
        self,
        items: List[Esc50Item],
        feature_cache,
        feature_type: str,
        cfg,
        postprocess=None,
    ):
        self.items = items
        self.feature_cache = feature_cache
        self.feature_type = feature_type
        self.cfg = cfg
        self.postprocess = postprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        feat = self.feature_cache.get_feature(item, self.feature_type, self.cfg)
        if self.postprocess is not None:
            feat = self.postprocess(feat)
        return feat, item.target


def get_fold_splits(meta: Esc50Meta):
    train_items = meta.by_folds([1, 2, 3, 4])
    test_items = meta.by_folds([5])
    return train_items, test_items
