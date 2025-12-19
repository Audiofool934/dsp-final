from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from src.dsp.mfcc import MfccConfig, log_mel_spectrogram, mfcc
from src.utils.audio import load_audio, normalize_audio


class FeatureCache:
    def __init__(self, root: str | Path = "outputs/features", enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled

    def _cfg_dict(self, cfg: MfccConfig) -> dict:
        if is_dataclass(cfg):
            params = asdict(cfg)
        else:
            params = dict(cfg)
        params["n_fft"] = cfg.n_fft or cfg.frame_length
        if params.get("f_max") is None:
            params["f_max"] = cfg.sample_rate / 2
        return params

    def params_hash(self, feature_type: str, cfg: MfccConfig | dict) -> tuple[str, dict]:
        if isinstance(cfg, MfccConfig):
            params = {"feature_type": feature_type, **self._cfg_dict(cfg)}
        else:
            params = {"feature_type": feature_type, **dict(cfg)}
        payload = json.dumps(params, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return digest, params

    def feature_dir(self, feature_type: str, cfg: MfccConfig | dict) -> Path:
        digest, _ = self.params_hash(feature_type, cfg)
        return self.root / feature_type / digest

    def feature_path(self, item, feature_type: str, cfg: MfccConfig | dict) -> Path:
        base = self.feature_dir(feature_type, cfg)
        return base / f"fold{item.fold}" / f"{item.filename}.npy"

    def load_feature(self, item, feature_type: str, cfg: MfccConfig | dict) -> np.ndarray | None:
        if not self.enabled:
            return None
        path = self.feature_path(item, feature_type, cfg)
        if path.exists():
            return np.load(path)
        return None

    def compute_feature(self, item, feature_type: str, cfg: MfccConfig) -> np.ndarray:
        audio, _ = load_audio(item.path, target_sr=cfg.sample_rate)
        audio = normalize_audio(audio)
        if feature_type == "log_mel":
            feat = log_mel_spectrogram(audio, cfg)
        elif feature_type == "mfcc":
            feat = mfcc(audio, cfg)
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
        return feat.astype(np.float32)

    def get_feature(self, item, feature_type: str, cfg: MfccConfig | dict) -> np.ndarray:
        cached = self.load_feature(item, feature_type, cfg)
        if cached is not None:
            return cached
        if isinstance(cfg, MfccConfig):
            feat = self.compute_feature(item, feature_type, cfg)
        else:
            raise ValueError("get_feature requires MfccConfig for mfcc/log_mel")
        if self.enabled:
            path = self.feature_path(item, feature_type, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, feat)
        return feat

    def write_manifest(self, feature_type: str, cfg: MfccConfig | dict, records: Iterable[dict]) -> Path:
        digest, params = self.params_hash(feature_type, cfg)
        records = list(records)
        manifest = {
            "feature_type": feature_type,
            "hash": digest,
            "params": params,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "num_files": len(records),
        }
        manifest["files"] = records
        out_dir = self.feature_dir(feature_type, cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        return path

    def precompute(
        self,
        items,
        feature_type: str,
        cfg: MfccConfig,
        progress: bool = False,
        desc: str | None = None,
    ) -> Path:
        items = list(items)
        records = []
        iterator = items
        if progress:
            iterator = tqdm(items, desc=desc or f"precompute {feature_type}", total=len(items))
        for item in iterator:
            feat = self.get_feature(item, feature_type, cfg)
            path = self.feature_path(item, feature_type, cfg)
            records.append(
                {
                    "filename": item.filename,
                    "fold": item.fold,
                    "path": str(path),
                    "shape": list(feat.shape),
                }
            )
        return self.write_manifest(feature_type, cfg, records)


def get_or_compute_embedding(
    cache: FeatureCache,
    item,
    feature_type: str,
    cfg: dict,
    compute_fn,
) -> np.ndarray:
    cached = cache.load_feature(item, feature_type, cfg)
    if cached is not None:
        return cached
    feat = compute_fn(item)
    path = cache.feature_path(item, feature_type, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feat.astype(np.float32))
    return feat.astype(np.float32)
