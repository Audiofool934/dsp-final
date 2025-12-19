from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from src.datasets.esc50 import Esc50Item
from src.dsp.mfcc import MfccConfig, mfcc
from src.utils.audio import load_audio, normalize_audio


@dataclass
class RetrievalResult:
    k: int
    precision: float


def _mfcc_embedding(signal: np.ndarray, cfg: MfccConfig) -> np.ndarray:
    feats = mfcc(signal, cfg)
    mean = np.mean(feats, axis=0)
    std = np.std(feats, axis=0)
    return np.concatenate([mean, std], axis=0)


def compute_embeddings(
    items: List[Esc50Item],
    cfg: MfccConfig,
    feature_cache=None,
) -> np.ndarray:
    embeddings = []
    for item in items:
        if feature_cache is None:
            audio, _ = load_audio(item.path, target_sr=cfg.sample_rate)
            audio = normalize_audio(audio)
            emb = _mfcc_embedding(audio, cfg)
        else:
            feats = feature_cache.get_feature(item, "mfcc", cfg)
            mean = np.mean(feats, axis=0)
            std = np.std(feats, axis=0)
            emb = np.concatenate([mean, std], axis=0)
        embeddings.append(emb)
    return np.stack(embeddings, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def evaluate_retrieval(
    db_items: List[Esc50Item],
    query_items: List[Esc50Item],
    db_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k_list: Iterable[int],
) -> List[RetrievalResult]:
    sims = cosine_similarity(query_embeddings, db_embeddings)
    targets_db = np.array([item.target for item in db_items])
    targets_query = np.array([item.target for item in query_items])

    results = []
    for k in k_list:
        topk_idx = np.argsort(-sims, axis=1)[:, :k]
        hits = 0
        for i in range(len(query_items)):
            if np.any(targets_db[topk_idx[i]] == targets_query[i]):
                hits += 1
        precision = hits / len(query_items)
        results.append(RetrievalResult(k=k, precision=precision))
    return results
