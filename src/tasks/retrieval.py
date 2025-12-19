from __future__ import annotations

from typing import Iterable

from src.datasets.esc50 import Esc50Meta
from src.dsp.mfcc import MfccConfig
from src.retrieval.retrieval import compute_embeddings, evaluate_retrieval
from src.retrieval.retrieval_ml import run_model_retrieval


def run_mfcc_retrieval(
    meta: Esc50Meta,
    cfg: MfccConfig,
    k_list: Iterable[int] = (10, 20),
    feature_cache=None,
):
    db_items = meta.by_folds([1, 2, 3, 4])
    query_items = meta.by_folds([5])
    db_embeddings = compute_embeddings(db_items, cfg, feature_cache=feature_cache)
    query_embeddings = compute_embeddings(query_items, cfg, feature_cache=feature_cache)
    return evaluate_retrieval(db_items, query_items, db_embeddings, query_embeddings, k_list=k_list)


def run_ml_retrieval(
    meta: Esc50Meta,
    cfg: MfccConfig,
    model_path: str,
    device,
    batch_size: int = 32,
    feature_cache=None,
    model_type: str = "cnn",
    model_id: str | None = None,
):
    db_items = meta.by_folds([1, 2, 3, 4])
    query_items = meta.by_folds([5])
    return run_model_retrieval(
        model_path=model_path,
        db_items=db_items,
        query_items=query_items,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        feature_cache=feature_cache,
        model_type=model_type,
        model_id=model_id,
    )
