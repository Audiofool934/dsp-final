from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.esc50 import Esc50Item, Esc50FeatureDataset, Esc50TorchDataset
from src.dsp.mfcc import MfccConfig
from src.models.ast import extract_embeddings as extract_ast_embeddings
from src.models.ast import load_ast
from src.models.clap import extract_audio_embeddings, load_clap_audio_model
from src.models.cnn import SimpleCnn

from src.features.cache import FeatureCache, get_or_compute_embedding
from src.models.panns import extract_embeddings as extract_panns_embeddings
from src.models.panns import load_panns
from src.train.transforms import LogMelTransform
from src.utils.metrics import accuracy


@dataclass
class RetrievalResult:
    k: int
    precision: float


def _extract_embeddings(
    model: SimpleCnn,
    dataset: Esc50TorchDataset,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    outputs = []
    with torch.no_grad():
        for feats, _ in loader:
            feats = feats.to(device)
            x = model.features(feats)
            x = x.view(x.size(0), -1)
            outputs.append(x.cpu().numpy())
    return np.concatenate(outputs, axis=0)


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


def build_datasets(
    db_items: List[Esc50Item],
    query_items: List[Esc50Item],
    cfg: MfccConfig,
    feature_cache=None,
):
    if feature_cache is None:
        transform = LogMelTransform(cfg)
        db_ds = Esc50TorchDataset(db_items, cfg.sample_rate, transform=transform)
        query_ds = Esc50TorchDataset(query_items, cfg.sample_rate, transform=transform)
    else:
        def to_tensor(feat):
            return torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)

        db_ds = Esc50FeatureDataset(db_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)
        query_ds = Esc50FeatureDataset(query_items, feature_cache, "log_mel", cfg, postprocess=to_tensor)
    return db_ds, query_ds


def run_model_retrieval(
    model_path: str,
    db_items: List[Esc50Item],
    query_items: List[Esc50Item],
    cfg: MfccConfig,
    device: torch.device,
    batch_size: int = 32,
    feature_cache=None,
    model_type: str = "cnn",
    model_id: str | None = None,
):
    model_type = model_type.lower()
    if model_type == "cnn":
        db_ds, query_ds = build_datasets(db_items, query_items, cfg, feature_cache=feature_cache)
        model = SimpleCnn(n_classes=50)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        db_embeddings = _extract_embeddings(model, db_ds, device, batch_size=batch_size)
        query_embeddings = _extract_embeddings(model, query_ds, device, batch_size=batch_size)
    elif model_type == "panns":
        model = load_panns(device)
        cache = feature_cache or FeatureCache()
        cfg = {
            "model_type": "panns",
            "sample_rate": 32000,
        }

        def compute(item):
            emb = extract_panns_embeddings(model, [item], device, batch_size=1)
            return emb[0]

        db_embeddings = np.stack(
            [get_or_compute_embedding(cache, item, "embedding_panns", cfg, compute) for item in db_items],
            axis=0,
        )
        query_embeddings = np.stack(
            [get_or_compute_embedding(cache, item, "embedding_panns", cfg, compute) for item in query_items],
            axis=0,
        )
    elif model_type == "ast":
        model_id = model_id or "MIT/ast-finetuned-audioset-10-10-0.4593"
        model, feature_extractor = load_ast(model_id, device)
        cache = feature_cache or FeatureCache()
        cfg = {
            "model_type": "ast",
            "model_id": model_id,
            "sample_rate": feature_extractor.sampling_rate,
        }

        def compute(item):
            emb = extract_ast_embeddings(model, feature_extractor, [item], device, batch_size=1)
            return emb[0]

        db_embeddings = np.stack(
            [get_or_compute_embedding(cache, item, "embedding_ast", cfg, compute) for item in db_items],
            axis=0,
        )
        query_embeddings = np.stack(
            [get_or_compute_embedding(cache, item, "embedding_ast", cfg, compute) for item in query_items],
            axis=0,
        )
    elif model_type == "clap":
        model_id = model_id or "laion/clap-htsat-unfused"
        model, processor = load_clap_audio_model(model_id, device)
        cache = feature_cache or FeatureCache()
        cfg = {
            "model_type": "clap",
            "model_id": model_id,
            "sample_rate": processor.feature_extractor.sampling_rate,
        }

        def compute(item):
            emb = extract_audio_embeddings(model, processor, [item], device, batch_size=1)
            return emb[0]

        db_embeddings = np.stack(
            [get_or_compute_embedding(cache, item, "embedding_clap", cfg, compute) for item in db_items],
            axis=0,
        )
        query_embeddings = np.stack(
            [get_or_compute_embedding(cache, item, "embedding_clap", cfg, compute) for item in query_items],
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    results = evaluate_retrieval(db_items, query_items, db_embeddings, query_embeddings, k_list=[10, 20])
    return results
