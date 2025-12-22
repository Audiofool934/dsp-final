from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.datasets.esc50 import Esc50Meta
from src.features.cache import FeatureCache
from src.models.ast import extract_embeddings as extract_ast_embeddings
from src.models.ast import load_ast
from src.models.clap import extract_audio_embeddings, load_clap_audio_model
from src.models.panns import extract_embeddings as extract_panns_embeddings
from src.models.panns import load_panns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer fold-5 predictions for transfer models.")
    parser.add_argument("--model-type", type=str, required=True, choices=["panns", "ast", "clap"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--model", type=str, default=None, help="AST/CLAP model id override")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-cache", type=str, default="outputs/features")
    parser.add_argument("--output", type=str, default="outputs/predictions/transfer_fold5.csv")
    return parser.parse_args()


def load_linear_head(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    if "weight" not in state:
        raise ValueError(f"Checkpoint missing linear head weights: {checkpoint_path}")
    in_dim = state["weight"].shape[1]
    model = nn.Linear(in_dim, 50)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_checkpoint_model_id(checkpoint_path: str) -> str | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    return cfg.get("model")


def get_embeddings_with_cache(
    items,
    cache: FeatureCache,
    feature_type: str,
    cfg: dict,
    compute_fn,
    batch_size: int,
) -> np.ndarray:
    embeddings: list[np.ndarray | None] = [None] * len(items)
    missing_items = []
    missing_idx = []
    for i, item in enumerate(items):
        cached = cache.load_feature(item, feature_type, cfg)
        if cached is not None:
            embeddings[i] = cached
        else:
            missing_items.append(item)
            missing_idx.append(i)

    if missing_items:
        batch_emb = compute_fn(missing_items, batch_size)
        for j, item in enumerate(missing_items):
            vec = batch_emb[j]
            path = cache.feature_path(item, feature_type, cfg)
            cache.save_feature(path, vec)
            embeddings[missing_idx[j]] = vec

    return np.stack(embeddings, axis=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    meta = Esc50Meta(args.data_root)
    test_items = meta.by_folds([5])
    target_to_label = {item.target: item.category.replace("_", " ") for item in meta.items}

    model_type = args.model_type
    cache = FeatureCache(args.feature_cache)

    if model_type == "panns":
        model = load_panns(device)
        cfg = {"model_type": "panns", "sample_rate": 32000}

        def compute(items, batch_size):
            return extract_panns_embeddings(model, items, device, batch_size=batch_size)

        embeddings = get_embeddings_with_cache(
            test_items, cache, "embedding_panns", cfg, compute, args.batch_size
        )
    elif model_type == "ast":
        model_id = args.model or get_checkpoint_model_id(args.checkpoint) or "MIT/ast-finetuned-audioset-10-10-0.4593"
        model, feature_extractor = load_ast(model_id, device)
        cfg = {
            "model_type": "ast",
            "model_id": model_id,
            "sample_rate": feature_extractor.sampling_rate,
        }

        def compute(items, batch_size):
            return extract_ast_embeddings(model, feature_extractor, items, device, batch_size=batch_size)

        embeddings = get_embeddings_with_cache(test_items, cache, "embedding_ast", cfg, compute, args.batch_size)
    elif model_type == "clap":
        model_id = args.model or get_checkpoint_model_id(args.checkpoint) or "laion/clap-htsat-unfused"
        model, processor = load_clap_audio_model(model_id, device)
        cfg = {
            "model_type": "clap",
            "model_id": model_id,
            "sample_rate": processor.feature_extractor.sampling_rate,
        }

        def compute(items, batch_size):
            return extract_audio_embeddings(model, processor, items, device, batch_size=batch_size)

        embeddings = get_embeddings_with_cache(test_items, cache, "embedding_clap", cfg, compute, args.batch_size)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    linear_head = load_linear_head(args.checkpoint, device)
    with torch.no_grad():
        feats = torch.tensor(embeddings, dtype=torch.float32, device=device)
        logits = linear_head(feats)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    rows = []
    correct = 0
    for item, pred in zip(test_items, preds):
        if int(pred) == item.target:
            correct += 1
        rows.append(
            {
                "filename": item.filename,
                "target": item.target,
                "target_label": target_to_label[item.target],
                "predicted_target": int(pred),
                "predicted_label": target_to_label[int(pred)],
            }
        )

    acc = correct / max(1, len(test_items))
    print(f"{model_type} transfer fold5 accuracy: {acc:.4f} ({correct}/{len(test_items)})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "target", "target_label", "predicted_target", "predicted_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
