from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import ASTModel, AutoFeatureExtractor

from src.utils.audio import load_audio, normalize_audio


def load_ast(model_id: str, device: torch.device):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = ASTModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, feature_extractor


def extract_embeddings(
    model,
    feature_extractor,
    items,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    sample_rate = feature_extractor.sampling_rate
    if batch_size < 1:
        batch_size = 1
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        audios = []
        for item in batch:
            audio, _ = load_audio(item.path, target_sr=sample_rate)
            audios.append(normalize_audio(audio).astype(np.float32))
        inputs = feature_extractor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        pooled = output.pooler_output
        if pooled is None:
            pooled = output.last_hidden_state.mean(dim=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)
