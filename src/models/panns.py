from __future__ import annotations

from typing import List

import numpy as np
import torch
from panns_inference import AudioTagging

from src.utils.audio import load_audio, normalize_audio

PANN_SAMPLE_RATE = 32000


def load_panns(device: torch.device):
    device_str = str(device)
    if device_str.startswith("cuda"):
        device_str = "cuda"
    return AudioTagging(device=device_str)


def extract_embeddings(
    model,
    items,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for item in items:
        audio, _ = load_audio(item.path, target_sr=PANN_SAMPLE_RATE)
        audio = normalize_audio(audio).astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        _, embedding = model.inference(audio)
        embedding = np.asarray(embedding)
        if embedding.ndim == 3:
            embedding = embedding[0]
        if embedding.ndim == 2:
            if embedding.shape[0] == 1:
                embedding = embedding[0]
            else:
                embedding = embedding.mean(axis=0)
        if embedding.ndim != 1:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
        embeddings.append(embedding.astype(np.float32))
    return np.stack(embeddings, axis=0)
