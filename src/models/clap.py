from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor, pipeline
from tqdm import tqdm

from src.datasets.esc50 import Esc50Item, Esc50Meta
from src.utils.audio import load_audio


def build_candidate_labels(meta: Esc50Meta) -> list[str]:
    categories = sorted({item.category for item in meta.items})
    return [f"Sound of {c.replace('_', ' ')}" for c in categories]


def build_category_to_target(meta: Esc50Meta) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in meta.items:
        if item.category not in mapping:
            mapping[item.category] = item.target
    return mapping


def run_zero_shot(
    items: Iterable[Esc50Item],
    meta: Esc50Meta,
    model_id: str,
    sample_rate: int,
    batch_size: int = 1,
    progress: bool = False,
):
    labels = build_candidate_labels(meta)
    category_to_target = build_category_to_target(meta)
    clf = pipeline(task="zero-shot-audio-classification", model=model_id)

    results = []
    correct = 0
    items = list(items)
    if batch_size < 1:
        batch_size = 1
    indices = range(0, len(items), batch_size)
    iterator = tqdm(indices, desc="clap zeroshot", total=(len(items) + batch_size - 1) // batch_size) if progress else indices
    for start in iterator:
        batch = items[start : start + batch_size]
        audios = []
        for item in batch:
            audio, _ = load_audio(item.path, target_sr=sample_rate)
            audios.append(np.asarray(audio, dtype=np.float32))
        preds = clf(audios, sampling_rate=sample_rate, candidate_labels=labels)
        if preds and isinstance(preds[0], dict):
            preds = [preds]
        for item, pred in zip(batch, preds):
            top = pred[0]
            pred_label = top["label"].replace("Sound of ", "").replace(" ", "_")
            pred_target = category_to_target[pred_label]
            if pred_target == item.target:
                correct += 1
            results.append(
                {
                    "filename": item.filename,
                    "target": item.target,
                    "predicted_target": pred_target,
                    "predicted_label": top["label"],
                    "score": top["score"],
                }
            )
    total = len(items)
    acc = correct / max(1, total)
    return results, acc, correct, total


def load_clap_audio_model(model_id: str, device: torch.device):
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def extract_audio_embeddings(
    model,
    processor,
    items: Iterable[Esc50Item],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    items = list(items)
    embeddings: list[np.ndarray] = []
    sample_rate = processor.feature_extractor.sampling_rate
    if batch_size < 1:
        batch_size = 1
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        audios = []
        for item in batch:
            audio, _ = load_audio(item.path, target_sr=sample_rate)
            audios.append(np.asarray(audio, dtype=np.float32))
        inputs = processor(audios=audios, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_features = model.get_audio_features(**inputs)
        embeddings.append(audio_features.cpu().numpy())
    return np.concatenate(embeddings, axis=0)
