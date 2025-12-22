from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.datasets.esc50 import Esc50Meta

_STOPWORDS = {"a", "an", "the", "of", "sound", "sounds", "audio"}


@dataclass(frozen=True)
class LabelLookup:
    display_labels: list[str]
    display_to_category: dict[str, str]
    category_to_target: dict[str, int]
    target_to_display: dict[int, str]
    normalized_to_display: dict[str, str]
    collapsed_to_display: dict[str, str]


def normalize_label_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    tokens = [token for token in text.split() if token not in _STOPWORDS]
    return " ".join(tokens)


def build_label_lookup(meta: Esc50Meta) -> LabelLookup:
    category_to_target: dict[str, int] = {}
    target_to_category: dict[int, str] = {}
    for item in meta.items:
        category_to_target.setdefault(item.category, item.target)
        target_to_category.setdefault(item.target, item.category)

    targets = sorted(target_to_category.keys())
    display_labels = [target_to_category[t].replace("_", " ") for t in targets]
    display_to_category = {c.replace("_", " "): c for c in target_to_category.values()}
    target_to_display = {t: target_to_category[t].replace("_", " ") for t in targets}

    normalized_to_display: dict[str, str] = {}
    collapsed_to_display: dict[str, str] = {}
    for display in display_labels:
        normalized = normalize_label_text(display)
        if normalized:
            normalized_to_display[normalized] = display
            collapsed_to_display[normalized.replace(" ", "")] = display

    return LabelLookup(
        display_labels=display_labels,
        display_to_category=display_to_category,
        category_to_target=category_to_target,
        target_to_display=target_to_display,
        normalized_to_display=normalized_to_display,
        collapsed_to_display=collapsed_to_display,
    )


def match_label_text(label_text: str, lookup: LabelLookup) -> str | None:
    if not label_text:
        return None
    normalized = normalize_label_text(label_text)
    if not normalized:
        return None
    if normalized in lookup.normalized_to_display:
        return lookup.normalized_to_display[normalized]
    collapsed = normalized.replace(" ", "")
    if collapsed in lookup.collapsed_to_display:
        return lookup.collapsed_to_display[collapsed]

    label_tokens = set(normalized.split())
    best_label = None
    best_score = 0.0
    for norm_label, display in lookup.normalized_to_display.items():
        tokens = set(norm_label.split())
        if not tokens:
            continue
        score = len(label_tokens & tokens) / len(tokens)
        if score > best_score:
            best_score = score
            best_label = display
    if best_score >= 0.8:
        return best_label
    return None


def _iter_json_payloads(text: str):
    if not text:
        return
    fence_re = re.compile(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", re.DOTALL | re.IGNORECASE)
    for block in fence_re.findall(text):
        yield block
    for block in re.findall(r"\\{[^{}]*\\}", text):
        yield block


def extract_label_from_response(response: str, lookup: LabelLookup) -> str | None:
    for block in _iter_json_payloads(response):
        try:
            payload = json.loads(block)
        except Exception:
            continue
        if isinstance(payload, dict):
            if "label_id" in payload:
                try:
                    label_id = int(payload["label_id"])
                except (TypeError, ValueError):
                    label_id = None
                if label_id is not None and label_id in lookup.target_to_display:
                    return lookup.target_to_display[label_id]
            if "label" in payload:
                label = match_label_text(str(payload["label"]), lookup)
                if label:
                    return label

    match = re.search(r"label_id\\s*[:=]\\s*(\\d+)", response, re.IGNORECASE)
    if match:
        label_id = int(match.group(1))
        if label_id in lookup.target_to_display:
            return lookup.target_to_display[label_id]

    match = re.search(r"label\\s*[:=]\\s*[\"']?([^\\n\\r\"'}]+)", response, re.IGNORECASE)
    if match:
        label = match_label_text(match.group(1).strip(), lookup)
        if label:
            return label

    normalized_response = normalize_label_text(response)
    matches = [
        display
        for norm_label, display in lookup.normalized_to_display.items()
        if norm_label and norm_label in normalized_response
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        response_lower = response.lower()
        best_label = None
        best_pos = -1
        for display in matches:
            pos = response_lower.rfind(display.lower())
            if pos > best_pos:
                best_pos = pos
                best_label = display
        return best_label

    return None


def evaluate_llm_predictions(meta: Esc50Meta, predictions_path: str | Path) -> tuple[float, int, int]:
    fold5 = {item.filename: item.target for item in meta.by_folds([5])}
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(
            "Missing predictions CSV. Create outputs/llm_predictions.csv with columns: filename,predicted_target"
        )

    correct = 0
    total = 0
    with pred_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            pred = int(row["predicted_target"])
            if filename not in fold5:
                continue
            total += 1
            if pred == fold5[filename]:
                correct += 1

    if total == 0:
        raise ValueError("No matching predictions for fold 5")

    acc = correct / total
    return acc, correct, total
