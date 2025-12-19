from __future__ import annotations

import csv
from pathlib import Path

from src.datasets.esc50 import Esc50Meta


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
