from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.datasets.esc50 import Esc50Meta
from src.tasks.llm_baseline import (
    build_label_lookup,
    evaluate_llm_predictions,
    extract_label_from_response,
    match_label_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Gemini predictions and recompute accuracy.")
    parser.add_argument("--input", type=str, default="outputs/llm_predictions.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input CSV")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {input_path}")

    if args.in_place:
        output_path = input_path
    else:
        output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_normalized.csv")

    meta = Esc50Meta(args.data_root)
    lookup = build_label_lookup(meta)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            response = row.get("response", "")
            label = extract_label_from_response(response, lookup)
            if label is None:
                label = match_label_text(row.get("predicted_label", ""), lookup)

            if label is None:
                row["predicted_label"] = "unknown"
                row["predicted_target"] = -1
            else:
                category = lookup.display_to_category[label]
                row["predicted_label"] = label
                row["predicted_target"] = lookup.category_to_target[category]

            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = list(fieldnames)
    for field in ["filename", "predicted_target", "predicted_label", "response"]:
        if field not in output_fields:
            output_fields.append(field)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    acc, correct, total = evaluate_llm_predictions(meta, output_path)
    print(f"Normalized predictions written to {output_path}")
    print(f"LLM baseline accuracy (fold 5): {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
