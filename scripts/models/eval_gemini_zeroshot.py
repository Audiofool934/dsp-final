from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from src.datasets.esc50 import Esc50Meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot classification with Gemini")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="outputs/llm_predictions.csv")
    parser.add_argument("--resume", action="store_true", help="Skip files already in output CSV")
    return parser.parse_args()


def build_labels(meta: Esc50Meta):
    categories = sorted({item.category for item in meta.items})
    display_labels = [c.replace("_", " ") for c in categories]
    category_to_target = {}
    for item in meta.items:
        if item.category not in category_to_target:
            category_to_target[item.category] = item.target
    display_to_category = {c.replace("_", " "): c for c in categories}
    return categories, display_labels, display_to_category, category_to_target


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("sound of", "")
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return " ".join(text.split())


def extract_label(text: str, display_labels: list[str]) -> str | None:
    try:
        match = re.search(r"\\{.*\\}", text, re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and "label" in payload:
                label = payload["label"].strip().lower()
                for candidate in display_labels:
                    if label == candidate.lower():
                        return candidate
    except Exception:
        pass

    norm = normalize_text(text)
    for candidate in display_labels:
        if candidate in norm:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass --api-key.")

    meta = Esc50Meta(args.data_root)
    items = meta.by_folds([5])
    categories, display_labels, display_to_category, category_to_target = build_labels(meta)

    client = genai.Client(api_key=api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["filename"])

    rows = []
    for idx, item in enumerate(items):
        if args.max_samples and idx >= args.max_samples:
            break
        if item.filename in processed:
            continue

        with open(item.path, "rb") as f:
            audio_bytes = f.read()

        prompt = (
            "Classify this audio into one of the ESC-50 categories. "
            "Return JSON only: {\"label\": \"<one label>\"}. "
            "Valid labels: " + ", ".join(display_labels)
        )

        response = client.models.generate_content(
            model=args.model,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav",
                ),
            ],
        )

        text = response.text or ""
        label = extract_label(text, display_labels)
        if label is None:
            predicted_target = -1
            predicted_label = "unknown"
        else:
            category = display_to_category[label]
            predicted_target = category_to_target[category]
            predicted_label = label

        rows.append(
            {
                "filename": item.filename,
                "predicted_target": predicted_target,
                "predicted_label": predicted_label,
                "response": text,
            }
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    write_header = not output_path.exists() or not args.resume
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "predicted_target", "predicted_label", "response"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
