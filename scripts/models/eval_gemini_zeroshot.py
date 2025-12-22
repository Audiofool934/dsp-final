from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

from google import genai
from google.genai import types
from tqdm import tqdm

from src.datasets.esc50 import Esc50Meta
from src.tasks.llm_baseline import build_label_lookup, extract_label_from_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot classification with Gemini")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", type=str, default="outputs/llm_predictions.csv")
    parser.add_argument("--resume", action="store_true", help="Skip files already in output CSV")
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="simple",
        choices=["simple", "guided"],
        help="Prompt format: simple classification or guided (summary + label).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass --api-key.")

    if args.prompt_style != "simple" and args.output == "outputs/llm_predictions.csv":
        args.output = "outputs/llm_predictions_guided.csv"

    meta = Esc50Meta(args.data_root)
    items = meta.by_folds([5])
    lookup = build_label_lookup(meta)
    display_labels = lookup.display_labels

    thread_local = threading.local()

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = genai.Client(api_key=api_key)
            thread_local.client = client
        return client

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["filename"])

    if args.prompt_style == "guided":
        label_lines = [f"{idx} = {lookup.target_to_display[idx]}" for idx in sorted(lookup.target_to_display)]
        prompt = (
            "You are an ESC-50 audio classifier.\n\n"
            "Step 1 (brief understanding):\n"
            "Describe the PRIMARY sound in 1 short sentence.\n\n"
            "Step 2 (classification):\n"
            "Pick exactly ONE label from the list below.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            "  \"summary\": \"<1 short sentence>\",\n"
            "  \"label_id\": <int>,\n"
            "  \"label\": \"<label name>\",\n"
            "  \"confidence\": <float 0-1>\n"
            "}\n\n"
            "Rules:\n"
            "- The label MUST be from the list. No other labels.\n"
            "- Do NOT output Markdown or any extra text.\n"
            "- If unsure, choose the closest label.\n\n"
            "Label list:\n"
            + "\n".join(label_lines)
        )
    else:
        prompt = (
            "Classify this audio into one of the ESC-50 categories. "
            "Return JSON only: {\"label\": \"<one label>\"}. "
            "Valid labels: " + ", ".join(display_labels)
        )

    pending_items = []
    for idx, item in enumerate(items):
        if args.max_samples and idx >= args.max_samples:
            break
        if item.filename in processed:
            continue
        pending_items.append(item)

    def process_item(item):
        with open(item.path, "rb") as f:
            audio_bytes = f.read()
        client = get_client()
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
        label = extract_label_from_response(text, lookup)
        if label is None:
            predicted_target = -1
            predicted_label = "unknown"
        else:
            category = lookup.display_to_category[label]
            predicted_target = lookup.category_to_target[category]
            predicted_label = label

        if args.sleep > 0:
            time.sleep(args.sleep)

        return {
            "filename": item.filename,
            "predicted_target": predicted_target,
            "predicted_label": predicted_label,
            "response": text,
        }

    write_header = not output_path.exists() or not args.resume
    rows_written = 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "predicted_target", "predicted_label", "response"],
        )
        if write_header:
            writer.writeheader()
        if args.workers and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_item, item) for item in pending_items]
                with tqdm(total=len(pending_items), desc="gemini", unit="audio") as pbar:
                    for future in as_completed(futures):
                        row = future.result()
                        writer.writerow(row)
                        f.flush()
                        rows_written += 1
                        if args.log_every > 0 and rows_written % args.log_every == 0:
                            tqdm.write(f"gemini progress: {rows_written}/{len(pending_items)}", end="\n")
                        pbar.update(1)
        else:
            for item in tqdm(pending_items, desc="gemini", unit="audio"):
                row = process_item(item)
                writer.writerow(row)
                f.flush()
                rows_written += 1
                if args.log_every > 0 and rows_written % args.log_every == 0:
                    tqdm.write(f"gemini progress: {rows_written}/{len(pending_items)}", end="\n")

    print(f"Saved {rows_written} predictions to {output_path}")


if __name__ == "__main__":
    main()
