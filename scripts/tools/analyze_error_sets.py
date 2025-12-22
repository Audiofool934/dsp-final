from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.datasets.esc50 import Esc50Meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze error CSVs and copy audio examples.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--errors-dir", type=str, default=None)
    parser.add_argument("--predictions-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--copy-audio", action="store_true")
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--clean-audio", action="store_true", help="Clear existing audio folders first")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        errors_dir = run_dir / "errors"
        predictions_dir = run_dir / "predictions"
        output_path = Path(args.output) if args.output else errors_dir / "error_analysis.md"
        audio_base = Path(args.audio_dir) if args.audio_dir else errors_dir / "audio"
        return errors_dir, predictions_dir, output_path, audio_base
    if args.errors_dir:
        errors_dir = Path(args.errors_dir)
        predictions_dir = Path(args.predictions_dir) if args.predictions_dir else None
        output_path = Path(args.output) if args.output else errors_dir / "error_analysis.md"
        audio_base = Path(args.audio_dir) if args.audio_dir else errors_dir / "audio"
        return errors_dir, predictions_dir, output_path, audio_base
    raise ValueError("Provide --run-dir or --errors-dir.")


def format_top(counter: Counter, top_k: int = 5) -> str:
    if not counter:
        return "none"
    items = counter.most_common(top_k)
    return ", ".join([f"{label} ({count})" for label, count in items])


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\s+", "-", text)
    text = re.sub(r"[^a-z0-9_-]", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def main() -> None:
    args = parse_args()
    errors_dir, predictions_dir, output_path, audio_base = resolve_paths(args)
    if not errors_dir.exists():
        raise FileNotFoundError(f"Errors directory not found: {errors_dir}")

    meta = Esc50Meta(args.data_root)
    filename_to_item = {item.filename: item for item in meta.items}
    target_to_label = {item.target: item.category.replace("_", " ") for item in meta.items}
    fold5_items = meta.by_folds([5])
    fold5_label_counts = Counter([item.category.replace("_", " ") for item in fold5_items])

    error_files = sorted(errors_dir.glob("*_errors.csv"))
    if not error_files:
        print(f"No error CSVs found in {errors_dir}")
        return

    lines = [
        "# Error Analysis",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Errors dir: {errors_dir}",
        "",
    ]

    for path in error_files:
        model_name = path.stem.replace("_errors", "")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            continue

        filenames = [row.get("filename", "") for row in rows if row.get("filename")]
        true_labels = []
        pred_labels = []
        pred_by_true: dict[str, Counter] = {}
        resolved_rows = []
        for row in rows:
            filename = row.get("filename", "")
            item = filename_to_item.get(filename)
            true_label = row.get("true_label")
            if not true_label and item is not None:
                true_label = item.category.replace("_", " ")
            pred_label = row.get("predicted_label")
            if not pred_label:
                try:
                    pred_target = int(row.get("predicted_target", -1))
                except (TypeError, ValueError):
                    pred_target = -1
                pred_label = target_to_label.get(pred_target, "unknown")
            true_labels.append(true_label or "unknown")
            pred_labels.append(pred_label or "unknown")
            pred_by_true.setdefault(true_label or "unknown", Counter())[pred_label or "unknown"] += 1
            resolved_rows.append(
                {
                    "filename": filename,
                    "true_label": true_label or "unknown",
                    "pred_label": pred_label or "unknown",
                }
            )

        if args.copy_audio:
            dest_dir = audio_base / model_name
            if args.clean_audio and dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            used_names = set()
            for row in resolved_rows:
                filename = row["filename"]
                item = filename_to_item.get(filename)
                if item is None or not item.path.exists():
                    continue
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                gt = slugify(row["true_label"])
                pred = slugify(row["pred_label"])
                name = f"{stem}__gt_{gt}__pred_{pred}{suffix}"
                if name in used_names:
                    idx = 1
                    while True:
                        candidate = f"{stem}__gt_{gt}__pred_{pred}__{idx}{suffix}"
                        if candidate not in used_names:
                            name = candidate
                            break
                        idx += 1
                used_names.add(name)
                shutil.copy2(item.path, dest_dir / name)

        total_errors = len(rows)
        total_preds = None
        if predictions_dir is not None:
            pred_path = predictions_dir / f"{model_name}.csv"
            if pred_path.exists():
                with pred_path.open("r", encoding="utf-8") as f:
                    pred_reader = csv.DictReader(f)
                    total_preds = sum(1 for _ in pred_reader)

        true_counts = Counter(true_labels)
        pred_counts = Counter(pred_labels)
        pair_counts = Counter(zip(true_labels, pred_labels))
        error_rates = []
        for label, err_count in true_counts.items():
            total = fold5_label_counts.get(label, 0)
            if total:
                error_rates.append((label, err_count, total, err_count / total))
        error_rates.sort(key=lambda x: (-x[3], -x[1], x[0]))

        lines.append(f"## Model: {model_name}")
        if total_preds:
            acc = 1.0 - total_errors / total_preds
            lines.append(f"- errors: {total_errors} / {total_preds} (acc {acc:.4f})")
        else:
            lines.append(f"- errors: {total_errors}")
        lines.append(f"- top true labels: {format_top(true_counts)}")
        lines.append(f"- top predicted labels: {format_top(pred_counts)}")
        top_pairs = pair_counts.most_common(5)
        if top_pairs:
            confusions = ", ".join([f"{t} -> {p} ({c})" for (t, p), c in top_pairs])
            lines.append(f"- top confusions: {confusions}")
        if error_rates:
            hardest = ", ".join(
                [f"{label} ({err}/{total}, {rate:.2f})" for label, err, total, rate in error_rates[:5]]
            )
            lines.append(f"- hardest labels: {hardest}")
        if args.copy_audio:
            lines.append(f"- audio copied to: {audio_base / model_name}")
        if error_rates:
            lines.append("")
            lines.append("| true_label | errors | total | error_rate | top_pred |")
            lines.append("| --- | --- | --- | --- | --- |")
            for label, err, total, rate in error_rates[:10]:
                top_pred = format_top(pred_by_true.get(label, Counter()), top_k=1)
                lines.append(f"| {label} | {err} | {total} | {rate:.2f} | {top_pred} |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
