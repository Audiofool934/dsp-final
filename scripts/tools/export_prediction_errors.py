from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.datasets.esc50 import Esc50Meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export error cases from prediction CSVs.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--predictions-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    return parser.parse_args()


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        preds_dir = run_dir / "predictions"
        out_dir = run_dir / "errors"
        return preds_dir, out_dir
    if args.predictions_dir:
        preds_dir = Path(args.predictions_dir)
        out_dir = Path(args.output_dir) if args.output_dir else preds_dir.parent / "errors"
        return preds_dir, out_dir
    raise ValueError("Provide --run-dir or --predictions-dir.")


def main() -> None:
    args = parse_args()
    preds_dir, out_dir = _resolve_dirs(args)
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")

    meta = Esc50Meta(args.data_root)
    fold5 = {item.filename: item for item in meta.by_folds([5])}

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(preds_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {preds_dir}")
        return

    for path in csv_files:
        if "_errors" in path.stem:
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "filename" not in fieldnames or "predicted_target" not in fieldnames:
                print(f"Skipping {path.name}: missing filename or predicted_target")
                continue

            out_rows = []
            for row in reader:
                filename = row["filename"]
                item = fold5.get(filename)
                if item is None:
                    continue
                try:
                    pred = int(row["predicted_target"])
                except (TypeError, ValueError):
                    pred = None
                if pred == item.target:
                    continue

                row_out = dict(row)
                row_out["true_target"] = item.target
                row_out["true_label"] = item.category.replace("_", " ")
                out_rows.append(row_out)

        if not out_rows:
            print(f"{path.name}: no errors found")
            continue

        out_fields = list(fieldnames)
        for field in ("true_target", "true_label"):
            if field not in out_fields:
                out_fields.append(field)

        out_path = out_dir / f"{path.stem}_errors.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"{path.name}: wrote {len(out_rows)} errors -> {out_path}")


if __name__ == "__main__":
    main()
