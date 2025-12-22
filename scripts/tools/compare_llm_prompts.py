from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.datasets.esc50 import Esc50Meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two LLM prediction CSVs.")
    parser.add_argument("--base", type=str, required=True, help="Base prompt predictions CSV")
    parser.add_argument("--compare", type=str, required=True, help="Comparison prompt predictions CSV")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--output", type=str, required=True, help="Markdown output path")
    return parser.parse_args()


def load_preds(path: Path, fold5: dict[str, int]) -> dict[str, int]:
    preds = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename")
            if filename in fold5:
                try:
                    preds[filename] = int(row.get("predicted_target", -1))
                except (TypeError, ValueError):
                    preds[filename] = -1
    return preds


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    cmp_path = Path(args.compare)
    out_path = Path(args.output)
    if not base_path.exists() or not cmp_path.exists():
        raise FileNotFoundError("Missing prediction CSVs for comparison.")

    meta = Esc50Meta(args.data_root)
    fold5 = {item.filename: item.target for item in meta.by_folds([5])}
    target_to_label = {item.target: item.category.replace("_", " ") for item in meta.items}

    base_preds = load_preds(base_path, fold5)
    cmp_preds = load_preds(cmp_path, fold5)
    common = sorted(set(base_preds) & set(cmp_preds))
    if not common:
        raise ValueError("No overlapping predictions to compare.")

    base_correct = sum(1 for fn in common if base_preds[fn] == fold5[fn])
    cmp_correct = sum(1 for fn in common if cmp_preds[fn] == fold5[fn])
    base_only = [fn for fn in common if base_preds[fn] == fold5[fn] and cmp_preds[fn] != fold5[fn]]
    cmp_only = [fn for fn in common if cmp_preds[fn] == fold5[fn] and base_preds[fn] != fold5[fn]]
    changed = [fn for fn in common if base_preds[fn] != cmp_preds[fn]]

    base_only_true = Counter(target_to_label[fold5[fn]] for fn in base_only)
    cmp_only_true = Counter(target_to_label[fold5[fn]] for fn in cmp_only)
    diff_pairs = Counter(
        (target_to_label[fold5[fn]], target_to_label.get(base_preds[fn], "unknown"), target_to_label.get(cmp_preds[fn], "unknown"))
        for fn in changed
    )

    lines = []
    lines.append("# Prompt Comparison")
    lines.append("")
    lines.append(f"- base: `{base_path}`")
    lines.append(f"- compare: `{cmp_path}`")
    lines.append(f"- common samples: {len(common)}")
    lines.append(f"- base accuracy: {base_correct}/{len(common)} ({base_correct/len(common):.4f})")
    lines.append(f"- compare accuracy: {cmp_correct}/{len(common)} ({cmp_correct/len(common):.4f})")
    lines.append(f"- changed predictions: {len(changed)}")
    lines.append(f"- base-only correct: {len(base_only)}")
    lines.append(f"- compare-only correct: {len(cmp_only)}")
    lines.append("")
    lines.append("## Base-only correct (top true labels)")
    if base_only_true:
        lines.append(", ".join([f"{label} ({count})" for label, count in base_only_true.most_common(10)]))
    else:
        lines.append("none")
    lines.append("")
    lines.append("## Compare-only correct (top true labels)")
    if cmp_only_true:
        lines.append(", ".join([f"{label} ({count})" for label, count in cmp_only_true.most_common(10)]))
    else:
        lines.append("none")
    lines.append("")
    lines.append("## Most frequent changes (true -> base -> compare)")
    if diff_pairs:
        for (true_label, base_label, cmp_label), count in diff_pairs.most_common(10):
            lines.append(f"- {true_label} -> {base_label} -> {cmp_label} ({count})")
    else:
        lines.append("none")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote comparison to {out_path}")


if __name__ == "__main__":
    main()
