from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def write_history_csv(history: Iterable[dict], path: str | Path, fieldnames: Sequence[str]) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
