from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extended grids: precompute (mfcc/log-mel), retrieval, classification."
    )
    parser.add_argument("--run-id", type=str, default="run_20251221_003516")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument(
        "--frame-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--hop-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--precompute-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_cmd(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def merge_csv(out_path: Path, new_path: Path) -> None:
    new_df = pd.read_csv(new_path)
    if out_path.exists():
        old_df = pd.read_csv(out_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["frame_length", "hop_length"], keep="last")
    else:
        df = new_df
    df = df.sort_values(["frame_length", "hop_length"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    run_dir = Path("outputs/results") / args.run_id
    feature_cache = run_dir / "features"
    logs_dir = run_dir / "logs"
    retrieval_out = run_dir / "retrieval" / "retrieval_mfcc.csv"
    classification_out = run_dir / "history" / "classification_grid.csv"
    history_dir = run_dir / "history" / "classification_grid"

    # 1) Precompute MFCC and log-mel for all combos
    for feature_type in ["mfcc", "log_mel"]:
        for fl in args.frame_lengths:
            for hl in args.hop_lengths:
                log_path = logs_dir / f"precompute_{feature_type}_fl{fl}_hl{hl}.log"
                cmd = [
                    sys.executable,
                    "scripts/tools/precompute_features.py",
                    "--feature-types",
                    feature_type,
                    "--frame-length",
                    str(fl),
                    "--hop-length",
                    str(hl),
                    "--n-mels",
                    str(args.n_mels),
                    "--n-mfcc",
                    str(args.n_mfcc),
                    "--workers",
                    str(args.precompute_workers),
                    "--feature-cache",
                    str(feature_cache),
                    "--data-root",
                    args.data_root,
                ]
                run_cmd(cmd, log_path)

    # 2) Retrieval grid (MFCC)
    retrieval_log = logs_dir / "retrieval_mfcc_grid.log"
    cmd = [
        sys.executable,
        "scripts/tasks/run_retrieval.py",
        "--data-root",
        args.data_root,
        "--sample-rate",
        str(args.sample_rate),
        "--frame-lengths",
        *map(str, args.frame_lengths),
        "--hop-lengths",
        *map(str, args.hop_lengths),
        "--n-mels",
        str(args.n_mels),
        "--n-mfcc",
        str(args.n_mfcc),
        "--feature-cache",
        str(feature_cache),
        "--output",
        str(retrieval_out),
    ]
    run_cmd(cmd, retrieval_log)

    # 3) Classification grid (log-mel)
    class_log = logs_dir / "classification_grid_full.log"
    cmd = [
        sys.executable,
        "scripts/tasks/run_classification_grid.py",
        "--data-root",
        args.data_root,
        "--sample-rate",
        str(args.sample_rate),
        "--frame-lengths",
        *map(str, args.frame_lengths),
        "--hop-lengths",
        *map(str, args.hop_lengths),
        "--n-mels",
        str(args.n_mels),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--output",
        str(history_dir / "classification_grid_latest.csv"),
        "--history-dir",
        str(history_dir),
        "--feature-cache",
        str(feature_cache),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    run_cmd(cmd, class_log)

    # Merge classification results into master CSV
    merge_csv(classification_out, history_dir / "classification_grid_latest.csv")
    print(f"Updated classification grid at {classification_out}")
    print(f"Retrieval grid at {retrieval_out}")


if __name__ == "__main__":
    main()
