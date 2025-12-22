from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MFCC retrieval heatmap.")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/retrieval/retrieval_mfcc.csv",
        help="CSV with columns frame_length,hop_length,top10,top20",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="top10",
        choices=["top10", "top20"],
        help="Metric column to visualize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/plots/retrieval_mfcc_heatmap.png",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="custom",
        help="Matplotlib colormap name or 'custom' for light green scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not in CSV columns: {df.columns}")

    frame_vals = sorted(df["frame_length"].unique())
    hop_vals = sorted(df["hop_length"].unique())

    mat = np.full((len(frame_vals), len(hop_vals)), np.nan)
    for _, row in df.iterrows():
        i = frame_vals.index(row["frame_length"])
        j = hop_vals.index(row["hop_length"])
        mat[i, j] = row[args.metric]

    masked = np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=(8, 5))
    if args.cmap == "custom":
        from matplotlib.colors import LinearSegmentedColormap

        colors = [
            (0.90, 0.97, 0.92),
            (0.75, 0.90, 0.80),
            (0.60, 0.82, 0.70),
            (0.45, 0.74, 0.60),
        ]
        cmap = LinearSegmentedColormap.from_list("light_green", colors, N=256)
    else:
        cmap = plt.get_cmap(args.cmap)

    im = ax.imshow(
        masked,
        cmap=cmap,
        origin="lower",
        vmin=np.nanmin(mat),
        vmax=np.nanmax(mat),
    )

    ax.set_xticks(range(len(hop_vals)))
    ax.set_yticks(range(len(frame_vals)))
    ax.set_xticklabels(hop_vals)
    ax.set_yticklabels(frame_vals)
    ax.set_xlabel("hop_length")
    ax.set_ylabel("frame_length")
    ax.set_title(f"MFCC Retrieval ({args.metric})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.metric)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
