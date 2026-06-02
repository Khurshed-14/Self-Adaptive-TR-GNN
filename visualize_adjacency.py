"""
Visualize a saved adjacency matrix (.npy) with ISO-NE feature names on both axes.
By default plots the raw (denormalized) learned A from training, with values in each cell.
Use --normalize for row-normalized weights (same as DenselyResidualGCN at runtime).

Usage:
    .\\load_new\\Scripts\\python.exe visualize_adjacency.py
    .\\load_new\\Scripts\\python.exe visualize_adjacency.py adjacency_matrices/epoch50.npy
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dataset_classes import ISO_NE


def load_feature_names(
    csv_path: str = "data/iso_ne/selected_data_ISONE.csv",
    T_in: int = 72,
    T_out: int = 240,
    lag_hours=None,
    rolling_windows=None,
):
    if lag_hours is None:
        lag_hours = [1, 12, 24, 168]
    if rolling_windows is None:
        rolling_windows = [12, 24]

    dataset = ISO_NE(
        csv_path=csv_path,
        T_in=T_in,
        T_out=T_out,
        lag_hours=lag_hours,
        rolling_windows=rolling_windows,
    )
    return dataset.feature_names, dataset.N


def load_adjacency(path: str) -> np.ndarray:
    A = np.load(path)
    if A.ndim == 2:
        A = A[np.newaxis, ...]
    elif A.ndim != 3:
        raise ValueError(f"Expected (N, N) or (num_heads, N, N), got shape {A.shape}")
    return A


def row_normalize(A_heads: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Match DenselyResidualGCN: A_tilde = D^{-1} A (each row sums to 1)."""
    out = A_heads.astype(np.float64, copy=True)
    row_sums = out.sum(axis=-1, keepdims=True)
    return out / np.maximum(row_sums, eps)


def shorten_labels(names, max_len: int = 14):
    out = []
    for name in names:
        s = str(name)
        out.append(s if len(s) <= max_len else s[: max_len - 1] + "…")
    return out


def plot_adjacency(
    A_heads: np.ndarray,
    feature_names: list,
    title: str,
    out_path: str | None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
    dpi: int = 120,
    fig_scale: float = 0.28,
    decimals: int = 2,
    row_normalized: bool = False,
):
    num_heads, n, _ = A_heads.shape
    if len(feature_names) != n:
        raise ValueError(
            f"Feature name count ({len(feature_names)}) does not match "
            f"adjacency size N={n}. Check dataset settings vs training run."
        )

    labels = shorten_labels(feature_names)

    side = max(3.2, n * fig_scale)
    fig, axes = plt.subplots(
        1, num_heads, figsize=(side * num_heads + 0.8, side + 0.6), constrained_layout=True
    )
    if num_heads == 1:
        axes = [axes]

    last_im = None
    for h, ax in enumerate(axes):
        mat = A_heads[h]
        last_im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"Head {h + 1}" if num_heads > 1 else "Adjacency", fontsize=9, fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("From feature (j)", fontsize=7)
        if h == 0:
            ax.set_ylabel("To feature (i)", fontsize=7)

        if annotate:
            cell_fs = max(3.5, min(7.0, 90 / n))
            fmt = f"{{:.{decimals}f}}"
            mid = (vmin + vmax) * 0.5
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    color = "white" if val > mid else "black"
                    ax.text(
                        j, i, fmt.format(val),
                        ha="center", va="center", fontsize=cell_fs, color=color,
                    )

    fig.suptitle(title, fontsize=10, fontweight="bold")
    cbar_label = (
        r"Row-normalized $\tilde{A}_{ij}$ (rows sum to 1)"
        if row_normalized
        else r"Raw learned $A_{ij}$ from graph module (denormalized)"
    )
    fig.colorbar(last_im, ax=axes, label=cbar_label, shrink=0.75, fraction=0.046, pad=0.02)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a saved adjacency matrix with feature labels.")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default="adjacency_matrices/epoch50.npy",
        help="Path to .npy file (shape (N,N) or (num_heads, N, N))",
    )
    parser.add_argument(
        "--csv",
        default="data/iso_ne/selected_data_ISONE.csv",
        help="ISO-NE CSV (must match training feature construction)",
    )
    parser.add_argument("--out", default=None, help="Optional output image path (e.g. epoch50_labeled.png)")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--no-annotate", action="store_true", help="Hide numbers inside cells")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Row-normalize before plotting (A_tilde = A / row_sum); default is raw/denormalized A",
    )
    parser.add_argument("--fig-scale", type=float, default=0.28, help="Inches per matrix row/col")
    parser.add_argument("--decimals", type=int, default=2, help="Decimal places in cell labels")
    args = parser.parse_args()

    if not os.path.isfile(args.npy_path):
        raise FileNotFoundError(f"Adjacency file not found: {args.npy_path}")

    feature_names, n_expected = load_feature_names(csv_path=args.csv)
    A_heads = load_adjacency(args.npy_path)
    if args.normalize:
        A_heads = row_normalize(A_heads)
        print("Applied row normalization (A_tilde = A / row_sum).")
    else:
        print("Plotting raw denormalized A (as saved from TemporalGraphLearning).")

    print(f"Loaded {args.npy_path}  shape={A_heads.shape}")
    print(f"ISO-NE features (N={n_expected}):")
    for i, name in enumerate(feature_names):
        print(f"  {i:2d}  {name}")

    epoch_label = os.path.splitext(os.path.basename(args.npy_path))[0]
    out_path = args.out or os.path.join(
        os.path.dirname(args.npy_path) or ".",
        f"{epoch_label}_labeled.png",
    )

    norm_tag = "row-normalized" if args.normalize else "raw"
    plot_adjacency(
        A_heads,
        feature_names,
        title=f"Adjacency matrix ({norm_tag}) — {epoch_label}",
        out_path=out_path,
        vmin=args.vmin,
        vmax=args.vmax,
        annotate=not args.no_annotate,
        fig_scale=args.fig_scale,
        decimals=args.decimals,
        row_normalized=args.normalize,
    )


if __name__ == "__main__":
    main()
