#!/usr/bin/env python3
"""
analysis_from_csv.py

Analysis from a compiled L2 CSV only (no downloads).

A) KDE + smoothed histogram of CH4 by qa group (equal-area weighting)
B) Weighted binned LOWESS: P(qa==1 | CH4)

Improvements vs. the draft:
- No hardcoded absolute paths (CSV provided via --csv)
- CLI args for key parameters (CH4 window, QA targets, smoothing)
- Output directory configurable (default: same folder as CSV)
- Safer defaults + clearer console output

Requires:
  pip install numpy pandas matplotlib scipy statsmodels

Examples:
  python3 analysis_from_csv.py --csv tmp_run/l2_2020-07-01_2020-07-07_ch4.csv
  python3 analysis_from_csv.py --csv tmp_run/l2_2020-07-01_2020-07-07_ch4.csv --ch4-min 1700 --ch4-max 2000
  python3 analysis_from_csv.py --csv tmp_run/l2_2020-07-01_2020-07-07_ch4.csv --qa-targets 0.4 1.0
  python3 analysis_from_csv.py --csv tmp_run/l2_2020-07-01_2020-07-07_ch4.csv --no-smooth-hist
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess


# -------------------- Helpers --------------------
def gaussian_smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian smoothing in 'bin units' without extra deps."""
    if sigma_bins is None or sigma_bins <= 0:
        return y
    radius = int(np.ceil(4 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_bins**2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze L2 CH4 distributions by QA group from a compiled CSV.")

    # Input/output
    p.add_argument("--csv", required=True, help="Path to compiled L2 CSV (contains ch4, qa_value)")
    p.add_argument("--outdir", default=None, help="Directory to save plots/tables (default: same dir as CSV)")

    # Columns
    p.add_argument("--ch4-col", default="ch4")
    p.add_argument("--qa-col", default="qa_value")

    # QA grouping
    p.add_argument("--qa-targets", nargs="+", type=float, default=[0.0, 0.4, 1.0],
                   help="QA groups to compare after rounding")
    p.add_argument("--qa-round-decimals", type=int, default=2)

    # CH4 window for distribution plots
    p.add_argument("--ch4-min", type=float, default=1750.0)
    p.add_argument("--ch4-max", type=float, default=1950.0)

    # KDE settings
    p.add_argument("--bw-mult", type=float, default=0.6, help="KDE bandwidth multiplier")
    p.add_argument("--grid-n", type=int, default=600)

    # Smoothed histogram settings
    p.add_argument("--no-smooth-hist", action="store_true", help="Disable smoothed histogram plot")
    p.add_argument("--hist-bins", type=int, default=140)
    p.add_argument("--smooth-sigma-bins", type=float, default=2.0)

    # LOWESS (weighted binned)
    p.add_argument("--lowess-drop-ch4-leq", type=float, default=0.0)
    p.add_argument("--n-bins", type=int, default=500)
    p.add_argument("--min-bin-count", type=int, default=30)
    p.add_argument("--lowess-frac", type=float, default=0.15)
    p.add_argument("--lowess-it", type=int, default=1)

    # weighting-by-replication controls
    p.add_argument("--max-rep-per-bin", type=int, default=200)
    p.add_argument("--scale-rep-by", type=float, default=2000)

    return p.parse_args()


# -------------------- Analyses --------------------
def kde_and_hist(
    df: pd.DataFrame,
    outdir: str,
    ch4_col: str,
    qa_col: str,
    qa_targets,
    qa_round_decimals: int,
    ch4_min: float,
    ch4_max: float,
    bw_mult: float,
    grid_n: int,
    do_smooth_hist: bool,
    hist_bins: int,
    smooth_sigma_bins: float,
):
    # Clean
    df = df[np.isfinite(df[ch4_col]) & np.isfinite(df[qa_col])].copy()

    df["qa_round"] = df[qa_col].round(qa_round_decimals)
    df = df[df["qa_round"].isin(qa_targets)].copy()
    df = df[(df[ch4_col] >= ch4_min) & (df[ch4_col] <= ch4_max)].copy()

    if df.empty:
        raise RuntimeError("No rows left after QA filter + CH4 window. Adjust --ch4-min/--ch4-max or --qa-targets.")

    counts = df["qa_round"].value_counts().sort_index()
    print("\n[INFO] Counts by qa_value (rounded) after CH4 window:")
    print(counts)

    qa_present = [q for q in qa_targets if q in counts.index and counts.loc[q] >= 2]
    missing = [q for q in qa_targets if q not in counts.index]
    if missing:
        print(f"\n[WARN] Missing QA groups (skipping): {missing}")
    if not qa_present:
        raise RuntimeError("No QA group has >=2 points after filtering.")

    # Equal-area weighting: each group's total weight = 1
    df["w"] = df["qa_round"].map(lambda q: 1.0 / counts.loc[q])

    x_grid = np.linspace(ch4_min, ch4_max, grid_n)

    # KDE plot
    plt.figure(figsize=(10, 6))
    for q in qa_present:
        sub = df[df["qa_round"] == q]
        x = sub[ch4_col].to_numpy(float)
        w = sub["w"].to_numpy(float)

        kde = gaussian_kde(x, weights=w)
        if bw_mult != 1.0:
            kde.set_bandwidth(bw_method=kde.factor * bw_mult)
        y = kde.evaluate(x_grid)

        plt.plot(x_grid, y, linewidth=2.5, label=f"qa={q:g} (n={len(sub):,})")

    plt.title(f"CH4 distribution by QA (KDE, equal-area), window [{ch4_min}, {ch4_max}]")
    plt.xlabel("L2 CH4")
    plt.ylabel("Density (each curve area ≈ 1)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_kde = os.path.join(outdir, "ch4_kde_equal_area.png")
    plt.tight_layout()
    plt.savefig(out_kde, dpi=300, bbox_inches="tight")
    print(f"[DONE] Saved KDE: {out_kde}")
    plt.show()

    # Smoothed histogram plot (optional)
    if do_smooth_hist:
        plt.figure(figsize=(10, 6))
        bin_edges = np.linspace(ch4_min, ch4_max, hist_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for q in qa_present:
            sub = df[df["qa_round"] == q]
            x = sub[ch4_col].to_numpy(float)
            w = sub["w"].to_numpy(float)

            hist, _ = np.histogram(x, bins=bin_edges, weights=w, density=True)
            hist_s = gaussian_smooth_1d(hist, sigma_bins=smooth_sigma_bins)
            plt.plot(bin_centers, hist_s, linewidth=2.5, label=f"qa={q:g} (n={len(sub):,})")

        plt.title(f"CH4 distribution by QA (smoothed hist, equal-area), window [{ch4_min}, {ch4_max}]")
        plt.xlabel("L2 CH4")
        plt.ylabel("Density (each curve area ≈ 1)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        out_hist = os.path.join(outdir, "ch4_hist_smooth_equal_area.png")
        plt.tight_layout()
        plt.savefig(out_hist, dpi=300, bbox_inches="tight")
        print(f"[DONE] Saved smoothed histogram: {out_hist}")
        plt.show()


def weighted_binned_lowess(
    df: pd.DataFrame,
    outdir: str,
    ch4_col: str,
    qa_col: str,
    drop_ch4_leq: float,
    n_bins: int,
    min_bin_count: int,
    lowess_frac: float,
    lowess_it: int,
    max_rep_per_bin: int,
    scale_rep_by: float,
):
    df = df[[ch4_col, qa_col]].dropna().copy()

    before = len(df)
    df = df[df[ch4_col] > drop_ch4_leq].copy()
    print(f"\n[INFO] Dropped CH4 <= {drop_ch4_leq}: {before - len(df):,} removed, {len(df):,} remain")

    # y = 1(qa==1.0) else 0
    df["y"] = (np.abs(df[qa_col] - 1.0) <= 1e-9).astype(int)

    ch4_min = float(df[ch4_col].min())
    ch4_max = float(df[ch4_col].max())
    print(f"[INFO] CH4 range for LOWESS: {ch4_min:.6g} to {ch4_max:.6g}")

    bins = np.linspace(ch4_min, ch4_max, n_bins + 1)
    df["bin"] = pd.cut(df[ch4_col], bins=bins, include_lowest=True)

    g = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), y_mean=("y", "mean"), x=(ch4_col, "mean"))
        .reset_index()
        .sort_values("x")
    )

    before_bins = len(g)
    g = g[g["n"] >= min_bin_count].copy()
    print(f"[INFO] Dropped bins with n < {min_bin_count}: {before_bins - len(g)} bins removed, {len(g)} remain")
    if len(g) < 10:
        raise RuntimeError("Too few bins left; reduce --min-bin-count or --n-bins.")

    # Weighting via replication (simple + robust)
    reps = np.clip((g["n"] / scale_rep_by).round().astype(int), 1, max_rep_per_bin)
    x_rep = np.repeat(g["x"].to_numpy(), reps.to_numpy())
    y_rep = np.repeat(g["y_mean"].to_numpy(), reps.to_numpy())
    print(f"[INFO] Total replicated points for LOWESS: {len(x_rep):,}")

    smoothed = lowess(endog=y_rep, exog=x_rep, frac=lowess_frac, it=lowess_it, return_sorted=True)
    xs, ys = smoothed[:, 0], smoothed[:, 1]

    out_tbl = os.path.join(outdir, "binned_weighted_qa1_vs_ch4.csv")
    g[["x", "n", "y_mean"]].to_csv(out_tbl, index=False)
    print(f"[DONE] Saved binned table: {out_tbl}")

    plt.figure(figsize=(8, 6))
    sizes = np.clip(g["n"] / g["n"].max() * 60.0, 8.0, 60.0)
    plt.scatter(g["x"], g["y_mean"], s=sizes, alpha=0.6, label="Binned mean (size ~ n)")
    plt.plot(xs, ys, linewidth=2.0, label=f"Weighted LOWESS (frac={lowess_frac})")
    plt.xlabel("L2 CH4")
    plt.ylabel("P(qa_value == 1.0)")
    plt.title("Weighted binned LOWESS: qa==1 vs CH4")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(outdir, "weighted_binned_lowess_qa1_vs_ch4.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[DONE] Saved LOWESS plot: {out_png}")
    plt.show()


def main():
    args = parse_args()

    csv_in = os.path.abspath(args.csv)
    outdir = args.outdir or os.path.dirname(csv_in)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    print("[INFO] --------------------")
    print(f"[INFO] csv={csv_in}")
    print(f"[INFO] outdir={outdir}")
    print(f"[INFO] ch4_col={args.ch4_col} qa_col={args.qa_col}")
    print(f"[INFO] qa_targets={args.qa_targets} qa_round_decimals={args.qa_round_decimals}")
    print(f"[INFO] ch4_window=[{args.ch4_min}, {args.ch4_max}]")
    print(f"[INFO] bw_mult={args.bw_mult} grid_n={args.grid_n}")
    print(f"[INFO] smooth_hist={'no' if args.no_smooth_hist else 'yes'} hist_bins={args.hist_bins} sigma_bins={args.smooth_sigma_bins}")
    print(f"[INFO] lowess: drop_ch4_leq={args.lowess_drop_ch4_leq} n_bins={args.n_bins} min_bin_count={args.min_bin_count} frac={args.lowess_frac} it={args.lowess_it}")
    print("[INFO] --------------------")

    df = pd.read_csv(csv_in)

    kde_and_hist(
        df=df,
        outdir=outdir,
        ch4_col=args.ch4_col,
        qa_col=args.qa_col,
        qa_targets=args.qa_targets,
        qa_round_decimals=args.qa_round_decimals,
        ch4_min=args.ch4_min,
        ch4_max=args.ch4_max,
        bw_mult=args.bw_mult,
        grid_n=args.grid_n,
        do_smooth_hist=not args.no_smooth_hist,
        hist_bins=args.hist_bins,
        smooth_sigma_bins=args.smooth_sigma_bins,
    )

    weighted_binned_lowess(
        df=df,
        outdir=outdir,
        ch4_col=args.ch4_col,
        qa_col=args.qa_col,
        drop_ch4_leq=args.lowess_drop_ch4_leq,
        n_bins=args.n_bins,
        min_bin_count=args.min_bin_count,
        lowess_frac=args.lowess_frac,
        lowess_it=args.lowess_it,
        max_rep_per_bin=args.max_rep_per_bin,
        scale_rep_by=args.scale_rep_by,
    )

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()