#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Create stratified train/test split for age prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--outdir", default="data/splits")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_bins", type=int, default=5)
    return ap.parse_args()

def plot_train_test_age_distribution(y_train, y_test, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y, label, color in zip(
        axes,
        [y_train, y_test],
        ["Train", "Test"],
        ["steelblue", "coral"]
    ):
        ages = y["age"].dropna()
        ax.hist(ages, bins=20, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(ages.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {ages.mean():.1f}")
        ax.axvline(ages.median(), color="green", linestyle="--", linewidth=2,
                   label=f"Median: {ages.median():.1f}")
        ax.set_xlabel("Age (years)", fontsize=18)
        ax.set_ylabel("Count", fontsize=18)
        ax.set_title(f"{label} Set (n={len(ages)})", fontsize=21, fontweight="bold")
        ax.legend(fontsize=15)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(alpha=0.3)

    fig.suptitle("Age Distribution: Train vs Test", fontsize=20, fontweight="bold")
    plt.tight_layout()

    out_path = Path(outdir) / "age_distribution_train_test.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Train/test age distribution saved to: {out_path}")

def main():
    args = parse_args()
    
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    
    print(f"Output directory: {outdir}")
    print(f"Test size: {args.test_size*100:.0f}%")
    print(f"Random state: {args.random_state}")
    

    print("\nLoading data...")
    
    try:
        X = pd.read_csv(args.X, index_col=0)
        print(f"  X: {X.shape[0]:,} samples × {X.shape[1]:,} genes")
    except Exception as e:
        print(f"  ERROR loading X: {e}")
        return
    
    try:
        y = pd.read_csv(args.y)
        print(f"  y: {len(y):,} samples")
    except Exception as e:
        print(f"  ERROR loading y: {e}")
        return
    
    # Validate y structure
    if "sample_id" not in y.columns or "age" not in y.columns:
        print(f"  ERROR: y must contain columns: sample_id, age")
        print(f"     Found columns: {list(y.columns)}")
        return
    
    # Align Samples
    print("\nAligning samples...")
    
    X.index = X.index.astype(str)
    y["sample_id"] = y["sample_id"].astype(str)
    
    # Find common samples
    common = X.index.intersection(y["sample_id"])
    
    if len(common) == 0:
        print(f"  ERROR: No common samples between X and y!")
        print(f"     X sample IDs (first 5): {list(X.index[:5])}")
        print(f"     y sample IDs (first 5): {list(y['sample_id'][:5])}")
        return
    
    # Filter to common
    X = X.loc[common]
    y = y.set_index("sample_id").loc[common].reset_index()
    
    # Verify alignment
    if len(X) != len(y):
        print(f"  ERROR: Shape mismatch after alignment!")
        print(f"     X: {len(X)}, y: {len(y)}")
        return
    
    print(f"  Aligned: {len(X):,} samples × {X.shape[1]:,} genes")
    
    # Check for minimum sample size
    if len(X) < 50:
        print(f"  WARNING: Very small dataset ({len(X)} samples)")
        print(f"     Results may not be reliable")
    

    print("\nCreating age bins for stratification...")
    
    print(f"\n  Age distribution:")
    print(f"    Mean: {y['age'].mean():.1f} ± {y['age'].std():.1f} years")
    print(f"    Range: {y['age'].min():.0f} - {y['age'].max():.0f} years")
    print(f"    Median: {y['age'].median():.1f} years")
    
    # Use quantile-based bins for equal sample distribution
    
    try:
        age_bins, bin_edges = pd.qcut(
            y["age"], 
            q=args.n_bins, 
            labels=False,
            duplicates='drop',
            retbins=True
        )
    except ValueError as e:
        print(f"  Quantile binning failed, using fixed bins")
        age_bins, bin_edges = pd.cut(
            y["age"], 
            bins=[0, 30, 40, 50, 60, 70, 150], 
            labels=False,
            retbins=True
        )
    
    bin_counts = pd.Series(age_bins).value_counts().sort_index()
    print(f"\n  Age bins for stratification:")
    for bin_id, count in bin_counts.items():
        pct = 100 * count / len(age_bins)
        print(f"    Bin {bin_id}: {count:3d} samples ({pct:.1f}%)")
    
    # Check for bins with very few samples
    min_bin_count = bin_counts.min()
    if min_bin_count < 10:
        print(f"\n  WARNING: Some bins have very few samples (min={min_bin_count})")
        print(f"     Stratification may not work well")
    
    print(f"\nCreating stratified train/test split...")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=age_bins
        )
        print(f"  Split successful")
    except Exception as e:
        print(f"  ERROR during split: {e}")
        print(f"  Trying without stratification...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state
        )
        print(f"  Split completed without stratification")
    
    # Validate split sizes
    if len(X_test) < 30:
        print(f"\n  WARNING: Test set is small ({len(X_test)} samples)")
    
    print(f"\nSaving splits to {outdir}...")
    
    X_train.to_csv(outdir / "X_train_raw.csv")
    X_test.to_csv(outdir / "X_test_raw.csv")
    y_train.to_csv(outdir / "y_train.csv", index=False)
    y_test.to_csv(outdir / "y_test.csv", index=False)
    
    print(f"  {outdir / 'X_train_raw.csv'}")
    print(f"  {outdir / 'X_test_raw.csv'}")
    print(f"  {outdir / 'y_train.csv'}")
    print(f"  {outdir / 'y_test.csv'}")

    # Age distribution plot
    print("\nGenerating train/test age distribution plot...")
    plot_train_test_age_distribution(y_train, y_test, outdir)
    
    # Summary Statistics
    print("\n Summary")
    
    print(f"\nDataset:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Total genes: {X.shape[1]:,}")
    
    print(f"\nSplit:")
    print(f"  Train samples: {len(X_train):,} ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Test samples: {len(X_test):,} ({100*len(X_test)/len(X):.1f}%)")
    
    print(f"\nAge Distribution:")
    print(f"  Train: {y_train['age'].mean():.1f} ± {y_train['age'].std():.1f} years")
    print(f"         Range: {y_train['age'].min():.0f} - {y_train['age'].max():.0f}")
    print(f"  Test:  {y_test['age'].mean():.1f} ± {y_test['age'].std():.1f} years")
    print(f"         Range: {y_test['age'].min():.0f} - {y_test['age'].max():.0f}")
    
    # t-test, ks   
    t_stat, p_value = stats.ttest_ind(y_train['age'], y_test['age'])
    ks_stat, ks_p_value = stats.ks_2samp(y_train['age'], y_test['age'])
    
    print(f"\nDistribution similarity:")
    print(f"  t-test p-value:  {p_value:.4f}", end="")
    if p_value > 0.05:
        print(f" (means not significantly different)")
    else:
        print(f" (means may differ)")
    print(f"  KS test p-value: {ks_p_value:.4f}", end="")
    if ks_p_value > 0.05:
        print(f" (distributions not significantly different)")
    else:
        print(f" (distributions may differ)")
    
    # Save Metadata
    metadata = {
        "n_total": int(len(X)),
        "n_genes": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "n_bins": int(args.n_bins),
        "bin_edges": [float(e) for e in bin_edges],
        "train_age_mean": float(y_train["age"].mean()),
        "train_age_std": float(y_train["age"].std()),
        "train_age_min": float(y_train["age"].min()),
        "train_age_max": float(y_train["age"].max()),
        "test_age_mean": float(y_test["age"].mean()),
        "test_age_std": float(y_test["age"].std()),
        "test_age_min": float(y_test["age"].min()),
        "test_age_max": float(y_test["age"].max()),
        "t_test_p_value": float(p_value),
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p_value),
    }
    
    with open(outdir / "split_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Metadata saved to: {outdir / 'split_info.json'}")


if __name__ == "__main__":
    main()