#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="ComBat batch correction with quality control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--X_log2",          required=True)
    ap.add_argument("--metadata",        required=True)
    ap.add_argument("--y",               required=True)
    ap.add_argument("--outdir",          required=True)
    ap.add_argument("--batch_variable",  default="SMCENTER")
    ap.add_argument("--rin_threshold",   type=float, default=6.5)
    ap.add_argument("--outlier_sd",      type=float, default=2.0)
    ap.add_argument("--skip_qc",         action="store_true")
    ap.add_argument("--seed",            type=int, default=42)
    return ap.parse_args()


def assert_aligned(X, metadata, y, step=""):
    meta_ids = metadata.set_index('sample_id').index \
        if 'sample_id' in metadata.columns \
        else metadata.index
    y_ids = y.set_index('sample_id').index \
        if 'sample_id' in y.columns \
        else y.index

    assert list(X.index) == list(meta_ids), \
        f"[{step}] X and metadata sample order mismatch."
    assert list(X.index) == list(y_ids), \
        f"[{step}] X and y sample order mismatch!"
    print(f"  Alignment verified ({step}): all files in same sample order")


def validate_combat_output(X_corrected):
    issues = []
    if X_corrected.isna().any().any():
        issues.append(f"Contains {X_corrected.isna().sum().sum():,} NaN values")
    if np.isinf(X_corrected.values).any():
        issues.append(f"Contains {np.isinf(X_corrected.values).sum():,} infinite values")
    if (X_corrected < 0).any().any():
        issues.append(f"Contains {(X_corrected < 0).sum().sum():,} negative values")
    data_min = X_corrected.min().min()
    data_max = X_corrected.max().max()
    if data_max > 25:
        issues.append(f"Maximum value ({data_max:.2f}) is unusually high")
    if data_min < -1:
        issues.append(f"Minimum value ({data_min:.2f}) is unusually low")
    return issues, data_min, data_max


def plot_pca_after_combat(X_corrected, metadata, batch_variable, outdir,
                          n_top_genes=2000):
    gene_var = X_corrected.var(axis=0)
    topk     = min(n_top_genes, gene_var.shape[0])
    top_genes = gene_var.sort_values(ascending=False).head(topk).index
    Xp = X_corrected[top_genes].values

    Z    = StandardScaler().fit_transform(Xp)
    n_pcs = min(3, Z.shape[1], Z.shape[0] - 1)
    pca  = PCA(n_components=n_pcs, random_state=42)
    PCs  = pca.fit_transform(Z)
    ev   = pca.explained_variance_ratio_ * 100

    meta_indexed = metadata.set_index('sample_id').reindex(X_corrected.index)
    batch_labels = meta_indexed[batch_variable].astype(str).values

    palette    = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    categories = [c for c in pd.Categorical(batch_labels).categories
                  if c != 'nan']
    pairs      = [(0, 1), (0, 2), (1, 2)][:min(3, n_pcs)]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, pairs):
        for idx, cat in enumerate(categories):
            mask = batch_labels == cat
            ax.scatter(PCs[mask, i], PCs[mask, j],
                       s=30, alpha=0.6,
                       color=palette[idx % len(palette)],
                       edgecolors='black', linewidth=0.3,
                       label=cat)
        ax.set_xlabel(f'PC{i+1} ({ev[i]:.2f}%)', fontsize=20, fontweight='bold')
        ax.set_ylabel(f'PC{j+1} ({ev[j]:.2f}%)', fontsize=20, fontweight='bold')
        ax.set_title(f'PC{i+1} vs PC{j+1}', fontsize=21, fontweight='bold')
        ax.legend(title=batch_variable, fontsize=13, title_fontsize=10)
        ax.tick_params(axis='both', labelsize=15)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'PCA After ComBat — Top {topk} Genes (colored by {batch_variable})',
        fontsize=20, fontweight='bold')
    plt.tight_layout()
    out_path = Path(outdir) / 'pca_after_combat.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Post-ComBat PCA saved to: {out_path}")


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    np.random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    print(f"Output directory:  {args.outdir}")
    print(f"Batch variable:    {args.batch_variable}")
    print(f"RIN threshold:     {args.rin_threshold}")
    print(f"Outlier SD:        {args.outlier_sd}")

    # STEP 1: Load
    print("\n Loading data...")
    try:
        X        = pd.read_csv(args.X_log2, index_col=0)
        metadata = pd.read_csv(args.metadata)
        y        = pd.read_csv(args.y)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  X:        {X.shape[0]:,} samples × {X.shape[1]:,} genes")
    print(f"  metadata: {len(metadata):,} samples")
    print(f"  y:        {len(y):,} samples")

    n_samples_input = X.shape[0]
    n_genes_input   = X.shape[1]

    # STEP 2: Align 
    print("\nAligning data across files...")

    X.index              = X.index.astype(str)
    X.index.name         = 'sample_id'
    metadata['sample_id'] = metadata['sample_id'].astype(str)
    y['sample_id']        = y['sample_id'].astype(str)

    common = set(X.index) & set(metadata['sample_id']) & set(y['sample_id'])
    if len(common) == 0:
        print("ERROR: No common samples found!", file=sys.stderr)
        sys.exit(1)

    common = sorted(common)
    print(f"  Common samples: {len(common):,} ")

    X        = X.loc[common]
    metadata = metadata.set_index('sample_id').loc[common].reset_index()
    y        = y.set_index('sample_id').loc[common].reset_index()

    assert_aligned(X, metadata, y, step="after initial alignment")

    # STEP 3: Prepare batch variable
    print(f"\nPreparing batch variable: {args.batch_variable}")

    if args.batch_variable not in metadata.columns:
        print(f"ERROR: '{args.batch_variable}' not found in metadata!",
              file=sys.stderr)
        print(f"  Available: {list(metadata.columns)}", file=sys.stderr)
        sys.exit(1)

    batch        = metadata[args.batch_variable].values
    batch_counts = pd.Series(batch).value_counts()

    print("  Batch distribution:")
    for b, count in batch_counts.items():
        print(f"    {b}: {count:,} samples")

    small_batches        = batch_counts[batch_counts < 2]
    n_small_batch_removed = 0

    if len(small_batches) > 0:
        print(f"\n  WARNING: {len(small_batches)} batch(es) with < 2 samples — removing")
        keep_mask             = ~metadata[args.batch_variable].isin(small_batches.index)
        n_small_batch_removed = (~keep_mask).sum()

        metadata = metadata[keep_mask].reset_index(drop=True)
        X        = X.loc[metadata['sample_id']]

        y = y.set_index('sample_id').loc[X.index].reset_index()
        batch = metadata[args.batch_variable].values

        print(f"  Removed {n_small_batch_removed:,} samples. Remaining: {len(X):,}")

        assert_aligned(X, metadata, y, step="after small-batch removal")

    # STEP 4: Run ComBat
    print("\nRunning ComBat batch correction...")

    try:
        from combat.pycombat import pycombat
    except ImportError:
        print("ERROR: ComBat library not installed.", file=sys.stderr)
        X.to_csv(f"{args.outdir}/X_combat_qc.csv")
        metadata.to_csv(f"{args.outdir}/metadata_qc.csv", index=False)
        y.to_csv(f"{args.outdir}/y_qc.csv", index=False)
        print("Saved uncorrected data. Data is NOT batch-corrected.")
        sys.exit(1)

    print(f"  Input: {X.shape[0]:,} samples × {X.shape[1]:,} genes")

    X_T = X.T
    print(f"  Transposed: {X_T.shape[0]:,} genes × {X_T.shape[1]:,} samples")

    # Cleaning
    if X_T.isna().any().any():
        n_nan = X_T.isna().sum().sum()
        print(f"  Filling {n_nan:,} NaN values with 0")
        X_T = X_T.fillna(0)

    if np.isinf(X_T.values).any():
        n_inf = np.isinf(X_T.values).sum()
        print(f"  Replacing {n_inf:,} infinite values with 0")
        X_T = X_T.replace([np.inf, -np.inf], 0)

    gene_std  = X_T.std(axis=1)
    zero_var  = gene_std == 0
    n_zero_var = 0
    if zero_var.any():
        n_zero_var = zero_var.sum()
        print(f"  Removing {n_zero_var:,} zero-variance genes")
        X_T = X_T.loc[~zero_var]

    print(f"  Final input: {X_T.shape[0]:,} genes × {X_T.shape[1]:,} samples")

    combat_success = False
    try:
        print("\n  Calling ComBat (attempt 1 — standard parameters)...")
        X_combat       = pycombat(data=X_T, batch=batch)
        combat_success = True
        print("  ComBat completed successfully!")
    except Exception as e1:
        print(f"  Attempt 1 failed: {e1}")
        try:
            print("\n  Calling ComBat (attempt 2 — par_prior=False)...")
            X_combat       = pycombat(data=X_T, batch=batch, par_prior=False)
            combat_success = True
            print("  ComBat completed")
        except Exception as e2:
            print(f"  Attempt 2 failed: {e2}", file=sys.stderr)
            print("ERROR: Both ComBat attempts failed!", file=sys.stderr)
            X.to_csv(f"{args.outdir}/X_combat_qc.csv")
            metadata.to_csv(f"{args.outdir}/metadata_qc.csv", index=False)
            y.to_csv(f"{args.outdir}/y_qc.csv", index=False)
            print("Saved uncorrected data.")
            sys.exit(1)

    X_corrected         = X_combat.T
    X_corrected.index   = X.index
    X_corrected.columns = X_T.index
    print(f"\n  Output: {X_corrected.shape[0]:,} samples × "
          f"{X_corrected.shape[1]:,} genes")

    print("\n  Validating ComBat output...")
    issues, data_min, data_max = validate_combat_output(X_corrected)
    print(f"    Data range: [{data_min:.2f}, {data_max:.2f}]")
    if issues:
        for issue in issues:
            print(f"    WARNING: {issue}")
    else:
        print("    All validation checks passed")

    # STEP 5: QC
    if args.skip_qc:
        print("\nSkipping QC (--skip_qc flag set)")
        n_rin_removed = n_outliers_removed = 0
    else:
        print("\nQuality control filtering...")
        n_before_qc = len(X_corrected)
        n_rin_removed = 0

        if 'SMRIN' in metadata.columns:
            print(f"\n  [QC 1] RIN filter (threshold = {args.rin_threshold})...")
            rin      = pd.to_numeric(metadata['SMRIN'], errors='coerce')
            good_rin = rin >= args.rin_threshold
            n_rin_removed = (~good_rin).sum()

            if n_rin_removed > 0:
                print(f"    Removing {n_rin_removed:,} samples with RIN < "
                      f"{args.rin_threshold}")
                metadata     = metadata[good_rin].reset_index(drop=True)
                X_corrected  = X_corrected.loc[metadata['sample_id']]
                # FIX 3 (repeated): align y positionally after RIN filter
                y = y.set_index('sample_id').loc[X_corrected.index].reset_index()
                assert_aligned(X_corrected, metadata, y, step="after RIN filter")
            else:
                print(f"    All samples pass RIN filter")
        else:
            print("\n  [QC 1] SMRIN column not found, skipping RIN filter")

        print("\n  [QC 2] Outlier detection...")
        medians   = X_corrected.median(axis=1)
        z_scores  = (medians - medians.mean()) / medians.std()
        outliers  = np.abs(z_scores) > args.outlier_sd
        n_outliers_removed = outliers.sum()

        if n_outliers_removed > 0:
            print(f"    Removing {n_outliers_removed:,} outlier samples "
                  f"(|z| > {args.outlier_sd})")
            X_corrected = X_corrected[~outliers]
            metadata    = metadata[metadata['sample_id'].isin(X_corrected.index)]
            # FIX 3 (repeated): align y positionally after outlier filter
            y = y.set_index('sample_id').loc[X_corrected.index].reset_index()
            assert_aligned(X_corrected, metadata, y, step="after outlier filter")
        else:
            print(f"    No outliers detected")

        n_after_qc  = len(X_corrected)
        n_qc_removed = n_before_qc - n_after_qc
        print(f"\n  QC complete: {n_before_qc:,} → {n_after_qc:,} samples "
              f"({n_qc_removed:,} removed)")

    # Save
    print(f"\nSaving results to {args.outdir}...")
    X_corrected.to_csv(f"{args.outdir}/X_combat_qc.csv")
    metadata.to_csv(f"{args.outdir}/metadata_qc.csv", index=False)
    y.to_csv(f"{args.outdir}/y_qc.csv", index=False)

    # PCA plot
    print("\nGenerating post-ComBat PCA plot...")
    plot_pca_after_combat(X_corrected, metadata,
                          args.batch_variable, args.outdir)

    # Summary
    n_samples_output  = len(X_corrected)
    n_samples_removed = n_samples_input - n_samples_output
    n_genes_output    = X_corrected.shape[1]
    n_genes_removed   = n_genes_input - n_genes_output

    print("\nSUMMARY")
    print(f"\nSamples:")
    print(f"  Input:   {n_samples_input:,}")
    print(f"  Output:  {n_samples_output:,}")
    print(f"  Removed: {n_samples_removed:,} "
          f"({100*n_samples_removed/n_samples_input:.1f}%)")
    print(f"    - Small batches: {n_small_batch_removed:,}")
    print(f"    - Low RIN:       {n_rin_removed:,}")
    print(f"    - Outliers:      {n_outliers_removed:,}")
    print(f"\nGenes:")
    print(f"  Input:   {n_genes_input:,}")
    print(f"  Output:  {n_genes_output:,}")
    print(f"  Removed: {n_genes_removed:,} ({100*n_genes_removed/n_genes_input:.1f}%)")
    print(f"    - Zero-variance: {n_zero_var:,}")
    print(f"\nData Quality:")
    print(f"  Batch-corrected: YES")
    print(f"  RIN filtered:    "
          f"{'YES' if not args.skip_qc and 'SMRIN' in metadata.columns else 'NO'}")
    print(f"  Outlier filtered: {'YES' if not args.skip_qc else 'NO'}")


if __name__ == "__main__":
    main()