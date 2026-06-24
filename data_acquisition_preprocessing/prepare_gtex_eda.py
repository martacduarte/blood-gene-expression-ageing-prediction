#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, argparse, textwrap, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# helpers
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def age_range_to_midpoint(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.upper() == "NA":
        return None
    if s.endswith("+"):
        nums = re.sub(r"\D", "", s)
        return float(int(nums) + 5) if nums else None
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    try:
        return float(s)
    except:
        return None

def normalize_sex_series(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    tmp = s.astype(str).str.strip().str.lower()
    mapped = tmp.replace({
        "1": "male", "2": "female",
        "m": "male", "f": "female",
        "male": "male", "female": "female",
        "nan": np.nan, "na": np.nan, "": np.nan, "none": np.nan, "unknown": np.nan
    })
    mapped = mapped.map(lambda x: {"male": "Male", "female": "Female"}.get(x, np.nan))
    return mapped

def read_gct(gct_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gct_path, sep="\t", skiprows=2)
    if "Name" not in df.columns:
        raise ValueError("GCT missing 'Name' column.")
    df = df.set_index("Name")
    if "Description" in df.columns:
        df = df.drop(columns=["Description"])
    return df  # genes x SAMPID

def sampid_to_subjid(sampid: str) -> str:
    parts = str(sampid).split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else str(sampid)

def save_head_csv(df: pd.DataFrame, outpath: Path, n=5, index=False):
    try:
        df.head(n).to_csv(outpath, index=index)
    except Exception:
        df.head(n).to_csv(outpath)

def safe_mode(s: pd.Series):
    s = s.dropna()
    if s.empty: return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan

def missing_report(df: pd.DataFrame, col_thresh=0.2, row_thresh=0.1):
    col_missing = df.isna().sum()
    col_missing_frac = col_missing / len(df) if len(df) else 0
    cols_with_missing = int((col_missing > 0).sum())
    rows_missing_cnt = df.isna().sum(axis=1)
    rows_with_missing = int((rows_missing_cnt > 0).sum())
    flagged_cols = int((col_missing_frac > col_thresh).sum())
    flagged_rows = int(((rows_missing_cnt / max(1, df.shape[1])) > row_thresh).sum())
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "cols_with_missing": cols_with_missing,
        "cols_with_missing_pct": float(cols_with_missing / max(1, df.shape[1]) * 100),
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": float(rows_with_missing / max(1, df.shape[0]) * 100),
        "flagged_cols_over_thresh": flagged_cols,
        "flagged_rows_over_thresh": flagged_rows,
        "col_thresh": float(col_thresh),
        "row_thresh": float(row_thresh),
    }
    details = pd.DataFrame({
        "missing_count": col_missing,
        "missing_pct": col_missing_frac * 100.0 if len(df) else 0
    }).sort_values("missing_pct", ascending=False)
    return summary, details

# plot functions
def plot_age_distribution_enhanced(age_series, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(age_series.dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(age_series.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age_series.mean():.1f}')
    axes[0, 0].axvline(age_series.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {age_series.median():.1f}')
    axes[0, 0].set_xlabel('Age (years)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Age Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    bp = axes[0, 1].boxplot(age_series.dropna(), vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_ylabel('Age (years)', fontsize=12)
    axes[0, 1].set_title('Age Boxplot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Density plot
    age_series.dropna().plot(kind='density', ax=axes[1, 0], linewidth=2)
    axes[1, 0].set_xlabel('Age (years)', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Age Density Plot', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(axes[1, 0].get_lines()[0].get_xdata(), 
                             axes[1, 0].get_lines()[0].get_ydata(), 
                             alpha=0.3)
    
    # Summary statistics table
    stats = age_series.describe()
    table_data = [
        ['Count', f'{stats["count"]:.0f}'],
        ['Mean', f'{stats["mean"]:.2f}'],
        ['Std', f'{stats["std"]:.2f}'],
        ['Min', f'{stats["min"]:.2f}'],
        ['25%', f'{stats["25%"]:.2f}'],
        ['Median', f'{stats["50%"]:.2f}'],
        ['75%', f'{stats["75%"]:.2f}'],
        ['Max', f'{stats["max"]:.2f}']
    ]
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=table_data, colLabels=['Statistic', 'Value'],
                             cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for i in range(len(table_data) + 1):
        table[(i, 0)].set_facecolor('#E8E8E8' if i % 2 == 0 else 'white')
        table[(i, 1)].set_facecolor('#E8E8E8' if i % 2 == 0 else 'white')
    axes[1, 1].set_title('Age Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'age_distribution_complete.png', dpi=300, bbox_inches='tight')
    print("creating age distribution")
    plt.close()
    
def plot_sample_overview(X_shape, y_shape, meta_shape, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Expression\n(X)', 'Outcomes\n(y)', 'Metadata']
    samples = [X_shape[0], y_shape[0], meta_shape[0]]
    features = [X_shape[1], y_shape[1], meta_shape[1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, samples, width, label='Samples', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, features, width, label='Features/Columns', color='coral', edgecolor='black')
    
    ax.set_xlabel('Dataset Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Dimensions Overview', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'dataset_dimensions.png', dpi=300, bbox_inches='tight')
    print("creating dataset dimensions")
    plt.close()

def plot_expression_overview(X_df, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    n_samples_plot = min(20, X_df.shape[0])
    rng = np.random.default_rng(42) 
    sample_indices = rng.choice(X_df.shape[0], n_samples_plot, replace=False)
    data_to_plot = [X_df.iloc[i].values for i in sample_indices]
    
    bp = axes[0, 0].boxplot(data_to_plot, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    axes[0, 0].set_xlabel('Sample', fontsize=11)
    axes[0, 0].set_ylabel('Expression (TPM)', fontsize=11)
    axes[0, 0].set_title(f'Expression Distribution Across {n_samples_plot} Random Samples', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xticklabels([])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Gene expression means
    gene_means = X_df.mean(axis=0)
    axes[0, 1].hist(gene_means, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Mean Expression (TPM)', fontsize=11)
    axes[0, 1].set_ylabel('Number of Genes', fontsize=11)
    axes[0, 1].set_title('Distribution of Gene Mean Expression', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(gene_means.median(), color='red', linestyle='--', 
                       linewidth=2, label=f'Median: {gene_means.median():.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gene expression variances
    gene_vars = X_df.var(axis=0)
    axes[1, 0].hist(np.log10(gene_vars + 1), bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('log10(Variance + 1)', fontsize=11)
    axes[1, 0].set_ylabel('Number of Genes', fontsize=11)
    axes[1, 0].set_title('Distribution of Gene Expression Variance', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sparsity plot
    zero_pct = (X_df == 0).sum(axis=0) / X_df.shape[0] * 100
    axes[1, 1].hist(zero_pct, bins=50, edgecolor='black', alpha=0.7, color='plum')
    axes[1, 1].set_xlabel('Percentage of Zero Values (%)', fontsize=11)
    axes[1, 1].set_ylabel('Number of Genes', fontsize=11)
    axes[1, 1].set_title('Gene Expression Sparsity', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(zero_pct.median(), color='red', linestyle='--', 
                       linewidth=2, label=f'Median: {zero_pct.median():.1f}%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'expression_overview.png', dpi=300, bbox_inches='tight')
    print("creating expression overview")
    plt.close()

def plot_pca_enhanced(PCs, ev, color, color_name, topk, fig_dir):
    n_pcs = PCs.shape[1]
    
    # Scree plot with cumulative variance
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(1, len(ev) + 1)
    cumsum = np.cumsum(ev)
    
    ax.bar(x_pos, ev, alpha=0.7, color='steelblue', edgecolor='black', label='Individual')
    ax.plot(x_pos, cumsum, 'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
    
    ax.set_xlabel('Principal Component', fontsize=20, fontweight='bold')
    ax.set_ylabel('Variance Explained (%)', fontsize=20, fontweight='bold')
    ax.set_title(f'PCA Scree Plot (top {topk} most variable genes)', fontsize=21, fontweight='bold')
    ax.legend(fontsize=13)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'pca_scree_enhanced.png', dpi=300, bbox_inches='tight')
    print("creating pca enhaneced")
    plt.close()
    
    # PC scatter plots
    if n_pcs >= 2:
        fig, axes = plt.subplots(1, 3 if n_pcs >= 3 else 2, figsize=(18 if n_pcs >= 3 else 12, 5))
        if n_pcs < 3:
            axes = [axes[0], axes[1], None]
        
        def plot_pc_scatter(ax, i, j):

            if color is None:
                ax.scatter(PCs[:, i], PCs[:, j], s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
            else:
                categories = [c for c in pd.Categorical(color).categories if c != 'nan']
                palette = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800',
                           '#00BCD4', '#F44336', '#8BC34A', '#795548', '#607D8B']
                for idx, cat in enumerate(categories):
                    mask = np.array(color) == cat
                    ax.scatter(PCs[mask, i], PCs[mask, j], s=30,
                               color=palette[idx % len(palette)],
                               alpha=0.6, edgecolors='black', linewidth=0.5, label=cat)
                if len(categories) <= 10:
                    ax.legend(title=color_name, fontsize=13, title_fontsize=14, loc='best')

            ax.set_xlabel(f'PC{i+1} ({ev[i]:.2f}%)', fontsize=20, fontweight='bold')
            ax.set_ylabel(f'PC{j+1} ({ev[j]:.2f}%)', fontsize=20, fontweight='bold')
            ax.tick_params(axis='both', labelsize=20)
            ax.grid(True, alpha=0.3)
        
        plot_pc_scatter(axes[0], 0, 1)
        axes[0].set_title('PC1 vs PC2', fontsize=20, fontweight='bold')
        
        if n_pcs >= 3:
            plot_pc_scatter(axes[1], 0, 2)
            axes[1].set_title('PC1 vs PC3', fontsize=20, fontweight='bold')
            
            plot_pc_scatter(axes[2], 1, 2)
            axes[2].set_title('PC2 vs PC3', fontsize=20, fontweight='bold')
        else:

            axes[1].bar([1, 2], ev[:2], color=['steelblue', 'coral'], edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Principal Component', fontsize=20, fontweight='bold')
            axes[1].set_ylabel('Variance Explained (%)', fontsize=20, fontweight='bold')
            axes[1].set_title('Variance Explained by PCs', fontsize=20, fontweight='bold')
            axes[1].set_xticks([1, 2])
            axes[1].tick_params(axis='both', labelsize=15)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        color_label = f' (colored by {color_name})' if color is not None else ''
        fig.suptitle(f'PCA Analysis on Top {topk} Genes{color_label}', 
                    fontsize=21, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'pca_scatter_plots.png', dpi=300, bbox_inches='tight')
        print("creating pca scater")
        plt.close()

# main pipeline
def parse_args():
    ap = argparse.ArgumentParser(description="Prepare GTEx, EDA, Clean missingness")
    
    ap.add_argument("--gct", required=True)
    ap.add_argument("--phenotypes", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--tissue_filter", default="Whole Blood")
    ap.add_argument("--limit_samples", type=int, default=0)

    # EDA
    ap.add_argument("--eda_outdir", default=None)
    ap.add_argument("--pca_color", default="SMCENTER")
    ap.add_argument("--cat_cols", default="SEX,SMCENTER,SMGEBTCH")
    ap.add_argument("--cont_cols", default="SMRIN")
    ap.add_argument("--max_features_pca", type=int, default=2000)

    
    ap.add_argument("--expr_subset_cols", type=int, default=1000,
                    help="How many random gene columns to sample for skew/kurt (default 1000)")
    ap.add_argument(
        "--make_log2p1_preview",
        action="store_true"
    )
    ap.add_argument(
        "--make_log1p_preview",
        action="store_true",
        help=argparse.SUPPRESS
    )

    # Cleaning thresholds
    ap.add_argument("--meta_row_thresh", type=float, default=0.10)
    ap.add_argument("--meta_impute", action="store_true")
    ap.add_argument("--x_col_thresh", type=float, default=0.10)
    ap.add_argument("--x_fill_remaining_zeros", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir); ensure_dir(outdir)

    print("DATA PREPARATION")
    
    # Phenotypes
    ph = pd.read_csv(args.phenotypes, sep="\t")
    ph.columns = [c.strip().upper() for c in ph.columns]
    if "SUBJID" not in ph.columns or "AGE" not in ph.columns:
        raise ValueError("Phenotypes must contain SUBJID and AGE.")
    ph["AGE_NUM"] = ph["AGE"].apply(age_range_to_midpoint)
    if "SEX" in ph.columns:
        ph["SEX"] = normalize_sex_series(ph["SEX"])

    # Sample attributes
    try:
        sa = pd.read_csv(args.samples)
    except Exception:
        sa = pd.read_csv(args.samples, sep="\t")
    sa.columns = [c.strip().upper() for c in sa.columns]
    if "SAMPID" not in sa.columns:
        raise ValueError("Sample attributes must contain SAMPID.")
    if "SUBJID" not in sa.columns:
        sa["SUBJID"] = sa["SAMPID"].apply(sampid_to_subjid)
    if "SEX" in sa.columns:
        sa["SEX"] = normalize_sex_series(sa["SEX"])

    # Tissue filter (SMTS / SMTSD contains)
    if args.tissue_filter:
        tf = args.tissue_filter.lower()
        tissue_cols = [c for c in ["SMTS", "SMTSD"] if c in sa.columns]
        if tissue_cols:
            mask = False
            for tc in tissue_cols:
                mask = mask | sa[tc].astype(str).str.lower().str.contains(tf, regex=False)
            sa = sa.loc[mask]
            print(f"Tissue filter '{args.tissue_filter}' kept {len(sa)} samples")
        else:
            print("[warn] SMTS/SMTSD not found; tissue filter ignored")

    # Expression (genes x samples)
    Xg = read_gct(Path(args.gct))
    
    # Check for negative values
    if (Xg < 0).any().any():
        n_negative = (Xg < 0).sum().sum()
        print(f"\n[ERROR] Found {n_negative:,} negative values in expression data!")
        raise ValueError("Invalid expression values detected (negative TPM values)")
    
    # Check for reasonable range
    max_val = Xg.max().max()
    if max_val > 1e6:
        print(f"[WARN] Very high expression values detected (max: {max_val:.2e})")
        print("This might indicate non-TPM normalized data. Proceeding anyway...")
    
    before = Xg.shape[1]
    Xg = Xg.loc[:, Xg.columns.isin(sa["SAMPID"])]
    print(f"Aligned expression: kept {Xg.shape[1]:,} / {before:,} samples")
    
    if args.limit_samples and args.limit_samples > 0:
        Xg = Xg.iloc[:, :args.limit_samples]
        print(f"Quick mode: using first {args.limit_samples:,} samples only")
    
    X = Xg.T
    X.index.name = "sample_id"

    # y (sample_id, age)
    sid_to_subj = sa.set_index("SAMPID")["SUBJID"]
    subjid_aligned = sid_to_subj.reindex(X.index)
    
    # Check for sample ID mismatches
    n_missing_subj = subjid_aligned.isna().sum()
    if n_missing_subj > 0:
        pct_missing = 100 * n_missing_subj / len(subjid_aligned)
        print(f"\n[WARN] {n_missing_subj} ({pct_missing:.1f}%) samples missing SUBJID mapping")
        if pct_missing > 50:
            print("[ERROR] More than 50% of samples have no SUBJID!")
            raise ValueError("Sample ID mismatch")
        elif pct_missing > 10:
            print("[WARN] This is concerning - verify sample ID formats match across files")
    
    y = pd.DataFrame({"sample_id": X.index, "SUBJID": subjid_aligned.values})
    y = y.merge(ph[["SUBJID", "AGE_NUM"]], on="SUBJID", how="left").rename(columns={"AGE_NUM":"age"})
    y = y.dropna(subset=["age"])
    
    # Check for sufficient age data
    if len(y) == 0:
        print("\n[ERROR] No valid age values found!")
        raise ValueError("No valid age values - cannot proceed")
    
    if len(y) < 10:
        print(f"\n[WARN] Only {len(y)} samples have valid ages!")
        if len(y) < 5:
            raise ValueError(f"Too few samples ({len(y)}) - need at least 5 for meaningful analysis")

    # metadata
    keep = ["SAMPID","SUBJID"] + [c for c in ["SEX","SMTS","SMTSD","SMRIN","SMCENTER","SMGEBTCH","SMGEBTCHT"] if c in sa.columns]
    meta = sa[keep].rename(columns={"SAMPID":"sample_id"})
    if "SEX" not in meta.columns and "SEX" in ph.columns:
        meta = meta.merge(ph[["SUBJID","SEX"]], on="SUBJID", how="left")

    # Final SEX normalization on the assembled metadata (safety)
    if "SEX" in meta.columns:
        meta["SEX"] = normalize_sex_series(meta["SEX"])

    # Align X to samples that have age
    X = X.loc[y["sample_id"]]
    meta = meta.set_index("sample_id").loc[X.index].reset_index()

    # Harmonization accounting
    expr_n = int(Xg.shape[1])
    meta_n = int(sa["SAMPID"].nunique())
    y_n    = int(ph["SUBJID"].nunique())
    common_n = int(X.shape[0])

    loss_vs_expr = 100.0 * (1.0 - common_n / max(1, expr_n))
    loss_vs_meta = 100.0 * (1.0 - common_n / max(1, meta_n))
    
    print(f"\n Final dataset:")
    print(f"  - Samples: {common_n:,}")
    print(f"  - Genes: {X.shape[1]:,}")
    print(f"  - Loss vs expression: {loss_vs_expr:.2f}%")
    print(f"  - Loss vs metadata: {loss_vs_meta:.2f}%")

    harmo_tbl = pd.DataFrame([
        {"source": "expression_after_attr_filter", "n": expr_n},
        {"source": "metadata_after_tissue_filter", "n": meta_n},
        {"source": "phenotypes_donors",           "n": y_n},
        {"source": "common_sample_ids",           "n": common_n,
         "loss_pct_vs_expression": loss_vs_expr,
         "loss_pct_vs_metadata":  loss_vs_meta}
    ])

    # Save prepared
    X_out = outdir / "X.csv"
    X_log_out = outdir / "X_log2.csv" 
    y_out = outdir / "y.csv"
    meta_out = outdir / "metadata.csv"
    
    # Save raw TPM (for reference)
    X.to_csv(X_out)
    
    # Apply log2(TPM+1) transformation and save
    print("\nApplying log2(TPM+1) transformation...")
    X_log = np.log2(X.astype(float) + 1.0)
    X_log.to_csv(X_log_out)
    
    # Validation checks
    print(f"  Raw TPM range: {X.min().min():.2f} - {X.max().max():.2f}")
    print(f"  Log2 range: {X_log.min().min():.2f} - {X_log.max().max():.2f}")
    print(f"  Median of gene max values (raw): {X.max(axis=0).median():.2f}")
    print(f"  Median of gene max values (log2): {X_log.max(axis=0).median():.2f}")
    
    # Save metadata
    y[["sample_id","age"]].to_csv(y_out, index=False)
    meta.to_csv(meta_out, index=False)
    
    print(f"\nSaved prepared data:")
    print(f"  - {X_out} (raw TPM)")
    print(f"  - {X_log_out} (log2-transformed)")
    print(f"  - {y_out}")
    print(f"  - {meta_out}")

    # EDA
    print("EXPLORATORY DATA ANALYSIS")
    
    eda_dir = Path(args.eda_outdir or f"runs/eda_full_gtex_{ts}")
    fig_dir = eda_dir / "figs"; tbl_dir = eda_dir / "tables"
    ensure_dir(fig_dir); ensure_dir(tbl_dir); ensure_dir(eda_dir)

    # save harmonization counts
    harmo_tbl.to_csv(tbl_dir / "harmonization_counts.csv", index=False)

    # Heads & shapes
    save_head_csv(pd.read_csv(y_out), tbl_dir / "y_head.csv")
    save_head_csv(pd.read_csv(meta_out), tbl_dir / "metadata_head.csv")
    pd.read_csv(X_out, nrows=5).iloc[:, :11].to_csv(tbl_dir / "X_head_5x10.csv", index=False)

    # Load for analysis
    X_df = pd.read_csv(X_out, index_col=0)
    y_df = pd.read_csv(y_out)
    meta_df = pd.read_csv(meta_out)

    # Shapes
    with open(tbl_dir / "X_shape.json", "w") as f:
        json.dump({"n_samples": int(X_df.shape[0]), "n_genes": int(X_df.shape[1])}, f, indent=2)
    
    # plots
    plot_sample_overview(X_df.shape, y_df.shape, meta_df.shape, fig_dir)
    
    plot_expression_overview(X_df, fig_dir)
    
    plot_age_distribution_enhanced(pd.to_numeric(y_df["age"], errors="coerce"), fig_dir)

    # Expression EDA
    def _skew_kurt(series):
        x = pd.to_numeric(series, errors="coerce").dropna()
        if len(x) < 3: return np.nan, np.nan
        return float(pd.Series(x).skew()), float(pd.Series(x).kurt())

    def _sample_columns(cols, k):
        cols = list(cols)
        if len(cols) <= k: return cols
        import random; random.seed(42)
        return random.sample(cols, k)

    subset_cols = _sample_columns(X_df.columns, args.expr_subset_cols)
    xs = X_df[subset_cols]
    sk_rows = []
    for c in xs.columns:
        s, k = _skew_kurt(xs[c])
        sk_rows.append({"gene": c, "skewness": s, "kurtosis": k})
    pd.DataFrame(sk_rows).to_csv(tbl_dir / "X_random_subset_skew_kurt.csv", index=False)

    # log2(TPM+1) preview
    do_log_preview = bool(getattr(args, "make_log2p1_preview", False) or getattr(args, "make_log1p_preview", False))
    if do_log_preview:
        vals = X_df.to_numpy().astype(float).ravel()
        vals = vals[~np.isnan(vals)]
        if vals.size > 200_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(vals.size, size=200_000, replace=False)
            vals = vals[idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(vals, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title("Raw Expression Values", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("TPM", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        vals_log = np.log2(vals + 1.0)
        axes[1].hist(vals_log, bins=100, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_title("log2(TPM + 1) Transformed", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("log2(TPM + 1)", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "expression_transformation_comparison.png", dpi=300, bbox_inches='tight')
        print("creating transfornation")

        plt.close()

    # Missingness before cleaning
    meta_before_summary, meta_before_details = missing_report(meta_df)
    x_before_summary, x_before_details = missing_report(X_df)
    meta_before_details.to_csv(tbl_dir / "missing_metadata_before.csv")
    x_before_details.to_csv(tbl_dir / "missing_X_before.csv")

    # Age summary
    a = pd.to_numeric(y_df["age"], errors="coerce")
    a.describe().to_csv(tbl_dir / "age_summary.csv")

    # Metadata plots
    cat_cols = [c.strip() for c in args.cat_cols.split(",") if c.strip()]
    cont_cols = [c.strip() for c in args.cont_cols.split(",") if c.strip()]
    
    for c in cat_cols:
        if c in meta_df.columns:
            vc = meta_df[c].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(vc)), vc.values, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=20)
            ax.set_xlabel(c, fontsize=23, fontweight='bold')
            ax.set_ylabel('Count', fontsize=23, fontweight='bold')
            ax.set_title(f'{c} Distribution', fontsize=24, fontweight='bold')
            ax.tick_params(axis='y', labelsize=20)
            ax.grid(True, alpha=0.3, axis='y')
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=20)
            plt.tight_layout()
            plt.savefig(fig_dir / f"meta_{c}_counts.png", dpi=300, bbox_inches='tight')
            print("creating counts")

            plt.close()
    
    for c in cont_cols:
        if c in meta_df.columns:
            s = pd.to_numeric(meta_df[c], errors="coerce").dropna()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(s, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
            axes[0].set_xlabel(c, fontsize=20, fontweight='bold')
            axes[0].set_ylabel('Count', fontsize=20, fontweight='bold')
            axes[0].set_title(f'{c} Distribution', fontsize=21, fontweight='bold')
            axes[0].tick_params(axis='both', labelsize=15)
            axes[0].grid(True, alpha=0.3)
            bp = axes[1].boxplot(s, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            axes[1].set_ylabel(c, fontsize=20, fontweight='bold')
            axes[1].set_title(f'{c} Boxplot', fontsize=21, fontweight='bold')
            axes[1].tick_params(axis='both', labelsize=15)
            axes[1].grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(fig_dir / f"meta_{c}_distribution.png", dpi=300, bbox_inches='tight')
            print("creating distribution")

            plt.close()

    # Skewness & kurtosis
    num_meta = meta_df.select_dtypes(include=[np.number]).copy()
    if "age" not in num_meta.columns:
        num_meta["age"] = a.values
    sk = num_meta.apply(lambda col: skew(col.dropna().values) if col.dropna().nunique()>1 else np.nan)
    ku = num_meta.apply(lambda col: kurtosis(col.dropna().values, fisher=True) if col.dropna().nunique()>1 else np.nan)
    pd.DataFrame({"skewness": sk, "kurtosis": ku}).to_csv(tbl_dir / "skewness_kurtosis.csv")

    # Correlation heatmap
    if num_meta.shape[1] >= 2:
        corr = num_meta.corr(method="pearson")
        fig, ax = plt.subplots(figsize=(max(8, 0.8*num_meta.shape[1]), max(6, 0.8*num_meta.shape[1])))
        im = ax.imshow(corr.values, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=11)
        ax.set_xticks(range(corr.shape[1]))
        ax.set_yticks(range(corr.shape[0]))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(corr.index, fontsize=10)
        ax.set_title('Numeric Metadata Correlation Heatmap', fontsize=13, fontweight='bold')
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                text = ax.text(j, i, f'{corr.values[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "metadata_corr_heatmap.png", dpi=300, bbox_inches='tight')
        print("creating heatmap")

        plt.close()

    # PCA on top-variance genes
    try:
        gene_var = X_df.var(axis=0)
        topk = min(args.max_features_pca, gene_var.shape[0])
        top_genes = gene_var.sort_values(ascending=False).head(topk).index.tolist()
        Xp = X_df[top_genes]
        
        # Check dimensions for PCA
        if Xp.shape[1] < 2:
            print(f"[WARN] Too few genes ({Xp.shape[1]}) for PCA. Need at least 2. Skipping PCA.")
            raise Exception("Insufficient genes for PCA")
        
        if Xp.shape[0] < 3:
            print(f"[WARN] Too few samples ({Xp.shape[0]}) for PCA. Need at least 3. Skipping PCA.")
            raise Exception("Insufficient samples for PCA")

        Z = StandardScaler(with_mean=True, with_std=True).fit_transform(Xp.values)
        n_pcs = max(2, min(10, Xp.shape[1], Xp.shape[0] - 1))  # FIXED: Also check n_samples
        pca = PCA(n_components=n_pcs, random_state=42)
        PCs = pca.fit_transform(Z)
        ev = (pca.explained_variance_ratio_ * 100).tolist()

        color = meta_df[args.pca_color].astype(str).values if args.pca_color in meta_df.columns else None
        plot_pca_enhanced(PCs, ev, color, args.pca_color, topk, fig_dir)

        with open(tbl_dir / "pca_explained_variance.json", "w") as f:
            json.dump({"explained_variance_pct": ev, "topk_genes": int(topk)}, f, indent=2)

    except Exception as e:
        print(f"[warn] PCA step skipped: {e}")

    # EDA report
    eda_report = {
        "shapes": {"X": list(X_df.shape), "y": list(y_df.shape), "metadata": list(meta_df.shape)},
        "missing_metadata_before": meta_before_summary,
        "missing_X_before": x_before_summary,
        "harmonization": {
            "expression_after_attr_filter": expr_n,
            "metadata_after_tissue_filter": meta_n,
            "phenotypes_donors": y_n,
            "common_sample_ids": common_n,
            "loss_pct_vs_expression": loss_vs_expr,
            "loss_pct_vs_metadata":  loss_vs_meta
        },
        "notes": "EDA complete. See figs/ and tables/ for details."
    }
    with open(eda_dir / "report.json", "w") as f:
        json.dump(eda_report, f, indent=2)


    print(f"\nEDA complete. Results saved to: {eda_dir}")

    # Cleaning
    print("Data cleaning")

    X_log2_df = pd.read_csv(X_log_out, index_col=0) 

    meta_clean = meta_df.copy()
    row_missing_frac = meta_clean.isna().sum(axis=1) / max(1, meta_clean.shape[1])
    drop_mask = row_missing_frac > args.meta_row_thresh
    dropped_ids = meta_clean.loc[drop_mask, "sample_id"].astype(str).tolist() if "sample_id" in meta_clean.columns else []
    meta_clean = meta_clean.loc[~drop_mask].reset_index(drop=True)

    # Align X_log2 & y to remaining sample_id
    if "sample_id" in meta_df.columns:
        keep_ids = set(meta_clean["sample_id"].astype(str))
        X_clean = X_log2_df.loc[[i for i in X_log2_df.index.astype(str) if i in keep_ids]].copy()  # ← CHANGED: Use X_log2_df
        y_clean = y_df.loc[y_df["sample_id"].astype(str).isin(keep_ids)].copy()
    else:
        X_clean, y_clean = X_log2_df.copy(), y_df.copy()  # ← CHANGED: Use X_log2_df

    print(f"Dropped {len(dropped_ids)} samples due to missing metadata (>{args.meta_row_thresh*100:.0f}%)")

    # Impute metadata
    if args.meta_impute:
        num_cols = meta_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in meta_clean.columns if c not in num_cols]
        for c in num_cols:
            meta_clean[c] = pd.to_numeric(meta_clean[c], errors="coerce")
            meta_clean[c] = meta_clean[c].fillna(meta_clean[c].median())
        for c in cat_cols:
            meta_clean[c] = meta_clean[c].fillna(safe_mode(meta_clean[c]))
        print("Imputed remaining missing metadata values")

    # Handle any NaN in X
    if X_clean.isna().values.any():
        col_missing_frac = X_clean.isna().sum(axis=0) / max(1, X_clean.shape[0])
        drop_cols = list(col_missing_frac[col_missing_frac > args.x_col_thresh].index)
        if drop_cols:
            X_clean = X_clean.drop(columns=drop_cols)
            print(f"Dropped {len(drop_cols)} genes with >{args.x_col_thresh*100:.0f}% missing values")
        if args.x_fill_remaining_zeros:
            X_clean = X_clean.fillna(0.0)
            print("Filled remaining NaN values with 0")

    # Save cleaned
    Xc_out = outdir / "X_log2_clean.csv"
    yc_out = outdir / "y_clean.csv"
    mc_out = outdir / "metadata_clean.csv"
    X_clean.to_csv(Xc_out)
    y_clean.to_csv(yc_out, index=False)
    meta_clean.to_csv(mc_out, index=False)

    # Cleaning report
    meta_after_summary, _ = missing_report(meta_clean)
    x_after_summary, _ = missing_report(X_clean)

    cleaning = {
        "dropped_sample_ids": dropped_ids,
        "n_dropped_samples": len(dropped_ids),
        "meta_row_thresh": args.meta_row_thresh,
        "meta_impute": bool(args.meta_impute),
        "x_col_thresh": args.x_col_thresh,
        "x_fill_remaining_zeros": bool(args.x_fill_remaining_zeros),
        "metadata_before": meta_before_summary,
        "metadata_after": meta_after_summary,
        "X_before": x_before_summary,
        "X_after": x_after_summary,
        "outputs": {"X_clean": str(Xc_out), "y_clean": str(yc_out), "metadata_clean": str(mc_out)},
    }
    run_dir = Path(f"runs/prepare_clean_{ts}"); ensure_dir(run_dir)
    with open(run_dir / "cleaning_report.json", "w") as f:
        json.dump(cleaning, f, indent=2)

    print(f"\nClean files saved:")
    print(f"  - {Xc_out} (log2-transformed + filtered)")
    print(f"  - {yc_out}")
    print(f"  - {mc_out}")
    print(f"\nFull cleaning report: {run_dir / 'cleaning_report.json'}")

    print(f"\nFinal dataset: {X_clean.shape[0]:,} samples × {X_clean.shape[1]:,} genes")
    print(f"EDA outputs: {eda_dir}")
    print(f"Cleaned data: {outdir}")

if __name__ == "__main__":
    main()