#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import time
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet, ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed")


# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_system_info():
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }
    
    if HAS_PSUTIL:
        info['cpu_count'] = psutil.cpu_count(logical=False)
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        info['ram_total_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    else:
        info['cpu_count'] = 'N/A (psutil not installed)'
        info['cpu_count_logical'] = 'N/A'
        info['ram_total_gb'] = 'N/A'
    
    return info


def load_data(X_path, y_path):
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path)
    
    if "sample_id" not in y.columns or "age" not in y.columns:
        raise ValueError("y must contain: sample_id, age")
    
    common = sorted(X.index.astype(str).intersection(y["sample_id"].astype(str)))
    X = X.loc[common]
    y_aligned = y.set_index("sample_id").loc[common]["age"].astype(float).values
    
    return X, y_aligned, list(common)


def preprocess(X_train, X_test, log2p1=True, basic_filter=True, quantile_transform=False):    

    if log2p1:
        
        median_max = X_train.max(axis=0).median()
        data_max = X_train.max().max()
        data_min = X_train.min().min()
        
        already_log2 = (data_max < 25) or (data_min < -0.1)
        
        if already_log2:
            print("Data appears to be already log2-transformed")
            print(f"       Range: [{data_min:.2f}, {data_max:.2f}]")
            print(f"       Skipping log2 transformation")
        else:
            print("Data appears to be raw TPM")
            print(f"       Range: [{data_min:.2f}, {data_max:.2f}]")
            print(f"       Applying log2(TPM+1) transformation...")
            X_train = np.log2(X_train.astype(float) + 1.0)
            X_test = np.log2(X_test.astype(float) + 1.0)
            print(f"       After log2: [{X_train.min().min():.2f}, {X_train.max().max():.2f}]")
    

    if basic_filter:
        mean_expr = X_train.mean(axis=0)
        nonzero_frac = (X_train > 0).sum(axis=0) / len(X_train)
        
        # Count how many genes would be removed
        keep = (mean_expr >= 0.1) & (nonzero_frac >= 0.05)
        n_to_remove = (~keep).sum()
        pct_to_remove = 100 * n_to_remove / len(keep)
                
        already_filtered = pct_to_remove < 1.0
        
        if already_filtered:
            print(f"Basic filtering appears to be already done")
            print(f"       Would only remove {n_to_remove} genes ({pct_to_remove:.2f}%)")
            print(f"       Skipping basic filtering")
        else:
            print(f"Applying basic gene filtering...")
            print(f"       Removing {n_to_remove} genes ({pct_to_remove:.2f}%)")
            print(f"       Criteria: mean_expr >= 0.1 AND nonzero_frac >= 0.05")
            
            X_train = X_train.loc[:, keep]
            X_test = X_test.loc[:, X_train.columns]
            
            print(f"       Kept {keep.sum()} genes")
    

    if quantile_transform:
        print("Applying quantile transformation...")
        qt = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(1000, len(X_train)),
            random_state=42
        )
        
        # Fit on training data only
        X_train_qt = qt.fit_transform(X_train.values)
        X_test_qt = qt.transform(X_test.values)
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(
            X_train_qt, 
            index=X_train.index, 
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            X_test_qt, 
            index=X_test.index, 
            columns=X_test.columns
        )
        
        print("Quantile transformation applied")
    
    return X_train, X_test


def remove_correlated_features(X_train, threshold=0.95):
    print(f"\nRemoving genes with correlation > {threshold}")
    
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"  Dropped {len(to_drop)} highly correlated genes")
    print(f"  Kept {len(X_train.columns) - len(to_drop)} genes")
    
    keep_cols = [c for c in X_train.columns if c not in to_drop]
    
    return keep_cols


def lasso_selection(X_train, y_train, n_features):
    print(f"\nLASSO selection (target: {n_features} genes)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    
    alphas = np.logspace(-4, 2, 100) 
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y_train)
        
        n_selected = (lasso.coef_ != 0).sum()
        
        if n_selected <= n_features:
            break
    
    selected_idx = np.where(lasso.coef_ != 0)[0]
    selected_genes = X_train.columns[selected_idx].tolist()
    
    print(f"  Selected {len(selected_genes)} genes with alpha={alpha:.4f}")
    
    return selected_genes


def ridge_selection(X_train, y_train, n_features):
    print(f"\nRidge selection (target: {n_features} genes)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)

    # Cross-validated alpha selection over log-scale range
    alphas = np.logspace(-4, 2, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
    ridge_cv.fit(X_scaled, y_train)

    best_alpha = ridge_cv.alpha_
    print(f"  Best alpha (5-fold CV): {best_alpha:.6f}")

    top_idx = np.argsort(np.abs(ridge_cv.coef_))[::-1][:n_features]
    selected_genes = X_train.columns[top_idx].tolist()

    print(f"  Selected {len(selected_genes)} genes with alpha={best_alpha:.4f}")
    return selected_genes


def mutual_info_selection(X_train, y_train, n_features):
    print(f"\nMutual Information selection (target: {n_features} genes)")
    
    mi_scores = mutual_info_regression(
        X_train.values, y_train, 
        random_state=42, 
        n_neighbors=5
    )
    
    top_idx = np.argsort(mi_scores)[::-1][:n_features]
    selected_genes = X_train.columns[top_idx].tolist()
    
    print(f"  Selected {len(selected_genes)} genes")
    
    return selected_genes


def rfe_selection(X_train, y_train, n_features):
    print(f"\nRFE selection (target: {n_features} genes)")
    
    estimator = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    
    selector = RFE(estimator, n_features_to_select=n_features, step=100)
    selector.fit(X_scaled, y_train)
    
    selected_genes = X_train.columns[selector.support_].tolist()
    
    print(f"  Selected {len(selected_genes)} genes")
    
    return selected_genes


def variance_selection(X_train, n_features):
    print(f"\nVariance selection (target: {n_features} genes)")
    
    var = X_train.var(axis=0)
    top_idx = np.argsort(var)[::-1][:n_features]
    selected_genes = X_train.columns[top_idx].tolist()
    
    print(f"  Selected {len(selected_genes)} genes")
    
    return selected_genes


def rf_importance_selection(X_train, y_train, n_features, n_jobs=-1):
    print(f"\nRandom Forest importance (target: {n_features} genes)")
    
    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5,
        random_state=42,
        n_jobs=n_jobs
    )
    rf.fit(X_train.values, y_train)
    
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_features]
    selected_genes = X_train.columns[top_idx].tolist()
    
    print(f"  Selected {len(selected_genes)} genes")
    
    return selected_genes


def ard_selection(X_train, y_train, n_features):
    print(f"\nARD Bayesian regression (target: {n_features} genes)")

    scaler = StandardScaler()
    X_train_array = scaler.fit_transform(
        X_train.values if hasattr(X_train, 'values') else X_train
    )

    y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
    if len(y_train_array.shape) > 1:
        y_train_array = y_train_array.ravel()

    # Fit ARD model
    ard = ARDRegression(
        max_iter=300,
        compute_score=True,
        threshold_lambda=10000,
        verbose=False
    )
    ard.fit(X_train_array, y_train_array)

    # Get feature importance (absolute coefficients)
    importances = np.abs(ard.coef_)

    # Sort and select top features
    top_indices = np.argsort(importances)[::-1][:n_features]
    selected_genes = X_train.columns[top_indices].tolist()

    print(f"  Selected {len(selected_genes)} genes")

    return selected_genes


def evaluate_gene_set(X_train, y_train, X_test, y_test, genes, method_name):
    X_tr = X_train[genes]
    X_te = X_test[genes]
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42))
    ])
    
    model.fit(X_tr.values, y_train)
    
    y_pred_train = model.predict(X_tr.values)
    y_pred_test = model.predict(X_te.values)
    
    results = {
        'method': method_name,
        'n_genes': len(genes),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
    }
    
    return results, genes


def create_comprehensive_plots(results_df, outdir):
    
    print("\nCreating publication-quality plots...")
    
    plot_performance_vs_genes(results_df, outdir)
    
    plot_overfitting_analysis(results_df, outdir)
    
    plot_heatmap_summary(results_df, outdir)
        
    create_summary_table(results_df, outdir)
    
    print("  All visualizations saved.")


def plot_performance_vs_genes(results_df, outdir):
   
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette
    colors = {'variance': '#e74c3c', 'lasso': '#27ae60', 'mi': '#3498db', 
          'rf': '#f39c12', 'rfe': '#9b59b6', 'ridge': '#1abc9c'}
    markers = {'variance': 'o', 'lasso': 's', 'mi': '^', 'rf': 'D', 'rfe': 'v',
           'ridge': 'P'}
    
    # Extract method names (remove _XXX suffix)
    results_df['method_clean'] = results_df['method'].str.replace(r'_\d+', '', regex=True)
    
    # Plot MAE
    for method in results_df['method_clean'].unique():
        data = results_df[results_df['method_clean'] == method].sort_values('n_genes')
        
        color = colors.get(method, '#95a5a6')
        marker = markers.get(method, 'o')
        
        axes[0].plot(data['n_genes'], data['test_mae'], 
                    marker=marker, label=method.upper(), 
                    linewidth=2.5, markersize=8, color=color, alpha=0.8)
    
    axes[0].set_xlabel('Number of Features', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Test MAE (years)', fontsize=14, fontweight='bold')
    axes[0].set_title('Model Performance vs Feature Count', fontsize=16, fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    axes[0].tick_params(labelsize=11)
    
    # Plot R^2
    for method in results_df['method_clean'].unique():
        data = results_df[results_df['method_clean'] == method].sort_values('n_genes')
        
        color = colors.get(method, '#95a5a6')
        marker = markers.get(method, 'o')
        
        axes[1].plot(data['n_genes'], data['test_r2'], 
                    marker=marker, label=method.upper(), 
                    linewidth=2.5, markersize=8, color=color, alpha=0.8)
    
    axes[1].set_xlabel('Number of Features', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Test R²', fontsize=14, fontweight='bold')
    axes[1].set_title('Explained Variance vs Feature Count', fontsize=16, fontweight='bold', pad=20)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    axes[1].tick_params(labelsize=11)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, frameon=True, shadow=True,
               loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    plt.savefig(f"{outdir}/1_performance_vs_features.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/1_performance_vs_features.pdf", bbox_inches='tight')
    plt.close()
    
    print("  Plot: Performance vs Features")


def plot_overfitting_analysis(results_df, outdir):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Calculate gaps
    results_df['gap'] = results_df['test_mae'] - results_df['train_mae']
    
    # Sort by gap
    sorted_df = results_df.sort_values('gap').copy()
    
    # Color code by gap size
    colors = []
    for gap in sorted_df['gap']:
        if gap < 2:
            colors.append('#27ae60')
        elif gap < 4:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    # Bar chart
    y_pos = np.arange(len(sorted_df))
    ax.barh(y_pos, sorted_df['gap'], color=colors, alpha=0.8, 
            edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['method'], fontsize=9)
    ax.set_xlabel('Train-Test Gap (years)', fontsize=13, fontweight='bold')
    ax.set_title('Generalization Gap\n(Green=Good <2yr, Orange=Moderate 2-4yr, Red=Severe >4yr)', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Reference lines
    ax.axvline(x=2, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add values
    for i, gap in enumerate(sorted_df['gap']):
        ax.text(gap + 0.1, i, f'{gap:.1f}', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/2_overfitting_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/2_overfitting_analysis.pdf", bbox_inches='tight')
    plt.close()
    
    print("  Plot: Overfitting analysis")

def plot_heatmap_summary(results_df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    results_df['method_clean'] = results_df['method'].str.replace(r'_\d+', '', regex=True)
    
    # Pivot for heatmap
    pivot_mae = results_df.pivot(index='method_clean', columns='n_genes', values='test_mae')
    pivot_r2 = results_df.pivot(index='method_clean', columns='n_genes', values='test_r2')
    
    # Sort by best average performance
    pivot_mae = pivot_mae.loc[pivot_mae.mean(axis=1).sort_values().index]
    pivot_r2 = pivot_r2.loc[pivot_r2.mean(axis=1).sort_values(ascending=False).index]
    
    # MAE heatmap
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Test MAE (years)'}, 
                ax=axes[0], linewidths=0.5, linecolor='gray')
    axes[0].set_title('Test MAE by Method and Gene Count\n(Lower is Better)', 
                     fontsize=16, fontweight='bold', pad=20)
    axes[0].set_xlabel('Number of Genes', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Method', fontsize=14, fontweight='bold')
    axes[0].tick_params(labelsize=11)
    
    # R^2 heatmap
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Test R²'}, 
                ax=axes[1], linewidths=0.5, linecolor='gray')
    axes[1].set_title('Test R² by Method and Gene Count\n(Higher is Better)', 
                     fontsize=16, fontweight='bold', pad=20)
    axes[1].set_xlabel('Number of Genes', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Method', fontsize=14, fontweight='bold')
    axes[1].tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/3_heatmap_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/3_heatmap_summary.pdf", bbox_inches='tight')
    plt.close()
    
    print("  Plot: Heatmap Summary")



def create_summary_table(results_df, outdir):
    # Sort by test MAE
    results_df['gap'] = results_df['test_mae'] - results_df['train_mae']
    top_10 = results_df.nsmallest(10, 'test_mae').copy()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    for i, row in top_10.iterrows():
        gap_status = 'good' if row['gap'] < 3 else 'warning' if row['gap'] < 5 else 'not good'
        table_data.append([
            f"{len(table_data)+1}",
            row['method'],
            f"{int(row['n_genes'])}",
            f"{row['train_mae']:.2f}",
            f"{row['test_mae']:.2f}",
            f"{row['test_r2']:.3f}",
            f"{row['gap']:.2f}",
            gap_status
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Rank', 'Method', 'N Genes', 'Train MAE', 'Test MAE', 'Test R²', 'Gap', 'Status'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(8):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Color code rows
    for i in range(1, len(table_data) + 1):
        # Alternate row colors
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(8):
            cell = table[(i, j)]
            cell.set_facecolor(color)
        
        # Highlight best row
        if i == 1:
            for j in range(8):
                cell = table[(i, j)]
                cell.set_facecolor('#f1c40f')
                cell.set_text_props(weight='bold')
    
    plt.title('Feature Selection Results - Top 10 Configurations', 
             fontsize=18, fontweight='bold', pad=20)
    
    plt.savefig(f"{outdir}/5_summary_table.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/5_summary_table.pdf", bbox_inches='tight')
    plt.close()
    
    print("  Plot: Summary Table")


def plot_timing_comparison(timing_df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group by method for aggregated view
    method_agg = timing_df.groupby('method').agg({
        'time_minutes': 'sum',
        'memory_mb': 'mean',
        'n_genes': 'mean'
    }).reset_index()
    
    # Sort by time
    method_agg_sorted = method_agg.sort_values('time_minutes')
    
    # Plot 1: Total time per method (aggregated across all n_genes)
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(method_agg_sorted)))
    bars1 = ax1.barh(range(len(method_agg_sorted)), method_agg_sorted['time_minutes'],
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(method_agg_sorted)))
    ax1.set_yticklabels(method_agg_sorted['method'], fontsize=11)
    ax1.set_xlabel('Total Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Time per Method (across all gene counts)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add values
    for i, (idx, row) in enumerate(method_agg_sorted.iterrows()):
        ax1.text(row['time_minutes'] + 0.5, i, 
                f"{row['time_minutes']:.1f}m", 
                va='center', fontweight='bold', fontsize=10)
    
    # Plot 2: Time vs n_genes for each method
    ax2 = axes[0, 1]
    for method in timing_df['method'].unique():
        method_data = timing_df[timing_df['method'] == method].sort_values('n_genes')
        ax2.plot(method_data['n_genes'], method_data['time_minutes'], 
                marker='o', linewidth=2, markersize=8, label=method, alpha=0.7)
    ax2.set_xlabel('Number of Genes Selected', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_title('Selection Time vs Number of Genes', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Memory usage (if available)
    ax3 = axes[1, 0]
    if 'memory_mb' in timing_df.columns and timing_df['memory_mb'].notna().any():
        memory_data = method_agg_sorted[method_agg_sorted['memory_mb'].notna()]
        if len(memory_data) > 0:
            bars3 = ax3.barh(range(len(memory_data)), memory_data['memory_mb'],
                            color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax3.set_yticks(range(len(memory_data)))
            ax3.set_yticklabels(memory_data['method'], fontsize=11)
            ax3.set_xlabel('Average Memory Usage (MB)', fontsize=12, fontweight='bold')
            ax3.set_title('Memory Usage per Method', fontsize=13, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            # Add values
            for i, (idx, row) in enumerate(memory_data.iterrows()):
                if pd.notna(row['memory_mb']):
                    ax3.text(row['memory_mb'] + 10, i, 
                            f"{row['memory_mb']:.0f} MB", 
                            va='center', fontweight='bold', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Memory data not available\n(install psutil)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'Memory data not available\n(install psutil)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')
    
    # Plot 4: Efficiency - Time per 100 genes selected
    ax4 = axes[1, 1]
    timing_df['efficiency'] = timing_df['time_minutes'] / (timing_df['n_genes'] / 100)
    efficiency_agg = timing_df.groupby('method')['efficiency'].mean().sort_values()
    
    colors4 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(efficiency_agg)))
    bars4 = ax4.barh(range(len(efficiency_agg)), efficiency_agg.values,
                     color=colors4, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(efficiency_agg)))
    ax4.set_yticklabels(efficiency_agg.index, fontsize=11)
    ax4.set_xlabel('Time per 100 genes (minutes)', fontsize=12, fontweight='bold')
    ax4.set_title('Selection Efficiency', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add values
    for i, (method, eff) in enumerate(efficiency_agg.items()):
        ax4.text(eff + 0.1, i, 
                f"{eff:.2f}m", 
                va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/timing_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/timing_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print("  Timing comparison plot created!")


# Main

def parse_args():
    ap = argparse.ArgumentParser(description="Advanced Feature Selection")
    ap.add_argument("--X_train", required=True)
    ap.add_argument("--X_test", required=True)
    ap.add_argument("--y_train", required=True)
    ap.add_argument("--y_test", required=True)
    ap.add_argument("--outdir", default="data/feature_selected")
    
    ap.add_argument("--methods", nargs="+",
                   default=["variance", "lasso", "ridge", "mi", "rf", "ard"],
                   choices=["variance", "lasso", "ridge", "mi", "rfe", "rf", "ard", "correlation"])
    
    ap.add_argument("--target_sizes", nargs="+", type=int,
                   default=[50, 100, 150, 200])
    
    ap.add_argument("--corr_threshold", type=float, default=0.95)
    
    ap.add_argument("--quantile_transform", action="store_true")
    
    ap.add_argument("--n_jobs", type=int, default=-1)

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    np.random.seed(42)
    
    
    # Get and display system info
    system_info = get_system_info()
    print(f"System info")
    print(f"  Platform: {system_info['platform']}")
    print(f"  Processor: {system_info['processor']}")
    print(f"  CPU cores: {system_info['cpu_count']} physical, {system_info['cpu_count_logical']} logical")
    print(f"  RAM: {system_info['ram_total_gb']} GB")
    print(f"  Python: {system_info['python_version']}")
    print()
    
    # Start total timer
    total_start_time = time.time()
    
    # Load data
    print("Loading data...")
    X_train, y_train, ids_train = load_data(args.X_train, args.y_train)
    X_test, y_test, ids_test = load_data(args.X_test, args.y_test)
    
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Preprocess
    print("\nPreprocessing...")
    X_train, X_test = preprocess(
        X_train, X_test, 
        log2p1=True, 
        basic_filter=True,
        quantile_transform=args.quantile_transform
    )
    
    print(f"  After preprocessing: {X_train.shape[1]} genes")
    
    # Track timing for each preprocessing step
    preprocessing_times = {}
    
    # Optional: Remove highly correlated genes first
    if "correlation" in args.methods:
        print("\nRemoving highly correlated genes...")
        corr_start = time.time()
        if HAS_PSUTIL:
            process = psutil.Process()
            corr_start_mem = process.memory_info().rss / (1024**2)
        
        keep_genes = remove_correlated_features(X_train, args.corr_threshold)
        X_train = X_train[keep_genes]
        X_test = X_test[keep_genes]
        
        corr_time = time.time() - corr_start
        if HAS_PSUTIL:
            corr_mem = process.memory_info().rss / (1024**2) - corr_start_mem
        else:
            corr_mem = None
        
        preprocessing_times['correlation'] = {
            'time_seconds': corr_time,
            'time_minutes': corr_time / 60,
            'memory_mb': corr_mem,
            'features_before': len(keep_genes) + (X_train.shape[1] - len(keep_genes)),
            'features_after': X_train.shape[1]
        }
        
        print(f"  After correlation filter: {X_train.shape[1]} genes")
        print(f"  Time: {corr_time:.2f}s ({corr_time/60:.2f}m)")
        if HAS_PSUTIL:
            print(f"  Memory: {corr_mem:.2f} MB")
    
    # Test different methods and sizes
    print("\nTesting feature selection methods...")
    
    results = []
    gene_sets = {}
    method_times = {}  # Track timing per method
    
    for n_genes in args.target_sizes:
        if n_genes > X_train.shape[1]:
            print(f"\n[SKIP] {n_genes} genes > available {X_train.shape[1]}")
            continue
        
        print(f"\n{'-'*70}")
        print(f"TARGET: {n_genes} genes")
        
        for method in args.methods:
            if method == "correlation":
                continue
            
            try:
                # Start timer for this method
                method_start = time.time()
                if HAS_PSUTIL:
                    process = psutil.Process()
                    method_start_mem = process.memory_info().rss / (1024**2)
                
                # Select genes
                if method == "variance":
                    selected = variance_selection(X_train, n_genes)
                elif method == "lasso":
                    selected = lasso_selection(X_train, y_train, n_genes)
                elif method == "ridge":
                    selected = ridge_selection(X_train, y_train, n_genes)
                elif method == "mi":
                    selected = mutual_info_selection(X_train, y_train, n_genes)
                elif method == "rfe":
                    selected = rfe_selection(X_train, y_train, n_genes)
                elif method == "rf":
                    selected = rf_importance_selection(X_train, y_train, n_genes, args.n_jobs)
                elif method == "ard":
                    selected = ard_selection(X_train, y_train, n_genes)
                
                # End timer
                method_time = time.time() - method_start
                if HAS_PSUTIL:
                    method_mem = process.memory_info().rss / (1024**2) - method_start_mem
                else:
                    method_mem = None
                
                # Store timing
                method_key = f"{method}_{n_genes}"
                method_times[method_key] = {
                    'method': method,
                    'n_genes': n_genes,
                    'time_seconds': method_time,
                    'time_minutes': method_time / 60,
                    'memory_mb': method_mem
                }
                
                # Evaluate
                result, genes = evaluate_gene_set(
                    X_train, y_train, X_test, y_test,
                    selected, method_key
                )
                
                # Add timing to result
                result['selection_time_seconds'] = method_time
                result['selection_time_minutes'] = method_time / 60
                result['selection_memory_mb'] = method_mem
                
                results.append(result)
                gene_sets[method_key] = genes
                
                print(f"  [{method}] Test MAE: {result['test_mae']:.2f}, R²: {result['test_r2']:.3f}")
                print(f"  Time: {method_time:.2f}s ({method_time/60:.2f}m)")
                if HAS_PSUTIL and method_mem:
                    print(f"  Memory: {method_mem:.2f} MB")
                
            except Exception as e:
                print(f"  [{method}] FAILED: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.outdir, "feature_selection_results.csv"), index=False)
    
    # Create all visualizations
    create_comprehensive_plots(results_df, args.outdir)
    

    # Save top 10 gene sets
    print(f"\nSaving top 10 gene sets...")
    
    # Sort by test MAE and get top 10
    top_10 = results_df.nsmallest(10, 'test_mae')
    
    print(f"\nTop 10 methods by test MAE:")
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        gap = row['test_mae'] - row['train_mae']
        print(f"  {i:2d}. {row['method']:15s} | {int(row['n_genes']):3d} genes | "
              f"Test MAE: {row['test_mae']:.2f} | Gap: {gap:.2f}")
    print("="*80)
    
    # Save gene lists, filtered matrices, and y files for top 10
    print(f"\nSaving gene lists and filtered matrices...")
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        method_name = row['method']
        genes = gene_sets[method_name]

        # Save gene list
        pd.Series(genes, name='gene').to_csv(
            os.path.join(args.outdir, f"genes_{method_name}.txt"), index=False)

        # Save filtered expression matrices in subfolder
        method_dir = os.path.join(args.outdir, method_name)
        ensure_dir(method_dir)
        X_train[genes].to_csv(os.path.join(method_dir, "X_train.csv"))
        X_test[genes].to_csv(os.path.join(method_dir, "X_test.csv"))
        shutil.copy(args.y_train, os.path.join(method_dir, "y_train.csv"))
        shutil.copy(args.y_test, os.path.join(method_dir, "y_test.csv"))

        print(f"  Rank {i:2d}: {method_name}/ ({len(genes)} genes)")

    print(f"  y_train.csv and y_test.csv copied to all top 10 folders")

    # Save best as reference (backward compatibility)
    best = top_10.iloc[0]
    best_genes = gene_sets[best['method']]

    X_train[best_genes].to_csv(os.path.join(args.outdir, "X_train_best.csv"))
    X_test[best_genes].to_csv(os.path.join(args.outdir, "X_test_best.csv"))
    shutil.copy(args.y_train, os.path.join(args.outdir, "y_train.csv"))
    shutil.copy(args.y_test, os.path.join(args.outdir, "y_test.csv"))
    
    # End total timer
    total_time = time.time() - total_start_time
    
    # Save timing summary
    if method_times:
        timing_df = pd.DataFrame(method_times).T
        timing_df.to_csv(os.path.join(args.outdir, "timing_summary.csv"))
        
        # Create timing visualization
        plot_timing_comparison(timing_df, args.outdir)
    
    # Save preprocessing times
    if preprocessing_times:
        with open(os.path.join(args.outdir, "preprocessing_times.json"), "w") as f:
            json.dump(preprocessing_times, f, indent=2)
    
    # Save metadata with timing and system info
    meta = {
        'system_info': system_info,
        'total_runtime_seconds': float(total_time),
        'total_runtime_minutes': float(total_time / 60),
        'total_runtime_hours': float(total_time / 3600),
        'args': vars(args),
        'methods_tested': args.methods,
        'target_sizes': args.target_sizes,
        'best_method': best['method'],
        'best_n_genes': int(best['n_genes']),
        'best_test_mae': float(best['test_mae']),
        'best_test_r2': float(best['test_r2']),
        'train_shape': list(X_train.shape),
        'test_shape': list(X_test.shape),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n Best feature set saved to: {args.outdir}")
    print(f"   - X_train_best.csv ({len(best_genes)} genes)")
    print(f"   - X_test_best.csv ({len(best_genes)} genes)")

    # Print timing summary
    if method_times:
        print("\n Timing summary")
        timing_df_display = timing_df[['method', 'n_genes', 'time_minutes', 'memory_mb']].copy()
        print(timing_df_display.to_string(index=False))
        print(f"\nTotal runtime: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()