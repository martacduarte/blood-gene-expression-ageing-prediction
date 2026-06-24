#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import platform

warnings.filterwarnings('ignore')

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not installed")

import joblib


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_data(X_path, y_path):
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path)

    if "sample_id" not in y.columns or "age" not in y.columns:
        raise ValueError("y must contain columns: sample_id, age")

    common = X.index.astype(str).intersection(y["sample_id"].astype(str))
    if len(common) == 0:
        raise ValueError("No common samples between X and y")

    X = X.loc[common]
    y_aligned = y.set_index("sample_id").loc[common]["age"].astype(float).values

    return X, y_aligned, list(common)


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

    return info


def preprocess_features(X_train, X_test, log2p1=False, quantile_transform=False):

    # Log transform
    if log2p1:
        median_max = float(X_train.max(axis=0).median())
        if median_max >= 20:
            print("Applying log2(TPM+1) transformation")
            X_train = np.log2(X_train.astype(float) + 1.0)
            X_test = np.log2(X_test.astype(float) + 1.0)
        else:
            print("Data appears log-transformed, skipping")

    # Quantile transformation
    if quantile_transform:
        print("Applying quantile transformation...")
        from sklearn.preprocessing import QuantileTransformer

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


def get_model_configs():
    configs = {}

    # Linear models
    configs['ElasticNet'] = {
        'model': ElasticNet(max_iter=10000, random_state=42),
        'params': {
            'model__alpha':    np.logspace(-4, 2, 50),
            'model__l1_ratio': np.linspace(0.0, 1.0, 21),
        },
        'needs_scaling': True,
        'n_iter': 40
    }

    # Tree-based models
    configs['RandomForest'] = {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'model__n_estimators':     [100, 200, 500],
            'model__max_depth':        [5, 10, 15, 20], 
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features':     ['sqrt', 'log2', 0.3],
        },
        'needs_scaling': False,
        'n_iter': 50 
    }

    if HAS_XGB:
        configs['XGBoost'] = {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'params': {
                'model__n_estimators':     [100, 200, 300, 500],
                'model__learning_rate':    [0.01, 0.05, 0.1],
                'model__max_depth':        [3, 4, 5, 6],
                'model__min_child_weight': [1, 3, 5, 10],
                'model__subsample':        [0.6, 0.7, 0.8, 0.9],
                'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'model__reg_alpha':        [0, 0.01, 0.1, 1.0, 10.0], 
                'model__reg_lambda':       [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
            },
            'needs_scaling': False,
            'n_iter': 50  
        }

    # Neural network
    configs['NeuralNetwork'] = {
        'model': MLPRegressor(
            max_iter=1000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=10, random_state=42
        ),
        'params': {
            'model__hidden_layer_sizes': [(100,), (100, 50), (200, 100), (100, 100, 50)],
            'model__activation':         ['relu', 'tanh'],
            'model__alpha':              [0.0001, 0.001, 0.01],
            'model__learning_rate_init': [0.0001, 0.001, 0.01],  
        },
        'needs_scaling': True,
        'n_iter': 40 
    }

    return configs



def train_model(name, config, X_train, y_train, cv_folds, n_jobs, random_state):
    print(f"\n Training: {name}")

    start_time = time.time()
    if HAS_PSUTIL:
        process = psutil.Process()
        start_mem = process.memory_info().rss / (1024**2)

    # Build pipeline
    steps = []
    if config['needs_scaling']:
        steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
    else:
        steps.append(('scaler', 'passthrough'))

    steps.append(('model', config['model']))
    pipe = Pipeline(steps)

    # Hyperparameter search
    search = RandomizedSearchCV(
        pipe,
        param_distributions=config['params'],
        n_iter=config['n_iter'],
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        random_state=random_state,
        verbose=1
    )

    search.fit(X_train, y_train)

    elapsed = time.time() - start_time
    if HAS_PSUTIL:
        mem_used = process.memory_info().rss / (1024**2) - start_mem
    else:
        mem_used = None

    # CV fold variability for the best hyperparameter combination
    best_idx = search.best_index_
    fold_maes = np.array([
        -search.cv_results_[f'split{i}_test_score'][best_idx]
        for i in range(cv_folds)
    ])
    cv_mean = fold_maes.mean()
    cv_std  = fold_maes.std()

    print(f"\nBest CV MAE: {cv_mean:.3f} ± {cv_std:.3f} years")
    print(f"  Per-fold MAE: {[f'{v:.2f}' for v in fold_maes]}")
    print(f"Best params: {search.best_params_}")
    print(f"Training time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if mem_used is not None:
        print(f"Memory used: {mem_used:.2f} MB")

    return search.best_estimator_, elapsed, mem_used, cv_mean, cv_std


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Per-fold CV metrics on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        fold_records.append({
            'fold':  fold_idx,
            'mae':   mean_absolute_error(y_val, y_val_pred),
            'rmse':  np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'r2':    r2_score(y_val, y_val_pred),
            'n_val': len(val_idx),
        })

    fold_df = pd.DataFrame(fold_records)

    # Summary row
    summary = {
        'fold': 'mean ± std',
        'mae':  f"{fold_df['mae'].mean():.3f} ± {fold_df['mae'].std():.3f}",
        'rmse': f"{fold_df['rmse'].mean():.3f} ± {fold_df['rmse'].std():.3f}",
        'r2':   f"{fold_df['r2'].mean():.3f} ± {fold_df['r2'].std():.3f}",
        'n_val': '',
    }
    fold_df_out = pd.concat(
        [fold_df.astype(str), pd.DataFrame([summary])],
        ignore_index=True
    )

    # Print per-fold metrics
    print("\n  CV Fold Metrics (on training set):")
    for _, row in fold_df.iterrows():
        print(f"    Fold {int(row['fold'])}: MAE={row['mae']:.2f}, "
              f"RMSE={row['rmse']:.2f}, R²={row['r2']:.3f}  (n={int(row['n_val'])})")
    print(f"    Mean:  MAE={fold_df['mae'].mean():.2f} ± {fold_df['mae'].std():.2f}, "
          f"RMSE={fold_df['rmse'].mean():.2f} ± {fold_df['rmse'].std():.2f}, "
          f"R²={fold_df['r2'].mean():.3f} ± {fold_df['r2'].std():.3f}")

    # Re-fit on full training set after CV (CV fits left model in partial state)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = {
        'train': {
            'r2':   r2_score(y_train, y_pred_train),
            'mae':  mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        },
        'test': {
            'r2':   r2_score(y_test, y_pred_test),
            'mae':  mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        },
        'cv_folds': fold_df.to_dict(orient='records'),
        'cv_summary': {
            'mae_mean':  fold_df['mae'].mean(),
            'mae_std':   fold_df['mae'].std(),
            'rmse_mean': fold_df['rmse'].mean(),
            'rmse_std':  fold_df['rmse'].std(),
            'r2_mean':   fold_df['r2'].mean(),
            'r2_std':    fold_df['r2'].std(),
        }
    }

    # Alias for downstream compatibility
    metrics['test_calibrated'] = metrics['test']

    return metrics, y_pred_train, y_pred_test, fold_df_out


def plot_parity(y_true, y_pred, outpath, title, metrics):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

    lims = [
        np.min([y_true.min(), y_pred.min()]) - 2,
        np.max([y_true.max(), y_pred.max()]) + 2,
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, lw=2, label='Ideal prediction')

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), "b-", alpha=0.5, lw=2,
            label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Age (years)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Predicted Age (years)', fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=21, fontweight='bold')
    ax.legend(loc='lower right', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=15)

    textstr = '\n'.join([
        f'R² = {metrics["r2"]:.3f}',
        f'MAE = {metrics["mae"]:.2f} years',
        f'RMSE = {metrics["rmse"]:.2f} years',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true, y_pred, outpath, title):
    residuals = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Age (years)', fontweight='bold')
    axes[0, 0].set_ylabel('Residuals (years)', fontweight='bold')
    axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals vs True Age
    axes[0, 1].scatter(y_true, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('True Age (years)', fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (years)', fontweight='bold')
    axes[0, 1].set_title('Residuals vs True Age', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals (years)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title(f'Residual Distribution (mean={residuals.mean():.2f})', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_error_by_age(y_true, y_pred, outpath, title):
    errors = np.abs(y_pred - y_true)

    age_bins = pd.cut(y_true, bins=[0, 30, 40, 50, 60, 70, 100],
                      labels=['<30', '30-40', '40-50', '50-60', '60-70', '>70'])

    df = pd.DataFrame({'Age Group': age_bins, 'MAE': errors})

    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='MAE', by='Age Group', ax=ax)
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (years)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.suptitle('')
    ax.grid(True, alpha=0.3, axis='y')

    means = df.groupby('Age Group', observed=False)['MAE'].mean()
    for i, (group, mean_val) in enumerate(means.items()):
        ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean_val:.1f}',
                ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

    return df.groupby('Age Group', observed=False)['MAE'].agg(['mean', 'std', 'count'])


def extract_feature_importance(model, gene_names, top_n=None):
    importance_df = None

    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps['model']
    else:
        actual_model = model

    if hasattr(actual_model, 'coef_'):
        importance_df = pd.DataFrame({
            'gene': gene_names,
            'coefficient': actual_model.coef_,
            'abs_coefficient': np.abs(actual_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

    elif hasattr(actual_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'gene': gene_names,
            'importance': actual_model.feature_importances_
        }).sort_values('importance', ascending=False)

    if importance_df is not None:
        return importance_df if top_n is None else importance_df.head(top_n)

    return None


def plot_feature_importance(importance_df, outpath, title, top_n=20):
    if importance_df is None:
        return

    data = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    value_col = 'abs_coefficient' if 'abs_coefficient' in data.columns else 'importance'

    y_pos = np.arange(len(data))
    ax.barh(y_pos, data[value_col].values, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['gene'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def compute_extended_metrics(y_true, y_pred, name):
    errors_abs = np.abs(y_pred - y_true)
    delta_age  = y_pred - y_true  # age acceleration

    p95  = float(np.percentile(errors_abs, 95))
    pmax = float(errors_abs.max())

    pct_within_5  = float(100 * (errors_abs <= 5).mean())
    pct_within_10 = float(100 * (errors_abs <= 10).mean())

    mean_delta_age   = float(delta_age.mean())
    median_delta_age = float(np.median(delta_age))
    std_delta_age    = float(delta_age.std())
    q1_delta_age     = float(np.percentile(delta_age, 25))
    q3_delta_age     = float(np.percentile(delta_age, 75))
    pct_over         = float(100 * (delta_age > 0).mean())
    pct_under        = float(100 * (delta_age < 0).mean())

    bins   = [0, 30, 40, 50, 60, 70, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
    age_bins = pd.cut(y_true, bins=bins, labels=labels)
    df = pd.DataFrame({'bin': age_bins, 'delta_age': delta_age, 'abs': errors_abs})

    bin_grouped = df.groupby('bin', observed=False)['delta_age']
    bin_mean_delta_age   = bin_grouped.mean().to_dict()
    bin_median_delta_age = bin_grouped.median().to_dict()
    bin_std_delta_age    = bin_grouped.std().to_dict()
    bin_q1_delta_age      = bin_grouped.quantile(0.25).to_dict()
    bin_q3_delta_age      = bin_grouped.quantile(0.75).to_dict()
    bin_n                 = bin_grouped.count().to_dict()

    return {
        'model': name,
        'p95_abs_error': p95,
        'max_abs_error': pmax,
        'pct_within_5yr': pct_within_5,
        'pct_within_10yr': pct_within_10,
        'mean_delta_age': mean_delta_age,
        'median_delta_age': median_delta_age,
        'std_delta_age': std_delta_age,
        'q1_delta_age': q1_delta_age,
        'q3_delta_age': q3_delta_age,
        'pct_predicted_older': pct_over,
        'pct_predicted_younger': pct_under,
        'bin_mean_delta_age': bin_mean_delta_age,
        'bin_median_delta_age': bin_median_delta_age,
        'bin_std_delta_age': bin_std_delta_age,
        'bin_q1_delta_age': bin_q1_delta_age,
        'bin_q3_delta_age': bin_q3_delta_age,
        'bin_n': bin_n,
    }


def plot_extended_diagnostics(y_true, y_pred, y_train, outdir, name):
    from scipy import stats as scipy_stats

    errors_abs = np.abs(y_pred - y_true)
    delta_age  = y_pred - y_true 

    bins   = [0, 30, 40, 50, 60, 70, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']

    df_test = pd.DataFrame({
        'bin':       pd.cut(y_true, bins=bins, labels=labels),
        'abs_error': errors_abs,
        'delta_age': delta_age,
        'y_true':    y_true,
        'y_pred':    y_pred,
    })

    train_counts = pd.cut(y_train, bins=bins, labels=labels).value_counts().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{name} — Extended Diagnostics', fontsize=16, fontweight='bold')

    # 1. Absolute error per bin
    bin_mae = df_test.groupby('bin', observed=False)['abs_error'].mean()
    bin_std = df_test.groupby('bin', observed=False)['abs_error'].std().fillna(0)
    ax = axes[0, 0]
    ax.bar(range(len(labels)), bin_mae.values, yerr=bin_std.values,
           color='#3498db', alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Age Group', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (years)', fontweight='bold')
    ax.set_title('MAE per Age Bin (mean ± std)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(bin_mae.values, bin_std.values)):
        ax.text(i, m + s + 0.2, f'{m:.1f}', ha='center', fontsize=9, fontweight='bold')

    # 2. Age acceleration per bin — boxplot (median, IQR, outliers)
    ax = axes[0, 1]
    box_data, box_labels = [], []
    for lbl in labels:
        vals = df_test.loc[df_test['bin'] == lbl, 'delta_age'].dropna().values
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(lbl)

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(facecolor='#5dade2', alpha=0.7))
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Age Group', fontweight='bold', fontsize=20)
    ax.set_ylabel('Age Acceleration, ΔAge (years)', fontweight='bold', fontsize=20)
    ax.set_title('Age Acceleration by Age Group\n(median, IQR, and outliers)',
                 fontweight='bold', fontsize=21)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Correlation with data density
    bin_n_train = train_counts.reindex(labels).fillna(0)
    ax = axes[1, 0]
    ax.scatter(bin_n_train.values, bin_mae.values, s=120,
               color='#9b59b6', edgecolor='black', linewidth=1.5, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (bin_n_train.values[i], bin_mae.values[i]),
                    xytext=(6, 4), textcoords='offset points', fontsize=9)
    valid_mask = ~np.isnan(bin_mae.values) & ~np.isnan(bin_n_train.values.astype(float))
    if valid_mask.sum() > 2:
        x_valid = bin_n_train.values[valid_mask].astype(float)
        y_valid = bin_mae.values[valid_mask]
        r, pval = scipy_stats.pearsonr(x_valid, y_valid)
        z = np.polyfit(x_valid, y_valid, 1)
        xr = np.linspace(x_valid.min(), x_valid.max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), 'r--', alpha=0.6, linewidth=2,
                label=f'r={r:.2f}, p={pval:.3f}')
        ax.legend(fontsize=10)
    ax.set_xlabel('Training Samples in Bin', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (years)', fontweight='bold')
    ax.set_title('Error vs Data Density\n(more data → lower error?)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Extreme cases: highlight worst 5% 
    threshold_95 = np.percentile(errors_abs, 95)
    is_extreme = errors_abs >= threshold_95
    ax = axes[1, 1]
    ax.scatter(y_true[~is_extreme], y_pred[~is_extreme],
               alpha=0.4, s=20, color='#3498db', label='Normal predictions')
    ax.scatter(y_true[is_extreme], y_pred[is_extreme],
               alpha=0.9, s=60, color='#e74c3c', edgecolor='black',
               linewidth=1, zorder=5, label=f'Worst 5% (error≥{threshold_95:.1f}yr)')
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Age (years)', fontweight='bold')
    ax.set_ylabel('Predicted Age (years)', fontweight='bold')
    ax.set_title('Extreme Cases (worst 5% predictions)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'extended_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  extended_diagnostics.png saved")


def plot_model_extreme_comparison(all_results_ext, outdir):
    if not all_results_ext:
        return

    _bin_dict_keys = ['bin_mean_delta_age', 'bin_median_delta_age',
                       'bin_std_delta_age', 'bin_q1_delta_age',
                       'bin_q3_delta_age', 'bin_n']
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in _bin_dict_keys}
        for r in all_results_ext
    ])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Model Extreme Error Comparison', fontsize=14, fontweight='bold')

    # 95th pct error
    sorted_df = df.sort_values('p95_abs_error')
    axes[0].barh(range(len(sorted_df)), sorted_df['p95_abs_error'],
                 color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(len(sorted_df)))
    axes[0].set_yticklabels(sorted_df['model'])
    axes[0].set_xlabel('95th Percentile Absolute Error (years)', fontweight='bold')
    axes[0].set_title('Worst-5% Error Threshold', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    for i, v in enumerate(sorted_df['p95_abs_error']):
        axes[0].text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9, fontweight='bold')

    # % within 5 years
    sorted_df2 = df.sort_values('pct_within_5yr', ascending=False)
    axes[1].barh(range(len(sorted_df2)), sorted_df2['pct_within_5yr'],
                 color='#27ae60', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(range(len(sorted_df2)))
    axes[1].set_yticklabels(sorted_df2['model'])
    axes[1].set_xlabel('% Predictions Within 5 Years', fontweight='bold')
    axes[1].set_title('Clinical Accuracy (±5 years)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    for i, v in enumerate(sorted_df2['pct_within_5yr']):
        axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Mean age acceleration (ΔAge) — systematic bias
    sorted_df3 = df.sort_values('mean_delta_age')
    bar_colors = ['#e74c3c' if v > 0 else '#27ae60' for v in sorted_df3['mean_delta_age']]
    axes[2].barh(range(len(sorted_df3)), sorted_df3['mean_delta_age'],
                 color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[2].axvline(0, color='black', linewidth=1.5)
    axes[2].set_yticks(range(len(sorted_df3)))
    axes[2].set_yticklabels(sorted_df3['model'])
    axes[2].set_xlabel('Mean Age Acceleration, ΔAge (years)', fontweight='bold')
    axes[2].set_title('Systematic Bias (Mean ΔAge)\n(Red=predicts older, Green=younger)', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    axes[2].invert_yaxis()
    for i, v in enumerate(sorted_df3['mean_delta_age']):
        axes[2].text(v + (0.05 if v >= 0 else -0.35), i, f'{v:.2f}',
                     va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'extreme_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  extreme_comparison.png saved")


def create_comparison_plots(results_df, outdir):
    print(" Creating comparison plots...")

    plot_df = results_df[results_df['train_r2'].notna()].copy()

    plot_model_comparison(plot_df, outdir)
    plot_train_test_comparison(plot_df, outdir)
    plot_performance_heatmap(plot_df, outdir)
    plot_best_model_dashboard(results_df, outdir)

    print("  All comparison plots saved!")


def plot_model_comparison(results_df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = results_df['model'].values
    x = np.arange(len(models))
    width = 0.35

    # 1. MAE Comparison (Train vs Test)
    axes[0, 0].bar(x - width/2, results_df['train_mae'], width,
                   label='Train', alpha=0.8, color='#3498db')
    axes[0, 0].bar(x + width/2, results_df['test_mae'], width,
                   label='Test', alpha=0.8, color='#e74c3c')
    axes[0, 0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MAE (years)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('MAE: Train vs Test', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.5, linewidth=2)

    best_idx = results_df['test_mae'].idxmin()
    # Highlight best test bar
    bar_x = np.where(results_df.index == best_idx)[0][0]
    axes[0, 0].bar(bar_x + width/2, results_df.loc[best_idx, 'test_mae'],
                   width, color='gold', edgecolor='black', linewidth=2, alpha=0.9)

    # 2. R^2 Comparison
    axes[0, 1].bar(x - width/2, results_df['train_r2'], width,
                   label='Train', alpha=0.8, color='#3498db')
    axes[0, 1].bar(x + width/2, results_df['test_r2'], width,
                   label='Test', alpha=0.8, color='#e74c3c')
    axes[0, 1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('R²', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('R²: Train vs Test', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Test MAE Ranking
    sorted_df = results_df.sort_values('test_mae')
    colors = ['#27ae60' if i == 0 else '#3498db' for i in range(len(sorted_df))]

    axes[1, 0].barh(range(len(sorted_df)), sorted_df['test_mae'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_yticks(range(len(sorted_df)))
    axes[1, 0].set_yticklabels(sorted_df['model'])
    axes[1, 0].set_xlabel('Test MAE (years)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Models Ranked by Test MAE', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        axes[1, 0].text(row['test_mae'] + 0.1, i, f"{row['test_mae']:.2f}",
                        va='center', fontweight='bold', fontsize=10)

    # 4. Generalization Gap
    results_df = results_df.copy()
    results_df['gap'] = results_df['test_mae'] - results_df['train_mae']
    colors = ['#27ae60' if gap < 2 else '#f39c12' if gap < 4 else '#e74c3c'
              for gap in results_df['gap']]

    axes[1, 1].bar(x, results_df['gap'], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Train-Test Gap (years)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Generalization Gap\n(Green=Good, Orange=Moderate, Red=Severe)',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 1].axhline(y=4, color='red', linestyle='--', alpha=0.5, linewidth=2)
    for i, gap in enumerate(results_df['gap']):
        axes[1, 1].text(i, gap + 0.2, f'{gap:.1f}', ha='center',
                        fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{outdir}/comparison_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/comparison_performance.pdf", bbox_inches='tight')
    plt.close()
    print("  Plot 1: Model Comparison")


def plot_train_test_comparison(results_df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # MAE scatter
    for idx, row in results_df.iterrows():
        color = '#27ae60' if (row['test_mae'] - row['train_mae']) < 2 else \
                '#f39c12' if (row['test_mae'] - row['train_mae']) < 4 else '#e74c3c'
        axes[0].scatter(row['train_mae'], row['test_mae'], s=200,
                        alpha=0.7, color=color, edgecolor='black', linewidth=2)
        axes[0].annotate(row['model'], (row['train_mae'], row['test_mae']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    max_mae = max(results_df['train_mae'].max(), results_df['test_mae'].max())
    axes[0].plot([0, max_mae], [0, max_mae], 'k--', alpha=0.5, linewidth=2,
                 label='Perfect (no overfitting)')
    axes[0].fill_between([0, max_mae], [0, max_mae], [2, max_mae+2],
                         alpha=0.1, color='green', label='Good (<2 years gap)')
    axes[0].fill_between([0, max_mae], [2, max_mae+2], [4, max_mae+4],
                         alpha=0.1, color='orange', label='Moderate (2-4 years)')

    axes[0].set_xlabel('Train MAE (years)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Test MAE (years)', fontsize=13, fontweight='bold')
    axes[0].set_title('Train vs Test Performance\n(Distance from diagonal = overfitting)',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max_mae+1)
    axes[0].set_ylim(0, max_mae+1)

    # R^2 scatter
    for idx, row in results_df.iterrows():
        gap = row['train_r2'] - row['test_r2']
        color = '#27ae60' if gap < 0.15 else '#f39c12' if gap < 0.3 else '#e74c3c'
        axes[1].scatter(row['train_r2'], row['test_r2'], s=200,
                        alpha=0.7, color=color, edgecolor='black', linewidth=2)
        axes[1].annotate(row['model'], (row['train_r2'], row['test_r2']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
    axes[1].set_xlabel('Train R²', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Test R²', fontsize=13, fontweight='bold')
    axes[1].set_title('Train vs Test R²\n(Distance from diagonal = overfitting)',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.1, 1.0)
    axes[1].set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(f"{outdir}/comparison_overfitting.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/comparison_overfitting.pdf", bbox_inches='tight')
    plt.close()
    print("  Plot 2: Overfitting Analysis")


def plot_performance_heatmap(results_df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics_mae = results_df[['model', 'train_mae', 'test_mae']].set_index('model')
    metrics_r2  = results_df[['model', 'train_r2',  'test_r2' ]].set_index('model')

    metrics_mae = metrics_mae.sort_values('test_mae')
    metrics_r2  = metrics_r2.sort_values('test_r2', ascending=False)

    sns.heatmap(metrics_mae.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'MAE (years)'}, ax=axes[0],
                linewidths=0.5, linecolor='gray')
    axes[0].set_title('MAE Across Train/Test Sets\n(Lower is Better)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Split', fontsize=12, fontweight='bold')
    axes[0].set_yticklabels(['Train MAE', 'Test MAE'], rotation=0)

    sns.heatmap(metrics_r2.T, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'R²'}, ax=axes[1],
                linewidths=0.5, linecolor='gray')
    axes[1].set_title('R² Across Train/Test Sets\n(Higher is Better)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Split', fontsize=12, fontweight='bold')
    axes[1].set_yticklabels(['Train R²', 'Test R²'], rotation=0)

    plt.tight_layout()
    plt.savefig(f"{outdir}/comparison_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/comparison_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("  Plot 3: Performance Heatmap")


def plot_best_model_dashboard(results_df, outdir):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ranked_df = results_df[results_df['train_r2'].notna()].copy()
    best_idx = ranked_df['test_mae'].idxmin()
    best = ranked_df.loc[best_idx]

    fig.suptitle(f'BEST MODEL: {best["model"]}', fontsize=20, fontweight='bold', y=0.98)

    # 1. Summary text
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    summary_text = (
        f"\n"
        f"    Training performance                         Test performance\n"
        f"    MAE:  {best['train_mae']:.2f} years"
        f"                            MAE:  {best['test_mae']:.2f} years\n"
        f"    RMSE: {best['train_rmse']:.2f} years"
        f"                           RMSE: {best['test_rmse']:.2f} years\n"
        f"    R²:   {best['train_r2']:.3f}"
        f"                                   R²:   {best['test_r2']:.3f}\n"
        f"    Generalization Gap (MAE): {best['test_mae'] - best['train_mae']:.2f} years\n"
    )
    ax1.text(0.5, 0.5, summary_text, transform=ax1.transAxes, fontsize=13,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 2. Ranking bar chart
    ax2 = fig.add_subplot(gs[1, :2])
    sorted_df = ranked_df.sort_values('test_mae')
    colors = ['gold' if m == best['model'] else '#3498db' for m in sorted_df['model']]
    ax2.barh(range(len(sorted_df)), sorted_df['test_mae'],
             color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels(sorted_df['model'], fontsize=11)
    ax2.set_xlabel('Test MAE (years)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Ranking by Test MAE', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # 3. Improvement analysis
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    worst_mae = ranked_df['test_mae'].max()
    improvement = worst_mae - best['test_mae']
    improvement_pct = (improvement / worst_mae) * 100
    status = ('EXCELLENT' if best['test_mae'] < 8
              else 'GOOD' if best['test_mae'] < 10
              else 'MODERATE')
    table_text = (
        f"\n    Improvement analysis\n\n"
        f"    Worst MAE:   {worst_mae:.2f} years\n"
        f"    Best MAE:    {best['test_mae']:.2f} years\n\n"
        f"    Improvement: {improvement:.2f} years\n"
        f"                 ({improvement_pct:.1f}% better)\n\n"
        f"    Status: {status}\n"
    )
    ax3.text(0.5, 0.5, table_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    models = ranked_df['model'].values
    x = np.arange(len(models))
    width = 0.35

    # 4. MAE comparison
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(x - width/2, ranked_df['train_mae'], width, label='Train', alpha=0.7, color='skyblue')
    ax4.bar(x + width/2, ranked_df['test_mae'],  width, label='Test',  alpha=0.7, color='salmon')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('MAE (years)', fontsize=10, fontweight='bold')
    ax4.set_title('MAE Comparison', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. R² comparison
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(x - width/2, ranked_df['train_r2'], width, label='Train', alpha=0.7, color='skyblue')
    ax5.bar(x + width/2, ranked_df['test_r2'],  width, label='Test',  alpha=0.7, color='salmon')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('R²', fontsize=10, fontweight='bold')
    ax5.set_title('R² Comparison', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Overfitting gaps
    ax6 = fig.add_subplot(gs[2, 2])
    gaps = ranked_df['test_mae'] - ranked_df['train_mae']
    gap_colors = ['#27ae60' if g < 2 else '#f39c12' if g < 4 else '#e74c3c' for g in gaps]
    ax6.bar(x, gaps, color=gap_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Gap (years)', fontsize=10, fontweight='bold')
    ax6.set_title('Generalization Gap', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=2, color='orange', linestyle='--', alpha=0.5)
    ax6.axhline(y=4, color='red', linestyle='--', alpha=0.5)

    plt.savefig(f"{outdir}/best_model_dashboard.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/best_model_dashboard.pdf", bbox_inches='tight')
    plt.close()
    print("  Plot 4: Best Model Dashboard")


def plot_timing_comparison(results_df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_df = results_df[results_df['train_time_minutes'].notna()].copy()

    axes[0, 0].barh(range(len(plot_df)), plot_df['train_time_minutes'])
    axes[0, 0].set_yticks(range(len(plot_df)))
    axes[0, 0].set_yticklabels(plot_df['model'])
    axes[0, 0].set_xlabel('Training Time (minutes)', fontweight='bold')
    axes[0, 0].set_title('Model Training Time', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    axes[0, 1].scatter(plot_df['train_time_minutes'], plot_df['test_mae'],
                       s=200, alpha=0.6, edgecolor='black', linewidth=2)
    for idx, row in plot_df.iterrows():
        axes[0, 1].annotate(row['model'],
                            (row['train_time_minutes'], row['test_mae']),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('Training Time (minutes)', fontweight='bold')
    axes[0, 1].set_ylabel('Test MAE (years)', fontweight='bold')
    axes[0, 1].set_title('Performance vs Training Time', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    if 'memory_mb' in plot_df.columns and plot_df['memory_mb'].notna().any():
        mem_df = plot_df[plot_df['memory_mb'].notna()]
        axes[1, 0].barh(range(len(mem_df)), mem_df['memory_mb'], color='coral', alpha=0.7)
        axes[1, 0].set_yticks(range(len(mem_df)))
        axes[1, 0].set_yticklabels(mem_df['model'])
        axes[1, 0].set_xlabel('Memory Used (MB)', fontweight='bold')
        axes[1, 0].set_title('Memory Usage', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 0].text(0.5, 0.5, 'Memory data not available\n(install psutil)',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')

    plot_df = plot_df.copy()
    plot_df['efficiency'] = plot_df['test_mae'] / plot_df['train_time_minutes']
    axes[1, 1].barh(range(len(plot_df)), plot_df['efficiency'])
    axes[1, 1].set_yticks(range(len(plot_df)))
    axes[1, 1].set_yticklabels(plot_df['model'])
    axes[1, 1].set_xlabel('MAE / Training Time', fontweight='bold')
    axes[1, 1].set_title('Training Efficiency (lower = better)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f"{outdir}/timing_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Advanced Age Prediction - Multiple Models")
    ap.add_argument("--X_train", required=True)
    ap.add_argument("--y_train", required=True)
    ap.add_argument("--X_test", required=True)
    ap.add_argument("--y_test", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--log2p1", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--create_ensemble", action="store_true")
    ap.add_argument("--quantile_transform", action="store_true")
    ap.add_argument("--gene_list", default=None)
    return ap.parse_args()


def main():
    total_start = time.time()
    args = parse_args()
    np.random.seed(args.random_state)
    ensure_dir(args.outdir)

    print(f"# Output: {args.outdir}")

    system_info = get_system_info()
    print(f"  Platform:  {system_info['platform']}")
    print(f"  Processor: {system_info['processor']}")
    if HAS_PSUTIL:
        print(f"  CPU cores: {system_info['cpu_count']} physical, "
              f"{system_info['cpu_count_logical']} logical")
        print(f"  RAM: {system_info['ram_total_gb']} GB")
    print(f"  Python: {system_info['python_version']}\n")

    # Load data
    print("[STEP 1] Loading data...")
    X_train, y_train, ids_train = load_data(args.X_train, args.y_train)
    X_test,  y_test,  ids_test  = load_data(args.X_test,  args.y_test)

    print(f"  Train: {X_train.shape[0]} samples × {X_train.shape[1]} genes")
    print(f"  Test:  {X_test.shape[0]} samples × {X_test.shape[1]} genes")

    # Gene list filtering (optional)
    if args.gene_list:
        print(f"\nLoading gene list from: {args.gene_list}")
        with open(args.gene_list) as f:
            requested_genes = [line.strip() for line in f if line.strip()]
        # Strip version suffixes if present (e.g. ENSG00000162490.7 -> ENSG00000162490)
        requested_genes = [g.split(".")[0] for g in requested_genes]
        # Match against columns (also strip versions from column names if needed)
        col_map = {c.split(".")[0]: c for c in X_train.columns}
        matched = [col_map[g] for g in requested_genes if g in col_map]
        missing = [g for g in requested_genes if g not in col_map]
        if missing:
            print(f"  [WARN] {len(missing)} genes not found in expression matrix: {missing}")
        if not matched:
            raise ValueError("None of the requested genes were found in the expression matrix!")
        X_train = X_train[matched]
        X_test  = X_test[matched]
        print(f"  Filtered to {len(matched)} genes from gene list")
        print(f"  Train after filter: {X_train.shape[0]} samples × {X_train.shape[1]} genes")
        print(f"  Test  after filter: {X_test.shape[0]} samples × {X_test.shape[1]} genes")

    # Preprocess
    print("\nPreprocessing...")
    X_train, X_test = preprocess_features(
        X_train, X_test,
        log2p1=args.log2p1,
        quantile_transform=args.quantile_transform
    )
    gene_names = X_train.columns.tolist()

    # Get model configurations
    all_configs = get_model_configs()

    if args.models:
        configs = {k: v for k, v in all_configs.items() if k in args.models}
    else:
        configs = all_configs

    print(f"\nTraining {len(configs)} models...")
    print(f"  Models: {list(configs.keys())}")

    results = []
    trained_models = {}
    all_results_ext = []

    for name, config in configs.items():
        try:
            # Train
            model, train_time, mem_used, cv_mean, cv_std = train_model(
                name, config, X_train.values, y_train,
                args.cv_folds, args.n_jobs, args.random_state
            )

            # Evaluate — returns metrics, predictions, and per-fold CV dataframe
            metrics, y_pred_train, y_pred_test, fold_df = evaluate_model(
                model, X_train.values, y_train, X_test.values, y_test
            )

            print(f"\n{name} Results:")
            print(f"  Train: R²={metrics['train']['r2']:.3f}, "
                  f"MAE={metrics['train']['mae']:.2f}, "
                  f"RMSE={metrics['train']['rmse']:.2f}")
            print(f"  Test:  R²={metrics['test']['r2']:.3f}, "
                  f"MAE={metrics['test']['mae']:.2f}, "
                  f"RMSE={metrics['test']['rmse']:.2f}")

            # Save model
            model_dir = os.path.join(args.outdir, name)
            ensure_dir(model_dir)
            joblib.dump(model, os.path.join(model_dir, f"model_{name}.joblib"))

            # Save per-fold CV metrics
            fold_df.to_csv(os.path.join(model_dir, "cv_fold_metrics.csv"), index=False)

            # Save predictions
            pred_df = pd.DataFrame({
                'sample_id':  ids_test,
                'y_true':     y_test,
                'y_pred':     y_pred_test,
            })
            pred_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)

            # Single parity plot (test set)
            plot_parity(y_test, y_pred_test,
                        os.path.join(model_dir, "parity_test.png"),
                        f"{name} - Test Predictions", metrics['test'])

            plot_residuals(y_test, y_pred_test,
                           os.path.join(model_dir, "residual_analysis.png"),
                           f"{name} - Residual Analysis")

            age_errors = plot_error_by_age(
                y_test, y_pred_test,
                os.path.join(model_dir, "error_by_age.png"),
                f"{name} - MAE by Age Group"
            )
            age_errors.to_csv(os.path.join(model_dir, "error_by_age.csv"))

            # Extended diagnostics
            ext_metrics = compute_extended_metrics(y_test, y_pred_test, name)
            plot_extended_diagnostics(y_test, y_pred_test, y_train, model_dir, name)

            # Feature importance
            importance = extract_feature_importance(model, gene_names, top_n=None)
            if importance is not None:
                importance.to_csv(
                    os.path.join(model_dir, "feature_importance.csv"), index=False
                )
                plot_feature_importance(
                    importance,
                    os.path.join(model_dir, "feature_importance.png"),
                    f"{name} - Top 20 Important Genes", top_n=20
                )

            # Save metrics (including extended diagnostics)
            metrics['extended'] = {
                k: v for k, v in ext_metrics.items() if k != 'model'
            }
            with open(os.path.join(model_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            results.append({
                'model':              name,
                'train_r2':           metrics['train']['r2'],
                'train_mae':          metrics['train']['mae'],
                'train_rmse':         metrics['train']['rmse'],
                'test_r2':            metrics['test']['r2'],
                'test_mae':           metrics['test']['mae'],
                'test_rmse':          metrics['test']['rmse'],
                'cv_mae_mean':        cv_mean,
                'cv_mae_std':         cv_std,
                'train_time_minutes': train_time / 60,
                'memory_mb':          mem_used,
            })

            all_results_ext.append(ext_metrics)
            trained_models[name] = model

        except Exception as e:
            print(f"\n Error training {name}: {e}")
            continue

    # Ensemble
    if args.create_ensemble and len(trained_models) >= 3:
        print("\nCreating ensemble model...")

        results_df = pd.DataFrame(results)
        top_3 = results_df.nsmallest(3, 'cv_mae_mean')['model'].tolist()

        print(f"  Top 3 models: {top_3}")

        top_3_cv_maes = [
            results_df.loc[results_df['model'] == n, 'cv_mae_mean'].values[0]
            for n in top_3
        ]
        weights = 1 / np.array(top_3_cv_maes)
        weights /= weights.sum()
        print(f"  Ensemble weights: {dict(zip(top_3, weights.round(3)))}")

        y_pred_ensemble = np.average(
            [trained_models[n].predict(X_test.values) for n in top_3],
            axis=0, weights=weights
        )

        ensemble_metrics = {
            'test': {
                'r2':   r2_score(y_test, y_pred_ensemble),
                'mae':  mean_absolute_error(y_test, y_pred_ensemble),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
            }
        }

        print(f"\n  Ensemble: R²={ensemble_metrics['test']['r2']:.3f}, "
              f"MAE={ensemble_metrics['test']['mae']:.2f}")

        ens_dir = os.path.join(args.outdir, "Ensemble")
        ensure_dir(ens_dir)

        pd.DataFrame({
            'sample_id':       ids_test,
            'y_true':          y_test,
            'y_pred_ensemble': y_pred_ensemble,
        }).to_csv(os.path.join(ens_dir, "predictions.csv"), index=False)

        plot_parity(y_test, y_pred_ensemble,
                    os.path.join(ens_dir, "parity_ensemble.png"),
                    f"Ensemble ({', '.join(top_3)}) - Predictions",
                    ensemble_metrics['test'])

        with open(os.path.join(ens_dir, "metrics.json"), "w") as f:
            json.dump({**ensemble_metrics, 'models': top_3}, f, indent=2)

        results.append({
            'model':      'Ensemble',
            'train_r2':   None,
            'train_mae':  None,
            'train_rmse': None,
            'test_r2':    ensemble_metrics['test']['r2'],
            'test_mae':   ensemble_metrics['test']['mae'],
            'test_rmse':  ensemble_metrics['test']['rmse'],
        })

    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    # Comparison visualizations
    print("\nCreating comparison visualizations...")
    create_comparison_plots(results_df, args.outdir)
    plot_model_extreme_comparison(all_results_ext, args.outdir)

    if results_df['train_time_minutes'].notna().any():
        plot_timing_comparison(results_df, args.outdir)
        print("  Timing comparison plot created!")

    # Extended metrics summary
    if all_results_ext:
        _bin_dict_keys = ['bin_mean_delta_age', 'bin_median_delta_age',
                           'bin_std_delta_age', 'bin_q1_delta_age',
                           'bin_q3_delta_age', 'bin_n']
        ext_df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in _bin_dict_keys}
            for r in all_results_ext
        ])
        ext_df.to_csv(
            os.path.join(args.outdir, "extended_metrics_summary.csv"), index=False
        )

    # Run metadata
    meta = {
        'args':           vars(args),
        'models_trained': list(trained_models.keys()),
        'train_shape':    list(X_train.shape),
        'test_shape':     list(X_test.shape),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - total_start
    print(f"\n Total runtime: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")

    print("\n Final results")
    print(results_df[['model', 'test_r2', 'test_mae', 'test_rmse']].to_string(index=False))

    best = results_df.loc[results_df['test_mae'].idxmin()]
    print(f"\nBest model: {best['model']}")
    print(f"   R²   = {best['test_r2']:.3f}")
    print(f"   MAE  = {best['test_mae']:.2f} years")
    print(f"   RMSE = {best['test_rmse']:.2f} years")

    print(f"\n All outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()