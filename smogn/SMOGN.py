#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# Relevance function

def compute_mean_relevance(y, n_bins=20):
    counts, edges = np.histogram(y, bins=n_bins)
    mean_count = counts.mean()
    max_count  = counts.max()

    # Per-bin relevance
    bin_phi = np.zeros(n_bins)
    for i, c in enumerate(counts):
        if c < mean_count:
            # Rare bin: phi in [0.5, 1.0]
            bin_phi[i] = 0.5 + 0.5 * (1.0 - c / mean_count)
        else:
            # Common bin: phi in [0.0, 0.5)
            if max_count > mean_count:
                bin_phi[i] = 0.5 * (1.0 - (c - mean_count) / (max_count - mean_count))
            else:
                bin_phi[i] = 0.0

    # Map each sample to its bin's relevance
    phi = np.zeros(len(y))
    for i, val in enumerate(y):
        idx = np.searchsorted(edges[1:], val)          # which bin
        idx = int(np.clip(idx, 0, n_bins - 1))
        phi[i] = bin_phi[idx]

    bin_info = {
        'edges'      : edges,
        'counts'     : counts,
        'bin_phi'    : bin_phi,
        'mean_count' : float(mean_count),
        'max_count'  : float(max_count),
        'n_rare'     : int((phi >= 0.5).sum()),
        'n_common'   : int((phi < 0.5).sum()),
    }
    return phi, bin_info


# Bin construction

def build_bins(X, y, phi, rel_threshold=0.5):
    order = np.argsort(y)
    X_s = X.iloc[order].reset_index(drop=True)
    y_s = y[order]
    phi_s = phi[order]

    bins_rare, bins_common = [], []
    buf_X, buf_y, buf_rare = [], [], None

    def flush(is_rare):
        if buf_X:
            entry = {'X': pd.DataFrame(buf_X, columns=X.columns),
                     'y': np.array(buf_y)}
            (bins_rare if is_rare else bins_common).append(entry)

    for i in range(len(y_s)):
        is_rare = phi_s[i] >= rel_threshold
        if buf_rare is not None and is_rare != buf_rare:
            flush(buf_rare)
            buf_X, buf_y = [], []
        buf_X.append(X_s.iloc[i].values)
        buf_y.append(y_s[i])
        buf_rare = is_rare

    flush(buf_rare)
    return bins_rare, bins_common


# SMOGN oversampling for one rare bin

def oversample_bin(bin_X, bin_y, n_to_generate, k, pert=0.02, seed=42):
    np.random.seed(seed)
    n = len(bin_y)

    if n < 2 or n_to_generate == 0:
        return np.empty((0, bin_X.shape[1])), np.array([])

    k_eff = min(k, n - 1)
    nbrs  = NearestNeighbors(n_neighbors=k_eff + 1, algorithm='auto').fit(bin_X)

    synth_X, synth_y = [], []
    generated = 0

    # Cycle through seed examples until we have enough synthetic samples
    idx_cycle = list(range(n))
    np.random.shuffle(idx_cycle)
    cycle_pos = 0

    while generated < n_to_generate:
        seed_i  = idx_cycle[cycle_pos % len(idx_cycle)]
        cycle_pos += 1

        x_seed = bin_X[seed_i]
        y_seed = bin_y[seed_i]

        # Safe distance: median of distances to all other samples / 2
        dists_all = np.linalg.norm(bin_X - x_seed, axis=1)
        dists_all = dists_all[dists_all > 0]  
        safe_dist = (np.median(dists_all) / 2) if len(dists_all) > 0 else 0.0

        # k-NN (exclude self)
        nn_dists, nn_idx = nbrs.kneighbors([x_seed])
        nn_dists = nn_dists[0][1:]  
        nn_idx   = nn_idx[0][1:]

        # Pick one neighbour randomly
        pick     = np.random.randint(k_eff)
        nb_i     = nn_idx[pick]
        nb_dist  = nn_dists[pick]
        x_nb     = bin_X[nb_i]
        y_nb     = bin_y[nb_i]

        if nb_dist < safe_dist:
            # Safe → SmoteR interpolation
            alpha   = np.random.uniform(0, 1)
            new_x   = x_seed + alpha * (x_nb - x_seed)
            new_y   = y_seed + alpha * (y_nb - y_seed)
        else:
            # Unsafe → Gaussian noise
            p       = min(safe_dist, pert) if safe_dist > 0 else pert
            new_x   = x_seed + np.random.normal(0, p, size=x_seed.shape)
            new_y   = float(y_seed + np.random.normal(0, p))

        synth_X.append(new_x)
        synth_y.append(new_y)
        generated += 1

    return np.array(synth_X), np.array(synth_y)


# Main SMOGN algorithm

def run_smogn(X, y, perc_under=75, perc_over=200, k=5,
              n_bins=20, pert=0.02, seed=42):
   
    np.random.seed(seed)
    X_arr = X.values

    # Step 1: relevance
    phi, bin_info = compute_mean_relevance(y, n_bins=n_bins)
    n_rare   = bin_info['n_rare']
    n_common = bin_info['n_common']
    print(f"  Rare samples   (phi >= 0.5): {n_rare}  ({100*n_rare/len(y):.1f}%)")
    print(f"  Common samples (phi <  0.5): {n_common} ({100*n_common/len(y):.1f}%)")

    # Step 2: build consecutive bins
    bins_rare, bins_common = build_bins(X, y, phi)
    print(f"  Rare bins  : {len(bins_rare)}")
    print(f"  Common bins: {len(bins_common)}")

    # Step 3: undersample common bins
    kept_X, kept_y = [], []
    for b in bins_common:
        n_keep = max(1, int(round(len(b['y']) * perc_under / 100)))
        idx    = np.random.choice(len(b['y']), size=n_keep, replace=False)
        kept_X.append(b['X'].values[idx])
        kept_y.append(b['y'][idx])

    # Step 4: keep all rare + generate synthetic
    rare_X, rare_y     = [], []
    synth_X, synth_y   = [], []

    for b in bins_rare:
        bX = b['X'].values
        by = b['y']
        rare_X.append(bX)
        rare_y.append(by)

        n_gen = max(0, int(round(len(by) * perc_over / 100)))
        if n_gen > 0:
            sX, sy = oversample_bin(bX, by, n_gen, k, pert, seed)
            if len(sy) > 0:
                synth_X.append(sX)
                synth_y.append(sy)

    # Step 5: combine
    parts_X = [np.vstack(kept_X)] if kept_X else []
    parts_y = [np.concatenate(kept_y)] if kept_y else []

    if rare_X:
        parts_X.append(np.vstack(rare_X))
        parts_y.append(np.concatenate(rare_y))

    n_synthetic = 0
    if synth_X:
        sx = np.vstack(synth_X)
        sy = np.concatenate(synth_y)
        parts_X.append(sx)
        parts_y.append(sy)
        n_synthetic = len(sy)

    X_new = pd.DataFrame(np.vstack(parts_X), columns=X.columns)
    y_new = np.concatenate(parts_y)

    n_common_kept = sum(len(b['y']) for b in kept_X) if kept_X else 0

    info = {
        'bin_info'   : bin_info,
        'n_orig'     : len(y),
        'n_new'      : len(y_new),
        'n_rare_kept': sum(len(b['y']) for b in bins_rare),
        'n_synthetic': n_synthetic,
        'n_common_kept': int(sum(a.shape[0] for a in kept_X)) if kept_X else 0,
    }
    return X_new, y_new, phi, info


# Visualisations 

def plot_relevance(y, phi, bin_info, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mean-Based Relevance Function', fontsize=13, fontweight='bold')

    # Left: histogram coloured by rare/common
    ax = axes[0]
    edges  = bin_info['edges']
    counts = bin_info['counts']
    bin_phi = bin_info['bin_phi']
    for i in range(len(counts)):
        color = '#e74c3c' if bin_phi[i] >= 0.5 else '#3498db'
        ax.bar(edges[i], counts[i], width=edges[i+1]-edges[i],
               color=color, alpha=0.8, edgecolor='black', linewidth=0.4,
               align='edge')
    ax.axhline(bin_info['mean_count'], color='black', linestyle='--',
               linewidth=2, label=f"Mean = {bin_info['mean_count']:.1f}")
    ax.set_xlabel('Age (years)', fontweight='bold')
    ax.set_ylabel('Sample count', fontweight='bold')
    ax.set_title('Age Histogram\n(Red = rare bins, Blue = common bins)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Right: relevance scatter
    ax = axes[1]
    ax.scatter(y, phi, alpha=0.4, s=12, c=phi, cmap='RdBu_r', vmin=0, vmax=1)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2,
               label='Threshold = 0.5')
    ax.set_xlabel('Age (years)', fontweight='bold')
    ax.set_ylabel('Relevance φ(age)', fontweight='bold')
    ax.set_title('Relevance per Sample\n(φ ≥ 0.5 = rare → oversampled)',
                 fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_distribution(y_before, y_after, n_synthetic, outpath):
    bins   = [0, 30, 40, 50, 60, 70, 200]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
    cb = pd.cut(y_before, bins=bins, labels=labels, right=False).value_counts().sort_index()
    ca = pd.cut(y_after,  bins=bins, labels=labels, right=False).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Age Distribution Before vs After SMOGN\n'
        f'({len(y_before)} original → {len(y_after)} samples, '
        f'{n_synthetic} synthetic added)',
        fontsize=13, fontweight='bold')

    # Left: histogram + KDE
    ax = axes[0]
    ax.hist(y_before, bins=20, alpha=0.5, color='#3498db', density=True,
            edgecolor='black', linewidth=0.4, label='Before SMOGN')
    ax.hist(y_after,  bins=20, alpha=0.5, color='#e74c3c', density=True,
            edgecolor='black', linewidth=0.4, label='After SMOGN')
    from scipy.stats import gaussian_kde
    for yy, col in [(y_before, '#2980b9'), (y_after, '#c0392b')]:
        if len(yy) > 1:
            kde = gaussian_kde(yy, bw_method=0.3)
            xr  = np.linspace(yy.min()-2, yy.max()+2, 300)
            ax.plot(xr, kde(xr), color=col, linewidth=2)
    ax.set_xlabel('Age (years)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Histogram + KDE', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Right: bin bar chart
    ax = axes[1]
    x, w = np.arange(len(labels)), 0.35
    ax.bar(x - w/2, cb.values, w, color='#3498db', alpha=0.8,
           edgecolor='black', linewidth=0.4, label='Before SMOGN')
    ax.bar(x + w/2, ca.values, w, color='#e74c3c', alpha=0.8,
           edgecolor='black', linewidth=0.4, label='After SMOGN')
    for i, (b, a) in enumerate(zip(cb.values, ca.values)):
        ax.text(i-w/2, b+1, str(b), ha='center', fontsize=8,
                color='#2980b9', fontweight='bold')
        ax.text(i+w/2, a+1, str(a), ha='center', fontsize=8,
                color='#c0392b', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Age Group', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Sample Count per Age Bin', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


# Argument parsing

def parse_args():
    ap = argparse.ArgumentParser(
        description='SMOGN with mean-based relevance'
    )
    ap.add_argument('--X_train',    required=True)
    ap.add_argument('--y_train',    required=True)
    ap.add_argument('--X_test',     required=True)
    ap.add_argument('--y_test',     required=True)
    ap.add_argument('--outdir',     required=True)
    ap.add_argument('--k',          type=int,   default=5)
    ap.add_argument('--perc_over',  type=float, default=200)
    ap.add_argument('--perc_under', type=float, default=75)
    ap.add_argument('--n_bins',     type=int,   default=20)
    ap.add_argument('--pert',       type=float, default=0.02)
    ap.add_argument('--seed',       type=int,   default=42)
    return ap.parse_args()


# Main

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    print('SMOGN — Mean-Based Relevance')
    print(f'  k           = {args.k}')
    print(f'  perc_over   = {args.perc_over}%')
    print(f'  perc_under  = {args.perc_under}%')
    print(f'  n_bins      = {args.n_bins}')
    print(f'  pert        = {args.pert}')
    print(f'  seed        = {args.seed}')
    print(f'  outdir      = {args.outdir}\n')

    # Load
    print('Loading data...')
    X_train = pd.read_csv(args.X_train, index_col=0)
    y_df    = pd.read_csv(args.y_train)
    common  = X_train.index.astype(str).intersection(y_df['sample_id'].astype(str))
    X_train = X_train.loc[common]
    y_df    = y_df.set_index('sample_id').loc[common].reset_index()
    y_train = y_df['age'].values
    print(f'  X_train: {X_train.shape[0]} samples x {X_train.shape[1]} genes')
    print(f'  Age: min={y_train.min():.1f}  max={y_train.max():.1f}  '
          f'mean={y_train.mean():.1f}')

    # SMOGN 
    print('\nRunning SMOGN with mean-based relevance...')
    t0 = time.time()
    X_new, y_new, phi, info = run_smogn(
        X_train, y_train,
        perc_under=args.perc_under,
        perc_over=args.perc_over,
        k=args.k,
        n_bins=args.n_bins,
        pert=args.pert,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f'\n  Completed in {elapsed:.1f}s')
    print(f'  Original  : {info["n_orig"]} samples')
    print(f'  Final     : {info["n_new"]} samples')
    print(f'  Synthetic : {info["n_synthetic"]} samples')

    # Age distribution after
    bins_rep   = [0, 30, 40, 50, 60, 70, 200]
    labels_rep = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
    print('\n  Age distribution after SMOGN:')
    counts_after = pd.cut(y_new, bins=bins_rep,
                          labels=labels_rep, right=False).value_counts().sort_index()
    counts_before = pd.cut(y_train, bins=bins_rep,
                           labels=labels_rep, right=False).value_counts().sort_index()
    for lbl in labels_rep:
        print(f'    {lbl:<8}: {counts_before[lbl]:>4} → {counts_after[lbl]:>4}')

    # Save 
    print('\nSaving outputs...')
    n_orig = info['n_orig']
    n_syn  = info['n_synthetic']
    new_ids = (list(common) +
               [f'synthetic_{i+1:04d}' for i in range(n_syn)])
    # Pad if undersampling reduced count below n_orig
    if len(new_ids) > len(X_new):
        new_ids = new_ids[:len(X_new)]
    elif len(new_ids) < len(X_new):
        new_ids += [f'extra_{i+1:04d}' for i in range(len(X_new) - len(new_ids))]

    X_out = X_new.copy()
    X_out.index = new_ids
    X_out.index.name = 'sample_id'
    X_out.to_csv(os.path.join(args.outdir, 'X_train_smogn.csv'))
    print(f'  Saved: X_train_smogn.csv  ({len(X_out)} x {X_out.shape[1]})')

    y_out = pd.DataFrame({'sample_id': new_ids, 'age': y_new})
    y_out.to_csv(os.path.join(args.outdir, 'y_train.csv'), index=False)
    print(f'  Saved: y_train.csv  ({len(y_out)} rows)')

    shutil.copy(args.X_test, os.path.join(args.outdir, 'X_test.csv'))
    shutil.copy(args.y_test, os.path.join(args.outdir, 'y_test.csv'))
    print(f'  Copied: X_test.csv and y_test.csv (unchanged)')

    # Plots 
    print('\nSaving plots...')
    plot_relevance(y_train, phi, info['bin_info'],
                   os.path.join(args.outdir, 'relevance_mean_based.png'))
    plot_distribution(y_train, y_new, n_syn,
                      os.path.join(args.outdir, 'age_distribution_smogn.png'))

    # Metadata 
    meta = {
        'params': {
            'k'          : args.k,
            'perc_over'  : args.perc_over,
            'perc_under' : args.perc_under,
            'n_bins'     : args.n_bins,
            'pert'       : args.pert,
            'seed'       : args.seed,
        },
        'samples': {
            'original'  : info['n_orig'],
            'final'     : info['n_new'],
            'synthetic' : info['n_synthetic'],
        },
        'relevance': {
            'method'          : 'mean_based',
            'threshold'       : 0.5,
            'mean_count'      : info['bin_info']['mean_count'],
            'n_rare'          : info['bin_info']['n_rare'],
            'n_common'        : info['bin_info']['n_common'],
        },
    }
    with open(os.path.join(args.outdir, 'smogn_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()