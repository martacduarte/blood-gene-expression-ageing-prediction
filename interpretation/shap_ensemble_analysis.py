#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def style_shap_summary_plot():
    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)

    fig = plt.gcf()
    for cax in fig.axes:
        if cax is not ax:
            cax.tick_params(labelsize=15)
            if cax.get_ylabel():
                cax.set_ylabel(cax.get_ylabel(), fontsize=20)


def parse_gtf_gene_symbols(gtf_path):
    gene_id_re = re.compile(r'gene_id "([^"]+)"')
    gene_name_re = re.compile(r'gene_name "([^"]+)"')
    gene_type_re = re.compile(r'gene_type "([^"]+)"')

    records = []
    with open(gtf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = fields[8]
            gid_match = gene_id_re.search(attrs)
            gname_match = gene_name_re.search(attrs)
            gtype_match = gene_type_re.search(attrs)
            if not gid_match:
                continue
            gene_id = gid_match.group(1)
            gene_name = gname_match.group(1) if gname_match else None
            gene_type = gtype_match.group(1) if gtype_match else None
            records.append((gene_id, gene_name, gene_type))

    df = pd.DataFrame(records, columns=["gene_id", "gene_symbol", "gene_type"])
    df["gene_id_noversion"] = df["gene_id"].str.split(".").str[0]
    df = df.drop_duplicates(subset="gene_id")
    print(f"  Parsed GTF: {len(df)} gene records found")
    return df


def build_symbol_lookup(gtf_path):
    mapping_df = parse_gtf_gene_symbols(gtf_path)
    lookup_versioned = dict(zip(mapping_df["gene_id"], mapping_df["gene_symbol"]))
    lookup_unversioned = dict(zip(mapping_df["gene_id_noversion"], mapping_df["gene_symbol"]))

    def lookup(gene_id):
        gene_id = str(gene_id)
        if gene_id in lookup_versioned:
            return lookup_versioned[gene_id]
        return lookup_unversioned.get(gene_id.split(".")[0], None)

    return lookup, mapping_df


def load_xy(X_path, y_path):
    X = pd.read_csv(X_path, index_col=0)
    y = pd.read_csv(y_path)
    if "sample_id" not in y.columns or "age" not in y.columns:
        raise ValueError("y must contain columns: sample_id, age")
    common = X.index.astype(str).intersection(y["sample_id"].astype(str))
    if len(common) == 0:
        raise ValueError("No common samples between X and y!")
    X = X.loc[common]
    y_aligned = y.set_index("sample_id").loc[common]["age"].astype(float).values
    return X, y_aligned


def unwrap_pipeline_model(loaded_obj):
    if hasattr(loaded_obj, "named_steps"):
        scaler = loaded_obj.named_steps.get("scaler", None)
        if scaler == "passthrough":
            scaler = None
        model = loaded_obj.named_steps["model"]
        return scaler, model
    return None, loaded_obj


def apply_scaler(scaler, X):
    if scaler is None:
        return X.values
    return scaler.transform(X.values)


def compute_shap_for_model(name, model_path, X_train, X_test, feature_names, outdir, top_n, symbol_lookup=None):
    print(f"\n[MODEL] {name}")
    loaded = joblib.load(model_path)
    scaler, estimator = unwrap_pipeline_model(loaded)

    X_train_proc = apply_scaler(scaler, X_train)
    X_test_proc = apply_scaler(scaler, X_test)

    model_type = type(estimator).__name__
    print(f"  Estimator type: {model_type}")

    if model_type in ("XGBRegressor", "RandomForestRegressor"):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test_proc)
    elif model_type == "ElasticNet":
        if hasattr(shap, "LinearExplainer"):
            explainer = shap.LinearExplainer(estimator, X_train_proc)
        else:
            explainer = shap.explainers.Linear(estimator, X_train_proc)
        shap_values = explainer.shap_values(X_test_proc)
    else:
        # Fallback: model-agnostic KernelExplainer (slow, use a small background sample)
        print("  [WARN] Unrecognised model type, using KernelExplainer (slow).")
        background = shap.sample(X_train_proc, 100, random_state=42)
        explainer = shap.KernelExplainer(estimator.predict, background)
        shap_values = explainer.shap_values(X_test_proc)

    shap_values = np.asarray(shap_values)
    print(f"  SHAP values shape: {shap_values.shape}")

    # Mean absolute SHAP value per gene (global importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "gene": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": shap_values.mean(axis=0),  # signed, for direction
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if symbol_lookup is not None:
        importance_df["gene_symbol"] = importance_df["gene"].apply(symbol_lookup)

    importance_df.to_csv(os.path.join(outdir, f"shap_importance_{name}.csv"), index=False)

    raw_df = pd.DataFrame(shap_values, columns=feature_names)
    raw_df.to_csv(os.path.join(outdir, f"shap_raw_values_{name}.csv"), index=False)

    # Beeswarm summary plot
    if symbol_lookup is not None:
        plot_feature_names = [symbol_lookup(g) or g for g in feature_names]
    else:
        plot_feature_names = feature_names

    plt.figure()
    shap.summary_plot(
        shap_values, X_test_proc, feature_names=plot_feature_names,
        max_display=top_n, show=False
    )
    style_shap_summary_plot()
    plt.title(f"SHAP Summary — {name}", fontsize=21, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_summary_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Bar plot of mean |SHAP|
    top = importance_df.head(top_n).copy()
    if symbol_lookup is not None:
        top["label"] = top["gene_symbol"].fillna(top["gene"])
    else:
        top["label"] = top["gene"]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(top["label"][::-1], top["mean_abs_shap"][::-1], color="#3498db", edgecolor="black")
    ax.set_xlabel("Mean |SHAP value|", fontsize=20, fontweight="bold")
    ax.set_title(f"Top {top_n} Genes by SHAP Importance — {name}", fontsize=21, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"shap_bar_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved SHAP outputs for {name}")
    return shap_values, importance_df


def main():
    ap = argparse.ArgumentParser(description="SHAP interpretability analysis for the weighted ensemble")
    ap.add_argument("--model_elasticnet", required=True)
    ap.add_argument("--model_xgboost", required=True)
    ap.add_argument("--model_rf", required=True)
    ap.add_argument("--X_train", required=True)
    ap.add_argument("--X_test", required=True)
    ap.add_argument("--y_train", required=True)
    ap.add_argument("--y_test", required=True)
    ap.add_argument("--weight_elasticnet", type=float, required=True)
    ap.add_argument("--weight_xgboost", type=float, required=True)
    ap.add_argument("--weight_rf", type=float, required=True)
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gtf", default=None)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    symbol_lookup = None
    if args.gtf:
        print(f"Parsing GTF for gene symbol mapping: {args.gtf}")
        symbol_lookup, mapping_df = build_symbol_lookup(args.gtf)
        mapping_df.to_csv(os.path.join(args.outdir, "ensembl_to_symbol_mapping.csv"), index=False)

    print("Loading data...")
    X_train, y_train = load_xy(args.X_train, args.y_train)
    X_test, y_test = load_xy(args.X_test, args.y_test)
    feature_names = X_train.columns.tolist()
    print(f"  Train: {X_train.shape[0]} samples x {X_train.shape[1]} genes")
    print(f"  Test:  {X_test.shape[0]} samples x {X_test.shape[1]} genes")

    print("\nComputing SHAP values per ensemble member...")
    shap_en, imp_en = compute_shap_for_model(
        "ElasticNet", args.model_elasticnet, X_train, X_test, feature_names, args.outdir, args.top_n, symbol_lookup
    )
    shap_xgb, imp_xgb = compute_shap_for_model(
        "XGBoost", args.model_xgboost, X_train, X_test, feature_names, args.outdir, args.top_n, symbol_lookup
    )
    shap_rf, imp_rf = compute_shap_for_model(
        "RandomForest", args.model_rf, X_train, X_test, feature_names, args.outdir, args.top_n, symbol_lookup
    )

    print("\nCombining SHAP values using ensemble weights...")
    w_sum = args.weight_elasticnet + args.weight_xgboost + args.weight_rf
    if abs(w_sum - 1.0) > 1e-6:
        print(f"  [WARN] Weights sum to {w_sum:.4f}, not 1.0. Re-normalising.")
    w_en = args.weight_elasticnet / w_sum
    w_xgb = args.weight_xgboost / w_sum
    w_rf = args.weight_rf / w_sum

    ensemble_shap = (
        w_en * shap_en + w_xgb * shap_xgb + w_rf * shap_rf
    )

    mean_abs_shap = np.abs(ensemble_shap).mean(axis=0)
    ensemble_importance_df = pd.DataFrame({
        "gene": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": ensemble_shap.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if symbol_lookup is not None:
        ensemble_importance_df["gene_symbol"] = ensemble_importance_df["gene"].apply(symbol_lookup)

    ensemble_importance_df.to_csv(os.path.join(args.outdir, "shap_importance_Ensemble.csv"), index=False)
    pd.DataFrame(ensemble_shap, columns=feature_names).to_csv(
        os.path.join(args.outdir, "shap_raw_values_Ensemble.csv"), index=False
    )

    if symbol_lookup is not None:
        plot_feature_names = [symbol_lookup(g) or g for g in feature_names]
    else:
        plot_feature_names = feature_names

    plt.figure()
    shap.summary_plot(
        ensemble_shap, X_test.values, feature_names=plot_feature_names,
        max_display=args.top_n, show=False
    )
    style_shap_summary_plot()
    plt.title("SHAP Summary — Weighted Ensemble", fontsize=21, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "shap_summary_Ensemble.png"), dpi=300, bbox_inches="tight")
    plt.close()

    top = ensemble_importance_df.head(args.top_n).copy()
    if symbol_lookup is not None:
        top["label"] = top["gene_symbol"].fillna(top["gene"])
    else:
        top["label"] = top["gene"]
    fig, ax = plt.subplots(figsize=(8, max(4, args.top_n * 0.3)))
    ax.barh(top["label"][::-1], top["mean_abs_shap"][::-1], color="#27ae60", edgecolor="black")
    ax.set_xlabel("Mean |SHAP value| (ensemble-weighted)", fontsize=20, fontweight="bold")
    ax.set_title(f"Top {args.top_n} Genes by SHAP Importance — Weighted Ensemble", fontsize=21, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "shap_bar_Ensemble.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\nComparing SHAP ranking with ElasticNet coefficient ranking...")
    en_loaded = joblib.load(args.model_elasticnet)
    _, en_estimator = unwrap_pipeline_model(en_loaded)
    coef_df = pd.DataFrame({
        "gene": feature_names,
        "elasticnet_coef": en_estimator.coef_,
        "abs_coef": np.abs(en_estimator.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    top_shap_genes = set(ensemble_importance_df.head(args.top_n)["gene"])
    top_coef_genes = set(coef_df.head(args.top_n)["gene"])
    overlap = top_shap_genes.intersection(top_coef_genes)
    print(f"  Top-{args.top_n} overlap between ensemble SHAP and ElasticNet coefficients: "
          f"{len(overlap)}/{args.top_n} genes ({100*len(overlap)/args.top_n:.1f}%)")

    comparison_df = ensemble_importance_df.merge(coef_df, on="gene", how="left")
    comparison_df.to_csv(os.path.join(args.outdir, "shap_vs_elasticnet_comparison.csv"), index=False)

    print(f"\nDone. All outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()