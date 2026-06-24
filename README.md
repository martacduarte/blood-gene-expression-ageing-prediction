# Ageing prediction model from blood gene expression

Machine learning pipeline for predicting chronological age from whole blood RNA-seq gene expression data, developed for my Master's thesis (Faculdade de Ciências da Universidade do Porto, 2025/26). The pipeline uses the GTEx v11 whole blood cohort and combines differential gene expression pre-filtering, systematic feature selection, and a comparative regression framework (ElasticNet, Random Forest, XGBoost, Neural Network) to predict age from a 200-gene signature. Predicted age is used to derive age acceleration (ΔAge), a proxy for biological ageing.

The final model (XGBoost) achieves a test MAE of 7.35 years and R² of 0.506 on a held-out set of 152 donors. The gene signature is interpreted using ElasticNet coefficients and SHAP values, and cross-validated against published transcriptomic ageing signatures and a STRING protein-protein interaction network. Synthetic oversampling (SMOGN) was also evaluated to address training set age imbalance, but did not improve performance and was not adopted in the final pipeline (see `smogn/`).

Full methodological details, results, and discussion are available in the accompanying thesis.

## Repository structure

The pipeline is organised into folders matching the methodology described in Chapter 3 of the thesis:

```
.
├── data_acquisition_preprocessing/   # GTEx loading, log2(TPM+1), expression filtering, EDA
├── batch_correction_qc/              # ComBat batch correction, RIN filtering, outlier removal
├── train_test_split/                 # Stratified train/test split by age bin
├── differential_expression/          # limma-voom DGE analysis (R)
├── feature_selection/                # Variance, MI, LASSO, Ridge, RF, ARD comparison
├── regression_modeling/              # ElasticNet, Random Forest, XGBoost, Neural Network, ensemble
├── smogn/                            # SMOGN oversampling experiment (tested, not adopted)
├── interpretation/                   # SHAP analysis, ElasticNet coefficients, PPI network
└── README.md
```

Each folder contains the scripts for that pipeline stage, named to match the thesis section they correspond to.

## Pipeline overview

| Stage | Thesis section | Script(s) |
|---|---|---|
| Data acquisition & EDA | 3.1 | `data_acquisition_preprocessing/prepare_gtex_eda.py` |
| Batch correction & QC | 3.2 | `batch_correction_qc/combat_batch_correct.py` |
| Train/test split | 3.3 | `train_test_split/create_splits.py` |
| Differential gene expression | 3.4 | `differential_expression/` *(limma-voom, R)* |
| Feature selection | 3.5 | `feature_selection/feature_selection_advanced.py` |
| Regression modelling | 3.6 | `regression_modeling/` |
| SMOGN oversampling *(tested, not adopted)* | 3.7 | `smogn/` |
| SHAP / interpretation | 5.2 | `interpretation/shap_ensemble_analysis.py` |
| PPI network | 5.3 | `interpretation/` *(STRING analysis)* |

## Requirements

- Python 3.11
- R (for the differential expression step, via `limma`/`limma-voom`)
- Key Python packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `combat`, `matplotlib`, `scipy`

A `requirements.txt` will be added with pinned versions.

## Usage

Each script can be run independently via the command line, e.g.:

```bash
python batch_correction_qc/combat_batch_correct.py \
    --X_log2 data/X_log2.csv \
    --metadata data/metadata.csv \
    --y data/y.csv \
    --outdir data/processed_combat \
    --batch_variable SMCENTER
```

See each script's `--help` for the full list of arguments.

## Data availability

This repository contains pipeline code only. Raw GTEx expression data is not included, as it is subject to GTEx's own access terms. The GTEx v11 dataset can be obtained from the [GTEx Portal](https://gtexportal.org/).

## Citation

If you use this pipeline, please cite the accompanying thesis:

> Duarte, M. (2026). *Ageing prediction model from blood gene expression*. Master's thesis, Faculdade de Ciências da Universidade do Porto.

## Contact

Marta Duarte — Faculdade de Ciências da Universidade do Porto
