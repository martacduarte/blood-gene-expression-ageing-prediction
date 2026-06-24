# Ageing prediction model from blood gene expression

Machine learning pipeline for predicting chronological age from whole blood RNA-seq gene expression data, developed for my Master's thesis (Faculdade de Ciências da Universidade do Porto, 2025/26). The pipeline uses the GTEx v11 whole blood cohort and combines differential gene expression pre-filtering, systematic feature selection, and a comparative regression framework (ElasticNet, Random Forest, XGBoost, Neural Network) to predict age from a 200-gene signature. Predicted age is used to derive age acceleration (ΔAge), a proxy for biological ageing.

The final model (XGBoost) achieves a test MAE of 7.35 years and R² of 0.506 on a held-out set of 152 donors. The gene signature is interpreted using ElasticNet coefficients and SHAP values, and cross-validated against published transcriptomic ageing signatures and a STRING protein-protein interaction network. Synthetic oversampling (SMOGN) was also evaluated to address training set age imbalance, but did not improve performance and was not adopted in the final pipeline (see `smogn/`).

## Repository structure

The pipeline is organised into:

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

Each folder contains the scripts for that pipeline stage.

## Requirements

- Python 3.11
- R (for the differential expression step, via `limma`/`limma-voom`)
- Key Python packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `combat`, `matplotlib`, `scipy`

## Data availability

This repository contains pipeline code only. Raw GTEx expression data is not included, as it is subject to GTEx's own access terms. The GTEx v11 dataset can be obtained from the [GTEx Portal](https://gtexportal.org/).

## Citation

If you use this pipeline, please cite the accompanying thesis:

> Duarte, M. (2026). *Ageing prediction models from blood gene expression*. Master's thesis, Faculdade de Ciências da Universidade do Porto.
