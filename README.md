# AutoML Studio for Binary Classification

End-to-end AutoML pipeline for binary classification with business domain support, data cleaning, visualizations, binning, feature selection, validation, and export for Predictions Studio.

## Features

- **CSV Upload** — Load and preview business datasets
- **Business Domain Flags** — Banking, Retail, Healthcare, Insurance, Telecom, Marketing, HR, Manufacturing
- **Target & Column Selection** — Keep/drop variables with binary target validation
- **Data Cleaning Recommendations** — Auto-detect missing values, duplicates, outliers, ID columns
- **Visualizations** — Distributions, box plots, correlation heatmaps, missing value charts
- **Binning** — Equal-width, equal-frequency binning with WoE/IV analysis
- **Feature Selection** — Correlation, mutual information, chi-square, ANOVA, RFE, Lasso
- **Validation** — Data quality checks + quick model validation (Logistic Regression, Random Forest)
- **Export** — Full ZIP bundle for Predictions Studio integration

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or on Windows, double-click `run.bat`.

## Sample Data

Use `sample_data/banking_loan_default.csv` to test the banking loan default workflow.

## Export Artifacts

The export bundle includes:

- `processed_dataset.csv`
- `feature_list.json`
- `cleaning_config.json`
- `binning_config.json`
- `validation_report.json`
- `studio_manifest.json`

## Tech Stack

Python · Streamlit · Pandas · Scikit-learn · Plotly
