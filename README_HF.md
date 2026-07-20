---
title: AutoML Studio
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# AutoML Studio for Binary Classification

Free, open-source AutoML pipeline. Upload CSV, clean data, visualize, bin features, select variables, validate, and export.

## Features

- CSV upload with business domain flags (Banking, Retail, Healthcare, etc.)
- Target selection, keep/drop columns
- Data cleaning recommendations
- Distributions & visualizations
- Binning with WoE/IV
- Feature selection (6 methods)
- Validation & export for Predictions Studio

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
