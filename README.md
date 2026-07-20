# AutoML Studio for Binary Classification

**Free & open-source** AutoML pipeline for binary classification. Built with 100% free tools — Python, Streamlit, Pandas, Scikit-learn, Plotly. No paid APIs required.

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Vekri/AutoML-Studio-for-classification&branch=main&mainModule=streamlit_app.py)

> **Seeing "You do not have access to this app"?** The app is not deployed yet. Follow **[DEPLOY.md](DEPLOY.md)** step by step.

## Live Demo (Free Public Access)

Deploy for **free** so anyone can access it in the browser — no login required for users.

### Option 1: Streamlit Community Cloud (Recommended — Free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **New app**
3. Select repository: `Vekri/AutoML-Studio-for-classification`
4. Set **Main file path** to: `streamlit_app.py`
5. Click **Deploy**

Your public URL will be: `https://<your-app-name>.streamlit.app` — share this link with anyone.

### Option 2: Hugging Face Spaces (Free)

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** → Create new Space
2. Choose **Streamlit** SDK
3. Connect this GitHub repo or upload files
4. Set app file to `streamlit_app.py`

### Option 3: Run Locally (Free)

```bash
git clone https://github.com/Vekri/AutoML-Studio-for-classification.git
cd AutoML-Studio-for-classification
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open **http://localhost:8501** — click **Try Sample Data (Banking)** to test without uploading a file.

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

## Tech Stack (All Free & Open Source)

| Tool | License | Purpose |
|------|---------|---------|
| Python | PSF | Runtime |
| Streamlit | Apache 2.0 | Web UI |
| Pandas | BSD | Data processing |
| Scikit-learn | BSD | ML & feature selection |
| Plotly | MIT | Interactive charts |
| NumPy / SciPy | BSD | Numerics |

## Sample Data

Use `sample_data/banking_loan_default.csv` or click **Try Sample Data** in the app.

## Docker

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

## Export Artifacts

The export bundle includes:

- `processed_dataset.csv`
- `feature_list.json`
- `cleaning_config.json`
- `binning_config.json`
- `validation_report.json`
- `studio_manifest.json`

## License

MIT — free for anyone to use, modify, and share.
