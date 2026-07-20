# AutoML Studio for Binary Classification

**Free & open-source** AutoML pipeline. Built with 100% free tools.

## Public Live App

> **"You do not have access to this app"** = app not deployed yet. See **[DEPLOY.md](DEPLOY.md)**.

| Platform | Public URL (after you deploy) |
|----------|-------------------------------|
| Hugging Face (easiest) | `https://huggingface.co/spaces/Vekri/automl-studio-classification` |
| Streamlit Cloud | `https://vekri-automl-studio.streamlit.app` |
| Render.com | `https://automl-studio-classification.onrender.com` |

**[Full deploy guide → DEPLOY.md](DEPLOY.md)**

## Quick Start (Local)

Double-click **`START.bat`** or:

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501** → **Try Sample Data (Banking)**

## Features

- CSV upload + business domain flags (Banking, Retail, Healthcare, etc.)
- Target selection, keep/drop columns
- Data cleaning recommendations
- Visualizations, binning, feature selection
- Validation & export for Predictions Studio

## License

MIT — free for anyone to use.
