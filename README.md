---
title: AutoML Studio
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Free AutoML for binary classification — React + FastAPI
---

# AutoML Studio for Binary Classification

**Free & open-source** · React UI · FastAPI · Hugging Face Spaces

## Stack

| Layer | Tech |
|-------|------|
| UI | React + Vite + Recharts |
| API | FastAPI |
| ML | Pandas, Scikit-learn |
| Hosting | Hugging Face Spaces (Docker) — free & public |

## Quick start (local)

```bash
# Terminal 1 — API
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 7860

# Terminal 2 — React
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

Or double-click **`START_DEV.bat`** (dev) / **`START.bat`** (built UI on :7860).

## Deploy free on Hugging Face

1. Create a Space with **Docker** SDK: [huggingface.co/new-space](https://huggingface.co/new-space)
2. Connect GitHub repo `Vekri/AutoML-Studio-for-classification`
3. Public app: `https://huggingface.co/spaces/<you>/automl-studio-classification`

Anyone can open the link — no login required for visitors.

See **[DEPLOY.md](DEPLOY.md)** for step-by-step.

## Workflow

1. Upload CSV / sample data + business domain  
2. Target & keep/drop columns  
3. Cleaning recommendations  
4. Visualizations  
5. Binning + WoE/IV  
6. Feature selection  
7. Validation  
8. Export ZIP for Predictions Studio  

## License

MIT
