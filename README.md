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

**Free & open-source** · React UI · FastAPI · Deploy on **Render / Railway / Fly.io**

## Stack

| Layer | Tech |
|-------|------|
| UI | React + Vite + Recharts |
| API | FastAPI |
| ML | Pandas, Scikit-learn |
| Hosting | **Render.com (free)** · Railway · Fly.io · HF Spaces |

## Deploy free (public URL)

### Recommended: Render.com

1. Go to **[dashboard.render.com](https://dashboard.render.com)** → sign in with GitHub  
2. **New +** → **Blueprint** → select `Vekri/AutoML-Studio-for-classification`  
3. Click **Apply** → wait for build  

Public URL: **`https://automl-studio-classification.onrender.com`**

Full guide: **[DEPLOY.md](DEPLOY.md)**

## Local quick start

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

Or double-click **`START_DEV.bat`** / **`START.bat`**.

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
