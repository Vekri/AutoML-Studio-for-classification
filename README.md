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

# IntelliPredict AI — Binary Classification

**Repository:** [Vekri/Binary-Classification](https://github.com/Vekri/Binary-Classification)  
**Live target:** https://intellipredict-web.onrender.com

Full-stack no-code AutoML for binary classification — landing page, project dashboard, and 8-step pipeline.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Vekri/Binary-Classification)

## Routes

| Path | Page |
|------|------|
| `/` | Marketing landing |
| `/dashboard/projects` | Create & manage projects (templates like churn, fraud, credit approval) |
| `/dashboard/upload` … `/dashboard/deploy` | Pipeline steps inside dashboard shell |
| `/studio` | Standalone studio (legacy) |

## Deploy on Render

1. https://dashboard.render.com → **New +** → **Blueprint**
2. Connect **Vekri/Binary-Classification** → **Apply**
3. Service: `intellipredict-web` → https://intellipredict-web.onrender.com

Build: `pip install -r backend/requirements.txt`  
Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## Local

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860** (React UI is prebuilt in `frontend/dist`)

Dev mode: `START_DEV.bat` → http://localhost:5173

## License

MIT
