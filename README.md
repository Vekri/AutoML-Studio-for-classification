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

**Free & open-source** · React + FastAPI · Landing page + AutoML studio

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Vekri/AutoML-Studio-for-classification)

**Target URL:** https://intellipredict-web.onrender.com (Render service `intellipredict-web`)

## Routes

| Path | Page |
|------|------|
| `/` | Marketing landing (like intellipredict-web.onrender.com) |
| `/studio` | Full AutoML pipeline workspace |
| `/dashboard` | Redirects to `/studio` |

## Deploy on Render

1. **[dashboard.render.com](https://dashboard.render.com)** → **New +** → **Blueprint**
2. Connect repo `Vekri/AutoML-Studio-for-classification` → **Apply**
3. Uses `render.yaml` → service name `intellipredict-web`
4. Open your `*.onrender.com` URL

Or one-click: use the Deploy to Render button above.

Build command: `pip install -r backend/requirements.txt`  
Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## Local

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860** (React UI is prebuilt in `frontend/dist`)

Dev mode: `START_DEV.bat` → http://localhost:5173

## License

MIT
