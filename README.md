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

**Free & open-source** · React + FastAPI

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Vekri/AutoML-Studio-for-classification)

**Live Docker image:** `ghcr.io/vekri/automl-studio-for-classification:latest`

## Deploy to Hugging Face (recommended)

1. Create a write token at https://huggingface.co/settings/tokens  
2. GitHub → **Settings → Secrets → Actions** → add `HF_TOKEN`  
3. Push to `main` (auto-deploy) **or** **Actions → Deploy to Hugging Face Space → Run workflow**

App URL: https://vekri-automl-studio-classification.hf.space

## Deploy free on Render (works)

1. **[dashboard.render.com](https://dashboard.render.com)** → login with GitHub  
2. **New +** → **Web Service** → repo `Vekri/AutoML-Studio-for-classification`  
3. Settings:

```
Runtime:        Python 3
Build Command:  pip install -r backend/requirements.txt
Start Command:  uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Instance:       Free
PYTHON_VERSION: 3.11.0
```

4. Deploy → open your `*.onrender.com` URL  

Full steps: **[DEPLOY.md](DEPLOY.md)**

## Local

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860** (React UI is prebuilt in `frontend/dist`)

Dev mode: `START_DEV.bat` → http://localhost:5173

## License

MIT
