# Free deploy that WORKS — Render (Python, no Docker)

Docker builds often fail / OOM on free tiers. This app now deploys as **native Python** with a **prebuilt React UI** in `frontend/dist`.

---

## Deploy on Render (do this)

### 1. Open Render
**[https://dashboard.render.com](https://dashboard.render.com)** → Sign in with GitHub

### 2. Create Web Service (manual — most reliable)

1. Click **New +** → **Web Service**
2. Connect repo: **`Vekri/AutoML-Studio-for-classification`**
3. Fill in:

| Setting | Value |
|---------|-------|
| Name | `automl-studio-classification` |
| Region | Oregon |
| Runtime | **Python 3** |
| Branch | `main` |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Instance type | **Free** |

4. Environment → Add:

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.11.0` |

5. Click **Deploy Web Service**

### 3. Wait for build (green “Live”)

First deploy: 3–8 minutes.

### 4. Open your URL

Render shows a URL like:

```
https://automl-studio-classification.onrender.com
```

Test API:

```
https://automl-studio-classification.onrender.com/api/health
```

Should return: `{"status":"ok",...}`

Then open the root URL for the React app.

---

## OR use Blueprint

**New +** → **Blueprint** → select this repo → **Apply**  
(`render.yaml` is already set for Python free tier.)

---

## Important free-tier notes

- App **sleeps after ~15 min** idle  
- First open after sleep: wait **30–60 seconds**, then refresh  
- If build fails, open **Logs** and paste the error here  

---

## Alternative free hosts

### Railway
1. [railway.app](https://railway.app) → New Project → Deploy from GitHub  
2. Use **Dockerfile** or set start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`  
3. Generate domain under Settings → Networking  

### Local (always works)
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```
Open http://localhost:7860

Or `START_DEV.bat` for hot reload on http://localhost:5173

---

## If you still see errors

Paste **exact** text from Render **Logs** (build or runtime). Common fixes:

| Error | Fix |
|-------|-----|
| ModuleNotFoundError | Build command must use `backend/requirements.txt` |
| Port bind failed | Start command must use `$PORT` |
| Application failed to respond | Wait for cold start, hit `/api/health` |
| Out of memory | Free tier is 512MB — use sample data, not huge CSVs |
