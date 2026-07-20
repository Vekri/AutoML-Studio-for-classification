# Deploy AutoML Studio — FREE public servers

The Hugging Face 404 means that Space was never created. Use one of these **other free hosts** instead.

---

## Option 1 — Render.com (Recommended · Free)

### Steps

1. Open **[dashboard.render.com](https://dashboard.render.com)**
2. Sign up with **GitHub** (account that owns `Vekri`)
3. Click **New +** → **Blueprint**
4. Select repo: **`Vekri/AutoML-Studio-for-classification`**
5. Click **Apply**
6. Wait 5–10 minutes for Docker build

### Your public URL

```
https://automl-studio-classification.onrender.com
```

Anyone can open it — free, no login for visitors.

### If Blueprint fails — create Web Service manually

1. **New +** → **Web Service**
2. Connect the same GitHub repo
3. Settings:

| Field | Value |
|-------|-------|
| Runtime | **Docker** |
| Branch | `main` |
| Region | Oregon (or closest) |
| Instance type | **Free** |
| Health check path | `/api/health` |

4. Click **Deploy**

### First load note

Free Render sleeps after ~15 min idle. First open may take 30–60 seconds — wait, then refresh.

---

## Option 2 — Railway.app (Free trial / hobby)

1. Open **[railway.app](https://railway.app)** → Login with GitHub
2. **New Project** → **Deploy from GitHub repo**
3. Choose `Vekri/AutoML-Studio-for-classification`
4. Railway detects `Dockerfile` automatically
5. Generate a public domain: **Settings** → **Networking** → **Generate Domain**

URL looks like: `https://automl-studio-classification-production.up.railway.app`

---

## Option 3 — Fly.io (Free allowance)

Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/), then:

```bash
fly auth login
fly launch --name automl-studio-classification --no-deploy
fly deploy
```

App URL: `https://automl-studio-classification.fly.dev`

---

## Verify deploy worked

Open:

```
https://YOUR-URL/api/health
```

Expected:

```json
{"status":"ok","app":"AutoML Studio","version":"2.0.0"}
```

Then open the root URL — React AutoML Studio UI.

---

## Local (always works)

```bash
# Dev UI
START_DEV.bat

# Or single server
cd frontend && npm run build && cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

- UI (dev): http://localhost:5173  
- Built: http://localhost:7860  

---

## Which should you pick?

| Host | Best for | Catch |
|------|----------|--------|
| **Render** | Easiest free public URL | Sleeps when idle |
| **Railway** | Fast deploys | Free credits limited |
| **Fly.io** | Always-on feel | Needs CLI |
| Hugging Face | ML demos | Space must be created first |

**Start with Render** — it's the simplest free public option for this app.
