# DEPLOY — Make AutoML Studio Public (Free)

## Why you see "You do not have access to this app"

This error means **the URL you opened does not exist yet** OR **belongs to someone else**.

You must **create and deploy** the app first. Opening a random `.streamlit.app` link will always fail until YOU deploy it.

---

# EASIEST: Hugging Face Spaces (Recommended)

Public URL anyone can use — no access errors.

### Step 1 — Create Hugging Face account (free)
Go to **[huggingface.co/join](https://huggingface.co/join)** → sign up (free)

### Step 2 — Create a Space
1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Fill in:
   - **Space name:** `automl-studio-classification`
   - **License:** MIT
   - **SDK:** Streamlit
   - **Hardware:** CPU basic (free)
3. Click **Create Space**

### Step 3 — Connect GitHub repo
1. On your Space page → **Settings** → **Repository**
2. Link to: `Vekri/AutoML-Studio-for-classification`
3. Set **App file** to: `streamlit_app.py`
4. Space rebuilds automatically

### Step 4 — Your public URL
```
https://huggingface.co/spaces/Vekri/automl-studio-classification
```
Share this link — **anyone can use it free, no login required**.

---

# OPTION 2: Streamlit Cloud

### Important: use a UNIQUE app name
Do NOT use generic names — they may be taken by others.
Use: **`vekri-automl-studio`** or **`automl-vekri-2026`**

### Steps
1. Open: **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub **Vekri**
2. Click **Create app**
3. Set:
   - Repository: `Vekri/AutoML-Studio-for-classification`
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - App URL: `vekri-automl-studio`
4. Click **Deploy** → wait 3 minutes
5. Your URL: **https://vekri-automl-studio.streamlit.app**

Only open the URL **after** deploy shows Running in your dashboard.

---

# OPTION 3: Render.com (Free)

1. **[dashboard.render.com](https://dashboard.render.com)** → sign up with GitHub
2. **New +** → **Blueprint**
3. Connect `Vekri/AutoML-Studio-for-classification`
4. Click **Apply**
5. URL: `https://automl-studio-classification.onrender.com`

---

# OPTION 4: Run on Your PC (Instant)

Double-click **`START.bat`**

Or:
```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```
Open **http://localhost:8501**

---

# Auto-deploy with GitHub Actions (Hugging Face)

1. Create HF account + Space (above)
2. Get token: **[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)** → New token (Write)
3. GitHub repo → **Settings** → **Secrets** → **Actions** → New secret:
   - Name: `HF_TOKEN`
   - Value: your HF token
4. Push any change to `main` — auto-deploys to your Space
