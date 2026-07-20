# Deploy AutoML Studio for FREE (public access for anyone)

The error **"You do not have access to this app"** means the app was **never deployed** under your account, or you're opening an old/wrong URL.

Follow these steps exactly.

---

## Step 1 — Fix Streamlit ↔ GitHub connection

1. Open **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with **vekris363@gmail.com**
3. Click your **profile (top right)** → **Settings**
4. Under **Linked accounts**, confirm **GitHub** shows **`Vekri`**
   - If it shows a different GitHub account → **Disconnect** and reconnect the correct one
   - The GitHub user **`Vekri`** must own the repo

---

## Step 2 — Deploy a NEW app (do not open old URLs)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **Create app** (or **New app**)
3. Fill in exactly:

| Field | Value |
|-------|-------|
| Repository | `Vekri/AutoML-Studio-for-classification` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| App URL (optional) | `automl-studio-classification` |

4. Click **Deploy**
5. Wait 2–3 minutes for the build to finish

---

## Step 3 — Your public URL

After deploy succeeds, your live app will be at:

**https://automl-studio-classification.streamlit.app**

(or whatever App URL you chose)

Share this link — **anyone can use it for free**, no login required.

---

## One-click deploy link

Click this to pre-fill the deploy form:

**[Deploy to Streamlit Cloud](https://share.streamlit.io/deploy?repository=Vekri/AutoML-Studio-for-classification&branch=main&mainModule=streamlit_app.py)**

---

## Option B: Render.com (Free — Often Easier)

If Streamlit Cloud gives access errors, use Render instead:

1. Go to **[dashboard.render.com](https://dashboard.render.com)** → sign up free with GitHub
2. Click **New +** → **Blueprint**
3. Connect repo: `Vekri/AutoML-Studio-for-classification`
4. Render reads `render.yaml` automatically → click **Apply**
5. Wait ~5 minutes — you get a public URL like:
   `https://automl-studio-classification.onrender.com`

Anyone can access it — completely free.

---

## Option C: Run on Your PC (Instant)

Check the **Logs** tab in Streamlit Cloud. Common fixes:

- **Main file** must be `streamlit_app.py` (not `app.py`)
- **Repo must be public** — it is: [github.com/Vekri/AutoML-Studio-for-classification](https://github.com/Vekri/AutoML-Studio-for-classification)
- **GitHub access** — grant Streamlit access to the `Vekri` org/account when prompted

---

## Option C: Run on Your PC (Instant)

Double-click **`START.bat`** in the project folder.

Or run manually:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open **http://localhost:8501** → click **Try Sample Data (Banking)**

---

## If Streamlit deploy fails

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Space name: `automl-studio-classification`
3. SDK: **Streamlit**
4. Create Space, then upload this repo or connect GitHub
5. Set app file to `streamlit_app.py`

Public URL: `https://huggingface.co/spaces/YOUR_USERNAME/automl-studio-classification`
