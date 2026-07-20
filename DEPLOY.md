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

## If deploy fails

Check the **Logs** tab in Streamlit Cloud. Common fixes:

- **Main file** must be `streamlit_app.py` (not `app.py`)
- **Repo must be public** — it is: [github.com/Vekri/AutoML-Studio-for-classification](https://github.com/Vekri/AutoML-Studio-for-classification)
- **GitHub access** — grant Streamlit access to the `Vekri` org/account when prompted

---

## Alternative: Run locally (always works)

```bash
git clone https://github.com/Vekri/AutoML-Studio-for-classification.git
cd AutoML-Studio-for-classification
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open **http://localhost:8501** → click **Try Sample Data (Banking)**

---

## Alternative: Hugging Face Spaces (also free)

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Space name: `automl-studio-classification`
3. SDK: **Streamlit**
4. Create Space, then upload this repo or connect GitHub
5. Set app file to `streamlit_app.py`

Public URL: `https://huggingface.co/spaces/YOUR_USERNAME/automl-studio-classification`
