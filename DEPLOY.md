# AutoML Studio (React + Hugging Face)

Free open-source **binary classification** AutoML studio.

- **Frontend:** React (Vite) + Recharts  
- **Backend:** FastAPI + Pandas + Scikit-learn  
- **Deploy:** Hugging Face Spaces (Docker)

## Features

1. Upload CSV / sample banking data  
2. Business domain (Banking, Retail, Healthcare, …)  
3. Target + keep/drop columns  
4. Cleaning recommendations  
5. Visualizations  
6. Binning + WoE/IV  
7. Feature selection  
8. Validation + ZIP export for Predictions Studio  

## Local development

### 1. Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 7860
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** (Vite proxies `/api` → backend).

### One-command (production-like)

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860**

## Deploy to Hugging Face Spaces (free + public)

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Space name: `automl-studio-classification`
3. SDK: **Docker**
4. Create Space → **Files** → upload this repo **or** connect GitHub `Vekri/AutoML-Studio-for-classification`
5. Wait for build

Public URL:
`https://huggingface.co/spaces/Vekri/automl-studio-classification`

Anyone can use it — free, no login for visitors.

### Connect GitHub to Space

Space → **Settings** → **Repository** → link this GitHub repo.  
Pushes to `main` rebuild the Space automatically.

## API

- `GET /api/health`
- `GET /api/domains`
- `POST /api/upload`
- `POST /api/sample`
- Session endpoints under `/api/session/{id}/...`

## License

MIT
