# Hugging Face / Docker / any cloud — React + FastAPI
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY config.py /app/config.py
COPY modules /app/modules
COPY sample_data /app/sample_data
COPY backend /app/backend
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

EXPOSE 7860

# Direct command — avoids Windows CRLF issues with shell scripts
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
