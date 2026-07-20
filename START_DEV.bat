@echo off
echo Dev mode: API on :7860 + Vite on :5173
start "AutoML API" cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --reload --port 7860"
timeout /t 2 >nul
start "AutoML UI" cmd /k "cd /d %~dp0frontend && npm install && npm run dev"
start http://localhost:5173
