#!/bin/bash
set -e
PORT="${PORT:-7860}"
echo "=== AutoML Studio startup ==="
echo "Python: $(python --version)"
echo "PORT: ${PORT}"
if [ -f /app/frontend/dist/index.html ]; then
  echo "Frontend: OK"
else
  echo "WARNING: Frontend missing"
fi
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}"
