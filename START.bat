@echo off
title AutoML Studio
echo ========================================
echo   AutoML Studio - Binary Classification
echo   Free and Open Source
echo ========================================
echo.
echo Installing dependencies (first run only)...
pip install -r requirements.txt -q
echo.
echo Starting app at http://localhost:8501
echo Press Ctrl+C to stop
echo.
start http://localhost:8501
streamlit run streamlit_app.py
