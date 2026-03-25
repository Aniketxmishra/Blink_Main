@echo off
echo Starting Blink REST API...
echo Docs available at: http://localhost:8000/docs
echo.
cd /d %~dp0..
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
