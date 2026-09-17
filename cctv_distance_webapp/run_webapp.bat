@echo off
cd /d "%~dp0.."
call venv\Scripts\activate
py -m uvicorn cctv_distance_webapp.server:app --host 127.0.0.1 --port 8000
