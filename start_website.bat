@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GoodGPT Website (port 8080)
set PYTHONIOENCODING=utf-8
echo Запуск сайта GoodGPT на http://localhost:8080/ ...
echo (API модели нужен отдельно: start_server.bat)
start "" http://localhost:8080/
.venv\Scripts\python.exe website\server.py --host 0.0.0.0 --port 8080
pause
