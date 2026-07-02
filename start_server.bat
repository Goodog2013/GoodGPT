@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GoodGPT API Server (port 8000)
echo Запуск GoodGPT OpenAI-совместимого сервера на http://192.168.1.3:8000/v1 ...
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe server.py --host 0.0.0.0 --port 8000 --ckpt goodgpt_best.pt
pause
