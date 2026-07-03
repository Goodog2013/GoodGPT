@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GoodGPT API Server (port 8001)
echo Запуск GoodGPT OpenAI-совместимого сервера на http://192.168.1.3:8001/v1 ...
set PYTHONIOENCODING=utf-8
rem ВО ВРЕМЯ ОБУЧЕНИЯ запускать с --device cpu, иначе train.py может упасть с CUDA OOM!
.venv\Scripts\python.exe server.py --host 0.0.0.0 --port 8001 --ckpt goodgpt_best.pt --device cpu
pause
