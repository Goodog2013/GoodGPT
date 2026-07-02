@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GoodGPT WebUI (port 8080)
echo Веб-чат: http://localhost:8080  (в сети: http://192.168.1.3:8080)
echo Не забудь запустить сам сервер модели: start_server.bat
start "" http://localhost:8080
.venv\Scripts\python.exe -m http.server 8080 --bind 0.0.0.0 --directory webui
pause
