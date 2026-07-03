@echo off
chcp 65001 >nul
title GoodGPT - остановка серверов
echo Останавливаю API модели (порт 8001) и сайт (порт 8080)...
echo (обучение train.py и сторож watchdog.py НЕ трогаются)
rem фильтр узкий: только наш API (--port 8001) и наш сайт (website\server.py) — чужие server.py не трогаем
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'website.server\.py' -or $_.CommandLine -match '--port 8001') } | ForEach-Object { Write-Host ('  stop pid ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }"
echo Готово.
pause
