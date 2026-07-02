@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GoodGPT RL training
set PYTHONIOENCODING=utf-8
rem Доп. аргументы можно передать прямо батнику: train_rl.bat --updates 500
.venv\Scripts\python.exe rl\train_rl.py --updates 300 %*
pause
