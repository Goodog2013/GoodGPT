@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GoodGPT 100M training (block_size 1024)
set PYTHONIOENCODING=utf-8
rem 100M-модель: 12 слоёв, 768 dim, контекст 1024, reasoning-формат <|think|>.
rem Доп. аргументы можно передать батнику, напр.: train_big.bat --resume
.venv\Scripts\python.exe train.py --max_iters 12000 --batch_size 6 --grad_accum 20 --block_size 1024 --n_layer 12 --n_head 12 --n_embd 768 --lr 3e-4 --warmup 800 --dropout 0.05 %* > logs\train.log 2>&1
echo Обучение завершено. Лог: logs\train.log
pause
