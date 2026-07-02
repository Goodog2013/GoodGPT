"""Автономный сторож обучения GoodGPT.

Не зависит от агентов и соединения: следит за ростом logs/train.log и, если обучение
зависло/упало (лог не растёт дольше STALL_SEC, а метки завершения нет), перезапускает
train.py с --resume от последнего чекпоинта. События пишет в logs/watchdog.log.

Запуск (в фоне, отдельно от обучения):
  .\.venv\Scripts\python.exe watchdog.py
Остановить: закрыть процесс watchdog.py (обучение при этом не трогается).
"""
import os
import sys
import time
import subprocess
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(BASE, ".venv", "Scripts", "python.exe")
TRAIN_LOG = os.path.join(BASE, "logs", "train.log")
WLOG = os.path.join(BASE, "logs", "watchdog.log")
CKPT = os.path.join(BASE, "checkpoints", "goodgpt.pt")

DONE_MARKERS = ("Обучение завершено", "\xd0\x9e\xd0\xb1\xd1\x83\xd1\x87")  # текст + возможные кракозябры
STALL_SEC = 60 * 60      # считаем зависшим, если лог не рос столько секунд
CHECK_SEC = 120          # период проверки
MAX_RESTARTS = 8         # предохранитель от бесконечного цикла перезапусков

TRAIN_ARGS = [
    "train.py", "--resume",
    "--max_iters", "12000", "--batch_size", "6", "--grad_accum", "20",
    "--block_size", "1024", "--n_layer", "12", "--n_head", "12", "--n_embd", "768",
    "--lr", "3e-4", "--warmup", "800", "--dropout", "0.05",
]


def wlog(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(WLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_size():
    try:
        return os.path.getsize(TRAIN_LOG)
    except OSError:
        return -1


def training_done():
    try:
        with open(TRAIN_LOG, "r", encoding="utf-8", errors="ignore") as f:
            tail = f.read()[-4000:]
        return any(m in tail for m in DONE_MARKERS if m == "Обучение завершено")
    except OSError:
        return False


def kill_stale_training():
    """Убивает python-процессы, выполняющие train.py (чтобы не задвоить обучение)."""
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*train.py*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=30)
    except Exception as e:
        wlog(f"не удалось убить старый процесс: {e}")


def restart_training():
    kill_stale_training()
    time.sleep(8)  # дать освободиться видеопамяти
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = open(TRAIN_LOG, "a", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen([PY] + TRAIN_ARGS, cwd=BASE, env=env,
                     stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
    wlog("обучение перезапущено с --resume")


def main():
    wlog(f"сторож запущен. Порог зависания {STALL_SEC//60} мин, проверка каждые {CHECK_SEC} с.")
    last_size = log_size()
    last_change = time.time()
    restarts = 0

    while True:
        time.sleep(CHECK_SEC)

        if training_done():
            wlog("обнаружена метка завершения обучения — сторож выходит.")
            break

        sz = log_size()
        if sz > last_size:
            last_size = sz
            last_change = time.time()
            continue

        stalled = time.time() - last_change
        if stalled > STALL_SEC:
            if restarts >= MAX_RESTARTS:
                wlog(f"достигнут лимит перезапусков ({MAX_RESTARTS}) — сторож выходит, нужна помощь.")
                break
            if not os.path.exists(CKPT):
                wlog("лог не растёт, но чекпоинта нет — не перезапускаю (жду первый eval).")
                last_change = time.time()
                continue
            restarts += 1
            wlog(f"лог не рос {int(stalled)} с — считаю обучение упавшим. Перезапуск #{restarts}.")
            restart_training()
            time.sleep(60)
            last_size = log_size()
            last_change = time.time()


if __name__ == "__main__":
    sys.exit(main())
