# GoodGPT-01

Маленькая разговорная нейросеть (decoder-only GPT), обученная с нуля на русском чате.
Не супер-крутая — просто чтобы поговорить.

## Что это

- Архитектура: decoder-only Transformer (nanoGPT-стиль), ~33.6M параметров.
- L8 / H8 / E512 / block_size 512 / vocab 16384 (BPE, byte-level).
- Обучена с нуля, без предобученных весов.
- Данные: ~90M символов чистого русского чата (`IlyaGusev/saiga_scored`, opus_score ≥ 7)
  плюс ~53M символов русских инструкций/диалогов из локального корпуса.
- Итог обучения: 10000 итераций, best val loss ≈ 2.54 (RTX 3060, ~3 часа).

## Установка

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install tokenizers datasets numpy tqdm
```

## Поговорить с моделью

Интерактивный чат:

```powershell
.\.venv\Scripts\python.exe chat.py
```

Один вопрос:

```powershell
.\.venv\Scripts\python.exe chat.py --prompt "Привет! Как дела?" --temperature 0.7 --top_k 40
```

Прогон по набору тестовых вопросов:

```powershell
.\.venv\Scripts\python.exe test_chat.py
```

## Сайт (чат в браузере)

Полноценный веб-интерфейс: лендинг, мультичаты, аккаунты (SQLite), гостевой режим,
стриминг ответов и блок «размышлений».

```powershell
start_server.bat    # API модели (порт 8001, на CPU во время обучения)
start_website.bat   # сайт (порт 8080) -> http://localhost:8080/
stop_all.bat        # остановить сервер и сайт (обучение не трогает)
```

В локальной сети: `http://192.168.1.3:8080/`. Код сайта — в `website/`
(бэкенд на чистом stdlib + sqlite3, фронтенд без фреймворков).

## Воспроизвести обучение с нуля

```powershell
# 1. подготовка данных (качает saiga + собирает локальный микс)
.\.venv\Scripts\python.exe prepare_data.py local
.\.venv\Scripts\python.exe prepare_data.py saiga

# 2. токенизатор
.\.venv\Scripts\python.exe train_tokenizer.py

# 3. упаковка в бинарники
.\.venv\Scripts\python.exe pack_data.py

# 4. обучение (~3 часа на RTX 3060)
.\.venv\Scripts\python.exe train.py --max_iters 10000 --batch_size 32 --grad_accum 4 --dropout 0.1

# докрутить с последнего чекпоинта
.\.venv\Scripts\python.exe train.py --resume --max_iters 15000 --batch_size 32 --grad_accum 4
```

## Файлы

| Файл | Назначение |
|---|---|
| `model.py` | архитектура GPT |
| `prepare_data.py` | сбор корпуса в `data/raw/*.txt` |
| `train_tokenizer.py` | обучение BPE-токенизатора |
| `pack_data.py` | токенизация → `data/train.bin` / `data/val.bin` |
| `train.py` | цикл обучения (AdamW, cosine LR, bf16, авто-чекпоинты) |
| `chat.py` | интерактивный чат / одиночный запрос |
| `test_chat.py` | прогон по набору вопросов |
| `checkpoints/goodgpt_best.pt` | лучший чекпоинт по val loss |