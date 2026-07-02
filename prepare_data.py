"""Готовит корпус в data/raw/*.txt в едином формате:

<|u|>реплика пользователя
<|b|>ответ ассистента
<|endofdialog|>

Источники:
1. IlyaGusev/saiga_scored — большой чистый русский чат (основа диалогов);
2. локальные JSONL из data/raw_src — доп. русский язык/инструкции (как документы).
"""
import os
import re
import json
import glob
import argparse

from datasets import load_dataset

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
SRC_DIR = os.path.join(BASE, "data", "raw_src")
os.makedirs(RAW_DIR, exist_ok=True)

U, B, EOD = "<|u|>", "<|b|>", "<|endofdialog|>"


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def do_saiga(min_score: int, limit_chars: int):
    """Русский чат из saiga_scored, фильтр по opus_score."""
    ds = load_dataset("IlyaGusev/saiga_scored", split="train", streaming=True)
    path = os.path.join(RAW_DIR, "saiga.txt")
    total = 0
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in ds:
            score = row.get("opus_score")
            if score is not None and score < min_score:
                continue
            if row.get("is_bad_by_regex"):
                continue
            msgs = row.get("messages")
            if not msgs or len(msgs) < 2:
                continue
            parts = []
            bad = False
            for m in msgs:
                role = m.get("role")
                content = clean(m.get("content") or "")
                if role == "system":
                    continue
                if not content or len(content) > 4000:
                    bad = True
                    break
                tag = U if role == "user" else B
                parts.append(f"{tag}{content}")
            if bad or len(parts) < 2:
                continue
            s = "\n".join(parts) + f"\n{EOD}\n"
            f.write(s)
            total += len(s)
            n += 1
            if total >= limit_chars:
                break
    print(f"saiga.txt: {n} dialogs, {total/1e6:.1f}M chars")


def _looks_like_chat_pair(text: str):
    """Разбирает 'Пользователь: X\\nОтвет: Y' -> (user, bot) или None."""
    m = re.match(r"^\s*(?:Пользователь|Вопрос|User)\s*:\s*(.+?)\n\s*(?:Ответ|Assistant|Бот)\s*:\s*(.+)\s*$",
                 text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def do_local_jsonl(limit_chars: int):
    """Локальные JSONL: chat-пары -> u/b, остальное -> документ."""
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.jsonl")))
    print("Локальные файлы:", [os.path.basename(f) for f in files])
    path = os.path.join(RAW_DIR, "local_mix.txt")
    total = 0
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = clean(obj.get("text") or "")
                    if len(text) < 10 or len(text) > 8000:
                        continue
                    pair = _looks_like_chat_pair(text)
                    if pair:
                        u, b = pair
                        s = f"{U}{u}\n{B}{b}\n{EOD}\n"
                    else:
                        s = f"{text}\n{EOD}\n"
                    f.write(s)
                    total += len(s)
                    n += 1
                    if total >= limit_chars:
                        break
            if total >= limit_chars:
                break
    print(f"local_mix.txt: {n} docs, {total/1e6:.1f}M chars")


STAGES = {
    "saiga": lambda: do_saiga(min_score=7, limit_chars=200_000_000),
    "local": lambda: do_local_jsonl(limit_chars=200_000_000),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    args = parser.parse_args()
    if args.stage == "all":
        for name, fn in STAGES.items():
            print(f"=== {name} ===")
            fn()
    else:
        STAGES[args.stage]()
