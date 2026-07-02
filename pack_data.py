"""Токенизирует data/raw*/*.txt и пакует в data/train.bin / data/val.bin (uint16).

Потоковая запись: документы кодируются пачками и сразу дописываются на диск, без
накопления всего корпуса в RAM (корпус ~3.6ГБ / ~700M токенов не помещается в память).
Каждый документ с вероятностью VAL_FRAC уходит в val, остальное — в train.
"""
import os
import glob

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIRS = [os.path.join(BASE, "data", "raw"), os.path.join(BASE, "data", "raw_big")]
OUT_DIR = os.path.join(BASE, "data")

tokenizer = Tokenizer.from_file(os.path.join(BASE, "tokenizer", "tokenizer.json"))
eod_id = tokenizer.token_to_id("<|endofdialog|>")
assert eod_id is not None

files = []
for d in RAW_DIRS:
    files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))
print("Файлы:", [os.path.basename(f) for f in files])

CHUNK_DOCS = 8000       # документов на один encode_batch
VAL_FRAC = 0.003        # доля документов в валидацию
SEED = 42

rng = np.random.default_rng(SEED)
train_path = os.path.join(OUT_DIR, "train.bin")
val_path = os.path.join(OUT_DIR, "val.bin")
train_f = open(train_path, "wb")
val_f = open(val_path, "wb")
train_tok = val_tok = 0


def flush_batch(batch):
    """Кодирует пачку документов и дописывает токены на диск (train/val)."""
    global train_tok, val_tok
    if not batch:
        return
    encs = tokenizer.encode_batch(batch)
    for enc in encs:
        arr = np.array(enc.ids, dtype=np.uint16)
        if rng.random() < VAL_FRAC:
            arr.tofile(val_f)
            val_tok += len(arr)
        else:
            arr.tofile(train_f)
            train_tok += len(arr)


for path in files:
    docs_in_file = 0
    batch, buf = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endofdialog|>":
                buf.append("<|endofdialog|>")
                batch.append("\n".join(buf))
                buf = []
                if len(batch) >= CHUNK_DOCS:
                    flush_batch(batch)
                    docs_in_file += len(batch)
                    batch = []
            else:
                buf.append(line.rstrip("\n"))
        if buf:
            batch.append("\n".join(buf) + "\n<|endofdialog|>")
    flush_batch(batch)
    docs_in_file += len(batch)
    print(f"{os.path.basename(path)}: {docs_in_file} docs, "
          f"train {train_tok/1e6:.1f}M / val {val_tok/1e6:.2f}M токенов", flush=True)

train_f.close()
val_f.close()
print(f"\nГОТОВО: {train_path} {train_tok/1e6:.1f}M токенов, "
      f"{val_path} {val_tok/1e6:.2f}M токенов")
