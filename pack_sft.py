"""Собирает SFT-корпус: персональность (с апсемплом) + выборка обычных диалогов saiga.

Выход: data/sft_train.bin, data/sft_val.bin (uint16).
Микс: persona.txt x PERSONA_REPEAT + SAIGA_FRAC случайных диалогов из data/raw/saiga.txt —
чтобы модель выучила «кто она», но не разучилась разговаривать.
"""
import os
import random

import numpy as np
from tokenizers import Tokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
PERSONA = os.path.join(BASE, "data", "sft_raw", "persona.txt")
SAIGA = os.path.join(BASE, "data", "raw", "saiga.txt")

PERSONA_REPEAT = 20     # персона повторяется, чтобы весила ~5% корпуса
SAIGA_FRAC = 0.12       # доля диалогов saiga в миксе
VAL_FRAC = 0.02
rng = random.Random(42)

tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer", "tokenizer.json"))


def read_dialogs(path):
    """Режет файл на диалоги по строке <|endofdialog|>."""
    dialogs, buf = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endofdialog|>":
                buf.append("<|endofdialog|>")
                dialogs.append("\n".join(buf))
                buf = []
            else:
                buf.append(line.rstrip("\n"))
    if buf:
        dialogs.append("\n".join(buf) + "\n<|endofdialog|>")
    return dialogs


persona = read_dialogs(PERSONA)
saiga = read_dialogs(SAIGA)
saiga_sample = [d for d in saiga if rng.random() < SAIGA_FRAC]
docs = persona * PERSONA_REPEAT + saiga_sample
rng.shuffle(docs)
print(f"персона: {len(persona)} диалогов x{PERSONA_REPEAT}, saiga: {len(saiga_sample)} из {len(saiga)}")

train_f = open(os.path.join(BASE, "data", "sft_train.bin"), "wb")
val_f = open(os.path.join(BASE, "data", "sft_val.bin"), "wb")
train_tok = val_tok = 0
BATCH = 4000
for i in range(0, len(docs), BATCH):
    for enc in tok.encode_batch(docs[i:i + BATCH]):
        arr = np.array(enc.ids, dtype=np.uint16)
        if rng.random() < VAL_FRAC:
            arr.tofile(val_f)
            val_tok += len(arr)
        else:
            arr.tofile(train_f)
            train_tok += len(arr)
train_f.close(); val_f.close()
print(f"ГОТОВО: sft_train.bin {train_tok/1e6:.1f}M токенов, sft_val.bin {val_tok/1e6:.2f}M")
