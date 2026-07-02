"""Обучает BPE-токенизатор на data/raw*/*.txt -> tokenizer/tokenizer.json"""
import os
import glob

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIRS = [os.path.join(BASE, "data", "raw"), os.path.join(BASE, "data", "raw_big")]
OUT_DIR = os.path.join(BASE, "tokenizer")
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 32000
# think-токены нужны для reasoning-формата <|b|><|think|>...<|/think|>ответ
SPECIAL_TOKENS = ["<|u|>", "<|b|>", "<|endofdialog|>", "<|pad|>", "<|think|>", "<|/think|>"]

files = []
for d in RAW_DIRS:
    files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))
assert files, "Нет файлов в data/raw*"
print("Файлы:", [os.path.basename(f) for f in files])

tokenizer = Tokenizer(models.BPE(unk_token=None))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train(files, trainer)
out_path = os.path.join(OUT_DIR, "tokenizer.json")
tokenizer.save(out_path)
print("Сохранено:", out_path)

# быстрая проверка
t = Tokenizer.from_file(out_path)
s = ("<|u|>сколько будет 2+2?\n<|b|><|think|>складываю два и два, получается четыре"
     "<|/think|>Будет 4.\n<|endofdialog|>")
ids = t.encode(s).ids
print("Токенов в примере:", len(ids))
print("Раскодировано:", t.decode(ids))
for tok in ("<|think|>", "<|/think|>"):
    print(f"{tok} id =", t.token_to_id(tok))
