"""Быстрый прогон модели по набору вопросов."""
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
ck = torch.load("checkpoints/goodgpt_best.pt", map_location=device)
model = GPT(GPTConfig(**ck["config"])).to(device)
model.load_state_dict(ck["model"])
model.eval()

eod_id = tok.token_to_id("<|endofdialog|>")
u_id = tok.token_to_id("<|u|>")
b_id = tok.token_to_id("<|b|>")

questions = [
    "Привет! Как тебя зовут?",
    "Расскажи, что такое солнце?",
    "Посоветуй книгу для вечернего чтения.",
    "Почему небо голубое?",
    "Мне грустно сегодня.",
    "Как приготовить омлет?",
    "Сколько будет 2+2?",
    "Что ты умеешь?",
]

for q in questions:
    ids = tok.encode(f"<|u|>{q}\n<|b|>").ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=120, temperature=0.7, top_k=40, stop_token=eod_id)
    gen = out[0].tolist()[len(ids):]
    for stop in (eod_id, u_id, b_id):
        if stop in gen:
            gen = gen[:gen.index(stop)]
    print(f"Ты: {q}")
    print(f"GoodGPT: {tok.decode(gen).strip()}\n")
