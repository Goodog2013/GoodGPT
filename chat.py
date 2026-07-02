"""Общение с обученной моделью GoodGPT."""
import os
import argparse

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

BASE = os.path.dirname(os.path.abspath(__file__))


def load(ckpt_name, device):
    tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer", "tokenizer.json"))
    ck = torch.load(os.path.join(BASE, "checkpoints", ckpt_name), map_location=device)
    config = GPTConfig(**ck["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="goodgpt_best.pt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--prompt", default=None, help="Один вопрос вместо интерактива")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.ckpt, device)
    u, b = "<|u|>", "<|b|>"
    eod_id = tok.token_to_id("<|endofdialog|>")
    b_id = tok.token_to_id("<|b|>")
    think_open = "<|think|>"
    think_close = "<|/think|>"

    def split_think(text):
        """Отделяет блок рассуждений от финального ответа."""
        text = text.replace(think_open, "")
        if think_close in text:
            think, answer = text.split(think_close, 1)
            return think.strip(), answer.strip()
        return "", text.strip()

    def ask(user_text, history=""):
        prompt = history + f"{u}{user_text}\n{b}"
        ids = tok.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, stop_token=eod_id,
        )
        gen_ids = out[0].tolist()[len(ids):]
        # обрезаем по спец-токенам границ реплик (но НЕ по think — он часть ответа)
        for stop in (eod_id, b_id, tok.token_to_id("<|u|>")):
            if stop in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(stop)]
        return split_think(tok.decode(gen_ids))

    def render(think, answer):
        if think:
            return f"[мысли] {think}\n{answer}"
        return answer

    if args.prompt is not None:
        think, answer = ask(args.prompt)
        print(render(think, answer))
        return

    print("GoodGPT готов. Пустая строка — выход, 'сброс' — очистить историю.\n")
    history = ""
    while True:
        try:
            user_text = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            break
        if user_text.lower() == "сброс":
            history = ""
            print("(история очищена)\n")
            continue
        think, answer = ask(user_text, history)
        print("GoodGPT:", render(think, answer), "\n")
        # в историю кладём только финальный ответ, без мыслей (как принято у R1)
        history += f"<|u|>{user_text}\n<|b|>{answer}\n<|endofdialog|>\n"
        # держим историю короткой
        if len(history) > 4000:
            history = history[-4000:]


if __name__ == "__main__":
    main()
