"""GRPO-лайт дообучение GoodGPT с эвристической наградой (см. env.py).

На каждый промпт сэмплируется группа из K ответов, advantage = нормированная
внутри группы награда, обновление — policy gradient с KL-штрафом к исходной
(замороженной) модели, чтобы не разучиться говорить по-русски.

Запуск из корня проекта:
  .\\.venv\\Scripts\\python.exe rl\\train_rl.py --updates 300
Чекпоинт: checkpoints/goodgpt_rl.pt (совместим с chat.py --ckpt goodgpt_rl.pt)
"""
import os
import sys
import copy
import argparse

import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from model import GPT, GPTConfig  # noqa: E402
from rl.env import ChatEnv  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def load_policy(ckpt_name, device):
    tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer", "tokenizer.json"))
    ck = torch.load(os.path.join(BASE, "checkpoints", ckpt_name), map_location=device)
    config = GPTConfig(**ck["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ck["model"])
    return model, tok, config


@torch.no_grad()
def sample_group(model, prompt_ids, k, max_new, temperature, top_k, stop_ids, device):
    """K сэмплов на один промпт. Возвращает (seq [K,T], lens [K], stopped [K])."""
    block = model.config.block_size
    idx = torch.tensor([prompt_ids] * k, dtype=torch.long, device=device)
    done = torch.zeros(k, dtype=torch.bool, device=device)
    pad = stop_ids[0]

    for _ in range(max_new):
        logits, _ = model(idx[:, -block:])
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        nxt[done] = pad
        is_stop = torch.zeros_like(done)
        for s in stop_ids:
            is_stop |= nxt == s
        idx = torch.cat((idx, nxt.unsqueeze(1)), dim=1)
        done |= is_stop
        if done.all():
            break

    p = len(prompt_ids)
    lens, stopped = [], []
    for i in range(k):
        gen = idx[i, p:].tolist()
        L = len(gen)
        stop = False
        for j, t in enumerate(gen):
            if t in stop_ids:
                L = j + 1  # стоп-токен входит в градиент: учим останавливаться
                stop = True
                break
        lens.append(L)
        stopped.append(stop)
    return idx, lens, stopped


def sequence_logprobs(model, seq):
    """Пер-токенные log p(seq[t] | seq[<t]) формы [B, T-1]."""
    logits, _ = model(seq[:, :-1], seq[:, 1:].clone())  # targets -> полные логиты
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="goodgpt_best.pt")
    p.add_argument("--out", default="goodgpt_rl.pt")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--group_size", type=int, default=8, help="сэмплов на промпт (K)")
    p.add_argument("--prompts_per_update", type=int, default=2)
    p.add_argument("--max_new", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--kl_coef", type=float, default=0.2)
    p.add_argument("--kl_stop", type=float, default=1.0,
                   help="жёсткий стоп: |KL| выше порога = политика убежала от языка")
    p.add_argument("--save_interval", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    model, tok, config = load_policy(args.ckpt, device)
    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)

    env = ChatEnv()
    stop_ids = [tok.token_to_id("<|endofdialog|>"), tok.token_to_id("<|u|>"), tok.token_to_id("<|b|>")]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    out_path = os.path.join(BASE, "checkpoints", args.out)

    print(f"device={device}, политика из {args.ckpt}, {args.updates} обновлений, "
          f"K={args.group_size}, промптов/шаг={args.prompts_per_update}")

    running = None
    for upd in range(1, args.updates + 1):
        model.eval()
        batch = []  # (seq, prompt_len, lens, advantages, rewards, sample_text)
        for prompt in env.sample(args.prompts_per_update):
            prompt_ids = tok.encode(f"<|u|>{prompt}\n<|b|>").ids
            seq, lens, stopped = sample_group(
                model, prompt_ids, args.group_size, args.max_new,
                args.temperature, args.top_k, stop_ids, device,
            )
            rewards, texts = [], []
            for i in range(args.group_size):
                gen = seq[i, len(prompt_ids):len(prompt_ids) + lens[i]].tolist()
                gen_clean = [t for t in gen if t not in stop_ids]
                text = tok.decode(gen_clean).strip()
                rewards.append(env.reward(text, stopped[i]))
                texts.append(text)
            r = torch.tensor(rewards, device=device)
            adv = (r - r.mean()) / (r.std() + 1e-4)
            batch.append((seq, len(prompt_ids), lens, adv, r, texts[0]))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_kl, total_r = 0.0, 0.0, 0.0
        for seq, plen, lens, adv, r, _ in batch:
            logp = sequence_logprobs(model, seq)
            with torch.no_grad():
                ref_logp = sequence_logprobs(ref, seq)

            mask = torch.zeros_like(logp)
            for i, L in enumerate(lens):
                mask[i, plen - 1:plen - 1 + L] = 1.0

            kl = logp - ref_logp  # k1-оценка KL(policy || ref)
            per_tok = -adv.unsqueeze(1) * logp + args.kl_coef * kl
            loss = (per_tok * mask).sum() / mask.sum().clamp(min=1.0)
            (loss / len(batch)).backward()

            total_loss += loss.item()
            total_kl += ((kl.detach() * mask).sum() / mask.sum().clamp(min=1.0)).item()
            total_r += r.mean().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n = len(batch)
        mean_r = total_r / n
        running = mean_r if running is None else 0.95 * running + 0.05 * mean_r
        print(f"upd {upd}: reward {mean_r:+.3f} (сглаж. {running:+.3f}), "
              f"kl {total_kl / n:+.4f}, loss {total_loss / n:+.4f}", flush=True)
        if upd % 10 == 0:
            print(f"    пример: {batch[0][5][:120]!r}", flush=True)

        mean_kl = total_kl / n
        if abs(mean_kl) > args.kl_stop:
            print(f"СТОП: |KL|={abs(mean_kl):.3f} > {args.kl_stop} — политика убегает от "
                  f"исходной модели (reward hacking). Чекпоинт НЕ сохранён.", flush=True)
            break

        if upd % args.save_interval == 0 or upd == args.updates:
            torch.save(
                {"model": model.state_dict(), "config": config.__dict__, "iter": upd},
                out_path,
            )
            print(f"    чекпоинт -> {out_path}", flush=True)

    print("RL-дообучение завершено. Проверка: python chat.py --ckpt", args.out)


if __name__ == "__main__":
    main()
