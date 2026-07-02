"""Обучение GoodGPT на data/train.bin. Автосохранение чекпоинтов + resume."""
import os
import math
import time
import argparse

import numpy as np
import torch

from model import GPT, GPTConfig

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
CKPT_DIR = os.path.join(BASE, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, data_splits, block_size, batch_size, device, iters=50):
    model.eval()
    out = {}
    for split, data in data_splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_iters", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--grad_accum", type=int, default=10)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=800)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data)/1e6:.1f}M, val tokens: {len(val_data)/1e6:.1f}M")
    data_splits = {"train": train_data, "val": val_data}

    # vocab_size из токенизатора
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer", "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    config = GPTConfig(
        vocab_size=vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"параметров: {model.num_params()/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    ckpt_path = os.path.join(CKPT_DIR, "goodgpt.pt")
    best_path = os.path.join(CKPT_DIR, "goodgpt_best.pt")
    start_iter = 0
    best_val = float("inf")

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck["iter"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"resume с итерации {start_iter}, best_val={best_val:.4f}")

    if args.compile:
        model = torch.compile(model)

    t0 = time.time()
    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % 50 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it}: loss {loss.item()*args.grad_accum:.4f}, lr {lr:.2e}, {dt:.1f}s/50it", flush=True)

        if it % args.eval_interval == 0 and it > 0:
            losses = estimate_loss(model, data_splits, args.block_size, args.batch_size, device)
            print(f">>> iter {it}: train {losses['train']:.4f}, val {losses['val']:.4f}", flush=True)
            raw_model = getattr(model, "_orig_mod", model)
            ck = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it,
                "best_val": best_val,
                "config": config.__dict__,
            }
            torch.save(ck, ckpt_path)
            if losses["val"] < best_val:
                best_val = losses["val"]
                ck["best_val"] = best_val
                torch.save(ck, best_path)
                print(f"    новый best_val: {best_val:.4f}", flush=True)

    print("Обучение завершено.")


if __name__ == "__main__":
    main()
