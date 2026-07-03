"""OpenAI-совместимый API-сервер для GoodGPT (только stdlib, без зависимостей).

Эндпоинты:
  GET  /v1/models
  POST /v1/chat/completions  (поддерживает stream=true)

Запуск: python server.py [--host 0.0.0.0] [--port 8000] [--ckpt goodgpt_best.pt]
"""
import os
import json
import time
import uuid
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "goodgpt-01"

# --- глобальное состояние модели ---
g = {"model": None, "tok": None, "device": None, "eod": None, "u": None, "b": None}
gen_lock = threading.Lock()  # одна генерация за раз (одна GPU)


def load_model(ckpt_name, device):
    tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer", "tokenizer.json"))
    ck = torch.load(os.path.join(BASE, "checkpoints", ckpt_name), map_location=device)
    config = GPTConfig(**ck["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    g.update(
        model=model, tok=tok, device=device,
        eod=tok.token_to_id("<|endofdialog|>"),
        u=tok.token_to_id("<|u|>"),
        b=tok.token_to_id("<|b|>"),
        think_open=tok.token_to_id("<|think|>"),
        think_close=tok.token_to_id("<|/think|>"),
    )
    print(f"Модель загружена: {ckpt_name}, {model.num_params()/1e6:.1f}M параметров, device={device}")


def build_prompt(messages):
    """OpenAI messages -> формат обучения: <|u|>...\n<|b|>...\n<|endofdialog|>\n"""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    prompt = ""
    first_user = True
    for i, m in enumerate(turns):
        if m["role"] == "user":
            text = m["content"]
            if first_user and system:
                text = system + "\n" + text
                first_user = False
            prompt += f"<|u|>{text}\n<|b|>"
        else:
            prompt += f"{m['content']}\n<|endofdialog|>\n"
    if not prompt.endswith("<|b|>"):
        prompt += "<|b|>"
    return prompt


@torch.no_grad()
def stream_generate(prompt, max_new_tokens, temperature, top_k):
    """Потоковая генерация. Выдаёт кортежи:
        (delta_text, phase, finish)
      phase = "reasoning" (внутри <|think|>) или "answer"; finish != None в конце.
    """
    model, tok, device = g["model"], g["tok"], g["device"]
    stops = {g["eod"], g["u"], g["b"]}
    th_open, th_close = g["think_open"], g["think_close"]
    block = model.config.block_size

    ids = tok.encode(prompt).ids
    ids = ids[-(block - min(max_new_tokens, 256)):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # раздельно копим токены мыслей и ответа, чтобы decode давал корректный текст
    phase = "answer"
    think_ids, ans_ids = [], []
    prev_think, prev_ans = "", ""
    finish = "length"

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block else idx[:, -block:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        token = idx_next.item()

        if token in stops:
            finish = "stop"
            break
        if token == th_open:
            phase = "reasoning"
            idx = torch.cat((idx, idx_next), dim=1)
            continue
        if token == th_close:
            phase = "answer"
            idx = torch.cat((idx, idx_next), dim=1)
            continue

        idx = torch.cat((idx, idx_next), dim=1)
        if phase == "reasoning":
            think_ids.append(token)
            text = tok.decode(think_ids)
            if len(text) > len(prev_think):
                yield text[len(prev_think):], "reasoning", None
                prev_think = text
        else:
            ans_ids.append(token)
            text = tok.decode(ans_ids)
            if len(text) > len(prev_ans):
                yield text[len(prev_ans):], "answer", None
                prev_ans = text
    yield "", phase, finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}],
            })
        elif self.path in ("/", "/health"):
            self._json(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._json(404, {"error": {"message": f"Unknown path {self.path}", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"Unknown path {self.path}", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = req["messages"]
            max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 200)
            temperature = float(req.get("temperature", 0.8))
            top_k = int(req.get("top_k", 40))
            stream = bool(req.get("stream", False))
        except Exception as e:
            self._json(400, {"error": {"message": f"Bad request: {e}", "type": "invalid_request_error"}})
            return

        prompt = build_prompt(messages)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        with gen_lock:
            if stream:
                self._stream_response(rid, created, prompt, max_tokens, temperature, top_k)
            else:
                think_parts, ans_parts, finish = [], [], "stop"
                for delta, phase, fin in stream_generate(prompt, max_tokens, temperature, top_k):
                    if fin is not None:
                        finish = fin
                    elif phase == "reasoning":
                        think_parts.append(delta)
                    else:
                        ans_parts.append(delta)
                text = "".join(ans_parts).strip()
                reasoning = "".join(think_parts).strip()
                message = {"role": "assistant", "content": text}
                if reasoning:
                    message["reasoning_content"] = reasoning
                prompt_tokens = len(g["tok"].encode(prompt).ids)
                completion_tokens = len(g["tok"].encode(reasoning + text).ids) if (text or reasoning) else 0
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })

    def _stream_response(self, rid, created, prompt, max_tokens, temperature, top_k):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send_chunk(obj):
            data = f"data: {json.dumps(obj, ensure_ascii=False)}\r\n\r\n".encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME}
        try:
            send_chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            finish = "stop"
            for delta, phase, fin in stream_generate(prompt, max_tokens, temperature, top_k):
                if fin is not None:
                    finish = fin
                elif delta:
                    key = "reasoning_content" if phase == "reasoning" else "content"
                    send_chunk({**base, "choices": [{"index": 0, "delta": {key: delta}, "finish_reason": None}]})
            send_chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            done = b"data: [DONE]\r\n\r\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (ConnectionAbortedError, BrokenPipeError):
            pass  # клиент отключился


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ckpt", default="goodgpt_best.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="cpu — чтобы не отъедать VRAM у идущего обучения")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model(args.ckpt, device)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"OpenAI-совместимый сервер запущен: http://{args.host}:{args.port}/v1")
    print(f"В локальной сети: http://192.168.1.3:{args.port}/v1  (модель: {MODEL_NAME})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера.")


if __name__ == "__main__":
    main()
