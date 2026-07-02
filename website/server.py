"""Сайт GoodGPT: статика + API аккаунтов и чатов (только stdlib + sqlite3).

Генерацию текста делает отдельный сервер модели (server.py в корне, порт 8000) —
фронтенд ходит к нему напрямую. Здесь только:
  - раздача website/static/
  - POST /api/register, /api/login, /api/logout, GET /api/me
  - GET/POST /api/chats, PATCH/DELETE /api/chats/<id>
  - GET/POST /api/chats/<id>/messages

Запуск: python website/server.py [--host 0.0.0.0] [--port 8080]
"""
import os
import re
import json
import time
import uuid
import hmac
import hashlib
import secrets
import sqlite3
import argparse
import threading
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
DB_PATH = os.path.join(BASE, "goodgpt_site.db")

MAX_CHATS_PER_USER = 200
MAX_MESSAGES_PER_CHAT = 500
MAX_CONTENT_LEN = 20000

db_lock = threading.Lock()
db = None  # инициализируется в main()


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        pass_hash BLOB NOT NULL,
        salt BLOB NOT NULL,
        created REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions(
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS chats(
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created REAL NOT NULL,
        updated REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        reasoning TEXT,
        created REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated DESC);
    CREATE INDEX IF NOT EXISTS idx_msgs_chat ON messages(chat_id, id);
    """)
    conn.commit()
    return conn


def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


USERNAME_RE = re.compile(r"^[\w.\-а-яА-ЯёЁ]{3,32}$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    # ---------- помощники ----------
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, message):
        self._json(code, {"error": message})

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def _user_id(self):
        """user_id по Bearer-токену либо None."""
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth[7:].strip()
        with db_lock:
            row = db.execute("SELECT user_id FROM sessions WHERE token=?", (token,)).fetchone()
        return row[0] if row else None

    def _own_chat(self, uid, chat_id):
        with db_lock:
            row = db.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?", (chat_id, uid)).fetchone()
        return bool(row)

    # ---------- статика ----------
    def _serve_static(self, path):
        if path == "/":
            path = "/index.html"
        elif path == "/app":
            path = "/app.html"
        # защита от выхода за пределы static/
        safe = os.path.normpath(path.lstrip("/"))
        if safe.startswith("..") or os.path.isabs(safe):
            self._err(404, "not found")
            return
        full = os.path.join(STATIC, safe)
        if not os.path.isfile(full):
            self._err(404, "not found")
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # ---------- HTTP ----------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if not path.startswith("/api/"):
            self._serve_static(path)
            return
        uid = self._user_id()

        if path == "/api/me":
            if not uid:
                self._err(401, "не авторизован")
                return
            with db_lock:
                row = db.execute("SELECT username, created FROM users WHERE id=?", (uid,)).fetchone()
                n = db.execute("SELECT COUNT(*) FROM chats WHERE user_id=?", (uid,)).fetchone()[0]
            self._json(200, {"username": row[0], "created": row[1], "chats": n})

        elif path == "/api/chats":
            if not uid:
                self._err(401, "не авторизован")
                return
            with db_lock:
                rows = db.execute(
                    "SELECT id, title, created, updated FROM chats WHERE user_id=? ORDER BY updated DESC",
                    (uid,)).fetchall()
            self._json(200, [{"id": r[0], "title": r[1], "created": r[2], "updated": r[3]} for r in rows])

        elif re.fullmatch(r"/api/chats/[0-9a-f\-]+/messages", path):
            if not uid:
                self._err(401, "не авторизован")
                return
            chat_id = path.split("/")[3]
            if not self._own_chat(uid, chat_id):
                self._err(404, "чат не найден")
                return
            with db_lock:
                rows = db.execute(
                    "SELECT role, content, reasoning, created FROM messages WHERE chat_id=? ORDER BY id",
                    (chat_id,)).fetchall()
            self._json(200, [
                {"role": r[0], "content": r[1], "reasoning": r[2] or "", "created": r[3]} for r in rows
            ])
        else:
            self._err(404, "not found")

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()

        if path == "/api/register":
            username = str(body.get("username", "")).strip()
            password = str(body.get("password", ""))
            if not USERNAME_RE.fullmatch(username):
                self._err(400, "Имя: 3–32 символа (буквы, цифры, . _ -)")
                return
            if len(password) < 4:
                self._err(400, "Пароль: минимум 4 символа")
                return
            salt = secrets.token_bytes(16)
            ph = hash_password(password, salt)
            with db_lock:
                try:
                    cur = db.execute(
                        "INSERT INTO users(username, pass_hash, salt, created) VALUES(?,?,?,?)",
                        (username, ph, salt, time.time()))
                    uid = cur.lastrowid
                    token = secrets.token_hex(32)
                    db.execute("INSERT INTO sessions(token, user_id, created) VALUES(?,?,?)",
                               (token, uid, time.time()))
                    db.commit()
                except sqlite3.IntegrityError:
                    self._err(409, "Это имя уже занято")
                    return
            self._json(200, {"token": token, "username": username})

        elif path == "/api/login":
            username = str(body.get("username", "")).strip()
            password = str(body.get("password", ""))
            with db_lock:
                row = db.execute("SELECT id, pass_hash, salt FROM users WHERE username=?",
                                 (username,)).fetchone()
            if not row or not hmac.compare_digest(row[1], hash_password(password, row[2])):
                self._err(401, "Неверное имя или пароль")
                return
            token = secrets.token_hex(32)
            with db_lock:
                db.execute("INSERT INTO sessions(token, user_id, created) VALUES(?,?,?)",
                           (token, row[0], time.time()))
                db.commit()
            self._json(200, {"token": token, "username": username})

        elif path == "/api/logout":
            auth = self.headers.get("Authorization", "")
            token = auth[7:].strip() if auth.startswith("Bearer ") else ""
            with db_lock:
                db.execute("DELETE FROM sessions WHERE token=?", (token,))
                db.commit()
            self._json(200, {"ok": True})

        elif path == "/api/chats":
            uid = self._user_id()
            if not uid:
                self._err(401, "не авторизован")
                return
            title = str(body.get("title", "Новый чат")).strip()[:80] or "Новый чат"
            with db_lock:
                n = db.execute("SELECT COUNT(*) FROM chats WHERE user_id=?", (uid,)).fetchone()[0]
                if n >= MAX_CHATS_PER_USER:
                    self._err(400, "Слишком много чатов")
                    return
                cid = uuid.uuid4().hex
                now = time.time()
                db.execute("INSERT INTO chats(id, user_id, title, created, updated) VALUES(?,?,?,?,?)",
                           (cid, uid, title, now, now))
                db.commit()
            self._json(200, {"id": cid, "title": title, "created": now, "updated": now})

        elif re.fullmatch(r"/api/chats/[0-9a-f\-]+/messages", path):
            uid = self._user_id()
            if not uid:
                self._err(401, "не авторизован")
                return
            chat_id = path.split("/")[3]
            if not self._own_chat(uid, chat_id):
                self._err(404, "чат не найден")
                return
            role = str(body.get("role", ""))
            content = str(body.get("content", ""))[:MAX_CONTENT_LEN]
            reasoning = str(body.get("reasoning", ""))[:MAX_CONTENT_LEN]
            if role not in ("user", "assistant") or not content:
                self._err(400, "role: user|assistant, content: непустой")
                return
            with db_lock:
                n = db.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,)).fetchone()[0]
                if n >= MAX_MESSAGES_PER_CHAT:
                    self._err(400, "Чат переполнен — создайте новый")
                    return
                now = time.time()
                db.execute("INSERT INTO messages(chat_id, role, content, reasoning, created) VALUES(?,?,?,?,?)",
                           (chat_id, role, content, reasoning or None, now))
                db.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))
                db.commit()
            self._json(200, {"ok": True})
        else:
            self._err(404, "not found")

    def do_PATCH(self):
        path = self.path.split("?")[0]
        m = re.fullmatch(r"/api/chats/([0-9a-f\-]+)", path)
        uid = self._user_id()
        if not m or not uid:
            self._err(404 if m else 401, "not found" if m else "не авторизован")
            return
        chat_id = m.group(1)
        if not self._own_chat(uid, chat_id):
            self._err(404, "чат не найден")
            return
        title = str(self._body().get("title", "")).strip()[:80]
        if not title:
            self._err(400, "пустой title")
            return
        with db_lock:
            db.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
            db.commit()
        self._json(200, {"ok": True})

    def do_DELETE(self):
        path = self.path.split("?")[0]
        m = re.fullmatch(r"/api/chats/([0-9a-f\-]+)", path)
        uid = self._user_id()
        if not m or not uid:
            self._err(404 if m else 401, "not found" if m else "не авторизован")
            return
        chat_id = m.group(1)
        if not self._own_chat(uid, chat_id):
            self._err(404, "чат не найден")
            return
        with db_lock:
            db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
            db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
            db.commit()
        self._json(200, {"ok": True})


def main():
    global db
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    db = init_db()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Сайт GoodGPT запущен: http://localhost:{args.port}/")
    print(f"В локальной сети:     http://192.168.1.3:{args.port}/")
    print("API модели должен работать отдельно (start_server.bat, порт 8000).")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сайта.")


if __name__ == "__main__":
    main()
