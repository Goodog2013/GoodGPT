/* GoodGPT — чат-приложение: аккаунты, мультичаты, гостевой режим, стриминг, think-блоки. */
"use strict";
const $ = id => document.getElementById(id);

/* ================= состояние ================= */
const S = {
  token: localStorage.getItem("gg_token") || "",
  username: localStorage.getItem("gg_user") || "",
  chats: [],          // [{id, title, updated}]
  currentId: null,    // null = черновик (не сохранён)
  messages: [],       // сообщения текущего чата [{role, content, reasoning}]
  busy: false,
};
const isAuthed = () => !!S.token;

const SUGGESTIONS = [
  "Привет! Кто ты?",
  "Расскажи что-нибудь интересное",
  "Что ты умеешь?",
  "Придумай короткую историю",
  "Как у тебя дела?",
];

/* ================= настройки ================= */
const defaultApi = `http://${location.hostname || "localhost"}:8000/v1`;
$("apiBase").value = localStorage.getItem("gg_api") || defaultApi;
$("apiBase").onchange = e => localStorage.setItem("gg_api", e.target.value.trim());
for (const [id, key, def] of [["temp", "gg_temp", "0.8"], ["maxTok", "gg_maxtok", "250"], ["ctxMsgs", "gg_ctx", "6"]]) {
  $(id).value = localStorage.getItem(key) || def;
  $(id + "Val").textContent = $(id).value;
  $(id).oninput = e => { $(id + "Val").textContent = e.target.value; localStorage.setItem(key, e.target.value); };
}
$("gearBtn").onclick = () => $("settings").classList.toggle("open");

/* ================= API-помощники ================= */
async function api(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(S.token ? { "Authorization": "Bearer " + S.token } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

/* ================= локальное хранилище (гость) ================= */
function guestLoad() {
  try { return JSON.parse(localStorage.getItem("gg_guest_chats") || "{}"); }
  catch { return {}; }
}
function guestSave(obj) { localStorage.setItem("gg_guest_chats", JSON.stringify(obj)); }

/* ================= слой данных: гость ⇄ сервер ================= */
const store = {
  async listChats() {
    if (isAuthed()) return await api("GET", "/api/chats");
    const g = guestLoad();
    return Object.values(g).map(c => ({ id: c.id, title: c.title, updated: c.updated }))
      .sort((a, b) => b.updated - a.updated);
  },
  async createChat(title) {
    if (isAuthed()) return await api("POST", "/api/chats", { title });
    const g = guestLoad();
    const id = "g" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
    g[id] = { id, title, created: Date.now() / 1000, updated: Date.now() / 1000, messages: [] };
    guestSave(g);
    return g[id];
  },
  async renameChat(id, title) {
    if (isAuthed()) return await api("PATCH", "/api/chats/" + id, { title });
    const g = guestLoad();
    if (g[id]) { g[id].title = title; guestSave(g); }
  },
  async deleteChat(id) {
    if (isAuthed()) return await api("DELETE", "/api/chats/" + id);
    const g = guestLoad(); delete g[id]; guestSave(g);
  },
  async getMessages(id) {
    if (isAuthed()) return await api("GET", `/api/chats/${id}/messages`);
    return (guestLoad()[id]?.messages) || [];
  },
  async addMessage(id, role, content, reasoning) {
    if (isAuthed()) return await api("POST", `/api/chats/${id}/messages`, { role, content, reasoning });
    const g = guestLoad();
    if (g[id]) {
      g[id].messages.push({ role, content, reasoning: reasoning || "" });
      g[id].updated = Date.now() / 1000;
      guestSave(g);
    }
  },
};

/* ================= мини-markdown ================= */
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}
function renderMd(src) {
  // split без capture-группы даёт чередование: [текст, код, текст, код, ...]
  const parts = esc(src).split(/```\w*\n?/);
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      html += "<pre><code>" + parts[i].replace(/\n+$/, "") + "</code></pre>";
      continue;
    }
    const lines = parts[i].split("\n");
    let list = null, para = [];
    const flushPara = () => {
      if (para.length) { html += "<p>" + inlineMd(para.join("<br>")) + "</p>"; para = []; }
    };
    const flushList = () => { if (list) { html += `<${list.tag}>` + list.items.map(x => "<li>" + inlineMd(x) + "</li>").join("") + `</${list.tag}>`; list = null; } };
    for (const raw of lines) {
      const line = raw.trimEnd();
      const h = line.match(/^(#{1,3})\s+(.*)/);
      const ul = line.match(/^[-*•]\s+(.*)/);
      const ol = line.match(/^\d+[.)]\s+(.*)/);
      const bq = line.match(/^>\s?(.*)/);
      if (!line.trim()) { flushPara(); flushList(); }
      else if (h) { flushPara(); flushList(); html += `<h3>${inlineMd(h[2])}</h3>`; }
      else if (bq) { flushPara(); flushList(); html += `<blockquote>${inlineMd(bq[1])}</blockquote>`; }
      else if (ul) { flushPara(); if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; } list.items.push(ul[1]); }
      else if (ol) { flushPara(); if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; } list.items.push(ol[1]); }
      else { flushList(); para.push(line); }
    }
    flushPara(); flushList();
  }
  return html || "<p></p>";
}

/* ================= отрисовка сообщений ================= */
function scrollDown() { $("chat").scrollTop = $("chat").scrollHeight; }

function addMsgEl(role, contentHtml, reasoning) {
  document.querySelector(".hello")?.remove();
  const m = document.createElement("div");
  m.className = "msg " + (role === "user" ? "user" : "bot");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? (S.username ? S.username[0].toUpperCase() : "Т") : "G";
  const body = document.createElement("div");
  body.className = "body";
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? (S.username || "Ты") : "GoodGPT";
  body.appendChild(meta);
  if (reasoning) {
    const th = makeThink();
    th.body.textContent = reasoning;
    th.el.open = false;
    body.appendChild(th.el);
  }
  const text = document.createElement("div");
  text.className = "text";
  text.innerHTML = contentHtml;
  body.appendChild(text);
  m.appendChild(who); m.appendChild(body);
  $("messages").appendChild(m);
  scrollDown();
  return { body, text };
}

function makeThink() {
  const el = document.createElement("details");
  el.className = "think";
  const sum = document.createElement("summary");
  sum.innerHTML = "Размышления";
  const body = document.createElement("div");
  body.className = "think-body";
  el.appendChild(sum); el.appendChild(body);
  return { el, sum, body };
}

function showHello() {
  $("messages").innerHTML = `
    <div class="hello">
      <div class="glyph">G</div>
      <h2>Чем помочь?</h2>
      <p>GoodGPT — модель на 109M параметров, обученная с нуля.<br>
      Болтает по-русски и умеет «думать», но факты может сочинять.</p>
    </div>`;
}

/* ================= чаты ================= */
async function refreshChatList() {
  try { S.chats = await store.listChats(); }
  catch { S.chats = []; }
  const list = $("chatList");
  list.innerHTML = "";
  if (!S.chats.length) {
    list.innerHTML = `<div class="empty">Пока нет чатов.<br>Напиши первое сообщение!</div>`;
    return;
  }
  for (const c of S.chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === S.currentId ? " on" : "");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = c.title;
    const ren = document.createElement("button");
    ren.className = "act"; ren.title = "Переименовать"; ren.innerHTML = "&#9998;";
    ren.onclick = async e => {
      e.stopPropagation();
      const t = prompt("Название чата:", c.title);
      if (t && t.trim()) { await store.renameChat(c.id, t.trim().slice(0, 80)); refreshChatList(); if (c.id === S.currentId) $("chatTitle").textContent = t.trim(); }
    };
    const del = document.createElement("button");
    del.className = "act del"; del.title = "Удалить"; del.innerHTML = "&#128465;";
    del.onclick = async e => {
      e.stopPropagation();
      if (!confirm(`Удалить чат «${c.title}»?`)) return;
      await store.deleteChat(c.id);
      if (c.id === S.currentId) newChat();
      refreshChatList();
    };
    item.appendChild(name); item.appendChild(ren); item.appendChild(del);
    item.onclick = () => openChat(c.id);
    list.appendChild(item);
  }
}

function newChat() {
  S.currentId = null;
  S.messages = [];
  $("chatTitle").textContent = "Новый чат";
  showHello();
  renderSuggest();
  refreshChatList();
  closeMobileSidebar();
  $("input").focus();
}

async function openChat(id) {
  if (S.busy) return;
  S.currentId = id;
  const meta = S.chats.find(c => c.id === id);
  $("chatTitle").textContent = meta ? meta.title : "Чат";
  $("messages").innerHTML = "";
  $("suggest").innerHTML = "";
  try {
    S.messages = await store.getMessages(id);
  } catch { S.messages = []; }
  if (!S.messages.length) showHello();
  for (const m of S.messages) {
    addMsgEl(m.role, m.role === "user" ? esc(m.content).replace(/\n/g, "<br>") : renderMd(m.content), m.reasoning);
  }
  refreshChatList();
  closeMobileSidebar();
  scrollDown();
}

/* ================= отправка ================= */
async function send(textArg) {
  const text = (textArg || $("input").value).trim();
  if (!text || S.busy) return;
  S.busy = true; $("send").disabled = true;
  $("input").value = ""; autosize();
  $("suggest").innerHTML = "";

  // черновик -> создаём чат с названием из первого сообщения
  if (!S.currentId) {
    const title = text.length > 42 ? text.slice(0, 42) + "…" : text;
    try {
      const c = await store.createChat(title);
      S.currentId = c.id;
      $("chatTitle").textContent = title;
    } catch (e) { /* без сохранения продолжаем как есть */ }
    refreshChatList();
  }

  addMsgEl("user", esc(text).replace(/\n/g, "<br>"));
  S.messages.push({ role: "user", content: text });
  if (S.currentId) store.addMessage(S.currentId, "user", text).catch(() => {});

  // пузырь бота
  const { body, text: textEl } = addMsgEl("bot", "");
  textEl.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';

  let think = null, reasoning = "", answer = "";
  const ensureThink = () => {
    if (think) return;
    think = makeThink();
    think.el.open = true;
    think.sum.innerHTML = 'Размышляю… <span class="pulse"></span>';
    body.insertBefore(think.el, textEl);
  };

  // контекст: последние N реплик
  const ctxN = parseInt($("ctxMsgs").value);
  const ctx = S.messages.slice(-ctxN).map(m => ({ role: m.role, content: m.content }));

  const apiBase = $("apiBase").value.replace(/\/$/, "");
  try {
    const resp = await fetch(apiBase + "/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "goodgpt-01",
        messages: ctx,
        temperature: parseFloat($("temp").value),
        max_tokens: parseInt($("maxTok").value),
        stream: true,
      }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split(/\r?\n\r?\n/);
      buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const d = JSON.parse(payload).choices?.[0]?.delta || {};
          if (d.reasoning_content) {
            ensureThink();
            reasoning += d.reasoning_content;
            think.body.textContent = reasoning;
            scrollDown();
          }
          if (d.content) {
            answer += d.content;
            textEl.innerHTML = renderMd(answer);
            scrollDown();
          }
        } catch {}
      }
    }
    if (!answer.trim()) { answer = "(пустой ответ)"; textEl.innerHTML = "<p>(пустой ответ)</p>"; }
    if (think) { think.el.open = false; think.sum.textContent = "Размышления"; }

    S.messages.push({ role: "assistant", content: answer, reasoning });
    if (S.currentId) store.addMessage(S.currentId, "assistant", answer, reasoning).catch(() => {});
  } catch (e) {
    textEl.classList.add("error");
    textEl.textContent = "Ошибка: " + e.message + ". Сервер модели запущен? (start_server.bat)";
    S.messages.pop();
  } finally {
    S.busy = false; $("send").disabled = false; $("input").focus();
  }
}

$("send").onclick = () => send();
$("input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 150) + "px";
}
$("input").addEventListener("input", autosize);
$("newChatBtn").onclick = newChat;

function renderSuggest() {
  const box = $("suggest");
  box.innerHTML = "";
  for (const s of SUGGESTIONS) {
    const b = document.createElement("button");
    b.textContent = s;
    b.onclick = () => send(s);
    box.appendChild(b);
  }
}

/* ================= статус модели ================= */
async function ping() {
  try {
    const r = await fetch($("apiBase").value.replace(/\/$/, "") + "/models");
    $("status").classList.toggle("online", r.ok);
    $("statusText").textContent = r.ok ? "модель онлайн" : "сервер отвечает с ошибкой";
  } catch {
    $("status").classList.remove("online");
    $("statusText").textContent = "модель офлайн (start_server.bat)";
  }
}
ping(); setInterval(ping, 12000);

/* ================= сайдбар ================= */
$("collapseBtn").onclick = () => $("sidebar").classList.toggle("hidden");
$("menuBtn").onclick = () => { $("sidebar").classList.add("open"); $("scrim").classList.add("on"); };
$("scrim").onclick = closeMobileSidebar;
function closeMobileSidebar() { $("sidebar").classList.remove("open"); $("scrim").classList.remove("on"); }

/* ================= аккаунт ================= */
let authMode = "login";
function setAuthMode(m) {
  authMode = m;
  $("tabLogin").classList.toggle("on", m === "login");
  $("tabReg").classList.toggle("on", m === "register");
  $("authTitle").textContent = m === "login" ? "Вход в GoodGPT" : "Регистрация";
  $("authSubmit").textContent = m === "login" ? "Войти" : "Создать аккаунт";
  $("authErr").textContent = "";
}
$("tabLogin").onclick = () => setAuthMode("login");
$("tabReg").onclick = () => setAuthMode("register");
$("authClose").onclick = () => $("authModal").classList.remove("open");
$("authGuest").onclick = () => {
  localStorage.setItem("gg_guest", "1");
  $("authModal").classList.remove("open");
};
$("authSubmit").onclick = async () => {
  const username = $("authUser").value.trim();
  const password = $("authPass").value;
  $("authErr").textContent = "";
  try {
    const d = await api("POST", "/api/" + authMode, { username, password });
    S.token = d.token; S.username = d.username;
    localStorage.setItem("gg_token", d.token);
    localStorage.setItem("gg_user", d.username);
    $("authModal").classList.remove("open");
    renderUserBox();
    newChat();
  } catch (e) {
    $("authErr").textContent = e.message;
  }
};
$("authPass").addEventListener("keydown", e => { if (e.key === "Enter") $("authSubmit").click(); });

async function logout() {
  try { await api("POST", "/api/logout"); } catch {}
  S.token = ""; S.username = "";
  localStorage.removeItem("gg_token"); localStorage.removeItem("gg_user");
  renderUserBox();
  newChat();
}

function renderUserBox() {
  const box = $("userBox");
  box.innerHTML = "";
  if (isAuthed()) {
    const av = document.createElement("div");
    av.className = "avatar"; av.textContent = S.username[0].toUpperCase();
    const info = document.createElement("div");
    info.style.flex = "1"; info.style.minWidth = "0";
    info.innerHTML = `<div class="uname"></div><div class="usub">история на сервере</div>`;
    info.querySelector(".uname").textContent = S.username;
    const out = document.createElement("button");
    out.textContent = "Выйти"; out.onclick = logout;
    box.appendChild(av); box.appendChild(info); box.appendChild(out);
  } else {
    const b = document.createElement("button");
    b.className = "login-cta";
    b.textContent = "Войти / Регистрация";
    b.onclick = () => { setAuthMode("login"); $("authModal").classList.add("open"); };
    box.appendChild(b);
  }
}

/* ================= запуск ================= */
(async function init() {
  renderUserBox();
  if (isAuthed()) {
    try { await api("GET", "/api/me"); }
    catch {
      S.token = ""; S.username = "";
      localStorage.removeItem("gg_token"); localStorage.removeItem("gg_user");
      renderUserBox();
    }
  }
  if (!isAuthed() && !localStorage.getItem("gg_guest")) {
    $("authModal").classList.add("open");
  }
  await refreshChatList();
  if (S.chats.length) await openChat(S.chats[0].id);
  else { showHello(); renderSuggest(); }
  $("input").focus();
})();
