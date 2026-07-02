"""Сбор большого корпуса (~450-500M токенов) для 100M-модели -> data/raw_big/*.txt

Источники (всё стримингом, без полной загрузки датасетов на диск):
  fineweb   — HuggingFaceFW/fineweb-2, rus_Cyrl: качественный русский веб (~250M ток.)
  wiki      — wikimedia/wikipedia 20231101.ru: энциклопедия (~100M ток.)
  reasoning — ZeroAgency/ru-thinking-reasoning-r1-v2: думающие диалоги (~60M ток.)
              формат: <|u|>вопрос\n<|b|><|think|>мысли<|/think|>ответ\n<|endofdialog|>
              берём только короткие рассуждения — блок контекста всего 512 токенов.

Диалоги saiga уже лежат в data/raw/saiga.txt — их скрипт не трогает,
при упаковке возьмём оба каталога.

Запуск: python prepare_data_big.py all      (или отдельно: fineweb / wiki / reasoning)
Смоук:  python prepare_data_big.py all --smoke   (по ~3 МБ с источника)
"""
import os
import re
import argparse

from datasets import load_dataset

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "raw_big")
os.makedirs(OUT_DIR, exist_ok=True)

U, B, EOD = "<|u|>", "<|b|>", "<|endofdialog|>"
TH_O, TH_C = "<|think|>", "<|/think|>"

# ~5.5 символов на токен у нашего BPE => целевые лимиты в символах
LIMITS_CHARS = {
    "fineweb": 1_400_000_000,   # ~250M токенов
    "wiki": 550_000_000,        # ~100M токенов
    "reasoning": 165_000_000,   # ~30M токенов (датасет грязный — берём немного, только чистое)
    "nebo": 110_000_000,        # ~20M токенов чистого longCoT
}


def clean(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cyr_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё") / len(letters)


def run_stage(name, rows_iter, to_text, limit_chars, log_every=20000):
    """Общий цикл: пишет документы в OUT_DIR/name.txt до лимита символов."""
    path = os.path.join(OUT_DIR, f"{name}.txt")
    total, n, seen = 0, 0, 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows_iter:
            seen += 1
            s = to_text(row)
            if s:
                f.write(s)
                total += len(s)
                n += 1
            if seen % log_every == 0:
                print(f"[{name}] просмотрено {seen}, взято {n}, {total/1e6:.0f}M символов", flush=True)
            if total >= limit_chars:
                break
    print(f"[{name}] ГОТОВО: {n} докум., {total/1e6:.1f}M символов -> {path}", flush=True)


def do_fineweb(limit_chars):
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)

    def to_text(row):
        text = clean(row.get("text"))
        if len(text) < 300 or len(text) > 20000:
            return None
        if cyr_ratio(text[:2000]) < 0.7:
            return None
        return f"{text}\n{EOD}\n"

    run_stage("fineweb", ds, to_text, limit_chars)


def do_wiki(limit_chars):
    ds = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True)

    def to_text(row):
        text = clean(row.get("text"))
        if len(text) < 500:
            return None
        title = clean(row.get("title"))
        # оставляем начало статьи: хвосты часто списки/ссылки
        text = text[:12000]
        return f"{title}\n{text}\n{EOD}\n"

    run_stage("wiki", ds, to_text, limit_chars)


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# признаки «нежелательного» контента для крошечной русской чат-модели:
# математика/LaTeX, программирование, разметка задач.
_MATH_CHARS = set("\\${}^_=<>[]|")
_CODE_MARKERS = (
    "```", "def ", "class ", "import ", "return ", "print(", "for (", "while (",
    "public ", "void ", "int ", "#include", "console.", "function ", "() {", ");",
    "**Ввод", "**Выход", "**Пример", "**Ограничения", "leetcode", "patterns",
)


def _ends_midsentence(text: str) -> bool:
    """Обрыв мысли: не заканчивается нормальным терминатором."""
    return bool(text) and text[-1] not in ".!?…\"»)"


# «мусорные» маркеры: утечка instruct-шаблонов и самоназвания чужих ассистентов
_JUNK_MARKERS = (
    "###", "<|", "|>", "system:", "user:", "assistant:",
    "open-assistant", "open assistant", "openassistant", "chatgpt", "qwen",
    "alibaba", "как ии", "как искусственный интеллект", "языковая модель",
    "языковой моделью", "я — ии", "я - ии", "gpt", "llama",
)


def _has_junk(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _JUNK_MARKERS)


# частые русские слова — маркер настоящего русского текста (а не транслита испанского и т.п.)
_RU_STOP = {
    "и", "в", "не", "на", "что", "с", "по", "это", "как", "а", "то", "все", "он",
    "она", "но", "из", "у", "за", "от", "так", "же", "вы", "мы", "я", "для", "к",
    "или", "если", "бы", "был", "была", "быть", "есть", "который", "чтобы", "меня",
    "вас", "вам", "мне", "они", "его", "ее", "тебя", "очень", "может", "можно",
    "нужно", "когда", "только", "уже", "надо", "будет", "этот", "этой", "того",
}


def _russian_ok(text: str, min_ratio: float = 0.22) -> bool:
    """Доля частых русских слов среди всех — фильтрует машинный перевод/транслит."""
    words = re.findall(r"[а-яёА-ЯЁ]+", text.lower())
    if len(words) < 5:
        return False
    hits = sum(1 for w in words if w in _RU_STOP)
    return hits / len(words) >= min_ratio


def _looks_technical(text: str) -> bool:
    low = text.lower()
    if any(m.lower() in low for m in _CODE_MARKERS):
        return True
    # много «математических» символов на длину — почти наверняка формулы/код
    math_hits = sum(1 for c in text if c in _MATH_CHARS)
    if math_hits > 8 or math_hits / max(len(text), 1) > 0.02:
        return True
    # длинные числовые выкладки
    if len(re.findall(r"\d", text)) / max(len(text), 1) > 0.08:
        return True
    return False


def _extract_messages(row):
    """Достаёт список (role, content) из разных схем conversation-датасетов."""
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if not msgs:
        return None
    out = []
    for m in msgs:
        role = m.get("role") or m.get("from") or ""
        content = m.get("content") or m.get("value") or ""
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
        out.append((role, content))
    return out


def do_reasoning(limit_chars):
    ds = load_dataset("ZeroAgency/ru-thinking-reasoning-r1-v2", split="train", streaming=True)

    def to_text(row):
        msgs = _extract_messages(row)
        if not msgs:
            return None
        parts = []
        for role, content in msgs:
            content = clean(content)
            if role == "system" or not content:
                continue
            if role == "user":
                # только короткие бытовые вопросы, без математики/кода
                if not (10 <= len(content) <= 400):
                    return None
                if _looks_technical(content) or _has_junk(content) or cyr_ratio(content) < 0.85:
                    return None
                if not _russian_ok(content):
                    return None
                parts.append(f"{U}{content}")
            elif role == "assistant":
                m = THINK_RE.search(content)
                if not m:
                    return None  # берём только сэмплы с размеченными мыслями
                think = clean(m.group(1))
                answer = clean(content[m.end():])
                # контекст модели 512 токенов: длинные цепочки бесполезны
                if not (30 <= len(think) <= 500) or not (10 <= len(answer) <= 600):
                    return None
                # обрыв мысли/ответа на полуслове — брак источника
                if _ends_midsentence(think) or _ends_midsentence(answer):
                    return None
                # выкидываем математику/код/формулы и латиницу
                if _looks_technical(think) or _looks_technical(answer):
                    return None
                # шаблонный мусор и самоназвания чужих ассистентов
                if _has_junk(think) or _has_junk(answer):
                    return None
                if cyr_ratio(think) < 0.9 or cyr_ratio(answer) < 0.9:
                    return None
                # отсев машинного перевода/транслита (напр. испанский кириллицей)
                if not _russian_ok(think) or not _russian_ok(answer):
                    return None
                parts.append(f"{B}{TH_O}{think}{TH_C}{answer}")
        # только однораундовые пары user->assistant с мыслями
        if len(parts) != 2 or not parts[0].startswith(U) or not parts[1].startswith(B):
            return None
        return "\n".join(parts) + f"\n{EOD}\n"

    run_stage("reasoning", ds, to_text, limit_chars)


NEBO_UPSAMPLE = 15  # reasoning-сэмплов мало (~1k); дублируем, чтобы модель выучила формат


def do_nebo(limit_chars):
    """kristaller486/Nebo-T1-Russian: longCoT от DeepSeek-R1, только чистые русские сэмплы.

    Годных под ctx=1024 ~974 шт. Дублируем каждый NEBO_UPSAMPLE раз, чтобы think-формат
    получил заметную долю в корпусе (иначе <0.1% — модель его не выучит).
    """
    ds = load_dataset("kristaller486/Nebo-T1-Russian", split="train", streaming=True)

    def to_text(row):
        # доверяем разметке датасета: только русские и с корректным форматом мыслей
        if not row.get("russian_only") or not row.get("correct_format"):
            return None
        prompt = clean(row.get("prompt"))
        think = clean(row.get("think"))
        answer = clean(row.get("answer"))
        if not prompt or not think or not answer:
            return None
        # длинный CoT режем по бюджету контекста (512 токенов)
        if not (10 <= len(prompt) <= 400):
            return None
        # бюджет контекста 1024 токена: think до ~3500 симв (~640 токенов) + prompt + answer
        if not (40 <= len(think) <= 3500) or not (10 <= len(answer) <= 800):
            return None
        # Датасет сам гарантирует russian_only+correct_format, а reasoning здесь —
        # реальные безопасные R1-цепочки со STEM (формулы/числа/единицы). Поэтому НЕ режем
        # по STEM, обрывам (STEM-ответ часто кончается числом) и жёсткой доле кириллицы.
        # Оставляем только защиту от шаблонного мусора и минимальную русскость.
        for t in (prompt, think, answer):
            if _has_junk(t) or cyr_ratio(t) < 0.55:
                return None
        if not _russian_ok(prompt, min_ratio=0.15):
            return None
        return f"{U}{prompt}\n{B}{TH_O}{think}{TH_C}{answer}\n{EOD}\n"

    # первый проход: собираем уникальные годные сэмплы
    uniq = []
    seen = 0
    for row in ds:
        seen += 1
        s = to_text(row)
        if s:
            uniq.append(s)
        if seen % 2000 == 0:
            print(f"[nebo] просмотрено {seen}, уникальных {len(uniq)}", flush=True)
    # апсемплинг с перемешиванием
    import random
    rng = random.Random(0)
    path = os.path.join(OUT_DIR, "nebo.txt")
    total, n = 0, 0
    with open(path, "w", encoding="utf-8") as f:
        pool = uniq * NEBO_UPSAMPLE
        rng.shuffle(pool)
        for s in pool:
            f.write(s)
            total += len(s)
            n += 1
            if total >= limit_chars:
                break
    print(f"[nebo] ГОТОВО: {len(uniq)} уникальных x{NEBO_UPSAMPLE} -> {n} записей, "
          f"{total/1e6:.1f}M символов -> {path}", flush=True)


STAGES = {"fineweb": do_fineweb, "wiki": do_wiki, "reasoning": do_reasoning, "nebo": do_nebo}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    parser.add_argument("--smoke", action="store_true", help="по ~3 МБ с источника для проверки")
    args = parser.parse_args()

    # 'all' = рабочий корпус: fineweb + wiki + nebo (ZeroAgency reasoning забракован)
    all_stages = ["fineweb", "wiki", "nebo"]
    names = all_stages if args.stage == "all" else [args.stage]
    for nm in names:
        limit = 3_000_000 if args.smoke else LIMITS_CHARS[nm]
        print(f"=== {nm} (лимит {limit/1e6:.0f}M символов) ===", flush=True)
        STAGES[nm](limit)
    print("Все стадии завершены.", flush=True)
