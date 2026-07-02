"""Простая RL-среда для GoodGPT: промпт -> ответ модели -> скалярная награда.

Награда эвристическая (без reward-модели, в духе RLVR):
  + модель сама остановилась (выдала <|endofdialog|>)
  + разумная длина ответа
  + разнообразие n-грамм (борьба с "классическая классическая классическая...")
  + текст в основном на кириллице
  - повторы слов подряд, пустые ответы
"""
import os
import random


def load_prompts(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def compute_reward(response: str, stopped: bool) -> float:
    """Скалярная награда за текст ответа. Диапазон примерно [-3, +3]."""
    words = response.split()
    n = len(words)
    if n == 0:
        return -2.0

    r = 0.0
    # сам закончил ответ, а не упёрся в лимит токенов
    r += 0.5 if stopped else -0.5

    # длина: слишком коротко плохо, разумная длина хорошо
    if n < 3:
        r -= 1.0
    elif 5 <= n <= 70:
        r += 1.0

    low = [w.lower().strip('.,!?…—-:;()"«»') for w in words]

    # разнообразие би- и триграмм: 1.0 = все уникальны, 0 = сплошной повтор
    for k in (2, 3):
        if len(low) >= k:
            grams = list(zip(*[low[i:] for i in range(k)]))
            distinct = len(set(grams)) / len(grams)
            r += distinct - 0.5  # [-0.5, +0.5] за каждую длину n-граммы

    # одинаковые слова подряд — прямой штраф
    runs = sum(1 for a, b in zip(low, low[1:]) if a == b and a)
    r -= 0.5 * min(runs, 4)

    # доля кириллицы среди букв (модель русская, латинский мусор штрафуем)
    letters = [c for c in response if c.isalpha()]
    if letters:
        cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
        r += cyr / len(letters) - 0.5

    return r


class ChatEnv:
    """Итератор по промптам: выдаёт случайные батчи промптов и считает награды."""

    def __init__(self, prompts=None, seed=0):
        self.prompts = prompts or load_prompts()
        self.rng = random.Random(seed)

    def sample(self, n):
        return self.rng.sample(self.prompts, min(n, len(self.prompts)))

    def reward(self, response, stopped):
        return compute_reward(response, stopped)
