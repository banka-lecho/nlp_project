# Argument Mining for Film Reviews: воспроизведение бейзлайна

Воспроизводим эксперимент из статьи Serbina, Borisova, Rabinovich, *Argument Mining for Film Reviews* (Dialogue 2026).
Датасет: [otipl2125/film_review_argumentation](https://huggingface.co/datasets/otipl2125/film_review_argumentation), CC-BY-4.0.

## Постановка из статьи

- Вход: только `content`. Цели: `NEW_grade3` (stance) и `arg_label` (premise), по 3 класса в каждой.
- Разбиение 70/15/15 со стратификацией по `arg_label`: 2296 / 492 / 492.
- Энкодер и две параллельные линейные головы, лосс равен сумме двух CE. Обучение идёт 2 эпохи.
- Метрика: macro-F1, bootstrap с 1000 итераций, 95% ДИ по подгруппам (век, рейтинг, конгруэнтность).

В статье не указаны lr, batch size, max length и seed разбиения. Здесь взяты стандартные значения:
lr 2e-5, batch 16, max_length 512, warmup 10%, weight decay 0.01, seed 42.
Поэтому состав тестовой выборки отличается от авторского, и корректнее сравнивать результаты по ДИ, а не по точным числам.

## Запуск

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/train.py --model DeepPavlov/rubert-base-cased
.venv/bin/python src/train.py --model ai-forever/ruRoberta-large
```

Результаты сохраняются в `runs/<model>_seed<seed>/`: `metrics.json`, `predictions.csv`, `report.csv`.
Пересчитать отчёт по готовым предсказаниям: `python src/evaluate.py runs/<run>/predictions.csv`.

## Результаты авторов (Table 4, General)

| Модель | Stance | Premise |
|---|---|---|
| rubert-base-cased | 71.31 | 75.00 |
| ruRoberta-large | 75.56 | 78.10 |

## Воспроизведение (seed 42, Mac M4 / MPS)

macro-F1 на тесте (n=492), в скобках 95% bootstrap ДИ.

| Модель | Stance | Premise | Stance (статья) | Premise (статья) |
|---|---|---|---|---|
| rubert-base-cased | 72.44 (68.5–76.1) | 68.89 (64.7–72.6) | 71.31 | 75.00 |
| ruRoberta-large | 74.22 (70.4–77.7) | 73.66 (69.8–77.5) | 75.56 | 78.10 |

ruRoberta-large обучалась с `--batch-size 4 --grad-accum 4`: на 24 GB unified memory Mac batch 16 уходит в своп.
Полные отчёты по подгруппам лежат в `runs/*/report.csv`.
