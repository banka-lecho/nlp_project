"""Разведочный анализ датасета: баланс классов, длины, утечка оценки в текст, шум,
качество сплита. Таблицы печатаются и сохраняются в analysis/ для отчёта.

    python src/analyze.py [--tokenizer DeepPavlov/rubert-base-cased] [--runs runs]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data import DATA_PATH, GRADE_PATTERNS, LABELS, load_dataset, split_dataset, strip_grade

ROOT = Path(__file__).resolve().parent.parent
GRADE_VALUE = re.compile(
    r"(?<!\d)(\d{1,4}(?:[.,]\d)?)\s*\*?\s*(?:бал{1,2}\w*)?\s*из\s*(?:100|10|десяти)",
    re.IGNORECASE,
)
GRADE_TO_LABEL = [(4, "Bad"), (6, "Neutral"), (np.inf, "Good")]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(frame, index=False):
    print(frame.to_string(index=index))
    return frame


def raw_frame(path):
    df = pd.read_csv(path)
    df["content_raw"] = df["content"]
    df["content"] = df["content"].str.strip()
    return df


def overview(df):
    section("1. Состав таблицы")
    df = df.drop(columns=["content_raw"])
    show(pd.DataFrame({
        "колонка": df.columns,
        "тип": [str(t) for t in df.dtypes],
        "пропусков": df.isna().sum().values,
        "уникальных": df.nunique().values,
    }))
    print(f"\nстрок: {len(df)}")
    print(f"дубликатов content: {df.content.duplicated().sum()}")
    ids = df.review_id.value_counts()
    print(f"неуникальных review_id: {df.review_id.duplicated().sum()} "
          f"(максимум {ids.max()} строк на id)")
    reused = df.groupby("review_id").movie_name.nunique()
    print(f"из них с разными фильмами под одним id: {(reused > 1).sum()} — "
          f"тексты разные, это не утечка, а ненадёжный ключ")


def balance(df, out_dir):
    section("2. Баланс классов")
    both = pd.DataFrame({
        "stance (NEW_grade3)": df.NEW_grade3.value_counts().reindex(LABELS),
        "premise (arg_label)": df.arg_label.value_counts().reindex(LABELS),
    })
    both["stance %"] = (both["stance (NEW_grade3)"] / len(df) * 100).round(1)
    both["premise %"] = (both["premise (arg_label)"] / len(df) * 100).round(1)
    show(both.rename_axis("класс").reset_index())
    print(f"\nдисбаланс (max/min): stance {both.iloc[:, 0].max() / both.iloc[:, 0].min():.2f}, "
          f"premise {both.iloc[:, 1].max() / both.iloc[:, 1].min():.2f}")

    print("\nсовместное распределение stance × premise:")
    joint = pd.crosstab(df.NEW_grade3, df.arg_label).reindex(index=LABELS, columns=LABELS)
    show(joint.rename_axis("stance \\ premise").reset_index())
    print(f"\nконгруэнтных (stance == premise): {(df.NEW_grade3 == df.arg_label).mean():.1%}")
    print(f"grade3 == NEW_grade3 (доля без ручной переразметки): "
          f"{(df.grade3 == df.NEW_grade3).mean():.1%}")

    both.to_csv(out_dir / "class_balance.csv")
    joint.to_csv(out_dir / "joint_labels.csv")


def lengths(df, out_dir, tokenizer_names):
    section("3. Длина текстов")
    rows = [{"единица": "слова", **df.content.str.split().str.len()
             .describe(percentiles=[.5, .95, .99]).round(0).to_dict()}]
    for name in tokenizer_names:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        lengths = pd.Series([len(tok(t, truncation=False)["input_ids"]) for t in df.content])
        rows.append({"единица": f"токены {name.split('/')[-1]}",
                     **lengths.describe(percentiles=[.5, .95, .99]).round(0).to_dict()})
        print(f"{name}: обрезается при max_length=512 — {(lengths > 512).mean():.2%}, "
              f"при 320 — {(lengths > 320).mean():.2%}, при 256 — {(lengths > 256).mean():.2%}")
    table = show(pd.DataFrame(rows).drop(columns=["count"]))
    table.to_csv(out_dir / "lengths.csv", index=False)
    if tokenizer_names:
        print("\nвывод: max_length=512 никогда не срабатывает, запас памяти тратится впустую")


def grade_leak(df, out_dir):
    section("4. Утечка оценки в текст")
    value = df.content.apply(
        lambda t: (lambda m: float(m[-1].replace(",", ".")) if m else np.nan)(GRADE_VALUE.findall(t))
    )
    marked = df.content.apply(lambda t: any(p.search(t) for p in GRADE_PATTERNS))
    print(f"отзывов с явной оценкой: {marked.sum()} ({marked.mean():.1%})")

    grade10 = pd.to_numeric(df.grade10.astype(str).str.replace(",", "."), errors="coerce")
    known = value.notna() & grade10.notna()
    print(f"извлечённое число совпадает с колонкой grade10: {(value[known] == grade10[known]).mean():.1%} "
          f"(из {known.sum()} отзывов)")
    print("то есть в тексте лежит ровно тот балл, из которого выведен таргет NEW_grade3")

    sub = df[value.notna()].assign(value=value[value.notna()])
    pred = sub.value.apply(lambda x: next(lab for bound, lab in GRADE_TO_LABEL if x <= bound))
    rows = []
    for task, column in [("stance", "NEW_grade3"), ("premise", "arg_label")]:
        rows.append({
            "задача": task,
            "n": len(sub),
            "accuracy": round(accuracy_score(sub[column], pred), 3),
            "macro-F1": round(f1_score(sub[column], pred, average="macro"), 3),
        })
    print("\nправило «взять число из текста», без обучения, на отзывах с оценкой:")
    rule = show(pd.DataFrame(rows))
    rule.to_csv(out_dir / "grade_rule_baseline.csv", index=False)

    print("\nсредний балл в тексте по классам:")
    by_class = pd.DataFrame({
        "stance": sub.groupby("NEW_grade3").value.mean().reindex(LABELS).round(2),
        "premise": sub.groupby("arg_label").value.mean().reindex(LABELS).round(2),
    })
    show(by_class.rename_axis("класс").reset_index())

    position = df.content[marked].apply(
        lambda t: max(m.end() for p in GRADE_PATTERNS for m in p.finditer(t)) / len(t)
    )
    print(f"\nмаркер стоит в последних 15% текста у {(position >= 0.85).mean():.1%} отзывов "
          f"— модель видит его гарантированно")

    cleaned = df.content.map(strip_grade)
    changed = cleaned != df.content
    residual = cleaned.str.contains(r"из\s*10|из\s*100|\d\s*/\s*10", case=False)
    print(f"\nstrip_grade(): изменено {changed.sum()} текстов, остаточных маркеров {residual.sum()}, "
          f"пустых текстов {(cleaned.str.len() == 0).sum()}")
    print(f"удаляется слов: в среднем "
          f"{(df.content.str.split().str.len() - cleaned.str.split().str.len()).mean():.2f}")


def noise(df, out_dir):
    section("5. Шум в тексте")
    padded = df.content_raw.str.match(r"^\s|.*\s$")
    print(f"padding по краям в исходном CSV: {padded.mean():.1%} — снимается обычным strip()\n")
    checks = [
        ("двойные пробелы внутри текста", r"  "),
        ("склейка предложений (слово.Слово)", r"[а-яa-z]\.[А-ЯA-Z]"),
        ("буква приклеена к цифре (слово10)", r"[а-яa-z]\d"),
        ("латиница длиннее 2 символов", r"[A-Za-z]{3,}"),
        ("html-теги", r"<[^>]+>"),
        ("html-сущности", r"&[a-z]+;"),
        ("url", r"https?://"),
        ("буква ё", r"ё"),
    ]
    table = show(pd.DataFrame([
        {"признак": name, "доля отзывов": f"{df.content.str.contains(p, regex=True).mean():.1%}"}
        for name, p in checks
    ]))
    table.to_csv(out_dir / "noise.csv", index=False)
    print("\nсклейки вида «Темнота.Чужие руки» дают токенизатору мусорные сабтокены —")
    print("кандидат на отдельный флаг --normalize-text")


def splits(out_dir, split_seed):
    section("6. Качество сплита")
    df = load_dataset()
    train, val, test = split_dataset(df, split_seed)
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        row = {"выборка": name, "n": len(part)}
        for task, column in [("stance", "NEW_grade3"), ("premise", "arg_label")]:
            share = part[column].value_counts(normalize=True)
            for label in LABELS:
                row[f"{task}:{label}"] = round(share.get(label, 0), 3)
        rows.append(row)
    table = show(pd.DataFrame(rows))
    table.to_csv(out_dir / "splits.csv", index=False)
    print("\nsplit_dataset() стратифицирует только по arg_label; распределение stance")
    print("держится случайно и может поехать на другом split-seed")

    print(f"\nфильмов в test: {test.movie_name.nunique()}, "
          f"из них встречаются в train: {test.movie_name.isin(train.movie_name).mean():.1%}")
    authors = raw_frame(DATA_PATH).author
    print(f"уникальных авторов: {authors.nunique()}, максимум отзывов у одного: {authors.value_counts().max()}")
    print("сплит не групповой: один автор может попасть и в train, и в test")


def confusion(runs_dir):
    section("7. Ошибки обученных моделей")
    predictions = sorted(Path(runs_dir).glob("*/predictions.csv"))
    if not predictions:
        print(f"нет предсказаний в {runs_dir}/ — запустите train.py")
        return
    for path in predictions:
        print(f"\n{path.parent.name}")
        preds = pd.read_csv(path)
        for task in ["stance", "premise"]:
            matrix = pd.crosstab(preds[f"{task}_label"], preds[f"{task}_pred"])
            matrix.index = [LABELS[i] for i in matrix.index]
            matrix.columns = [LABELS[i] for i in matrix.columns]
            recall = (np.diag(matrix) / matrix.sum(axis=1)).round(2)
            print(f"  {task}: " + ", ".join(f"{lab} {r:.0%}" for lab, r in recall.items()))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--tokenizer", action="append", default=None,
                        help="повторяемый флаг; без него длины считаются только в словах")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = raw_frame(args.data)

    overview(df)
    balance(df, args.out_dir)
    lengths(df, args.out_dir, args.tokenizer or [])
    grade_leak(df, args.out_dir)
    noise(df, args.out_dir)
    splits(args.out_dir, args.split_seed)
    confusion(args.runs)
    print(f"\nтаблицы сохранены в {args.out_dir}")


if __name__ == "__main__":
    main()
