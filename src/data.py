import re
from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DATA_URL = "https://huggingface.co/datasets/otipl2125/film_review_argumentation/resolve/main/film_review_argumentation.csv"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "film_review_argumentation.csv"
LABELS = ["Bad", "Neutral", "Good"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
TASKS = {"stance": "NEW_grade3", "premise": "arg_label"}

GRADE_PATTERNS = [
    re.compile(
        r"\s*(?<!\d)\d{1,4}(?:[.,]\d)?\s*\*?\s*(?:бал{1,2}\w*)?\s*"
        r"из\s*(?:100|10|десяти)\s*\*?(?:\s*бал{1,2}\w*)?(?:\s*[.!]+)?\s*",
        re.IGNORECASE,
    ),
    re.compile(r"\s*(?<!\d)\d{1,2}(?:[.,]\d)?\s*/\s*10(?!\d)(?:\s*[.!]+)?\s*"),
    re.compile(r"\s*из\s*(?:100|10)(?:\s*бал{1,2}\w*)?(?:\s*[.!]+)?\s*", re.IGNORECASE),
]

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def strip_grade(text):
    """Убирает из текста явную числовую оценку фильма."""
    for pattern in GRADE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+([,.;:!?)])", r"\1", text).strip()


def drop_last_sentence(text):
    """Убирает последнее предложение целиком — оценка обычно живёт в нём."""
    boundaries = list(SENTENCE_BOUNDARY.finditer(text.strip()))
    if not boundaries:
        return text.strip()
    return text.strip()[: boundaries[-1].start()].strip()


def keep_prefix(fraction):
    """Оставляет первые fraction символов, не разрывая слово."""

    def cut(text):
        text = text.strip()
        head = text[: max(1, int(len(text) * fraction))]
        return (head if head == text else head.rsplit(" ", 1)[0]).strip()

    return cut


def after_grade(cut):
    """Ступень лестницы: сначала всегда снимается оценка, потом режется текст."""
    return lambda text: cut(strip_grade(text))


REDACTIONS = {
    "none": lambda text: text,
    "grade": strip_grade,
    "last-sentence": after_grade(drop_last_sentence),
    "tail15": after_grade(keep_prefix(0.85)),
    "first-half": after_grade(keep_prefix(0.5)),
}
REDACTION_SUFFIX = {"none": "", "grade": "_nograde"}


def load_dataset(path=DATA_PATH, redact="none"):
    if redact not in REDACTIONS:
        raise ValueError(f"неизвестный redact={redact}, доступно: {', '.join(REDACTIONS)}")
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    df = df[["review_id", "movie_name", "author", "part", "content", "NEW_grade3", "arg_label"]].copy()
    year = df["movie_name"].str.extract(r"\((\d{4})\)\s*$")[0].astype(int)
    df["century"] = (year > 2000).map({True: "XXI", False: "XX"})
    df["congruent"] = df["NEW_grade3"] == df["arg_label"]
    df["author"] = df["author"].fillna(pd.Series([f"__unknown_{i}" for i in df.index], index=df.index))
    df["content"] = df["content"].str.strip().map(REDACTIONS[redact])
    for task, column in TASKS.items():
        df[f"{task}_label"] = df[column].map(LABEL2ID)
    return df.reset_index(drop=True)


def split_dataset(df, seed=42, group=None):
    """70/15/15. group=None — стратификация по arg_label, как в статье.

    group="movie_name" или "author" — групповой сплит: ни один фильм (автор) не попадает
    сразу в train и test. Стратификации при этом нет, баланс классов плывёт — его печатает
    analyze.py. В исходном сплите 95.5% фильмов теста есть и в трейне, так что обобщение
    на новый фильм там не измеряется вовсе.
    """
    if group is None:
        train, rest = train_test_split(df, test_size=0.3, stratify=df["arg_label"], random_state=seed)
        val, test = train_test_split(rest, test_size=0.5, stratify=rest["arg_label"], random_state=seed)
    else:
        train, rest = group_split(df, df[group], 0.3, seed)
        val, test = group_split(rest, rest[group], 0.5, seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def group_split(df, groups, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    left, right = next(splitter.split(df, groups=groups))
    return df.iloc[left], df.iloc[right]
