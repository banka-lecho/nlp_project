from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://huggingface.co/datasets/otipl2125/film_review_argumentation/resolve/main/film_review_argumentation.csv"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "film_review_argumentation.csv"
LABELS = ["Bad", "Neutral", "Good"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
TASKS = {"stance": "NEW_grade3", "premise": "arg_label"}


def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    df = df[["review_id", "movie_name", "part", "content", "NEW_grade3", "arg_label"]].copy()
    year = df["movie_name"].str.extract(r"\((\d{4})\)\s*$")[0].astype(int)
    df["century"] = (year > 2000).map({True: "XXI", False: "XX"})
    df["congruent"] = df["NEW_grade3"] == df["arg_label"]
    for task, column in TASKS.items():
        df[f"{task}_label"] = df[column].map(LABEL2ID)
    return df.reset_index(drop=True)


def split_dataset(df, seed=42):
    train, rest = train_test_split(df, test_size=0.3, stratify=df["arg_label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["arg_label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
