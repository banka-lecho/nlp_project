"""Линейный пробинг замороженного энкодера по лестнице удаления текста.

Дообучения нет: энкодер не трогается, поверх его представлений учится логистическая
регрессия. Это дешёвая версия эксперимента с утечкой — видно, сколько сигнала лежит
на поверхности представлений и как быстро он исчезает, когда из текста убирают
сначала оценку, потом последнее предложение, потом хвост.

    python src/probe.py --model DeepPavlov/rubert-base-cased
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

from data import REDACTIONS, load_dataset, split_dataset
from evaluate import macro_f1, qwk
from model import TASKS, mean_pool, pick_device

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def encode(texts, tokenizer, encoder, device, max_length, batch_size, pooling):
    features = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt").to(device)
        hidden = encoder(**batch).last_hidden_state
        pooled = mean_pool(hidden, batch["attention_mask"]) if pooling == "mean" else hidden[:, 0]
        features.append(pooled.float().cpu().numpy())
        if sys.stdout.isatty():
            print(f"\r  {min(start + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    if sys.stdout.isatty():
        print("\r" + " " * 24 + "\r", end="")
    return np.vstack(features)


def fit_probe(train_x, train_y, val_x, val_y, grid=(0.0003, 0.001, 0.003, 0.01, 0.1, 1.0, 10.0)):
    """C подбирается на val, иначе пробинг меряет не представления, а удачу с регуляризацией."""
    best = None
    for c in grid:
        model = LogisticRegression(C=c, max_iter=3000).fit(train_x, train_y)
        score = macro_f1(val_y, model.predict(val_x))
        if best is None or score > best[0]:
            best = (score, c, model)
    return best[2], best[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--redact", nargs="+", choices=list(REDACTIONS), default=list(REDACTIONS))
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--group", choices=["movie_name", "author"], default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(device).eval()
    print(f"device={device} model={args.model} pooling={args.pooling}")

    rows = []
    for redact in args.redact:
        print(f"{redact}:")
        train, val, test = split_dataset(load_dataset(redact=redact), args.split_seed, args.group)
        scaler = StandardScaler()
        features = {}
        for name, part in [("train", train), ("val", val), ("test", test)]:
            raw = encode(part.content.tolist(), tokenizer, encoder, device,
                         args.max_length, args.batch_size, args.pooling)
            features[name] = scaler.fit_transform(raw) if name == "train" else scaler.transform(raw)
        for task in TASKS:
            probe, c = fit_probe(features["train"], train[f"{task}_label"],
                                 features["val"], val[f"{task}_label"])
            predicted = probe.predict(features["test"])
            truth = test[f"{task}_label"].to_numpy()
            rows.append({"redact": redact, "task": task, "n_test": len(test), "C": c,
                         "macro-F1": round(macro_f1(truth, predicted), 4),
                         "QWK": round(qwk(truth, predicted), 4)})
            print(f"  {task:<8} macro-F1={rows[-1]['macro-F1']:.4f}  QWK={rows[-1]['QWK']:.4f}  C={c}")

    table = pd.DataFrame(rows)
    print()
    print(table.pivot(index="redact", columns="task", values="macro-F1")
          .reindex([r for r in REDACTIONS if r in args.redact]).to_string())
    path = args.out_dir / f"probe_{args.model.split('/')[-1]}.csv"
    table.to_csv(path, index=False)
    print(f"\nсохранено в {path}")


if __name__ == "__main__":
    main()
