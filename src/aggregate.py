"""Сводка по нескольким прогонам: среднее и разброс по seed'ам.

Один seed на выборке из 492 примеров ничего не доказывает — разброс от инициализации
сопоставим с теми дельтами, которые мы измеряем. Здесь прогоны группируются по
конфигурации (имя каталога без seed), и разница между конфигурациями считается
по совпадающим seed'ам, а не по одному прогону.

    python src/aggregate.py
    python src/aggregate.py --delta rubert-base-cased rubert-base-cased_nograde
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from model import TASKS

ROOT = Path(__file__).resolve().parent.parent
SEED = re.compile(r"_seed(\d+)")


def collect(runs_dir):
    rows = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        summary = json.loads(path.read_text())
        name = path.parent.name
        match = SEED.search(name)
        rows.append({
            "config": SEED.sub("", name),
            "seed": int(match.group(1)) if match else -1,
            "minutes": round(summary.get("train_minutes", float("nan")), 1),
            "epoch": summary.get("selected_epoch"),
            **{task: summary["test"][task] for task in TASKS},
        })
    if not rows:
        raise SystemExit(f"нет прогонов в {runs_dir}/")
    return pd.DataFrame(rows).sort_values(["config", "seed"])


def summarise(runs):
    grouped = runs.groupby("config")
    table = pd.DataFrame({"seeds": grouped.size(), "minutes": grouped.minutes.mean().round(1)})
    for task in TASKS:
        table[task] = [
            f"{values.mean() * 100:.2f} ± {values.std(ddof=1) * 100:.2f}" if len(values) > 1
            else f"{values.mean() * 100:.2f}"
            for _, values in grouped[task]
        ]
    return table.reset_index()


def delta(runs, config_a, config_b):
    """Разница по совпадающим seed'ам: сравниваются прогоны с одинаковой инициализацией."""
    a = runs[runs.config == config_a].set_index("seed")
    b = runs[runs.config == config_b].set_index("seed")
    shared = sorted(set(a.index) & set(b.index))
    if not shared:
        raise SystemExit(f"нет общих seed'ов у {config_a} и {config_b}")
    rows = []
    for task in TASKS:
        diffs = np.array([a.loc[s, task] - b.loc[s, task] for s in shared]) * 100
        rows.append({
            "task": task, "seeds": len(shared),
            "A": round(a[task].mean() * 100, 2), "B": round(b[task].mean() * 100, 2),
            "delta": round(diffs.mean(), 2),
            "std": round(diffs.std(ddof=1), 2) if len(diffs) > 1 else np.nan,
            "по seed'ам": " ".join(f"{d:+.2f}" for d in diffs),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--delta", nargs=2, metavar=("A", "B"), default=None,
                        help="две конфигурации (имя каталога без _seedN)")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "seeds.csv")
    args = parser.parse_args()
    runs = collect(args.runs)

    if args.delta:
        print(delta(runs, *args.delta).to_string(index=False))
        return

    table = summarise(runs)
    print(table.to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nсохранено в {args.out}")
    single = table[table.seeds == 1].config.tolist()
    if single:
        print(f"один seed (разброс неизвестен): {', '.join(single)}")


if __name__ == "__main__":
    main()
