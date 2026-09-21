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
CHANCE_LEVEL = 0.35


def collect(runs_dir):
    rows = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        summary = json.loads(path.read_text())
        name = path.parent.name
        match = SEED.search(name)
        args = summary.get("args", {})
        rows.append({
            "config": SEED.sub("", name),
            "seed": int(match.group(1)) if match else -1,
            "minutes": round(summary.get("train_minutes", float("nan")), 1),
            "epoch": summary.get("selected_epoch"),
            "device": summary.get("device", "?"),
            "testset": f"group={args.get('group')} split={args.get('split_seed')}",
            **{task: summary["test"][task] for task in TASKS},
        })
    if not rows:
        raise SystemExit(f"нет прогонов в {runs_dir}/")
    return pd.DataFrame(rows).sort_values(["config", "seed"])


def degenerate(runs):
    """Прогоны на уровне случайного угадывания: обучение разошлось, а не «плохой seed».

    Для трёх классов вырожденное предсказание одного класса даёт macro-F1 около 0.17.
    Такой прогон незаметно утаскивает среднее и раздувает разброс, поэтому его лучше
    увидеть отдельной строкой, чем искать глазами в таблице.
    """
    mask = runs[list(TASKS)].max(axis=1) < CHANCE_LEVEL
    return runs[mask]


def summarise(runs):
    grouped = runs.groupby("config")
    table = pd.DataFrame({"seeds": grouped.size(), "minutes": grouped.minutes.mean().round(1),
                          "device": grouped.device.agg(lambda v: "/".join(sorted(set(v))))})
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
    if set(a.testset) != set(b.testset):
        print(f"ВНИМАНИЕ: тестовые выборки разные ({a.testset.iloc[0]} против {b.testset.iloc[0]}),")
        print("разность считается между разными выборками и парной не является\n")
    if set(a.device) != set(b.device):
        print(f"ВНИМАНИЕ: прогоны сделаны на разном железе ({set(a.device) | set(b.device)})\n")
    failed = degenerate(pd.concat([a.reset_index(), b.reset_index()]))
    if not failed.empty:
        print(f"ВНИМАНИЕ: среди сравниваемых прогонов есть разошедшиеся "
              f"(seed {', '.join(str(s) for s in sorted(set(failed.seed)))}),")
        print("дельта по ним не имеет смысла\n")

    rows = []
    for task in TASKS:
        values_a = np.array([a.loc[s, task] for s in shared]) * 100
        values_b = np.array([b.loc[s, task] for s in shared]) * 100
        diffs = values_a - values_b
        rows.append({
            "task": task, "seeds": len(shared),
            "A": round(values_a.mean(), 2), "B": round(values_b.mean(), 2),
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

    failed = degenerate(runs)
    if not failed.empty:
        print(f"РАЗОШЛИСЬ (macro-F1 ниже {CHANCE_LEVEL}, обе задачи на уровне угадывания):")
        for row in failed.itertuples():
            print(f"  {row.config}_seed{row.seed}: stance {row.stance * 100:.2f}, "
                  f"premise {row.premise * 100:.2f}")
        print("эти прогоны входят в среднее и разброс ниже — пересчитайте их или исключите\n")

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
