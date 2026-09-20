import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SUBGROUPS = [
    ("General", None),
    ("XX century", ("century", "XX")),
    ("XXI century", ("century", "XXI")),
    ("Good (250)", ("part", "top250")),
    ("Bad (100)", ("part", "bottom100")),
    ("Congruent", ("congruent", True)),
    ("Non-congruent", ("congruent", False)),
]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)


def bootstrap(y_true, y_pred, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores[i] = macro_f1(y_true[idx], y_pred[idx])
    return {
        "f1": macro_f1(y_true, y_pred),
        "mean": scores.mean(),
        "ci_low": np.percentile(scores, 2.5),
        "ci_high": np.percentile(scores, 97.5),
    }


def paired_bootstrap(predictions_a, predictions_b, task, n_iter=2000, seed=0):
    """Разница macro-F1 двух прогонов на одной тестовой выборке.

    Пересечение отдельных ДИ здесь ничего не говорит: выборка общая, ошибки моделей
    скоррелированы, поэтому ресэмплить надо одни и те же индексы для обоих.
    """
    y_true = predictions_a[f"{task}_label"].to_numpy()
    if not np.array_equal(y_true, predictions_b[f"{task}_label"].to_numpy()):
        raise ValueError("прогоны сделаны на разных тестовых выборках — сравнивать нельзя")
    pred_a, pred_b = predictions_a[f"{task}_pred"].to_numpy(), predictions_b[f"{task}_pred"].to_numpy()

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs[i] = macro_f1(y_true[idx], pred_a[idx]) - macro_f1(y_true[idx], pred_b[idx])
    return {
        "task": task,
        "f1_a": macro_f1(y_true, pred_a),
        "f1_b": macro_f1(y_true, pred_b),
        "delta": macro_f1(y_true, pred_a) - macro_f1(y_true, pred_b),
        "ci_low": np.percentile(diffs, 2.5),
        "ci_high": np.percentile(diffs, 97.5),
        "p_value": float((diffs <= 0).mean()),
    }


def subgroup_report(predictions, n_iter=1000, seed=0):
    rows = []
    for name, condition in SUBGROUPS:
        subset = predictions if condition is None else predictions[predictions[condition[0]] == condition[1]]
        for task in ["stance", "premise"]:
            stats = bootstrap(subset[f"{task}_label"], subset[f"{task}_pred"], n_iter, seed)
            rows.append({"sample": name, "n": len(subset), "task": task, **stats})
    return pd.DataFrame(rows)


def format_report(report):
    lines = []
    for row in report.itertuples():
        lines.append(
            f"{row.sample:<14} {row.n:>4}  {row.task:<8} "
            f"F1={row.f1 * 100:6.2f}  bootstrap mean={row.mean * 100:6.2f} "
            f"({row.ci_low * 100:.2f}, {row.ci_high * 100:.2f})"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--compare", type=Path, default=None,
                        help="второй predictions.csv: парный bootstrap разницы вместо отчёта")
    parser.add_argument("--n-iter", type=int, default=1000)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)

    if args.compare:
        other = pd.read_csv(args.compare)
        print(f"A = {args.predictions.parent.name}\nB = {args.compare.parent.name}  (n={len(predictions)})")
        for task in ["stance", "premise"]:
            row = paired_bootstrap(predictions, other, task, max(args.n_iter, 2000))
            print(f"{row['task']:<8} A={row['f1_a'] * 100:6.2f}  B={row['f1_b'] * 100:6.2f}  "
                  f"delta={row['delta'] * 100:+6.2f}  "
                  f"95% CI ({row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f})  "
                  f"p={row['p_value']:.3f}")
        return

    report = subgroup_report(predictions, args.n_iter)
    print(format_report(report))
    report.to_csv(args.predictions.with_name("report.csv"), index=False)
    print(json.dumps(report[report["sample"] == "General"][["task", "f1"]].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
