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
    parser.add_argument("--n-iter", type=int, default=1000)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    report = subgroup_report(predictions, args.n_iter)
    print(format_report(report))
    report.to_csv(args.predictions.with_name("report.csv"), index=False)
    print(json.dumps(report[report["sample"] == "General"][["task", "f1"]].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
