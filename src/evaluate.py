import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["Bad", "Neutral", "Good"]
TASKS = ("stance", "premise")

SUBGROUPS = [
    ("General", None),
    ("XX century", ("century", "XX")),
    ("XXI century", ("century", "XXI")),
    ("Good (250)", ("part", "top250")),
    ("Bad (100)", ("part", "bottom100")),
    ("Congruent", ("congruent", True)),
    ("Non-congruent", ("congruent", False)),
]


def prob_columns(task):
    return [f"{task}_p_{label}" for label in LABELS]


def probabilities(predictions, task):
    """Вероятности из predictions.csv; None для старых прогонов, где сохранялся только argmax."""
    columns = prob_columns(task)
    if not set(columns).issubset(predictions.columns):
        return None
    return predictions[columns].to_numpy(dtype=float)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)


def confusion(y_true, y_pred, k=3):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def qwk(y_true, y_pred, k=3):
    """Quadratic weighted kappa: штраф растёт как квадрат расстояния между классами.

    Для упорядоченных Bad < Neutral < Good это честнее macro-F1, который считает
    ошибку Bad→Neutral и Bad→Good одинаковой.
    """
    weights = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    observed = confusion(y_true, y_pred, k)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    denominator = (weights * expected).sum()
    return 1.0 - (weights * observed).sum() / denominator if denominator else 0.0


def mae(y_true, y_pred):
    return float(np.abs(y_true - y_pred).mean())


def severe(y_true, y_pred):
    """Доля ошибок через класс: Bad предсказан как Good или наоборот."""
    return float((np.abs(y_true - y_pred) == 2).mean())


METRICS = {"f1": macro_f1, "qwk": qwk, "mae": mae, "severe": severe}


def bootstrap(y_true, y_pred, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    samples = {name: np.empty(n_iter) for name in METRICS}
    for i in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        for name, metric in METRICS.items():
            samples[name][i] = metric(y_true[idx], y_pred[idx])
    stats = {}
    for name, metric in METRICS.items():
        stats[name] = metric(y_true, y_pred)
        stats[f"{name}_lo"] = np.percentile(samples[name], 2.5)
        stats[f"{name}_hi"] = np.percentile(samples[name], 97.5)
    stats["mean"] = samples["f1"].mean()
    stats["ci_low"], stats["ci_high"] = stats["f1_lo"], stats["f1_hi"]
    return stats


def paired_bootstrap(predictions_a, predictions_b, task, n_iter=2000, seed=0, metric="f1"):
    """Разница метрики двух прогонов на одной тестовой выборке.

    Пересечение отдельных ДИ здесь ничего не говорит: выборка общая, ошибки моделей
    скоррелированы, поэтому ресэмплить надо одни и те же индексы для обоих.
    """
    score = METRICS[metric]
    y_true = predictions_a[f"{task}_label"].to_numpy()
    if not np.array_equal(y_true, predictions_b[f"{task}_label"].to_numpy()):
        raise ValueError("прогоны сделаны на разных тестовых выборках — сравнивать нельзя")
    pred_a, pred_b = predictions_a[f"{task}_pred"].to_numpy(), predictions_b[f"{task}_pred"].to_numpy()

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs[i] = score(y_true[idx], pred_a[idx]) - score(y_true[idx], pred_b[idx])
    return {
        "task": task,
        "metric": metric,
        "f1_a": score(y_true, pred_a),
        "f1_b": score(y_true, pred_b),
        "delta": score(y_true, pred_a) - score(y_true, pred_b),
        "ci_low": np.percentile(diffs, 2.5),
        "ci_high": np.percentile(diffs, 97.5),
        "p_value": float((diffs <= 0).mean()),
    }


def subgroup_report(predictions, n_iter=1000, seed=0):
    rows = []
    for name, condition in SUBGROUPS:
        subset = predictions if condition is None else predictions[predictions[condition[0]] == condition[1]]
        for task in TASKS:
            stats = bootstrap(subset[f"{task}_label"], subset[f"{task}_pred"], n_iter, seed)
            rows.append({"sample": name, "n": len(subset), "task": task, **stats})
    return pd.DataFrame(rows)


def format_report(report):
    lines = []
    for row in report.itertuples():
        lines.append(
            f"{row.sample:<14} {row.n:>4}  {row.task:<8} "
            f"F1={row.f1 * 100:6.2f} ({row.ci_low * 100:.2f}, {row.ci_high * 100:.2f})  "
            f"QWK={row.qwk * 100:6.2f}  MAE={row.mae:.3f}  через класс={row.severe:.1%}"
        )
    return "\n".join(lines)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def ece(probs, labels, n_bins=15):
    """Expected calibration error: разрыв между уверенностью и точностью."""
    confidence, predicted = probs.max(1), probs.argmax(1)
    correct = (predicted == np.asarray(labels)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)[1:-1]
    bins = np.digitize(confidence, edges)
    return float(sum(
        (bins == b).mean() * abs(correct[bins == b].mean() - confidence[bins == b].mean())
        for b in range(n_bins) if (bins == b).any()
    ))


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(probs, labels, grid=np.logspace(-1, 1, 201)):
    """Температурное шкалирование по val. Для softmax-головы log p — это логиты
    с точностью до константы, которая в softmax сокращается. Для CORN это не логиты,
    а то же монотонное «размытие» распределения; F1 не меняется в обоих случаях.
    """
    logits = np.log(np.clip(probs, 1e-12, None))
    losses = [nll(softmax(logits / t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def apply_temperature(probs, temperature):
    return softmax(np.log(np.clip(probs, 1e-12, None)) / temperature)


def calibration_report(test_predictions, val_predictions):
    rows = []
    for task in TASKS:
        test_probs, val_probs = probabilities(test_predictions, task), probabilities(val_predictions, task)
        if test_probs is None or val_probs is None:
            raise ValueError("в predictions.csv нет колонок вероятностей — нужен прогон train.py заново")
        test_labels = test_predictions[f"{task}_label"].to_numpy()
        temperature = fit_temperature(val_probs, val_predictions[f"{task}_label"].to_numpy())
        scaled = apply_temperature(test_probs, temperature)
        rows.append({
            "task": task,
            "T": round(temperature, 3),
            "ECE": ece(test_probs, test_labels),
            "ECE_T": ece(scaled, test_labels),
            "NLL": nll(test_probs, test_labels),
            "NLL_T": nll(scaled, test_labels),
            "f1": macro_f1(test_labels, test_probs.argmax(1)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--compare", type=Path, default=None,
                        help="второй predictions.csv: парный bootstrap разницы вместо отчёта")
    parser.add_argument("--metric", choices=list(METRICS), default="f1", help="метрика для --compare")
    parser.add_argument("--calibrate", action="store_true",
                        help="ECE и температурное шкалирование по соседнему val_predictions.csv")
    parser.add_argument("--n-iter", type=int, default=1000)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)

    if args.compare:
        other = pd.read_csv(args.compare)
        print(f"A = {args.predictions.parent.name}\nB = {args.compare.parent.name}  "
              f"(n={len(predictions)}, метрика {args.metric})")
        for task in TASKS:
            row = paired_bootstrap(predictions, other, task, max(args.n_iter, 2000), metric=args.metric)
            print(f"{row['task']:<8} A={row['f1_a'] * 100:6.2f}  B={row['f1_b'] * 100:6.2f}  "
                  f"delta={row['delta'] * 100:+6.2f}  "
                  f"95% CI ({row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f})  "
                  f"p={row['p_value']:.3f}")
        return

    if args.calibrate:
        val_path = args.predictions.with_name("val_predictions.csv")
        if not val_path.exists():
            raise SystemExit(f"нет {val_path} — температура подбирается на val, а не на test")
        report = calibration_report(predictions, pd.read_csv(val_path))
        print(report.to_string(index=False))
        report.to_csv(args.predictions.with_name("calibration.csv"), index=False)
        return

    report = subgroup_report(predictions, args.n_iter)
    print(format_report(report))
    report.to_csv(args.predictions.with_name("report.csv"), index=False)
    print(json.dumps(report[report["sample"] == "General"][["task", "f1"]].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
