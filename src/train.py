import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import LABELS, REDACTION_SUFFIX, REDACTIONS, load_dataset, split_dataset
from evaluate import format_report, macro_f1, subgroup_report
from model import TASKS, TwoHeadClassifier, pick_device

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai-forever/ruRoberta-large")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--redact", choices=list(REDACTIONS), default="none",
                        help="что вырезать из текста: grade — явную оценку, дальше по лестнице")
    parser.add_argument("--group", choices=["movie_name", "author"], default=None,
                        help="групповой сплит: группа целиком уходит в одну выборку")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--head", choices=["softmax", "corn"], default="softmax")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--select-by", choices=["mean", "last"], default="last",
                        help="mean — брать эпоху с лучшим средним val macro-F1, last — последнюю, как в статье")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="не пересчитывать прогон, если в каталоге уже есть metrics.json")
    parser.add_argument("--tag", default=None, help="произвольный суффикс каталога прогона")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run_name(args):
    """Имя каталога: конфигурация по умолчанию даёт то же имя, что и раньше."""
    parts = [args.model.split("/")[-1], f"seed{args.seed}"]
    if args.split_seed != 42:
        parts.append(f"split{args.split_seed}")
    parts.append(REDACTION_SUFFIX.get(args.redact, f"_{args.redact}").lstrip("_"))
    if args.group:
        parts.append("group-" + args.group.replace("_name", ""))
    for value, default in [(args.pooling, "cls"), (args.head, "softmax"), (args.class_weights, "none")]:
        if value != default:
            parts.append(value)
    if args.select_by != "last":
        parts.append("best" + args.select_by)
    if args.tag:
        parts.append(args.tag)
    return "_".join(part for part in parts if part)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, max_length, batch_size, shuffle):
    def collate(rows):
        encoded = tokenizer(
            [row["content"] for row in rows],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = {task: torch.tensor([row[f"{task}_label"] for row in rows]) for task in TASKS}
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}, labels

    records = df[["content", "stance_label", "premise_label"]].to_dict("records")
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def class_weights(train_df, mode):
    if mode == "none":
        return None
    return {
        task: compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                                   y=train_df[f"{task}_label"].to_numpy())
        for task in TASKS
    }


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict(model, loader, device):
    """Вероятности классов по каждой задаче — из них же берутся метки argmax."""
    model.eval()
    batches = {task: [] for task in TASKS}
    for inputs, _ in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with autocast(device):
            logits = model(**inputs)
        for task, task_logits in zip(TASKS, logits):
            batches[task].append(model.probabilities(task_logits).cpu())
    return {task: torch.cat(value).numpy() for task, value in batches.items()}


def scores(df, probs):
    return {task: macro_f1(df[f"{task}_label"], probs[task].argmax(-1)) for task in TASKS}


def to_frame(df, probs):
    frame = df.drop(columns=["content"]).copy()
    for task in TASKS:
        frame[f"{task}_pred"] = probs[task].argmax(-1)
        for i, label in enumerate(LABELS):
            frame[f"{task}_p_{label}"] = probs[task][:, i]
    return frame


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    output_dir = args.output_dir or ROOT / "runs" / run_name(args)
    if args.skip_existing and (output_dir / "metrics.json").exists():
        print(f"пропуск: {output_dir.name} уже посчитан")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(redact=args.redact)
    train_df, val_df, test_df = split_dataset(dataset, args.split_seed, args.group)
    if args.limit:
        train_df, val_df, test_df = (d.head(args.limit) for d in (train_df, val_df, test_df))
    print(f"device={device} train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"redact={args.redact} group={args.group} head={args.head} -> {output_dir.name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader = make_loader(train_df, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_loader = make_loader(test_df, tokenizer, args.max_length, args.batch_size, shuffle=False)

    model = TwoHeadClassifier(args.model, pooling=args.pooling, head=args.head,
                              class_weights=class_weights(train_df, args.class_weights)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    history, best = [], None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            with autocast(device):
                loss = model.loss(model(**inputs), labels)
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
            running_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running_loss / step:.4f}", flush=True)
        val_probs = predict(model, val_loader, device)
        val_scores = scores(val_df, val_probs)
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader), "val": val_scores})
        print(f"epoch {epoch} val stance={val_scores['stance']:.4f} premise={val_scores['premise']:.4f}", flush=True)

        mean_score = sum(val_scores.values()) / len(val_scores)
        if best is None or mean_score > best["score"]:
            best = {"epoch": epoch, "score": mean_score, "val_probs": val_probs}
            if args.select_by == "mean":
                best["state"] = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    selected_epoch = best["epoch"] if args.select_by == "mean" else args.epochs
    if selected_epoch != args.epochs:
        print(f"выбрана эпоха {best['epoch']} (val mean={best['score']:.4f})")
        model.load_state_dict(best["state"])
        val_probs = best["val_probs"]

    to_frame(val_df, val_probs).to_csv(output_dir / "val_predictions.csv", index=False)
    test_probs = predict(model, test_loader, device)
    predictions = to_frame(test_df, test_probs)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    report = subgroup_report(predictions)
    report.to_csv(output_dir / "report.csv", index=False)
    print(format_report(report))

    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
        "run": output_dir.name,
        "train_minutes": (time.time() - start) / 60,
        "selected_epoch": selected_epoch,
        "history": history,
        "val": scores(val_df, val_probs),
        "test": scores(test_df, test_probs),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
